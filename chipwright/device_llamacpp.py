"""The llama.cpp-HTP board transport (spec 004 T-RT11) — drive the generative gate over SSH.

Unlike `RemoteQNN` (Genie/QNN context binaries), this runs the llama.cpp Hexagon backend: a GGUF, the
arch-matched `libggml-htp-<arch>.so`, and the binaries, all under an injected glibc loader (the container
build targets a newer glibc than the QDC board ships). It stages a runtime pack, pushes the GGUF, and runs
`llama-perplexity` on HTP0 and on CPU (`GGML_HEXAGON_NDEV=0`) — the two measurements the perplexity-
agreement gate compares. It NEVER runs `llama-cli` (free generation wedges the board).

Reuses the sshpass `_ssh`/`_scp` helpers from `device.py` so there is one SSH transport, not two.
"""
from __future__ import annotations

import os
import shlex

from . import cache
from .device import _require_sshpass, _scp, _ssh
from .runtimes import llamacpp_htp as lh


class LlamaHtpBoard:
    def __init__(self, host, user, pw, port=22, board_dir="/data/cw"):
        _require_sshpass()
        self.host, self.user, self.pw, self.port = host, user, pw, int(port)
        self.board_dir = board_dir
        self.dest = f"{user}@{host}"

    # --- board introspection (feeds host_preflight) ---------------------------------------------
    def probe(self) -> dict:
        """Return a capability dict for `llamacpp_htp.host_preflight` — what the route needs to run."""
        script = (
            "echo ARCH=$(cat /sys/devices/soc0/soc_id 2>/dev/null); "
            "test -e /usr/lib/libcdsprpc.so && echo CDSPRPC=1; "
            "ls /dev/fastrpc-cdsp* >/dev/null 2>&1 && echo FASTRPC=1; "
            f"ls {self.board_dir}/pkg-snapdragon/lib/libggml-htp-*.so 2>/dev/null; "
            f"ls {self.board_dir}/glibc238 2>/dev/null")
        r = _ssh(self.pw, self.port, self.dest, script, timeout=60)
        out = (r.stdout or "")
        import re
        soc = None
        m = re.search(r"ARCH=(\d+)", out)
        if m:
            from .env import SOC_ID
            soc = SOC_ID.get(int(m.group(1)), (None, None))[1]   # (name, arch) -> arch
        staged_htp = set(re.findall(r"libggml-htp-(v\d+)\.so", out))
        staged_loader = set(l for l in lh.REQUIRED_LOADER_LIBS if l in out)
        return {"has_fastrpc": "FASTRPC=1" in out, "has_cdsprpc": "CDSPRPC=1" in out,
                "htp_arch": soc, "staged_htp_libs": staged_htp, "staged_loader_libs": staged_loader,
                "raw": out}

    # --- staging --------------------------------------------------------------------------------
    def _exists(self, board_path: str) -> bool:
        r = _ssh(self.pw, self.port, self.dest, f"test -e {shlex.quote(board_path)} && echo OK", timeout=40)
        return "OK" in (r.stdout or "")

    def ensure_pack(self, pack_url: str, pack_sha: str) -> None:
        """Stage the runtime pack (pkg-snapdragon + glibc loader + wikitext corpus) if not already present.

        The pack is content-addressed+sha-verified into the host cache, scp'd once, and extracted. A
        stranger on a fresh v73 board gets the whole route staged from the hosted pack.
        """
        if self._exists(f"{self.board_dir}/pkg-snapdragon/lib"):
            return
        local = cache.fetch(pack_url, pack_sha)
        _ssh(self.pw, self.port, self.dest, f"mkdir -p {shlex.quote(self.board_dir)}", timeout=40)
        tar_board = f"{self.board_dir}/_pack.tgz"
        r = _scp(self.pw, self.port, local, f"{self.dest}:{tar_board}")
        if getattr(r, "returncode", 1) != 0:
            raise RuntimeError(f"failed to push runtime pack: {getattr(r, 'stderr', '')}")
        x = _ssh(self.pw, self.port, self.dest,
                 f"cd {shlex.quote(self.board_dir)} && tar xzf _pack.tgz && rm -f _pack.tgz", timeout=300)
        if not self._exists(f"{self.board_dir}/pkg-snapdragon/lib"):
            raise RuntimeError("runtime pack extracted but pkg-snapdragon/lib missing:\n" + (x.stdout or "")[-400:])

    def ensure_gguf(self, local_path: str, name: str) -> str:
        """Push the GGUF to the board if absent; return its board path."""
        board_path = f"{self.board_dir}/{name}"
        if not self._exists(board_path):
            r = _scp(self.pw, self.port, local_path, f"{self.dest}:{board_path}")
            if getattr(r, "returncode", 1) != 0:
                raise RuntimeError(f"failed to push GGUF: {getattr(r, 'stderr', '')}")
        return board_path

    # --- runs -----------------------------------------------------------------------------------
    def _paths(self, gguf_name: str):
        lib = f"{self.board_dir}/pkg-snapdragon/lib"
        return lib, f"{self.board_dir}/glibc238", f"{self.board_dir}/{gguf_name}", \
            f"{self.board_dir}/wikitext-2-raw/wiki.test.raw"

    def _env(self, lib_dir: str, ndev) -> str:
        e = f"ADSP_LIBRARY_PATH={shlex.quote(lib_dir)}"
        return e + " GGML_HEXAGON_NDEV=0" if ndev == 0 else e

    def perplexity(self, gguf_name: str, device="HTP0", ndev=None, chunks=16, timeout=1200):
        """Run llama-perplexity (HTP0 or CPU) and return (ppl, stdout)."""
        lib, loader, gguf, corpus = self._paths(gguf_name)
        argv = lh.perplexity_command(gguf, lib, loader, corpus, device=device, ndev=ndev, chunks=chunks)
        cmd = f"cd {shlex.quote(self.board_dir)} && {self._env(lib, ndev)} " + " ".join(shlex.quote(a) for a in argv)
        r = _ssh(self.pw, self.port, self.dest, cmd, timeout=timeout)
        out = (r.stdout or "") + (r.stderr or "")
        return lh.ppl_from_stdout(out), out

    def bench(self, gguf_name: str, device="HTP0", ndev=None, p=64, n=32, reps=2, timeout=600):
        """Run llama-bench (for the dispatch proof) and return stdout."""
        lib, loader, gguf, _ = self._paths(gguf_name)
        argv = lh.bench_command(gguf, lib, loader, device=device, ndev=ndev, p=p, n=n, r=reps)
        cmd = f"cd {shlex.quote(self.board_dir)} && {self._env(lib, ndev)} " + " ".join(shlex.quote(a) for a in argv)
        r = _ssh(self.pw, self.port, self.dest, cmd, timeout=timeout)
        return (r.stdout or "") + (r.stderr or "")

    # --- the agent gate (T-AG07b): push a prompt + generate one guarded tool call --------------------
    def push_text(self, text: str, name: str) -> str:
        """Write a prompt text to a board file (scp from a host temp file) and return its board path."""
        import tempfile
        board_path = f"{self.board_dir}/{name}"
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(text)
            local = f.name
        try:
            _ssh(self.pw, self.port, self.dest, f"mkdir -p {shlex.quote(self.board_dir)}", timeout=40)
            r = _scp(self.pw, self.port, local, f"{self.dest}:{board_path}")
            if getattr(r, "returncode", 1) != 0:
                raise RuntimeError(f"failed to push prompt: {getattr(r, 'stderr', '')}")
        finally:
            os.unlink(local)
        return board_path

    def generate(self, gguf_name: str, prompt_board_path: str, device="HTP0", ndev=None,
                 n_predict=200, ctx=4096, timeout=240):
        """Run ONE guarded llama-cli generation (HTP0 or CPU) and return (tool_call_raw, stdout).

        The guarded single-turn invocation does not wedge the board (proven on silicon). `ndev=0` => the CPU
        reference of the SAME GGUF. The parsed `<tool_call>` payload (or None = abstained) is returned for
        the gate; the full stdout is returned too for the record / debugging."""
        from . import verify_llm
        lib, loader, gguf, _ = self._paths(gguf_name)
        argv = lh.generate_command(gguf, lib, loader, prompt_board_path, device=device, ndev=ndev,
                                   n_predict=n_predict, ctx=ctx)
        inner = max(30, int(timeout) - 10)                 # belt-and-braces: a hard timeout under the SSH one
        cmd = (f"cd {shlex.quote(self.board_dir)} && {self._env(lib, ndev)} timeout {inner} "
               + " ".join(shlex.quote(a) for a in argv))
        r = _ssh(self.pw, self.port, self.dest, cmd, timeout=timeout)
        out = (r.stdout or "") + (r.stderr or "")
        return verify_llm.extract_tool_call(out), out
