"""Remote QNN runner — push a context binary to the board and execute it over SSH.

The device transport the `run` verb uses. It is deliberately quantization-agnostic: it moves raw
bytes in the graph's native dtype and runs `qnn-net-run --retrieve_context`. Quant/dequant belongs
to the modality runner, which owns the artifact's I/O scales. (Generalized from the proven
board-side driver; here it drives the board from the host.)
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import tempfile
import time

import numpy as np

_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def _safe_board_name(name):
    """`board_name` comes from the registry and is interpolated into on-board shell commands (and used
    as a board path), so it MUST be a plain identifier. A name with a shell metacharacter, space, slash
    or `..` is refused — otherwise a crafted registry entry could run arbitrary commands as root on the
    board (`board_name = 'x; curl … | sh'`)."""
    if not isinstance(name, str) or name in (".", "..") or not _SAFE_NAME.match(name):
        raise RuntimeError(
            f"unsafe board_name {name!r}: must match [A-Za-z0-9._-]+ (it is interpolated into on-board "
            f"shell commands — a separator or shell metacharacter is refused)")
    return name


def _require_sshpass():
    """`sshpass` is a system binary (not a pip package) the SSH transport shells out to.

    Fail here, by name, with the install hint — not later with a cryptic FileNotFoundError from
    subprocess — so a missing system dependency is reported the way the framework reports everything:
    plainly, naming the fix.
    """
    if shutil.which("sshpass") is None:
        raise RuntimeError(
            "chipwright run needs the `sshpass` system binary to reach the board over SSH, and it "
            "is not on PATH. Install it: `brew install hudochenkov/sshpass/sshpass` (macOS) or "
            "`apt-get install sshpass` (Debian/Ubuntu).")


# Connection-failure signatures: a flaky board/tunnel (or a dropped QDC double-tunnel) manifests as ssh's
# own exit 255 and/or one of these in stderr — distinct from a remote COMMAND that ran and returned non-zero.
_SSH_CONN_ERR = ("connection timed out", "banner exchange", "connection refused", "connection reset",
                 "lost connection", "connection closed", "broken pipe", "no route to host",
                 "operation timed out", "timed out", "kex_exchange_identification")


def _ssh(pw, port, dest, *cmd, timeout=600, capture=True, retries=3):
    """Run one SSH command on the board. Resilient to a transient tunnel/board blip: keepalives hold a long
    single call (e.g. a multi-minute perplexity run) open, and a CONNECTION failure (ssh exit 255 or a
    connection-error signature — NOT a remote command's non-zero exit) is retried with backoff. A gate makes
    many sequential board calls; one dropped connection must not fail the whole gate. Caught on a flaky QDC
    double-tunnel; the same resilience matters for a real user's board over a shaky link."""
    base = ["sshpass", "-p", pw, "ssh", "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=20",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
            "-o", "PreferredAuthentications=password", "-o", "PubkeyAuthentication=no",
            "-p", str(port), dest, *cmd]
    last = None
    for attempt in range(max(1, retries)):
        try:
            r = subprocess.run(base, capture_output=capture, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            last = e
            if attempt + 1 < retries:
                time.sleep(2 * (attempt + 1))
                continue
            raise
        err = ((r.stderr or "") + (r.stdout or "")).lower() if capture else ""
        conn_failed = r.returncode == 255 or any(s in err for s in _SSH_CONN_ERR)
        if conn_failed and attempt + 1 < retries:
            last = r
            time.sleep(2 * (attempt + 1))
            continue
        return r
    return last


def _scp(pw, port, *args, timeout=1800):
    base = ["sshpass", "-p", pw, "scp", "-P", str(port), "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=20",
            "-o", "PreferredAuthentications=password", "-o", "PubkeyAuthentication=no", *args]
    return subprocess.run(base, capture_output=True, text=True, timeout=timeout)


class RemoteQNN:
    def __init__(self, host, user, pw, port=22, backend="/usr/lib/libQnnHtp.so",
                 board_dir="/opt/chipwright"):
        _require_sshpass()
        self.host, self.user, self.pw, self.port = host, user, pw, int(port)
        self.backend, self.board_dir = backend, board_dir
        self.dest = f"{user}@{host}"

    def _retry(self, fn, tries=4, wait=3):
        last = None
        for i in range(tries):
            try:
                r = fn()
                if getattr(r, "returncode", 0) == 0:
                    return r
                last = r
            except Exception as e:  # pragma: no cover
                last = e
            time.sleep(wait)
        return last

    def ensure_ctx(self, local_path, name, ext=None):
        """Push an artifact to the board and return a runnable context-binary path (cached).

        A DLC is the durable artifact; the HTP context binary is per-target, so a `.dlc` is compiled
        into a context binary ON the board (2 MB VTCM) — regenerate, don't ship brittle. A prebuilt
        context binary (`.bin`/`.qnnctx`) is pushed as-is.

        `ext` is the registry-declared artifact kind ("dlc"/"bin"/…). It is authoritative: a hosted
        artifact is fetched into the content-addressed cache under a HASHED filename with no extension,
        so inferring "is this a DLC?" from `local_path` wrongly treats a fetched DLC as a prebuilt
        context binary (skips context-gen → "Could not create context from binary" on-device). When
        `ext` is given we trust it; only when it is absent do we fall back to the filename suffix.
        """
        name = _safe_board_name(name)  # registry-sourced; confine before it reaches any shell string
        self._retry(lambda: _ssh(self.pw, self.port, self.dest, f"mkdir -p {self.board_dir}", timeout=40))
        is_dlc = (ext == "dlc") if ext is not None else local_path.endswith(".dlc")
        if is_dlc:
            dlc_board = f"{self.board_dir}/{name}"
            ctx_board = f"{self.board_dir}/ctx_{name}/{name}_ctx.bin"
            present = _ssh(self.pw, self.port, self.dest, f"test -f {ctx_board} && echo OK", timeout=40)
            if "OK" not in (present.stdout or ""):
                r = self._retry(lambda: _scp(self.pw, self.port, local_path, f"{self.dest}:{dlc_board}"))
                if getattr(r, "returncode", 1) != 0:
                    raise RuntimeError(f"failed to push DLC: {getattr(r,'stderr','')}")
                env = "export ADSP_LIBRARY_PATH=/usr/lib/rfsa/adsp:/usr/lib LD_LIBRARY_PATH=/usr/lib"
                gen = (f"cd {self.board_dir} && {env} && rm -rf ctx_{name} && "
                       f"qnn-context-binary-generator --backend {self.backend} "
                       f"--model /usr/lib/libQnnModelDlc.so --dlc_path {name} "
                       f"--binary_file {name}_ctx --output_dir ctx_{name}")
                g = self._retry(lambda: _ssh(self.pw, self.port, self.dest, gen, timeout=1200))
                check = _ssh(self.pw, self.port, self.dest, f"test -f {ctx_board} && echo OK", timeout=40)
                if "OK" not in (check.stdout or ""):
                    raise RuntimeError("on-board context-gen failed:\n" + (getattr(g, "stdout", "") or "")[-500:])
            return ctx_board
        board_path = f"{self.board_dir}/{name}"
        present = _ssh(self.pw, self.port, self.dest, f"test -f {board_path} && echo OK", timeout=40)
        if "OK" not in (present.stdout or ""):
            r = self._retry(lambda: _scp(self.pw, self.port, local_path, f"{self.dest}:{board_path}"))
            if getattr(r, "returncode", 1) != 0:
                raise RuntimeError(f"failed to push context binary: {getattr(r,'stderr','')}")
        return board_path

    def run(self, ctx_board_path, feeds_native, outputs, native_io=True):
        """feeds_native: {name: ndarray}. outputs: [{name,dtype,shape}]. Returns {name: ndarray[shape]}.

        native_io=True: raws are the graph's native quantized dtype (the modality quantizes/dequantizes).
        native_io=False: feed float32, let qnn-net-run quantize the input and dequantize the output —
        the simple path for graphs with float I/O semantics (e.g. an image classifier).
        """
        dt = {"uint16": np.uint16, "int32": np.int32, "float32": np.float32}
        with tempfile.TemporaryDirectory() as td:
            parts = []
            for n, a in feeds_native.items():
                np.ascontiguousarray(a).tofile(f"{td}/{n}.raw")
                parts.append(f"{n}:={n}.raw")
            open(f"{td}/il.txt", "w").write(" ".join(parts) + "\n")
            rdir = f"{self.board_dir}/req_{int(time.time()*1000)}"
            self._retry(lambda: _ssh(self.pw, self.port, self.dest, f"mkdir -p {rdir}", timeout=40))
            files = [f"{td}/{f}" for f in os.listdir(td)]
            r = self._retry(lambda: _scp(self.pw, self.port, *files, f"{self.dest}:{rdir}/"))
            if getattr(r, "returncode", 1) != 0:
                raise RuntimeError("failed to push inputs")
            env = "export ADSP_LIBRARY_PATH=/usr/lib/rfsa/adsp:/usr/lib LD_LIBRARY_PATH=/usr/lib"
            flags = " --use_native_input_files --use_native_output_files" if native_io else ""
            cmd = (f"cd {rdir} && {env} && qnn-net-run --backend {self.backend} "
                   f"--retrieve_context {ctx_board_path} --input_list il.txt --output_dir out" + flags)
            run = self._retry(lambda: _ssh(self.pw, self.port, self.dest, cmd, timeout=600))
            if getattr(run, "returncode", 1) != 0:
                _ssh(self.pw, self.port, self.dest, f"rm -rf {rdir}", timeout=30)
                raise RuntimeError("qnn-net-run failed:\n" + (getattr(run, "stdout", "") or "")[-500:])
            self._retry(lambda: _scp(self.pw, self.port, "-r", f"{self.dest}:{rdir}/out", td))
            _ssh(self.pw, self.port, self.dest, f"rm -rf {rdir}", timeout=30)
            res = {}
            for spec in outputs:
                nm, d, sh = spec["name"], dt[spec.get("dtype", "float32")], tuple(spec["shape"])
                fs = (glob.glob(f"{td}/out/**/{nm}_native.raw", recursive=True)
                      or glob.glob(f"{td}/out/**/{nm}.raw", recursive=True)
                      or glob.glob(f"{td}/out/**/{nm}*.raw", recursive=True))
                if not fs:
                    raise RuntimeError(f"missing output {nm}")
                res[nm] = np.fromfile(fs[0], d).reshape(sh)
            return res
