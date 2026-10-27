"""The llama.cpp-HTP runtime adapter (spec 004 T-RT07; FR-RT06 / SC-RT03).

This is the route Genie structurally lacks: a >4-bit integer quant (Q8_0, Q4_0) dispatched to the Hexagon
HTP through the llama.cpp Hexagon backend. Classic Genie is w4a16-only; llama.cpp-HTP runs Q8_0 MUL_MAT on
the NPU on v73+, which is the quality/prefill win.

Two pieces are the adapter's contract, and both are pure + offline-testable:

  * `host_preflight(probe)` — refuse-by-name if the board lacks what the route needs (fastrpc / libcdsprpc,
    the matching-arch HTP lib, the staged glibc loader). The llama.cpp-HTP binaries are built against a
    newer glibc than the QDC board ships, so they run under an injected loader (`ld-linux-aarch64.so.1
    --library-path <staged-glibc>:<pkg-lib>:...`). If the loader or the arch lib isn't staged, we say so by
    name rather than letting the dynamic linker emit a raw error on-device.

  * `dispatch_proof(stdout)` — THE gate (SC-RT03). "It ran" is not "it ran on the NPU": llama.cpp silently
    falls back to CPU when the Hexagon registry is disabled (`GGML_HEXAGON_NDEV=0`) or a session can't open.
    This parses the ggml-hex log + the llama-bench table and returns `dispatched=True` ONLY when an HTP0
    session was opened on the compute-DSP (`_dom=cdsp`) AND a bench row actually executed on device `HTP0`.
    The CPU-fallback output (ndev 0, no HTP0 session, no `dev` column) returns `dispatched=False` — the gate
    distinguishes the two, the way the CV gate must FAIL random-noise.

SILICON_RECORD is the proof this adapter is wired on (Principle III: tested-is-a-record, not a badge).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# --- the proven on-silicon record (Principle I + III) ------------------------------------------------
# Re-verified on a fresh v73 board through this adapter's recipe on 2026-10-24 (Track 2). The dispatch
# proof and the PPL both reproduce the earlier sweep: PPL identical to 4 decimals, prefill speedup even
# larger on this part. This is what promotes the Q8_0-HTP variant from "proven@sweep" to silicon-backed.
SILICON_RECORD = {
    "runtime": "llamacpp-htp",
    "model": "qwen2.5-1.5b",
    "precision": "Q8_0",
    "board": "QCS9075 (Dragonwing IQ-9075)",
    "soc_id": 676,
    "htp_arch": "v73",
    "date": "2026-10-24",
    "gguf_sha256": "cedde4175edae2e9467d8b98d70843e57c8fb101afd54ca773bce022e80f7e4e",
    "llamacpp_build": "3278e921b",
    "dispatch": {                      # SC-RT03: Q8_0 MUL_MAT on HTP0, not CPU fallback
        "htp0_prefill_tps": 592.82,
        "cpu_prefill_tps": 217.55,
        "prefill_speedup": 2.72,
        "htp0_decode_tps": 23.51,      # decode regresses vs CPU on a model this small (memory-bound) — recorded, not hidden
        "cpu_decode_tps": 28.99,
    },
    "quality": {                       # near-lossless vs the model's own f16 (11.6388)
        "ppl_wikitext_q8_0": 11.6631,
        "ppl_stderr": 0.49427,
        "chunks": 16,
    },
    # The generative gate, driven BY chipwright on silicon (QCS8550 / kalama, soc_id 603, v73, 2026-10-25):
    # llama-perplexity on HTP0 vs CPU (GGML_HEXAGON_NDEV=0) of the SAME Q8_0 GGUF. Identical weights, only
    # the backend differs — they agree to the digit, isolating the backend (no chip bug).
    "gate": {
        "ppl_htp": 11.6631,
        "ppl_cpu": 11.6631,
        "backend_delta_pct": 0.0,      # |htp - cpu| / cpu — the gate (house rule <= 1.0%)
        "ppl_f16_baseline": 11.6388,
        "lossless_delta_pct": 0.209,   # recorded, not gated
        "board": "QCS8550 / kalama (soc_id 603, v73)",
        "date": "2026-10-25",
    },
    # dispatch also proven on QCS9075 (soc_id 676, v73): prefill 592.82 vs 217.55 = 2.72×.
}

# The glibc the container-built binaries need, staged next to the package (loader injection).
LOADER = "ld-linux-aarch64.so.1"
REQUIRED_LOADER_LIBS = ("ld-linux-aarch64.so.1", "libc.so.6", "libstdc++.so.6", "libm.so.6")


@dataclass
class DispatchProof:
    """The verdict of the on-device dispatch check (SC-RT03)."""
    dispatched: bool
    arch: Optional[str] = None            # 'v73' etc., from the ggml-hex log
    ndev: Optional[int] = None            # Hexagon registry device count (0 => CPU-only)
    on_cdsp: bool = False                 # an HTP0 session opened on the compute-DSP domain
    device_rows: List[Dict] = field(default_factory=list)   # bench rows that ran on dev=HTP0
    reason: str = ""

    def as_record(self) -> Dict:
        return {"dispatched": self.dispatched, "arch": self.arch, "ndev": self.ndev,
                "on_cdsp": self.on_cdsp, "device_rows": self.device_rows, "reason": self.reason}


_ARCH_RE = re.compile(r"Hexagon Arch version (v\d+)")
_NDEV_RE = re.compile(r"\bndev (\d+)\b")
_SESSION_RE = re.compile(r"HTP0 new session")
_CDSP_RE = re.compile(r"_dom=cdsp")


def _parse_bench_rows(stdout: str) -> List[Dict]:
    """Parse the llama-bench markdown table; return one dict per data row with its columns.

    The table has a header row, a separator row, then data rows, all `|`-delimited. The HTP build names a
    `dev` column only when a Hexagon device is in play; CPU-fallback output omits it. We key rows by header.
    """
    header: Optional[List[str]] = None
    rows: List[Dict] = []
    for line in stdout.splitlines():
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):      # the |---|---| separator
            continue
        if header is None:
            header = cells
            continue
        if len(cells) == len(header):
            rows.append(dict(zip(header, cells)))
    return rows


def dispatch_proof(stdout: str) -> DispatchProof:
    """THE gate (SC-RT03). Return a DispatchProof: did Q8_0 actually run on the HTP, or fall back to CPU?

    `dispatched` is True only when ALL of: the Hexagon registry had a device (ndev >= 1), an HTP0 session
    opened on the compute-DSP (`_dom=cdsp`), and at least one bench row executed on device `HTP0`. Any one
    missing => CPU fallback (or no run), and we say which. This is pure: feed it captured stdout.
    """
    arch_m = _ARCH_RE.search(stdout)
    ndev_m = _NDEV_RE.search(stdout)
    arch = arch_m.group(1) if arch_m else None
    ndev = int(ndev_m.group(1)) if ndev_m else None
    on_cdsp = bool(_SESSION_RE.search(stdout) and _CDSP_RE.search(stdout))

    rows = _parse_bench_rows(stdout)
    # a device column may be named 'dev' or 'device'
    def devcell(r: Dict) -> str:
        return (r.get("dev") or r.get("device") or "").strip()
    device_rows = [r for r in rows if devcell(r).upper() == "HTP0"]

    if ndev == 0:
        return DispatchProof(False, arch, ndev, on_cdsp, device_rows,
                             "Hexagon registry disabled (ndev 0) — ran on CPU, not the NPU")
    if not on_cdsp:
        return DispatchProof(False, arch, ndev, on_cdsp, device_rows,
                             "no HTP0 session opened on the compute-DSP (_dom=cdsp) — not dispatched to the NPU")
    if not device_rows:
        return DispatchProof(False, arch, ndev, on_cdsp, device_rows,
                             "HTP session opened but no bench row executed on device HTP0 — nothing dispatched")
    tests = ", ".join(f"{r.get('test','?')}={r.get('t/s','?')}t/s" for r in device_rows)
    return DispatchProof(True, arch, ndev, on_cdsp, device_rows,
                         f"Q8_0 dispatched to HTP0 on Hexagon {arch} ({tests})")


def host_preflight(probe: Dict) -> Tuple[bool, Optional[str]]:
    """Refuse-by-name if the board can't host the llama.cpp-HTP route. Pure: `probe` is a board capability dict.

    Expected keys: `has_cdsprpc` (bool — THE requirement: the fastrpc user driver llama.cpp loads),
    `htp_arch` (str, e.g. 'v73'), `staged_htp_libs` (iterable of arch tags present, e.g. {'v73'}),
    `staged_loader_libs` (iterable of loader lib filenames staged). `has_fastrpc` (the `/dev/fastrpc*`
    device nodes) is informational only — proven on silicon that a board with libcdsprpc but no
    `/dev/fastrpc*` nodes (QCS8550) still dispatches to HTP0, so it is NOT gated on. Returns (ok, reason).
    """
    if not probe.get("has_cdsprpc"):
        return False, ("runtime=llamacpp-htp needs libcdsprpc.so on the board to reach the Hexagon DSP "
                       "(the fastrpc user driver llama.cpp loads); this image lacks it")
    arch = probe.get("htp_arch")
    staged = set(probe.get("staged_htp_libs") or ())
    if arch not in staged:
        return False, (f"runtime=llamacpp-htp needs libggml-htp-{arch}.so staged for this board's arch "
                       f"({arch}); staged: {sorted(staged) or 'none'}")
    missing = [lib for lib in REQUIRED_LOADER_LIBS if lib not in set(probe.get("staged_loader_libs") or ())]
    if missing:
        return False, (f"runtime=llamacpp-htp binaries need an injected glibc loader; missing staged libs: "
                       f"{missing}")
    return True, None


def _loader_prefix(lib_dir: str, loader_dir: str) -> List[str]:
    return [f"{loader_dir}/{LOADER}", "--library-path", f"{loader_dir}:{lib_dir}:/usr/lib:/lib"]


def bench_command(gguf: str, lib_dir: str, loader_dir: str, *, device: str = "HTP0",
                  ndev: Optional[int] = None, p: int = 64, n: int = 32, r: int = 2) -> List[str]:
    """The exact loader-injected llama-bench argv proven on silicon. `ndev=0` forces the CPU reference.

    Grounded in the verified run:
        ADSP_LIBRARY_PATH=<lib_dir> <loader_dir>/ld-linux-aarch64.so.1 \\
          --library-path <loader_dir>:<lib_dir>:/usr/lib:/lib \\
          <bin>/llama-bench -m <gguf> [--device HTP0] -p <p> -n <n> -r <r>
    The caller sets ADSP_LIBRARY_PATH and (for CPU) GGML_HEXAGON_NDEV in the environment.
    """
    bin_dir = lib_dir.rsplit("/lib", 1)[0] + "/bin"
    argv = _loader_prefix(lib_dir, loader_dir) + [f"{bin_dir}/llama-bench", "-m", gguf]
    if ndev != 0:
        argv += ["--device", device]
    argv += ["-p", str(p), "-n", str(n), "-r", str(r)]
    return argv


def perplexity_command(gguf: str, lib_dir: str, loader_dir: str, corpus: str, *,
                       device: str = "HTP0", ndev: Optional[int] = None, chunks: int = 16) -> List[str]:
    """The loader-injected llama-perplexity argv (the generative gate's measurement). `ndev=0` => CPU ref.

    Uses teacher-forced perplexity — NOT `llama-cli` free generation, which wedges the board. `--device
    HTP0` (default) measures on the NPU; `ndev=0` (GGML_HEXAGON_NDEV=0 in the env) measures the identical
    GGUF on CPU, so the two perplexities isolate the backend.
    """
    bin_dir = lib_dir.rsplit("/lib", 1)[0] + "/bin"
    argv = _loader_prefix(lib_dir, loader_dir) + [f"{bin_dir}/llama-perplexity",
                                                  "-m", gguf, "-f", corpus, "--chunks", str(chunks)]
    if ndev != 0:
        argv += ["--device", device]
    return argv


def generate_command(gguf: str, lib_dir: str, loader_dir: str, prompt_file: str, *,
                     device: str = "HTP0", ndev: Optional[int] = None,
                     n_predict: int = 200, ctx: int = 4096) -> List[str]:
    """The loader-injected `llama-cli` argv for ONE guarded tool-call generation (the agent gate, T-AG07b).

    The guard flags (`-no-cnv -st --temp 0 -s 0 --special --no-display-prompt`, greedy) are what make
    llama-cli NOT wedge the board — the wedge was interactive/conversation mode, proven on silicon. `--device
    HTP0` measures on the NPU; `ndev=0` (GGML_HEXAGON_NDEV=0 in the env) generates from the SAME GGUF on CPU,
    so the two generations isolate the backend. The caller wraps this in `timeout` as a belt-and-braces guard.
    """
    bin_dir = lib_dir.rsplit("/lib", 1)[0] + "/bin"
    argv = _loader_prefix(lib_dir, loader_dir) + [
        f"{bin_dir}/llama-cli", "-m", gguf, "-f", prompt_file,
        "-no-cnv", "-st", "--temp", "0", "-s", "0", "--special", "--no-display-prompt",
        "-n", str(n_predict), "-c", str(ctx)]
    if ndev != 0:
        argv += ["--device", device, "-ngl", "99"]
    return argv


_PPL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9]+\.[0-9]+)")


def ppl_from_stdout(stdout: str) -> Optional[float]:
    """Parse `Final estimate: PPL = 11.6631` from llama-perplexity output; None if absent (so a crashed or
    wedged run never reads as a PPL of 0)."""
    m = _PPL_RE.search(stdout or "")
    return float(m.group(1)) if m else None
