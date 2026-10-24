"""Board probe → TargetKey. The thing the resolver matches artifacts against.

On Qualcomm silicon the question that decides everything is *which chip* — specifically the Hexagon
architecture, the installed QAIRT version, and the SoC. `chipwright doctor` reads those off the board (over
SSH/ADB) rather than guessing. This mirrors the `qualcomm-env-discovery` skill's `probe-env.sh`, kept
self-contained here so chipwright-core has no hard dependency on a skill checkout.

Offline (no board reachable) is a first-class state, not an error: resolution can still run against a
TargetKey the user names by hand (`--target htpv68,qnn2.37`).
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Optional

# soc_id -> (marketing name, Hexagon HTP arch). Extend as boards are added.
SOC_ID = {
    498: ("QCS6490 / RB3 Gen2", "v68"),
    35:  ("QCS6490 (Advantech)", "v68"),
    30:  ("QCS8250 / RB5", "v66"),
    57:  ("QCS8550", "v73"),
    69:  ("QCS8650", "v75"),
    # QDC fleet soc_ids (verified on-device via `chipwright doctor`):
    603: ("QCS8550 / kalama", "v73"),
    676: ("QCS9075 / Dragonwing IQ-9075", "v73"),  # verified on-silicon 2026-10-24 (libQnnHtpV73Skel + _dom=cdsp)
    660: ("SM8850 / Snapdragon 8 Elite Gen 5", "v81"),
    # Snapdragon 8 Elite (SM8750) — Hexagon v79, the arch the LLM path reaches over adb. The arch is the
    # load-bearing fact (it is what fails at load and the resolver's hard axis); the on-device lib-name
    # fallback in `probe`/`probe_adb` recovers it even if this marketing soc_id drifts across images.
    618: ("SM8750 / Snapdragon 8 Elite", "v79"),
}


@dataclass(frozen=True)
class TargetKey:
    """What an artifact must be compatible with to load on this board."""
    htp_arch: str                 # e.g. "v68" — the axis that most often fails at load
    qairt: Optional[str] = None   # on-device QAIRT version, e.g. "2.37.1"
    soc_id: Optional[int] = None
    soc_name: Optional[str] = None
    reachable: bool = False
    qnn_libs: tuple = field(default_factory=tuple)   # e.g. ("libQnnHtpV68.so",)
    thermal_c: Optional[float] = None
    ram_kb: Optional[int] = None    # MemTotal from the board — an LLM bundle's load headroom check

    def __str__(self) -> str:
        sdk = f", qnn{self.qairt}" if self.qairt else ""
        soc = f" [{self.soc_name}]" if self.soc_name else ""
        return f"htp{self.htp_arch}{sdk}{soc}"

    @classmethod
    def manual(cls, spec: str) -> "TargetKey":
        """Build from a hand-typed spec like 'htpv68,qnn2.37' — for offline resolution."""
        arch, qairt = None, None
        for tok in spec.replace(" ", "").split(","):
            if tok.startswith("htp"):
                arch = tok[3:]
            elif tok.startswith("qnn"):
                qairt = tok[3:]
        if not arch:
            raise ValueError(f"target spec must name an arch, e.g. 'htpv68': got {spec!r}")
        return cls(htp_arch=arch, qairt=qairt)


def _arch_from_libs(libs) -> Optional[str]:
    """Recover the Hexagon arch from the on-device HTP lib names, or None when it is ambiguous.

    A board carrying one `libQnnHtpV<NN>.so` answers the arch (libQnnHtpV68.so -> v68). We strip any
    suffix (…Stub, …CalculatorStub) so we never emit a garbage arch like "v68CalculatorStub", and we
    return None unless *every* HTP arch lib agrees on one version — a board with multiple arch stubs is
    ambiguous and we must NOT guess. Shared by the SSH probe and the adb probe so the hardened fallback
    behaves identically on both transports.
    """
    found = set()
    for lib in libs:
        if lib.startswith("libQnnHtpV"):
            digits = ""
            for ch in lib[len("libQnnHtpV"):]:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            if digits:
                found.add("v" + digits)
    return found.pop() if len(found) == 1 else None


def _ssh_opts(pw):
    return ["sshpass", "-p", pw, "ssh", "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=15",
            "-o", "PreferredAuthentications=password", "-o", "PubkeyAuthentication=no"]


def probe(host: str, user: str, pw: str, port: int = 22) -> TargetKey:
    """SSH to a board and read SoC id, Hexagon arch, on-device QNN libs, temperature.

    Returns a TargetKey with reachable=False (and whatever it could read) if the board is down.
    """
    dest = f"{user}@{host}"
    script = (
        "echo SOC=$(cat /sys/devices/soc0/soc_id 2>/dev/null); "
        "echo NETRUN=$(command -v qnn-net-run); "
        "echo LIBS=$(ls /usr/lib/libQnnHtpV*.so 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\\n' ' '); "
        # QAIRT version from the device runtime: 'QNN SDK v2.38.0.<build>' -> 2.38.0 (printed to stderr)
        "echo QAIRT=$(qnn-net-run --version 2>&1 | grep -oE 'v[0-9]+\\.[0-9]+\\.[0-9]+' | head -n 1 | tr -d v); "
        "echo TEMP=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null)"
    )
    try:
        r = subprocess.run(_ssh_opts(pw) + ["-p", str(port), dest, script],
                           capture_output=True, text=True, timeout=30)
    except Exception:
        return TargetKey(htp_arch="unknown", reachable=False)
    if r.returncode != 0:
        return TargetKey(htp_arch="unknown", reachable=False)

    out = dict(line.split("=", 1) for line in r.stdout.strip().splitlines() if "=" in line)
    soc_id = int(out["SOC"]) if out.get("SOC", "").strip().isdigit() else None
    soc_name, arch = SOC_ID.get(soc_id, (None, "unknown"))
    libs = tuple(out.get("LIBS", "").split()) if out.get("LIBS", "").strip() else ()
    # arch fallback from the on-device lib name (libQnnHtpV68.so -> v68) — only when unambiguous.
    if arch == "unknown":
        arch = _arch_from_libs(libs) or arch
    temp = None
    if out.get("TEMP", "").strip().isdigit():
        temp = round(int(out["TEMP"]) / 1000.0, 1)
    qairt = out.get("QAIRT", "").strip() or None
    return TargetKey(htp_arch=arch, qairt=qairt, soc_id=soc_id, soc_name=soc_name,
                     reachable=True, qnn_libs=libs, thermal_c=temp)


# --- adb (Android) probe ------------------------------------------------------------------------
# A Snapdragon *phone*-class target (the LLM path) is reached over `adb`, not SSH. The question is the
# same — which chip, which arch, which QAIRT — so probe_adb mirrors `probe` and returns the same
# TargetKey; only the transport (an `adb -s <serial> shell …`) and the extra RAM read differ.

def _adb_base(serial, adb_bin="adb"):
    return [adb_bin] + (["-s", serial] if serial else [])


def probe_adb(serial: Optional[str] = None, adb_bin: str = "adb") -> TargetKey:
    """adb to a board and read SoC id, Hexagon arch, on-device QNN libs, RAM, temperature.

    The adb parallel of `probe`. Returns a TargetKey with reachable=False (and whatever it could read)
    if the board is down or `adb` is missing — offline is a first-class state, not an error, exactly as
    on the SSH path. RAM (`MemTotal`) is read too: an LLM Genie bundle is GBs of context binaries, so the
    load-headroom number is worth carrying in the key.
    """
    script = (
        "echo SOC=$(cat /sys/devices/soc0/soc_id 2>/dev/null); "
        "echo NETRUN=$(command -v qnn-net-run); "
        "echo LIBS=$(ls /usr/lib/libQnnHtpV*.so 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\\n' ' '); "
        "echo QAIRT=$(qnn-net-run --version 2>&1 | grep -oE 'v[0-9]+\\.[0-9]+\\.[0-9]+' | head -n 1 | tr -d v); "
        "echo RAM=$(grep -oE '[0-9]+' /proc/meminfo 2>/dev/null | head -n 1); "
        "echo TEMP=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null)"
    )
    try:
        r = subprocess.run(_adb_base(serial, adb_bin) + ["shell", script],
                           capture_output=True, text=True, timeout=30)
    except Exception:
        return TargetKey(htp_arch="unknown", reachable=False)
    if r.returncode != 0:
        return TargetKey(htp_arch="unknown", reachable=False)

    out = dict(line.split("=", 1) for line in r.stdout.strip().splitlines() if "=" in line)
    soc_id = int(out["SOC"]) if out.get("SOC", "").strip().isdigit() else None
    soc_name, arch = SOC_ID.get(soc_id, (None, "unknown"))
    libs = tuple(out.get("LIBS", "").split()) if out.get("LIBS", "").strip() else ()
    if arch == "unknown":
        arch = _arch_from_libs(libs) or arch
    temp = None
    if out.get("TEMP", "").strip().isdigit():
        temp = round(int(out["TEMP"]) / 1000.0, 1)
    ram_kb = int(out["RAM"]) if out.get("RAM", "").strip().isdigit() else None
    qairt = out.get("QAIRT", "").strip() or None
    return TargetKey(htp_arch=arch, qairt=qairt, soc_id=soc_id, soc_name=soc_name,
                     reachable=True, qnn_libs=libs, thermal_c=temp, ram_kb=ram_kb)


# --- skel-load capability -----------------------------------------------------------------------
# Some device images (observed: SM8750/v79 on QDC) block loading the unsigned DSP skel — a trivial
# backend init dies with "Failed to load skel, error 4000" and the LLM simply cannot run there. This is
# a *device-capability* fact, not a model fault: we detect it up front and report a clean FAIL (naming
# the axis) instead of letting a run hang on a backend that will never initialize.

SKEL_ERR_CODE = 4000


@dataclass(frozen=True)
class SkelCheck:
    """The result of a trivial on-device backend init. `ok=False` => the DSP skel would not load."""
    ok: bool
    code: Optional[int] = None     # the error code when it failed (4000 = skel lockdown)
    reason: str = ""               # a plain, axis-naming sentence
    raw: str = ""                  # the tail of the init output, for the record

    def __bool__(self) -> bool:
        return self.ok


# The default trivial init: force the HTP backend (and thus the fastRPC DSP skel) to load. On a locked
# image this is exactly where "Failed to load skel, error 4000" surfaces, before any model is involved.
_SKEL_INIT_CMD = ("export ADSP_LIBRARY_PATH=/usr/lib/rfsa/adsp:/usr/lib LD_LIBRARY_PATH=/usr/lib && "
                  "qnn-net-run --backend /usr/lib/libQnnHtp.so --version 2>&1")


def detect_skel_load(shell, init_cmd: str = _SKEL_INIT_CMD, timeout: int = 60) -> SkelCheck:
    """Run a trivial backend init through `shell` and classify whether the DSP skel loads.

    `shell` is any callable `shell(cmd, timeout=…) -> CompletedProcess` — it duck-types to both
    `AdbDevice.shell` and the SSH transport, so the same detector serves either board. A
    `Failed to load skel … 4000` in the output is a device-capability FAIL (named, not a hang); a
    timeout or an unreachable shell is reported as its own FAIL rather than being swallowed.
    """
    try:
        r = shell(init_cmd, timeout=timeout)
    except Exception as e:
        return SkelCheck(ok=False, reason=f"could not run the backend-init probe on the board ({e})")
    if r is None:
        return SkelCheck(ok=False, reason="backend-init probe returned no result (board unreachable?)")

    raw = ((getattr(r, "stdout", "") or "") + (getattr(r, "stderr", "") or ""))
    low = raw.lower()
    if "failed to load skel" in low or f"error {SKEL_ERR_CODE}" in low or f"0x{SKEL_ERR_CODE:04x}" in low:
        return SkelCheck(
            ok=False, code=SKEL_ERR_CODE,
            reason=("device image blocks loading the DSP skel (Failed to load skel, error 4000) — the "
                    "LLM cannot run on this board; this is a device-capability FAIL, not a model fault"),
            raw=raw[-500:])
    return SkelCheck(ok=True, reason="DSP skel loads; backend initialises", raw=raw[-500:])
