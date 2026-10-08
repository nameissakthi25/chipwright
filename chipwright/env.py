"""Board probe → TargetKey. The thing the resolver matches artifacts against.

On Qualcomm silicon the question that decides everything is *which chip* — specifically the Hexagon
architecture, the installed QAIRT version, and the SoC. `qcf doctor` reads those off the board (over
SSH/ADB) rather than guessing. This mirrors the `qualcomm-env-discovery` skill's `probe-env.sh`, kept
self-contained here so qcf-core has no hard dependency on a skill checkout.

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
    # arch fallback from the on-device lib name (libQnnHtpV68.so -> v68)
    if arch == "unknown":
        for lib in libs:
            if lib.startswith("libQnnHtpV") and lib.endswith(".so"):
                arch = "v" + lib[len("libQnnHtpV"):-len(".so")]
                break
    temp = None
    if out.get("TEMP", "").strip().isdigit():
        temp = round(int(out["TEMP"]) / 1000.0, 1)
    qairt = out.get("QAIRT", "").strip() or None
    return TargetKey(htp_arch=arch, qairt=qairt, soc_id=soc_id, soc_name=soc_name,
                     reachable=True, qnn_libs=libs, thermal_c=temp)
