"""Host topology — the `.qualcomm-env` file and the build-host preflight.

`Env` is the small record the build verb needs about *this* machine: the boards it can reach, the
QAIRT SDK roots installed, the SSH keys, and a cache dir. `read_env` / `write_env` are a typed wrapper
over the skills' `write-env.sh` (the desktop app's Settings screen is a GUI over the same file).

`host_preflight` answers the one question that decides whether a build can even start here: is this an
x86_64-linux host with the QAIRT converter/quantizer/context-generator on PATH? The HTP toolchain is
x86_64-Linux-centric (no macOS host), so a build on the wrong host must fail early and by name, not
halfway through a convert.
"""
from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass, field
from typing import List, Optional

import yaml

DEFAULT_ENV_PATH = os.path.expanduser("~/.qualcomm-env")

# The host tools a build shells out to (verified pipeline). All three must be on PATH to build here.
REQUIRED_TOOLS = ("qairt-converter", "qairt-quantizer", "qnn-context-binary-generator")


@dataclass
class Env:
    """Host topology for the build verb. Serialized to `.qualcomm-env` (YAML)."""
    hosts: List[dict] = field(default_factory=list)       # [{name, host, user, key_or_pw, port, transport}]
    sdk_roots: List[str] = field(default_factory=list)    # QAIRT/QNN SDK install roots on this machine
    keys: List[str] = field(default_factory=list)         # SSH key paths
    cache_dir: str = os.path.expanduser("~/.cache/chipwright")

    def to_dict(self) -> dict:
        return {"hosts": self.hosts, "sdk_roots": self.sdk_roots,
                "keys": self.keys, "cache_dir": self.cache_dir}

    @classmethod
    def from_dict(cls, doc: Optional[dict]) -> "Env":
        doc = doc or {}
        return cls(hosts=list(doc.get("hosts", []) or []),
                   sdk_roots=list(doc.get("sdk_roots", []) or []),
                   keys=list(doc.get("keys", []) or []),
                   cache_dir=doc.get("cache_dir") or os.path.expanduser("~/.cache/chipwright"))

    @property
    def sdk_root(self) -> Optional[str]:
        """The first installed SDK root, or None — what the build executors add to PATH when present."""
        return self.sdk_roots[0] if self.sdk_roots else None


@dataclass
class PreflightReport:
    ok: bool
    missing: List[str] = field(default_factory=list)   # required tools not found on PATH
    arch: str = ""                                      # host machine arch, e.g. x86_64
    system: str = ""                                    # host OS, e.g. Linux
    note: Optional[str] = None

    def __bool__(self) -> bool:
        return self.ok


def read_env(path: str = DEFAULT_ENV_PATH) -> Env:
    """Load `.qualcomm-env` into an `Env`. Returns an empty `Env` when the file is absent."""
    if not os.path.exists(path):
        return Env()
    with open(path) as f:
        return Env.from_dict(yaml.safe_load(f))


def write_env(topology: Env, path: str = DEFAULT_ENV_PATH) -> str:
    """Write an `Env` to `.qualcomm-env` (YAML). Returns the path written. Mirrors skills' write-env.sh."""
    if not isinstance(topology, Env):
        topology = Env.from_dict(topology)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(topology.to_dict(), f, sort_keys=False)
    return path


def host_preflight(env: Optional[Env] = None) -> PreflightReport:
    """Check this is an x86_64-linux build host with the QAIRT toolchain on PATH.

    `env.sdk_root/bin` is added to the search path when present (the SDK often ships the tools there
    without being on the global PATH). A non-x86_64 or non-Linux host is flagged in `note` but still
    reports the tool scan honestly — on macOS the tools are expected to be missing.
    """
    env = env or Env()
    arch, system = platform.machine(), platform.system()
    extra = os.path.join(env.sdk_root, "bin") if env.sdk_root else None
    search = os.environ.get("PATH", "")
    if extra and os.path.isdir(extra):
        search = extra + os.pathsep + search

    missing = [t for t in REQUIRED_TOOLS if shutil.which(t, path=search) is None]

    note = None
    host_ok = arch in ("x86_64", "AMD64") and system == "Linux"
    if not host_ok:
        note = (f"host is {system}/{arch}; the QAIRT HTP toolchain needs x86_64-linux "
                f"(no macOS host) — builds must run on a Linux x86_64 build host")
    return PreflightReport(ok=(host_ok and not missing), missing=missing,
                           arch=arch, system=system, note=note)
