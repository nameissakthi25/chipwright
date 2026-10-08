"""Artifact tags — the wheel-tag grammar for QNN artifacts.

A QNN context binary is valid for exactly one combination of five axes. We name that combination
in the filename (the parts a resolver matches on) and expand the rest in the manifest, the same way
a PyPI wheel names `cp312-manylinux_x86_64` and leaves the rest to metadata.

    whisper-small-en-1.2.0+qnn2.37-htpv68-w8a16-win30s.qnnctx
    └──────┬───────┘ └─┬─┘ └──┬──┘ └──┬──┘ └─┬─┘ └─┬─┘ └─┬──┘
         name       version  sdk    arch   quant  shape  ext

Filename carries arch + quant + shape + a coarse SDK (brief lean); the manifest carries the full
SDK compatibility range and everything else. Parsing is deliberately strict: a tag that doesn't
parse is a bug, not a near-miss to paper over.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_EXT = ("qnnctx", "dlc", "bin")
# name is greedy up to the LAST "-<semver>+"; version is semver; then the +qnn... compat tag.
_RE = re.compile(
    r"^(?P<name>.+?)-(?P<version>\d+\.\d+\.\d+)"
    r"\+qnn(?P<sdk>\d+\.\d+)"
    r"-htp(?P<arch>v\d+)"
    r"-(?P<quant>w\d+a\d+|fp16|fp32)"
    r"-(?P<shape>[A-Za-z0-9]+)"
    r"\.(?P<ext>" + "|".join(_EXT) + r")$"
)


@dataclass(frozen=True)
class ArtifactTag:
    name: str
    version: str
    sdk: str        # coarse, e.g. "2.37" — the manifest holds the full compatible range
    arch: str       # Hexagon arch, e.g. "v68"
    quant: str      # e.g. "w8a16"
    shape: str      # shape profile, e.g. "win30s"
    ext: str        # "qnnctx" | "dlc" | "bin"

    @classmethod
    def parse(cls, filename: str) -> "ArtifactTag":
        m = _RE.match(filename.strip())
        if not m:
            raise ValueError(f"not a valid QNN artifact tag: {filename!r}")
        return cls(**m.groupdict())

    def __str__(self) -> str:
        return (f"{self.name}-{self.version}+qnn{self.sdk}"
                f"-htp{self.arch}-{self.quant}-{self.shape}.{self.ext}")

    @property
    def is_context_binary(self) -> bool:
        """Context binaries are brittle (pinned to arch×SDK×backend); DLCs are forward-compatible."""
        return self.ext in ("qnnctx", "bin")
