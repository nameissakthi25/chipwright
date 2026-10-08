"""qcf-core — the resolver, registry client, artifact cache, and device transport that the CLI,
the desktop app, and a Claude Code session all share, so they always agree about what will run on
your board. See the Qualcomm AI Model Framework design brief.
"""
from .env import TargetKey, probe
from .registry import Registry
from .resolver import Decision, Outcome, Variant, resolve
from .tags import ArtifactTag

__all__ = ["TargetKey", "probe", "Registry", "Decision", "Outcome", "Variant", "resolve", "ArtifactTag"]
__version__ = "0.1.0"
