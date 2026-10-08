"""Typed build errors — the build verb fails by naming the axis, never by a bare traceback.

A build shells out to the QAIRT toolchain, which may be absent (no x86_64-linux host, no SDK) or may
reject the graph. Each failure mode is a distinct type so the route orchestrator can stop at the right
stage, set `BuildResult.failed_axis`, and let the CLI print an honest reason — the same name-the-axis
discipline the resolver uses for a FAIL.
"""
from __future__ import annotations


class BuildError(Exception):
    """Base for every typed build failure. `axis` names the stage that stopped the route.

    The specific subclasses fix `axis` at the class level; a bare `BuildError` may name its stage
    per-instance (e.g. the convert/quantize executors, which have no dedicated subclass)."""
    axis: str = "build"

    def __init__(self, *args, axis: str = None):
        super().__init__(*args)
        if axis is not None:
            self.axis = axis


class PreflightError(BuildError):
    """A required host tool (qairt-converter / -quantizer / context-binary-generator) is missing."""
    axis = "preflight"


class OpUnsupported(BuildError):
    """The op-support gate found an op that will not run on the target HTP (before any tool runs)."""
    axis = "op_support"


class ContextGenError(BuildError):
    """qnn-context-binary-generator failed (e.g. SDK×arch mismatch, VTCM sizing, missing backend .so)."""
    axis = "context"


class FidelityBelowThreshold(BuildError):
    """The built artifact loaded but missed its cosine / task-metric gate — never published."""
    axis = "fidelity"
