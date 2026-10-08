"""Resolver — match registry variants against a board's TargetKey, with an explicit miss path.

The brief's resolution rules, in order (design brief §"Resolution rules"):

  1. Exact match on arch, quant, shape; SDK inside the compatible range  -> USE
  2. Match, but SDK outside the tested point (still in range)            -> USE_WITH_WARN
  3. No artifact, but a recipe whose target can be built                 -> BUILD (offer, estimate)
  4. Nothing                                                             -> FAIL, name the axis

Never silently downgrade to a near-miss artifact. The arch axis is hard (a mismatch fails at load);
the SDK axis is a range, not a point (warn outside the tested point, don't refuse).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .env import TargetKey


class Outcome(Enum):
    USE = "use"                   # exact match — run it
    USE_WITH_WARN = "use_warn"    # in-range but outside the tested SDK point
    BUILD = "build"               # no artifact, but a recipe can produce one
    FAIL = "fail"                 # nothing — the named axis is why


@dataclass
class Variant:
    """One build variant from the registry (the registry's unit is a variant, not a model)."""
    model: str
    version: str
    arch: str
    quant: str
    shape: str
    sdk_tested: str                      # e.g. "2.37.1"
    sdk_compatible: str                  # e.g. ">=2.34,<2.40"
    ext: str = "qnnctx"
    url: Optional[str] = None            # present => a built artifact exists (remote)
    sha256: Optional[str] = None
    local: Optional[str] = None          # present => an artifact file on disk (not yet hosted)
    run: Optional[dict] = None           # how `run` executes it: {modality, meta, cpu_onnx, board_name, input}
    recipe: Optional[str] = None         # present => can be built when no artifact matches
    verified: Optional[dict] = None      # the verification record (loads / cosine / task / latency)

    @property
    def has_artifact(self) -> bool:
        return bool(self.url or self.local)

    @property
    def has_recipe(self) -> bool:
        return bool(self.recipe)


@dataclass
class Decision:
    outcome: Outcome
    variant: Optional[Variant] = None
    reason: str = ""
    axis: Optional[str] = None        # on FAIL: which axis could not be satisfied
    warning: Optional[str] = None     # on USE_WITH_WARN

    @property
    def ok(self) -> bool:
        return self.outcome in (Outcome.USE, Outcome.USE_WITH_WARN)


# --- tiny semver range check (enough for QAIRT x.y.z) -----------------------------------------
def _ver(v: str):
    return tuple(int(x) for x in (v.split("-")[0].split("+")[0].split(".") + ["0", "0"])[:3])


def _in_range(point: str, rng: str) -> bool:
    """point like '2.37.1' against a comma range like '>=2.34,<2.40'."""
    p = _ver(point)
    for clause in rng.split(","):
        clause = clause.strip()
        for op in (">=", "<=", "==", ">", "<"):
            if clause.startswith(op):
                bound = _ver(clause[len(op):])
                ok = {">=": p >= bound, "<=": p <= bound, "==": p == bound,
                      ">": p > bound, "<": p < bound}[op]
                if not ok:
                    return False
                break
    return True


def resolve(target: TargetKey, variants: List[Variant],
            want_quant: Optional[str] = None, want_shape: Optional[str] = None) -> Decision:
    """Return the single Decision for this board. Pure function — easy to test without a board."""
    if not variants:
        return Decision(Outcome.FAIL, reason="no variants registered for this model", axis="model")

    # Axis 1 (hard): Hexagon architecture must match exactly, or the binary fails at load.
    arch_ok = [v for v in variants if v.arch == target.htp_arch]
    if not arch_ok:
        have = sorted({v.arch for v in variants})
        return Decision(Outcome.FAIL, axis="htp_arch",
                        reason=f"no variant for arch {target.htp_arch}; registry has {have}")

    # Optional narrowing on quant / shape when the caller asked for a specific one.
    cand = arch_ok
    if want_quant:
        c2 = [v for v in cand if v.quant == want_quant]
        if not c2:
            have = sorted({v.quant for v in cand})
            return Decision(Outcome.FAIL, axis="quant",
                            reason=f"no {want_quant} for {target.htp_arch}; have {have}")
        cand = c2
    if want_shape:
        c2 = [v for v in cand if v.shape == want_shape]
        if not c2:
            have = sorted({v.shape for v in cand})
            return Decision(Outcome.FAIL, axis="shape",
                            reason=f"no shape {want_shape} for {target.htp_arch}; have {have}")
        cand = c2

    built = [v for v in cand if v.has_artifact]

    # Axis: SDK range. If we don't know the board's QAIRT (offline), we can't check the point —
    # prefer a built artifact and warn that the SDK match is unverified.
    if built:
        if target.qairt is None:
            v = built[0]
            return Decision(Outcome.USE_WITH_WARN, variant=v,
                            warning="board QAIRT unknown (offline) — SDK compatibility unverified",
                            reason="artifact matches arch/quant/shape")
        in_range = [v for v in built if _in_range(target.qairt, v.sdk_compatible)]
        if in_range:
            exact = [v for v in in_range if _ver(v.sdk_tested) == _ver(target.qairt)]
            if exact:
                return Decision(Outcome.USE, variant=exact[0], reason="exact match")
            v = in_range[0]
            return Decision(Outcome.USE_WITH_WARN, variant=v,
                            warning=(f"board QAIRT {target.qairt} is in range {v.sdk_compatible} "
                                     f"but the artifact was tested at {v.sdk_tested}"),
                            reason="in-range, outside tested point")
        # artifact exists but SDK is out of every range -> fall through to recipe/fail on the SDK axis
        sdk_axis = True
    else:
        sdk_axis = False

    # Axis 3: no usable artifact, but a recipe can build one for this target.
    buildable = [v for v in cand if v.has_recipe]
    if buildable:
        return Decision(Outcome.BUILD, variant=buildable[0],
                        reason=("no artifact matches the board SDK; a recipe can build one"
                                if sdk_axis else "no built artifact, but a recipe exists"))

    # Axis 4: nothing.
    if sdk_axis:
        ranges = sorted({v.sdk_compatible for v in built})
        return Decision(Outcome.FAIL, axis="qairt",
                        reason=f"board QAIRT {target.qairt} outside every artifact range {ranges}, and no recipe")
    return Decision(Outcome.FAIL, axis="artifact",
                    reason="arch matches but no artifact and no recipe for this target")
