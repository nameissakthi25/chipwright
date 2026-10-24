"""Resolver — match registry variants against a board's TargetKey, with an explicit miss path.

The resolution rules, in order:

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
    precision_support: Optional[dict] = None  # LLM: {precision: "supported"|"blocked:<r>"|"unsupported:<r>"}
    runtime: Optional[str] = None        # the inference runtime this variant binds to (None => classic genie)
    runtime_support: Optional[dict] = None    # {runtime: {arch: "supported"|"deprecated:<s>"|"unsupported:<r>"}} — FR-RT04

    @property
    def has_artifact(self) -> bool:
        return bool(self.url or self.local)

    @property
    def effective_runtime(self) -> str:
        """The runtime this variant runs on. Historically every variant silently assumed classic Genie
        (spec 004), so an unset runtime IS `genie` — the deprecating-but-default floor."""
        return self.runtime or "genie"

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


# --- precision-support axis (LLM) -------------------------------------------------------------
# Not every precision survives the qai-hub -> Genie pipeline (spec 002, FR-L06): w4a16 is supported,
# w8a16 is blocked on a KV-cache-encoding divergence at convert, fp16 is unsupported because Genie's
# transformer path is integer-only. A recipe declares this per model; `resolve` refuses a non-supported
# precision and NAMES the reason — the same discipline as FAIL-names-the-axis, applied one axis earlier.
_SUPPORTED = "supported"


def precision_refusal(precision: Optional[str], support: Optional[dict]) -> Optional[str]:
    """Return the named reason to refuse `precision`, or None if it is supported / not constrained.

    Pure and tiny so it is trivial to test. A precision absent from the matrix is *not* refused here
    (the matrix only constrains what it declares); the normal variant/quant matching handles the rest.
    A declared non-"supported" status (e.g. "blocked:kv-cache-encoding") is returned verbatim so the
    caller can quote the reason by name.
    """
    if not precision or not support:
        return None
    status = support.get(precision)
    if status is None or status == _SUPPORTED:
        return None
    return str(status)


# --- runtime axis (spec 004) ------------------------------------------------------------------
# The same shape as precision, one axis over. A model+board can be served by different inference runtimes
# ({genie, geniex-qairt, llamacpp-htp}) with genuinely different arch + precision support (classic Genie is
# w4a16-only and deprecating; the llama.cpp HTP backend does Q4_0/Q8_0 on v73+ but NOT v68). So `resolve`
# must (a) refuse a runtime the board's arch cannot host, BY NAME (`axis=runtime`), and (b) among the
# runtimes the board CAN host, pick the one matching the caller's priority. The support table is per-arch:
# `{runtime: {arch: status}}`, validated in recipe.py (`_parse_runtime_support`).
#
# Priority -> the order runtimes are preferred in, among those runnable on the board's arch. Grounded in the
# v73 measurement: llamacpp-htp Q8_0 is the quality/prefill win (WikiText PPL 11.66, near-lossless, the
# >4-bit integer route Genie structurally lacks) but regresses decode (14.4 vs 22.5 tok/s); genie w4a16 is
# the decode win and the default floor. So quality-first prefers llamacpp-htp, decode-latency-first prefers
# genie, and the no-priority default is the conservative floor (genie) with a warn to choose explicitly.
_RUNTIME_PRIORITY = {
    "quality":        ("llamacpp-htp", "geniex-qairt", "genie"),
    "decode-latency": ("genie", "geniex-qairt", "llamacpp-htp"),
}
_DEFAULT_RUNTIME_ORDER = ("genie", "geniex-qairt", "llamacpp-htp")
_RUNNABLE = ("supported", "deprecated")   # 'unsupported' is the only hard refusal; 'deprecated' still runs


def runtime_refusal(runtime: Optional[str], arch: str, support: Optional[dict]) -> Optional[str]:
    """Return the named reason to refuse `runtime` on `arch`, or None if it is runnable / not constrained.

    Pure and tiny, mirroring `precision_refusal`. A runtime absent from the matrix (or an arch the matrix
    doesn't declare for it) is *not* refused here — the matrix only constrains what it declares. Only an
    explicit `unsupported:<reason>` is a refusal; a `deprecated:<successor>` status still runs (resolve warns).
    """
    if not runtime or not support:
        return None
    archmap = support.get(runtime)
    if not isinstance(archmap, dict):
        return None
    status = archmap.get(arch)
    if status is None or str(status).split(":", 1)[0] != "unsupported":
        return None
    return str(status)


def runnable_runtimes(arch: str, support: Optional[dict]) -> List[str]:
    """The runtimes the matrix declares runnable on `arch` (status 'supported' or 'deprecated')."""
    if not support:
        return []
    return sorted(rt for rt, am in support.items()
                  if isinstance(am, dict) and str(am.get(arch, "")).split(":", 1)[0] in _RUNNABLE)


def pick_runtime(runtimes: List[str], priority: Optional[str]):
    """Choose one runtime from `runtimes` (all runnable on the board's arch) by `priority`.

    Returns (chosen, note). `note` records the trade-off (which runtime was chosen over which) so the
    Decision can carry it — tested-is-a-record, now across runtimes.
    """
    order = _RUNTIME_PRIORITY.get(priority)
    if order:
        for rt in order:
            if rt in runtimes:
                others = [r for r in runtimes if r != rt]
                return rt, f"priority={priority}: chose runtime {rt} over {others}"
    # no (or unknown) priority -> the conservative default floor, and say how to choose
    for rt in _DEFAULT_RUNTIME_ORDER:
        if rt in runtimes:
            others = [r for r in runtimes if r != rt]
            return rt, (f"multiple runtimes available {sorted(runtimes)}; defaulted to {rt} "
                        f"(pass priority=quality|decode-latency to choose)" + (f"; alternatives {others}" if others else ""))
    rt = sorted(runtimes)[0]                      # only unrecognised runtimes present
    return rt, f"multiple runtimes {sorted(runtimes)}; defaulted to {rt}"


def _merge_warn(*parts: Optional[str]) -> Optional[str]:
    """Join the non-empty warning fragments, or None."""
    return " | ".join(p for p in parts if p) or None


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
            want_quant: Optional[str] = None, want_shape: Optional[str] = None,
            precision_support: Optional[dict] = None,
            want_runtime: Optional[str] = None, priority: Optional[str] = None,
            runtime_support: Optional[dict] = None) -> Decision:
    """Return the single Decision for this board. Pure function — easy to test without a board."""
    if not variants:
        return Decision(Outcome.FAIL, reason="no variants registered for this model", axis="model")

    # The runtime-support matrix may be passed explicitly (e.g. straight from the recipe) or carried on the
    # variants (the registry populates it). Resolve it once so both the refusal and the selection use it.
    rsupport = runtime_support
    if rsupport is None:
        rsupport = next((v.runtime_support for v in variants if v.runtime_support), None)

    # Axis 0b (runtime refusal): mirror the precision refusal one axis over. If the caller names a runtime
    # the board's arch cannot host, refuse up front and NAME it — listing the runtimes that ARE runnable on
    # this arch — before arch/SDK matching, because the runtime is unavailable on this silicon regardless of
    # which artifact is in hand (FAIL-names-the-axis, FR-RT03 / SC-RT02).
    if want_runtime:
        reason = runtime_refusal(want_runtime, target.htp_arch, rsupport)
        if reason:
            ok = runnable_runtimes(target.htp_arch, rsupport)
            return Decision(Outcome.FAIL, axis="runtime",
                            reason=f"runtime {want_runtime} is not available on {target.htp_arch}: {reason} "
                                   f"(supported on {target.htp_arch}: {ok})")

    # Axis 0 (LLM precision-support): if the caller asked for a precision the model declares as
    # blocked/unsupported, refuse up front and name the reason — before arch/SDK matching, because the
    # precision is unavailable through the pipeline regardless of which board is in hand. The matrix may
    # be passed explicitly or carried on the variants (the registry populates it per model).
    if want_quant:
        support = precision_support
        if support is None:
            support = next((v.precision_support for v in variants if v.precision_support), None)
        reason = precision_refusal(want_quant, support)
        if reason:
            return Decision(Outcome.FAIL, axis="precision",
                            reason=f"precision {want_quant} is not available: {reason}")

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

    # Runtime axis (spec 004): pick WHICH runtime among the candidates runnable on this board's arch.
    # If the caller named a runtime, narrow to it (its refusal was already handled above). Otherwise, only
    # when the candidates span more than one runtime do we choose by priority — a single-runtime model
    # resolves exactly as before, so nothing that pre-dates the runtime axis changes behaviour.
    runtime_warning = None
    if want_runtime:
        c2 = [v for v in cand if v.effective_runtime == want_runtime]
        if not c2:
            have = sorted({v.effective_runtime for v in cand})
            return Decision(Outcome.FAIL, axis="runtime",
                            reason=f"runtime {want_runtime} has no variant for {target.htp_arch}; have {have}")
        cand = c2
    else:
        runtimes_here = sorted({v.effective_runtime for v in cand})
        if len(runtimes_here) > 1:
            chosen, runtime_warning = pick_runtime(runtimes_here, priority)
            cand = [v for v in cand if v.effective_runtime == chosen]
    # Warn if the chosen runtime is marked deprecated on this arch (its successor is named in the matrix).
    if rsupport and cand:
        dep = rsupport.get(cand[0].effective_runtime, {})
        dep = dep.get(target.htp_arch, "") if isinstance(dep, dict) else ""
        if str(dep).split(":", 1)[0] == "deprecated":
            runtime_warning = _merge_warn(runtime_warning,
                                          f"runtime {cand[0].effective_runtime} on {target.htp_arch} is {dep}")

    built = [v for v in cand if v.has_artifact]

    # Axis: SDK range. If we don't know the board's QAIRT (offline), we can't check the point —
    # prefer a built artifact and warn that the SDK match is unverified.
    if built:
        if target.qairt is None:
            v = built[0]
            return Decision(Outcome.USE_WITH_WARN, variant=v,
                            warning=_merge_warn("board QAIRT unknown (offline) — SDK compatibility unverified",
                                                runtime_warning),
                            reason="artifact matches arch/quant/shape")
        in_range = [v for v in built if _in_range(target.qairt, v.sdk_compatible)]
        if in_range:
            exact = [v for v in in_range if _ver(v.sdk_tested) == _ver(target.qairt)]
            if exact:
                return Decision(Outcome.USE, variant=exact[0], reason="exact match", warning=runtime_warning)
            v = in_range[0]
            return Decision(Outcome.USE_WITH_WARN, variant=v,
                            warning=_merge_warn(f"board QAIRT {target.qairt} is in range {v.sdk_compatible} "
                                                f"but the artifact was tested at {v.sdk_tested}", runtime_warning),
                            reason="in-range, outside tested point")
        # artifact exists but SDK is out of every range -> fall through to recipe/fail on the SDK axis
        sdk_axis = True
    else:
        sdk_axis = False

    # Axis 3: no usable artifact, but a recipe can build one for this target.
    buildable = [v for v in cand if v.has_recipe]
    if buildable:
        return Decision(Outcome.BUILD, variant=buildable[0], warning=runtime_warning,
                        reason=("no artifact matches the board SDK; a recipe can build one"
                                if sdk_axis else "no built artifact, but a recipe exists"))

    # Axis 4: nothing.
    if sdk_axis:
        ranges = sorted({v.sdk_compatible for v in built})
        return Decision(Outcome.FAIL, axis="qairt",
                        reason=f"board QAIRT {target.qairt} outside every artifact range {ranges}, and no recipe")
    return Decision(Outcome.FAIL, axis="artifact",
                    reason="arch matches but no artifact and no recipe for this target")
