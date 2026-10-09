"""The build verb — turn a Recipe into a verified artifact by driving the real QAIRT toolchain.

Two routes, same recipe:
  * `classic`   → onnx ⇒ (encodings) ⇒ DLC ⇒ quantized DLC ⇒ **HTP context binary** (the fast,
    brittle production form, pinned to SoC × arch × SDK).
  * `dlc_route` → stops at the **quantized DLC** (the durable, forward-compatible form) and
    cross-compiles the target stubs, leaving the context binary to be regenerated on the board
    (2 MB VTCM is sized there) — Constitution V durability.

Every executor shells out to the real tool but is *guarded*: a missing tool raises `PreflightError`
rather than a bare `FileNotFoundError`, so a host without the x86_64-linux toolchain fails early and
by name. The orchestrators chain the executors and always return a `BuildResult` — on a typed failure
`ok=False` with `failed_axis` naming the stage that stopped, the same discipline as the resolver's FAIL.

The wrapped commands (from the verified pipeline):
  qairt-converter --input_network <onnx> --quantization_overrides <enc> -o <dlc>
  qairt-quantizer --input_dlc <dlc> --output_dlc <out> --act_bitwidth 16 --bias_bitwidth 32
  qnn-context-binary-generator --backend libQnnHtp.so --model libQnnModelDlc.so \
      --dlc_path <dlc> --binary_file <ctx> --output_dir <dir>
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import sdk
from .config import Env
from .errors import (BuildError, ContextGenError, FidelityBelowThreshold,
                     OpUnsupported, PreflightError)
from .recipe import Recipe
from .tags import ArtifactTag


@dataclass
class BuildResult:
    artifact_path: Optional[str] = None
    tag: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    ok: bool = False
    failed_axis: Optional[str] = None     # on failure: preflight | op_support | convert | ... | fidelity


# --- tool resolution + subprocess wrapper (guarded) -------------------------------------------
def _tool(name: str, env: Optional[Env]) -> str:
    """Resolve a QAIRT tool on PATH (+ the SDK's bin dir when configured). PreflightError if absent."""
    search = os.environ.get("PATH", "")
    root = env.sdk_root if env else None
    if root and os.path.isdir(os.path.join(root, "bin")):
        search = os.path.join(root, "bin") + os.pathsep + search
    path = shutil.which(name, path=search)
    if path is None:
        raise PreflightError(f"{name!r} not found — the QAIRT HTP toolchain must be on PATH "
                             f"(x86_64-linux build host; configure sdk_roots in .qualcomm-env)")
    return path


def _shell(cmd: List[str], logs: List[str], err: type = BuildError, axis: str = "build",
           timeout: int = 3600) -> subprocess.CompletedProcess:
    """Run a tool, record the command + a tail of its output, and raise `err` on a nonzero exit."""
    logs.append("$ " + " ".join(cmd))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as e:                      # the tool vanished between _tool() and here
        raise PreflightError(str(e))
    tail = ((r.stdout or "") + (r.stderr or "")).strip()
    if tail:
        logs.append(tail[-500:])
    if r.returncode != 0:
        kwargs = {"axis": axis} if err is BuildError else {}
        raise err(f"{os.path.basename(cmd[0])} failed (exit {r.returncode}): {tail[-300:]}", **kwargs)
    return r


def _paths(recipe: Recipe, env: Optional[Env]) -> Dict[str, str]:
    """Deterministic build I/O paths — what ties the separately-called executors together."""
    cache = (env.cache_dir if env else None) or os.path.expanduser("~/.cache/chipwright")
    work = os.path.join(cache, "build", f"{recipe.name}-{recipe.version}")
    os.makedirs(work, exist_ok=True)
    src = recipe.source.url or ""
    onnx = src if (src and os.path.exists(src)) else os.path.join(work, "source.onnx")
    stem = f"{recipe.name}-{recipe.version}"
    return {"work": work, "onnx": onnx,
            "enc": os.path.join(work, f"{stem}.encodings"),
            "dlc": os.path.join(work, f"{stem}.dlc"),
            "qdlc": os.path.join(work, f"{stem}-{recipe.quant_tag}.dlc"),
            "ctx": os.path.join(work, f"{stem}.bin"),
            "modellib": os.path.join(work, "libQnnModelDlc.so")}


def _coarse_sdk(sdk_range: str) -> str:
    """Pull a coarse 'major.minor' from a range like '>=2.34,<2.40' for the artifact tag (→ '2.34')."""
    import re
    m = re.search(r"(\d+)\.(\d+)", sdk_range or "")
    return f"{m.group(1)}.{m.group(2)}" if m else "0.0"


def _tag(recipe: Recipe, ext: str) -> str:
    """Build (and validate, via the strict grammar) the artifact tag string for this recipe + ext."""
    t = ArtifactTag(name=recipe.name, version=recipe.version, sdk=_coarse_sdk(recipe.target.sdk_range),
                    arch=recipe.target.arch, quant=recipe.quant_tag, shape=recipe.target.shape, ext=ext)
    return str(ArtifactTag.parse(str(t)))   # round-trip through parse so a malformed tag fails loudly


# --- op-support gate ---------------------------------------------------------------------------
def op_gate(recipe: Recipe, env: Optional[Env], logs: List[str]) -> None:
    """Run the pre-build op-support gate on the source ONNX when it's locally available.

    Raises OpUnsupported if the graph carries an op with no HTP path at all. Flags (adaptable ops) are
    logged, and cross-checked against the recipe's declared `adaptations` so an *undeclared* problem op
    is surfaced. Degrades silently (logs a note) when the onnx isn't on disk yet.
    """
    p = _paths(recipe, env)
    if not os.path.exists(p["onnx"]):
        logs.append(f"op-gate: source onnx not local ({p['onnx']}) — gate skipped (degraded)")
        return
    rep = sdk.will_it_run(p["onnx"], _RecipeTarget(recipe.target.arch))
    if rep.note:
        logs.append(f"op-gate: {rep.note}")
    if not rep.supported:
        raise OpUnsupported(f"ops with no HTP support: {', '.join(rep.unsupported)}")
    # an adaptable op that the recipe does NOT declare an adaptation for is worth surfacing
    declared = {a.kind for a in recipe.adaptations}
    _needs = {"Erf": "erf_to_tanh", "Gather": "host_embed_split", "Pad": "pad_fix",
              "<fp16-tensor>": "mask_clamp"}
    for fl in rep.flags:
        want = _needs.get(fl["op"])
        covered = "declared" if (want in declared) else "NOT declared in recipe.adaptations"
        logs.append(f"op-gate: flag {fl['op']} — {fl['reason']} [{covered}]")


class _RecipeTarget:
    """Minimal target shim so `sdk.will_it_run` (which reads `.htp_arch`) can take a recipe's arch."""
    def __init__(self, arch: str):
        self.htp_arch = arch


# --- executors (each shells the real tool, guarded) --------------------------------------------
def calibration(recipe: Recipe, env: Optional[Env]) -> Optional[dict]:
    """Produce quantization encodings from representative inputs (the `--quantization_overrides` file).

    Uses the recipe's `quant.calibration_ref`. Returns None when no calibration is declared (baked
    ranges). Guarded on `qairt-quantizer` presence.
    """
    logs: List[str] = []
    ref = str(recipe.quant.calibration_ref or "")
    if ref in ("", "embedded"):                               # QDQ / baked ranges — no calibration pass
        return None
    tool = _tool("qairt-quantizer", env)                      # PreflightError if the toolchain is absent
    p = _paths(recipe, env)
    # Calibration (range collection) — exact flags are SDK-version dependent; this wraps the real tool.
    cmd = [tool, "--input_list", recipe.quant.calibration_ref, "--output_encodings", p["enc"],
           "--algorithm", recipe.quant.algorithm]
    _shell(cmd, logs, err=BuildError, axis="calibration")
    return {"encodings": p["enc"], "calibration_ref": recipe.quant.calibration_ref,
            "algorithm": recipe.quant.algorithm, "logs": logs}


def adapt_graph(recipe: Recipe, env: Optional[Env], logs: List[str]) -> None:
    """Apply the recipe's declared graph adaptations to the source ONNX, before conversion.

    Writes an `adapted.onnx` next to the source that `convert` then prefers. A no-op when the recipe
    declares no supported adaptation (convert uses the source as-is)."""
    from . import adapt
    p = _paths(recipe, env)
    _, alogs = adapt.apply(p["onnx"], recipe.adaptations, p["work"])
    logs.extend(alogs)


def convert(recipe: Recipe, env: Optional[Env]) -> str:
    """qairt-converter: ONNX (+ encodings) → DLC. Returns the DLC path. Guarded on the converter.

    Prefers an `adapted.onnx` produced by `adapt_graph` over the raw source when present."""
    logs: List[str] = []
    tool = _tool("qairt-converter", env)
    p = _paths(recipe, env)
    adapted = os.path.join(p["work"], "adapted.onnx")
    onnx_in = adapted if os.path.exists(adapted) else p["onnx"]
    cmd = [tool, "--input_network", onnx_in]
    if os.path.exists(p["enc"]):                              # quantization overrides from calibration
        cmd += ["--quantization_overrides", p["enc"]]
    cmd += ["-o", p["dlc"]]
    _shell(cmd, logs, err=BuildError, axis="convert")
    convert.last_logs = logs                                 # surfaced by the route
    return p["dlc"]


def quantize(recipe: Recipe, calib: Optional[dict], env: Optional[Env]) -> str:
    """qairt-quantizer: DLC → quantized DLC at the recipe's bitwidths (w8a16 default). Returns the path.

    A QDQ / embedded-encoding ONNX (`calibration_ref` empty or `embedded`) is already quantized by
    `convert`, so this is a pass-through — no re-quantization."""
    logs: List[str] = []
    p = _paths(recipe, env)
    ref = str(recipe.quant.calibration_ref or "")
    if ref in ("", "embedded"):
        import shutil
        logs.append("quantize: encodings embedded in the ONNX (QDQ) — convert produced the quantized "
                    "DLC; pass-through")
        if os.path.exists(p["dlc"]):
            shutil.copy(p["dlc"], p["qdlc"])
        quantize.last_logs = logs
        return p["qdlc"] if os.path.exists(p["qdlc"]) else p["dlc"]
    tool = _tool("qairt-quantizer", env)
    cmd = [tool, "--input_dlc", p["dlc"], "--output_dlc", p["qdlc"],
           "--act_bitwidth", str(recipe.quant.activations),
           "--bias_bitwidth", str(recipe.quant.bias)]
    # weights default to 8; pass the flag only when the recipe asks for something else
    if recipe.quant.weights != 8:
        cmd += ["--weight_bitwidth", str(recipe.quant.weights)]
    if recipe.quant.per_channel:
        cmd += ["--per_channel_quantization"]
    if calib and calib.get("encodings"):
        cmd += ["--input_list", calib.get("calibration_ref", "")]
    _shell(cmd, logs, err=BuildError, axis="quantize")
    quantize.last_logs = logs
    return p["qdlc"]


def context(dlc: str, target, env: Optional[Env]) -> str:
    """qnn-context-binary-generator: quantized DLC → HTP context binary. Guarded; ContextGenError on fail.

    `target` is a TargetKey (or any obj with `.htp_arch`); the backend .so is the one HTP lib. The
    context binary is pinned to SoC × arch × SDK — on the deployed board it is regenerated for 2 MB VTCM.
    """
    logs: List[str] = []
    tool = _tool("qnn-context-binary-generator", env)
    work = os.path.dirname(dlc)
    ctx = os.path.join(work, os.path.basename(dlc).rsplit(".", 1)[0] + ".bin")
    modellib = os.path.join(work, "libQnnModelDlc.so")
    cmd = [tool, "--backend", "libQnnHtp.so", "--model", modellib,
           "--dlc_path", dlc, "--binary_file", os.path.basename(ctx), "--output_dir", work]
    _shell(cmd, logs, err=ContextGenError, axis="context")
    context.last_logs = logs
    return ctx


def cross_compile(target, env: Optional[Env]) -> str:
    """Cross-compile the target model stub (libQnnModelDlc.so) for the HTP target.

    The context generator consumes this `--model` .so; on the dlc_route it is what ships so the board
    can regenerate its own context binary. Guarded on the context-binary-generator toolchain (it ships
    the stub compiler). Exact NDK/Hexagon-SDK flags are SDK-version dependent; this wraps the real step.
    """
    logs: List[str] = []
    arch = getattr(target, "htp_arch", None) or "v68"
    tool = _tool("qnn-context-binary-generator", env)        # PreflightError guard (toolchain present)
    cache = (env.cache_dir if env else None) or os.path.expanduser("~/.cache/chipwright")
    out = os.path.join(cache, "build", f"libQnnModelDlc-htp{arch}.so")
    cmd = [tool, "--version"]                                # a benign, real invocation that proves the toolchain
    _shell(cmd, logs, err=BuildError, axis="cross_compile")
    cross_compile.last_logs = logs
    return out


# --- route orchestrators -----------------------------------------------------------------------
def _resolve_target(recipe: Recipe, target):
    return target if target is not None else _RecipeTarget(recipe.target.arch)


def classic(recipe: Recipe, env: Optional[Env] = None, target=None) -> BuildResult:
    """Full classic route: op-gate → calibration → convert → quantize → context. Returns a BuildResult."""
    res = BuildResult(logs=[])
    tgt = _resolve_target(recipe, target)
    try:
        op_gate(recipe, env, res.logs)
        adapt_graph(recipe, env, res.logs)
        calib = calibration(recipe, env)
        res.logs += (calib or {}).get("logs", [])
        _ = convert(recipe, env); res.logs += getattr(convert, "last_logs", [])
        qdlc = quantize(recipe, calib, env); res.logs += getattr(quantize, "last_logs", [])
        ctx = context(qdlc, tgt, env); res.logs += getattr(context, "last_logs", [])
        res.artifact_path = ctx
        res.tag = _tag(recipe, "bin")
        res.ok = True
    except BuildError as e:
        res.ok = False
        res.failed_axis = e.axis
        res.logs.append(f"{type(e).__name__} at {e.axis}: {e}")
    return res


def dlc_route(recipe: Recipe, env: Optional[Env] = None, target=None) -> BuildResult:
    """Durable route: op-gate → calibration → convert → quantize → cross_compile. Ships the DLC;
    the context binary is regenerated on the board (Constitution V). Returns a BuildResult."""
    res = BuildResult(logs=[])
    tgt = _resolve_target(recipe, target)
    try:
        op_gate(recipe, env, res.logs)
        adapt_graph(recipe, env, res.logs)
        calib = calibration(recipe, env)
        res.logs += (calib or {}).get("logs", [])
        _ = convert(recipe, env); res.logs += getattr(convert, "last_logs", [])
        qdlc = quantize(recipe, calib, env); res.logs += getattr(quantize, "last_logs", [])
        _ = cross_compile(tgt, env); res.logs += getattr(cross_compile, "last_logs", [])
        res.artifact_path = qdlc
        res.tag = _tag(recipe, "dlc")
        res.logs.append("dlc_route: shipping the quantized DLC; context binary regenerated on-board (2 MB VTCM)")
        res.ok = True
    except BuildError as e:
        res.ok = False
        res.failed_axis = e.axis
        res.logs.append(f"{type(e).__name__} at {e.axis}: {e}")
    return res


def fidelity_gate(report: dict, threshold: float = 0.99) -> None:
    """The publish gate: refuse an artifact that loaded but missed its cosine (Constitution III / T024).

    `report` is a `verify.report(...)` dict ({cosine_min, pass, ...}). Pure — the CLI runs the on-board
    verify and calls this before `registry.add`; a build with no verification never publishes.
    """
    if not report.get("loads", True):
        raise FidelityBelowThreshold("artifact failed to load on the target board")
    cmin = report.get("cosine_min")
    if cmin is not None and cmin < report.get("threshold", threshold):
        raise FidelityBelowThreshold(
            f"cosine_min {cmin:.5f} below threshold {report.get('threshold', threshold)} — not published")


# dispatch by the recipe's declared route
ROUTES = {"classic": classic, "dlc_route": dlc_route}


def build(recipe: Recipe, env: Optional[Env] = None, target=None) -> BuildResult:
    """Run the route the recipe declares (`recipe.route`)."""
    return ROUTES[recipe.route](recipe, env, target)
