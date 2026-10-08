"""SDK knowledge — the pre-build op-support gate and a thin docs lookup.

`will_it_run(onnx_path, target)` scans an ONNX graph's op types against a small built-in table of ops
known-problematic on the HTP and returns an `OpReport` *before* any QAIRT tool is shelled out. It is a
gate, not a converter: a fast, honest "this won't convert cleanly" beats a 20-minute build that dies at
the context stage. It degrades gracefully — with no `onnx` installed it parses what it can from the raw
file and says so, rather than throwing.

The rules encode what the recipe's `adaptations` exist to fix (data-model Recipe): a surviving FP16
tensor on a v68 context, `Erf` with no HTP layout inferer, an on-chip integer `Gather`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OpReport:
    supported: bool
    unsupported: List[str] = field(default_factory=list)      # op types with no HTP support at all
    flags: List[Dict[str, str]] = field(default_factory=list)  # {op, reason} — problematic but adaptable
    note: Optional[str] = None                                 # set when the scan degraded (no onnx, etc.)

    def __bool__(self) -> bool:
        return self.supported


# Ops that HTP cannot run at all (the converter will reject them outright).
_UNSUPPORTED = {
    "NonMaxSuppression": "no HTP implementation",
    "RoiAlign": "no HTP implementation",
    "GridSample": "no HTP implementation",
}

# Ops that convert only with a declared adaptation. reason -> what the recipe must carry.
_PROBLEMATIC = {
    "Erf": "Erf has no HTP layout inferer — rewrite GELU via tanh (adaptation: erf_to_tanh)",
    "Gather": "integer Gather lands on-chip and is slow/unsupported for a large table — split the embed to host (adaptation: host_embed_split)",
    "Pad": "some Pad configs are rejected by the converter — pin mode=constant (adaptation: pad_fix)",
}


def _onnx_op_types(onnx_path: str) -> (List[str], Dict[str, Any], Optional[str]):
    """Return (op_types, tensor_dtype_info, note). Uses onnx if available; else a lightweight byte scan."""
    try:
        import onnx  # type: ignore
        m = onnx.load(onnx_path)
        ops = [n.op_type for n in m.graph.node]
        # look for a float16 value_info / initializer surviving into the graph
        fp16 = 1  # onnx.TensorProto.FLOAT16
        has_fp16 = any(getattr(i, "data_type", None) == fp16 for i in m.graph.initializer)
        for vi in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
            if vi.type.tensor_type.elem_type == fp16:
                has_fp16 = True
                break
        return ops, {"fp16": has_fp16}, None
    except ImportError:
        pass
    except Exception as e:  # a corrupt/partial file — degrade rather than crash the whole gate
        return [], {"fp16": False}, f"onnx failed to parse ({e}); op scan skipped"

    # Lightweight fallback: scan the raw protobuf bytes for op-type strings we know about.
    try:
        with open(onnx_path, "rb") as f:
            blob = f.read()
    except OSError as e:
        return [], {"fp16": False}, f"could not read {onnx_path} ({e})"
    known = list(_UNSUPPORTED) + list(_PROBLEMATIC) + [
        "Conv", "MatMul", "Add", "Mul", "Softmax", "LayerNormalization", "Relu", "Transpose"]
    found = [op for op in known if op.encode() in blob]
    # a byte scan cannot reliably tell a tensor's dtype, so fp16 is left undetected here
    return found, {"fp16": False}, "lightweight byte scan (install `onnx` for a full op + dtype gate)"


def will_it_run(onnx_path: str, target: Any) -> OpReport:
    """Scan `onnx_path` for ops problematic on `target` (a TargetKey or any obj with `.htp_arch`).

    Returns an OpReport: `supported=False` if any op has no HTP path at all; otherwise `supported=True`
    with `flags` for ops that need a declared adaptation. Never raises for a missing graph/tool — it
    reports the degradation in `note`.
    """
    arch = getattr(target, "htp_arch", None) or (target.get("arch") if isinstance(target, dict) else None) or "?"

    if not onnx_path or not os.path.exists(onnx_path):
        return OpReport(supported=True, note=f"onnx not found at {onnx_path!r}; op gate skipped (degraded)")

    ops, info, note = _onnx_op_types(onnx_path)
    if not ops and note is None:
        note = "no op types recognised; using a lightweight byte scan (install `onnx` for a full gate)"

    unsupported = sorted({op for op in ops if op in _UNSUPPORTED})
    flags: List[Dict[str, str]] = []
    for op in sorted(set(ops)):
        if op in _PROBLEMATIC:
            flags.append({"op": op, "reason": _PROBLEMATIC[op]})
    # a surviving FP16 tensor on a v68/v6x context overflows the attention mask fill and loses range
    if info.get("fp16") and str(arch).startswith("v6"):
        flags.append({"op": "<fp16-tensor>",
                      "reason": f"a surviving FP16 tensor on an {arch} context — clamp the mask "
                                f"(-3.4e38 → -1e4) and quantize (adaptation: mask_clamp)"})

    return OpReport(supported=not unsupported, unsupported=unsupported, flags=flags, note=note)


def docs(query: str, sdk_root: Optional[str] = None) -> dict:
    """Thin docs lookup: grep the SDK docs dir for `query` when an SDK root is present.

    A stub by design — returns matching lines (capped) or a clear 'no SDK docs' result. Wraps the
    skills' `extract-sdk-docs.py` surface without taking a hard dependency on the SDK being installed.
    """
    root = sdk_root or os.environ.get("QNN_SDK_ROOT") or os.environ.get("QAIRT_SDK_ROOT")
    if not root or not os.path.isdir(root):
        return {"query": query, "sdk_root": root, "found": False,
                "note": "no SDK docs dir (set QNN_SDK_ROOT); docs lookup unavailable"}
    docs_dir = next((d for d in (os.path.join(root, "docs"), root) if os.path.isdir(d)), root)
    hits: List[Dict[str, str]] = []
    for base, _dirs, files in os.walk(docs_dir):
        for fn in files:
            if not fn.lower().endswith((".txt", ".md", ".html", ".rst")):
                continue
            path = os.path.join(base, fn)
            try:
                with open(path, errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if query.lower() in line.lower():
                            hits.append({"file": path, "line": str(i), "text": line.strip()[:200]})
                            if len(hits) >= 20:
                                return {"query": query, "sdk_root": root, "found": True, "hits": hits}
            except OSError:
                continue
    return {"query": query, "sdk_root": root, "found": bool(hits), "hits": hits}
