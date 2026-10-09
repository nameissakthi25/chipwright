"""Graph adaptations — the declared, auditable ONNX edits a recipe applies before conversion.

The converter and the HTP reject a handful of op/attribute patterns that are harmless to fix. Each
adaptation is named in `recipe.adaptations`, applied here, and logged, so a build never silently
mutates a graph — it does exactly what the recipe says, on the record.

Implemented so far:
  - reshape_allowzero : set Reshape `allowzero` 1 → 0. Lossless when the target shape has no literal
    0 (the common case in ViT / detector exports); qairt-converter rejects allowzero=1 outright.
"""
from __future__ import annotations

import os
from typing import List, Tuple


def _reshape_allowzero(model) -> int:
    n = 0
    for node in model.graph.node:
        if node.op_type == "Reshape":
            for a in node.attribute:
                if a.name == "allowzero" and a.i == 1:
                    a.i = 0
                    n += 1
    return n


_HANDLERS = {"reshape_allowzero": _reshape_allowzero}


def apply(onnx_path: str, adaptations, out_dir: str = None) -> Tuple[str, List[str]]:
    """Apply the declared adaptations to `onnx_path`; return (path_to_use, logs).

    Returns the original path unchanged when there is nothing to do (no supported adaptation declared,
    onnx not installed, or the file isn't on disk) — the caller then converts the source as-is.
    """
    kinds = [a.kind for a in (adaptations or [])]
    todo = [k for k in kinds if k in _HANDLERS]
    if not todo:
        return onnx_path, []
    if not os.path.exists(onnx_path):
        return onnx_path, [f"adapt: source onnx not local ({onnx_path}) — adaptations skipped"]
    try:
        import onnx
    except ImportError:
        return onnx_path, ["adapt: onnx not installed — adaptations skipped"]

    model = onnx.load(onnx_path)
    logs = []
    for kind in todo:
        count = _HANDLERS[kind](model)
        logs.append(f"adapt: {kind} applied to {count} node(s)")
    dest = out_dir or os.path.dirname(onnx_path) or "."
    os.makedirs(dest, exist_ok=True)
    out = os.path.join(dest, "adapted.onnx")
    onnx.save(model, out)
    return out, logs
