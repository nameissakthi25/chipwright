"""The `run` verb — resolve → fetch/locate the artifact → push + execute on the board → verify.

Ties the resolver, the device transport and a modality runner together, and ends in the thing that
makes the whole framework trustworthy: a cosine against a CPU reference, so a fast wrong answer is
caught here instead of shipped.
"""
from __future__ import annotations

import importlib
import os

from . import verify
from .device import RemoteQNN


def _board_env():
    return dict(host=os.environ.get("CW_BOARD_HOST"),
                user=os.environ.get("CW_BOARD_USER", "root"),
                pw=os.environ.get("CW_BOARD_PASS"),
                port=int(os.environ.get("CW_BOARD_PORT", 22)))


def run_variant(variant, input_path=None):
    """Execute a resolved variant on the board and return the verification report."""
    r = variant.run or {}
    if not r.get("modality"):
        raise RuntimeError(f"variant for {variant.model} has no run block (modality/meta/cpu_onnx)")

    # locate the artifact: a local file, or fetched by hash into the content-addressed cache
    if variant.local:
        art = os.path.expanduser(variant.local)
        if not os.path.exists(art):
            raise RuntimeError(f"local artifact not found: {art}")
    else:
        from . import cache
        art = cache.fetch(variant.url, variant.sha256)

    b = _board_env()
    if not (b["host"] and b["pw"]):
        raise RuntimeError("no board configured — set CW_BOARD_HOST / CW_BOARD_PASS")
    dev = RemoteQNN(b["host"], b["user"], b["pw"], b["port"])

    board_name = r.get("board_name", os.path.basename(art))
    print(f"  pushing artifact to board ({board_name}) …", flush=True)
    ctx_board = dev.ensure_ctx(art, board_name)

    mod = importlib.import_module(f".modalities.{r['modality']}", __package__)
    meta = os.path.expanduser(r["meta"])
    cpu_onnx = os.path.expanduser(r["cpu_onnx"])
    inp = os.path.expanduser(input_path or r["input"])
    print("  running on NPU + CPU reference …", flush=True)
    npu_out, cpu_out = mod.run(dev, ctx_board, meta, cpu_onnx, inp)
    return verify.report(npu_out, cpu_out, threshold=float(r.get("threshold", 0.99)))
