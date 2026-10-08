"""Image-classifier modality runner (first slice of a `chipwright-cv` package).

Runs one preprocessed image two ways on the same input and hands both outputs to the verifier:
  - NPU: the w8a16 context binary, float I/O (qnn-net-run quantizes the input and dequantizes the
    logits), so the runner needs no per-tensor scales.
  - CPU: the float ONNX through ONNX Runtime — the reference.

A single-graph classifier is the cheapest clean end-to-end proof: fixed input shape, one output, a
cosine that directly measures how faithful the quantized graph is to float.
"""
from __future__ import annotations

import numpy as np


def run(device, ctx_board_path, meta, cpu_onnx, input_npy):
    """Return (npu_out, cpu_out) dicts keyed by the logits output name. `meta` is unused here
    (float I/O needs no scales); kept for a uniform modality signature."""
    import onnxruntime as ort

    x = np.load(input_npy).astype(np.float32)        # [1, 3, H, W]

    sess = ort.InferenceSession(cpu_onnx, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    out_shape = [1 if (isinstance(d, str) or d is None) else d for d in sess.get_outputs()[0].shape]
    cpu_logits = sess.run(None, {in_name: x})[0]
    cpu_out = {out_name: cpu_logits}

    npu = device.run(ctx_board_path, {in_name: x},
                     [{"name": out_name, "dtype": "float32", "shape": tuple(out_shape)}],
                     native_io=False)
    npu_out = {out_name: npu[out_name].reshape(cpu_logits.shape)}
    return npu_out, cpu_out
