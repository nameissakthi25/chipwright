"""Single-image CV modality runner (first slice of a `chipwright-cv` package).

Works for any model whose input is one preprocessed image tensor and whose output is a single tensor
— classifier logits, a super-resolved image, a segmentation map, a depth field. Runs it two ways on
the same input and hands both outputs to the verifier:
  - NPU: the quantized context binary, float I/O (qnn-net-run quantizes the input and dequantizes the
    output), so the runner needs no per-tensor scales.
  - CPU: the float ONNX through ONNX Runtime — the reference.

A single-graph image model is the cheapest clean end-to-end proof: fixed input shape, one output, a
cosine that directly measures how faithful the quantized graph is to float.
"""
from __future__ import annotations

import numpy as np


def run(device, ctx_board_path, meta, cpu_onnx, input_npy):
    """Return (npu_out, cpu_out) dicts keyed by the model's output name. `meta` is unused (float I/O
    needs no scales); kept for a uniform modality signature."""
    import onnxruntime as ort

    x = np.load(input_npy).astype(np.float32)        # [1, C, H, W]

    sess = ort.InferenceSession(cpu_onnx, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    cpu_y = sess.run(None, {in_name: x})[0]
    cpu_out = {out_name: cpu_y}

    npu = device.run(ctx_board_path, {in_name: x},
                     [{"name": out_name, "dtype": "float32", "shape": tuple(cpu_y.shape)}],
                     native_io=False)
    npu_out = {out_name: npu[out_name].reshape(cpu_y.shape)}
    return npu_out, cpu_out
