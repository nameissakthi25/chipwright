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


def cpu_reference(cpu_onnx, input_npy):
    """The CPU/ONNX-Runtime reference half of the gate: run the float ONNX on the preprocessed input and
    return (cpu_out, x, in_name) — cpu_out keyed by the model's output name.

    Shared by the live `run` (NPU vs this) and the offline `replay.rederive` (pinned-NPU vs this), so both
    front doors compute the reference the same way (Principle IV — one implementation). No board here.
    """
    import onnxruntime as ort

    x = np.load(input_npy).astype(np.float32)        # [1, C, H, W]

    sess = ort.InferenceSession(cpu_onnx, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    cpu_vals = sess.run(None, {in_name: x})
    cpu_out = {n: v for n, v in zip(out_names, cpu_vals)}
    return cpu_out, x, in_name


def run(device, ctx_board_path, meta, cpu_onnx, input_npy):
    """Return (npu_out, cpu_out) dicts keyed by the model's output name. `meta` is unused (float I/O
    needs no scales); kept for a uniform modality signature."""
    cpu_out, x, in_name = cpu_reference(cpu_onnx, input_npy)

    specs = [{"name": n, "dtype": "float32", "shape": tuple(cpu_out[n].shape)} for n in cpu_out]
    npu = device.run(ctx_board_path, {in_name: x}, specs, native_io=False)
    npu_out = {n: npu[n].reshape(cpu_out[n].shape) for n in cpu_out}
    return npu_out, cpu_out
