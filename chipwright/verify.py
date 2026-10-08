"""Numerical verification — the check that catches a fast wrong answer.

A fast wrong answer is the failure this toolchain produces most, so `run` always compares the NPU
output to a CPU/ONNX-Runtime reference on the same input and reports the cosine per output. A high
cosine is necessary, not sufficient (a cosine of 0.99 can still cost task accuracy), so the record
keeps this separate from the task metric.
"""
from __future__ import annotations

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def report(npu_out: dict, cpu_out: dict, threshold: float = 0.99) -> dict:
    per = {k: cosine(npu_out[k], cpu_out[k]) for k in npu_out if k in cpu_out}
    lo = min(per.values()) if per else 0.0
    return {"per_output": per, "cosine_min": lo, "threshold": threshold, "pass": lo >= threshold}
