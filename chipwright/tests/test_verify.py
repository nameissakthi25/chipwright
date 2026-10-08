"""Numerical verification — cosine similarity and the per-output pass/fail report.

Pure numpy, no board. Runnable as `python3 test_verify.py`.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chipwright.verify import cosine, report


def test_cosine_of_identical_vectors_is_one():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert cosine(a, a) == 1.0


def test_cosine_of_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine(a, b)) < 1e-12


def test_cosine_of_opposite_vectors_is_minus_one():
    a = np.array([1.0, 2.0, 3.0])
    assert math.isclose(cosine(a, -a), -1.0, rel_tol=1e-12)


def test_cosine_is_shape_agnostic_via_ravel():
    # a 2-D NPU output and its flattened reference compare equal
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    b = np.arange(6, dtype=np.float64)
    assert math.isclose(cosine(a, b), 1.0, rel_tol=1e-12)


def test_cosine_handles_zero_vector_without_crashing():
    a = np.zeros(4)
    b = np.array([1.0, 2.0, 3.0, 4.0])
    assert cosine(a, b) == 0.0
    assert cosine(a, a) == 0.0


def test_report_passes_when_all_outputs_meet_threshold():
    npu = {"logits": np.array([1.0, 2.0, 3.0]), "hidden": np.array([0.5, 0.5])}
    cpu = {"logits": np.array([1.0, 2.0, 3.0]), "hidden": np.array([0.5, 0.5])}
    r = report(npu, cpu, threshold=0.99)
    assert r["pass"] is True
    assert math.isclose(r["cosine_min"], 1.0, rel_tol=1e-12)
    assert r["threshold"] == 0.99


def test_report_fails_when_an_output_is_below_threshold():
    npu = {"good": np.array([1.0, 1.0]), "bad": np.array([1.0, 0.0])}
    cpu = {"good": np.array([1.0, 1.0]), "bad": np.array([0.0, 1.0])}  # orthogonal -> cosine 0
    r = report(npu, cpu, threshold=0.99)
    assert r["pass"] is False
    assert math.isclose(r["cosine_min"], 0.0, abs_tol=1e-12)


def test_report_per_output_dict_is_keyed_by_output_name():
    npu = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    cpu = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    r = report(npu, cpu)
    assert set(r["per_output"]) == {"a", "b"}
    assert math.isclose(r["per_output"]["a"], 1.0, rel_tol=1e-12)
    # cosine_min is the minimum over the per-output values
    assert r["cosine_min"] == min(r["per_output"].values())


def test_report_only_compares_shared_keys():
    # keys present in npu but missing from cpu are skipped, not an error
    npu = {"shared": np.array([1.0, 1.0]), "npu_only": np.array([1.0])}
    cpu = {"shared": np.array([1.0, 1.0]), "cpu_only": np.array([1.0])}
    r = report(npu, cpu)
    assert set(r["per_output"]) == {"shared"}


def test_report_empty_inputs_do_not_crash():
    r = report({}, {})
    assert r["per_output"] == {}
    assert r["cosine_min"] == 0.0
    assert r["pass"] is False  # 0.0 < default 0.99


if __name__ == "__main__":
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for f in fns:
        f(); print("ok", f.__name__)
    print(f"\n{len(fns)} passed")
