"""Offline replay — re-derive a recorded verification from pinned tensors, with no board (spec 005).

No network, no board, no onnxruntime: `cache.fetch` / `cache.fetch_named` are monkeypatched to hand back
small numpy tensors the test controls, so `replay.rederive` runs the real gate math (`verify.report`) over
known operands. Covers SC-RR01 (recompute == recorded within ±0.001), SC-RR04 (replay and the live gate
call the SAME `verify.report`), and the FR-RR04 refusals (no pinned output; recomputed ≠ recorded).
Runnable as `python3 test_replay.py`.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chipwright import cache, replay, verify


def _variant(**kw):
    from chipwright.resolver import Variant
    base = dict(model="synth-cv", version="1", arch="v68", quant="w8a16", shape="img224",
                sdk_tested="2.37.1", sdk_compatible=">=2.34,<2.40", ext="dlc")
    base.update(kw)
    return Variant(**base)


def _record(value, *, with_npu=True, out_name="logits"):
    """A synthetic verification record. `with_npu` toggles the pinned on-device output (the spec-005 field)."""
    vs = {"backend": "onnxruntime-cpu-float", "metric": "cosine_min", "value": value}
    if with_npu:
        vs["npu_output"] = [{
            "name": out_name,
            "url": "https://example.test/synth/npu_logits.npy",
            "sha256": "cd" * 32, "dtype": "float32", "shape": [4],
        }]
    return {"on": "QCS6490 (soc_id 498)", "date": "2026-10-09", "loads": True, "vs_reference": vs}


def _run_block():
    """A run block whose CPU reference resolves (via the mocked cache) to a pinned .npy tensor."""
    return {"modality": "cv", "threshold": 0.95,
            "cpu_onnx": "https://example.test/synth/cpu_ref.npy", "cpu_onnx_sha256": "ab" * 32}


def _mock_cache(monkeypatch, npu_path, cpu_path):
    monkeypatch.setattr(cache, "fetch", lambda url, sha256: npu_path)
    monkeypatch.setattr(cache, "fetch_named", lambda url, sha256, name, sidecars=None: cpu_path)


def _tensors(tmp_path, npu, cpu):
    npu_p, cpu_p = str(tmp_path / "npu.npy"), str(tmp_path / "cpu.npy")
    np.save(npu_p, np.asarray(npu, dtype=np.float32))
    np.save(cpu_p, np.asarray(cpu, dtype=np.float32))
    return npu_p, cpu_p


# --- SC-RR01: the headline number recomputes offline, within tolerance ------------------------
def test_rederive_recomputes_cosine_equal_to_recorded(monkeypatch, tmp_path):
    npu, cpu = [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.03]       # a realistic sub-1.0 cosine
    npu_p, cpu_p = _tensors(tmp_path, npu, cpu)
    _mock_cache(monkeypatch, npu_p, cpu_p)
    recorded = round(verify.cosine(np.asarray(npu), np.asarray(cpu)), 3)   # the *published* (rounded) value

    v = _variant(run=_run_block(), verified=_record(recorded))
    res = replay.rederive(v)

    assert res["recorded"] == recorded
    assert abs(res["cosine_min"] - recorded) <= 1e-3          # within the stated ±0.001 tolerance
    assert res["matches"] is True
    assert res["pass"] is True                                # cosine >= threshold 0.95
    assert set(res["outputs"]) == {"logits"}


def test_rederive_exact_match_is_verified(monkeypatch, tmp_path):
    npu_p, cpu_p = _tensors(tmp_path, [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    _mock_cache(monkeypatch, npu_p, cpu_p)
    v = _variant(run=_run_block(), verified=_record(1.0))
    res = replay.rederive(v)
    assert res["matches"] is True and res["pass"] is True
    assert abs(res["cosine_min"] - 1.0) < 1e-12


# --- SC-RR04: replay and the live gate call the SAME verify.report (one implementation) --------
def test_replay_and_live_gate_produce_identical_math(monkeypatch, tmp_path):
    npu, cpu = [0.2, 0.9, -0.4, 1.3], [0.21, 0.88, -0.35, 1.29]
    npu_p, cpu_p = _tensors(tmp_path, npu, cpu)
    _mock_cache(monkeypatch, npu_p, cpu_p)

    # the live gate's own call, on the same operands
    live = verify.report({"logits": np.asarray(npu, dtype=np.float32)},
                         {"logits": np.asarray(cpu, dtype=np.float32)}, threshold=0.95)

    v = _variant(run=_run_block(), verified=_record(round(live["cosine_min"], 3)))
    res = replay.rederive(v)

    assert res["cosine_min"] == live["cosine_min"]           # identical math, not merely close
    assert res["outputs"] == live["per_output"]


# --- FR-RR04: refusals ------------------------------------------------------------------------
def test_rederive_refuses_when_no_pinned_output():
    """A record that predates reproducible-record (no npu_output) must refuse by name, not fabricate."""
    v = _variant(run=_run_block(), verified=_record(0.956, with_npu=False))
    with pytest.raises(RuntimeError, match="no pinned on-device output to replay"):
        replay.rederive(v)


def test_rederive_reports_mismatch_without_passing(monkeypatch, tmp_path):
    """A recomputed cosine that disagrees with the recorded value → matches False (caller refuses)."""
    npu_p, cpu_p = _tensors(tmp_path, [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])  # true cosine 1.0
    _mock_cache(monkeypatch, npu_p, cpu_p)
    v = _variant(run=_run_block(), verified=_record(0.50))     # recorded is a lie vs the pinned tensors
    res = replay.rederive(v)
    assert res["matches"] is False
    assert abs(res["cosine_min"] - 1.0) < 1e-12


def test_pinned_npu_output_parsing():
    assert replay.pinned_npu_output(_variant(verified=_record(0.9))) is not None
    assert replay.pinned_npu_output(_variant(verified=_record(0.9, with_npu=False))) is None
    assert replay.pinned_npu_output(_variant(verified=None)) is None
    assert replay.is_rederivable(_variant(verified=_record(0.9))) is True


def test_sha_mismatch_refuses_never_passes(monkeypatch, tmp_path):
    """A pinned tensor whose bytes don't hash to the pinned sha → the hardened cache raises; replay must
    propagate the refusal (it never reports a recomputed pass). We simulate the cache's refusal."""
    def _boom(url, sha256):
        raise RuntimeError("sha256 mismatch for %s" % url)
    monkeypatch.setattr(cache, "fetch", _boom)
    v = _variant(run=_run_block(), verified=_record(0.956))
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        replay.rederive(v)


# --- CLI replay path (FR-RR03): labelled replay + refusals, correct exit codes ----------------
class _Reg:
    def __init__(self, variants):
        self._v = variants

    def variants(self, _model):
        return self._v


def test_cli_replay_verified_exit0(monkeypatch, tmp_path, capsys):
    from chipwright import cli
    npu_p, cpu_p = _tensors(tmp_path, [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    _mock_cache(monkeypatch, npu_p, cpu_p)
    v = _variant(run=_run_block(), verified=_record(1.0))
    rc = cli._replay(_Reg([v]), "synth-cv")
    out = capsys.readouterr().out
    assert rc == 0
    assert "REPLAY" in out and "recorded on QCS6490 (soc_id 498), 2026-10-09" in out
    assert "VERIFIED" in out
    # the line must read as a replay of a recorded result, never as a fresh on-silicon verification.
    assert "NPU vs CPU-reference" not in out


def test_cli_replay_no_pin_refuses_by_name_exit2(capsys):
    from chipwright import cli
    v = _variant(run=_run_block(), verified=_record(0.956, with_npu=False))
    rc = cli._replay(_Reg([v]), "synth-cv")
    out = capsys.readouterr().out
    assert rc == 2
    assert "no pinned on-device output to replay" in out


def test_cli_replay_mismatch_refuses_exit1(monkeypatch, tmp_path, capsys):
    from chipwright import cli
    npu_p, cpu_p = _tensors(tmp_path, [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    _mock_cache(monkeypatch, npu_p, cpu_p)
    v = _variant(run=_run_block(), verified=_record(0.50))
    rc = cli._replay(_Reg([v]), "synth-cv")
    out = capsys.readouterr().out
    assert rc == 1
    assert "REFUSING" in out and "not reproducible" in out


if __name__ == "__main__":
    # These tests use pytest fixtures (monkeypatch / tmp_path / capsys), so run them under pytest.
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__, "-q"]))
