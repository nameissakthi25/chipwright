"""`run --emit-record` capture (spec 005 FR-RR05): the on-device output tensor is written to the cache
and a ready-to-host `vs_reference.npu_output` fragment (sha256 + dtype + shape) is printed. This is the
capture half that makes a record re-derivable; it was exercised on real silicon (QCS6490/v68) producing
mobilenet-v2's class_logits. No board here — we call the emitter with a tensor directly."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from chipwright import run as runmod, cache


class _V:
    model = "mobilenet-v2"


def test_emit_npu_record_writes_tensor_and_prints_fragment(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))
    npu_out = {"class_logits": np.zeros((1, 1000), dtype=np.float32)}
    runmod._emit_npu_record(_V(), npu_out)
    out = capsys.readouterr().out
    # the fragment names the field, the output, its sha256, dtype and shape
    assert "npu_output:" in out
    assert "name: class_logits" in out
    assert "sha256:" in out
    assert "dtype: float32" in out
    assert "shape: [1, 1000]" in out
    # and the tensor is actually written under the content cache (so it can be hosted)
    saved = tmp_path / "records" / "mobilenet-v2" / "npu_class_logits.npy"
    assert saved.exists()
    back = np.load(saved)
    assert back.shape == (1, 1000) and back.dtype == np.float32


def test_emit_npu_record_sha_is_the_saved_bytes(capsys, monkeypatch, tmp_path):
    """The printed sha256 is the sha of the saved .npy — so hosting that file and pasting the fragment
    yields a record the hardened cache will accept."""
    import hashlib
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))
    runmod._emit_npu_record(_V(), {"class_logits": np.arange(6, dtype=np.float32).reshape(1, 6)})
    out = capsys.readouterr().out
    printed_sha = [ln.split("sha256:")[1].strip() for ln in out.splitlines() if "sha256:" in ln][0]
    saved = tmp_path / "records" / "mobilenet-v2" / "npu_class_logits.npy"
    assert hashlib.sha256(saved.read_bytes()).hexdigest() == printed_sha
