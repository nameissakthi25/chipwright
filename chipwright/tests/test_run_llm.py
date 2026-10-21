"""The LLM run path refuses by name when it can't gate — integration guards for run_llm_variant.

Mirrors the CV/ASR refusal-path tests: a pass is never reported without the gate, and the LLM path
refuses with a message naming the fix (no Genie bundle / no pinned reference), not a raw traceback.
No board needed — these never reach adb.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chipwright import run as runmod
from chipwright.resolver import Variant


def _llm_variant(**kw):
    base = dict(model="qwen-x", version="1", arch="v81", quant="w4a16", shape="ctx4096",
                sdk_tested="2.51", sdk_compatible=">=2.34,<2.60")
    base.update(kw)
    return Variant(**base)


def test_llm_run_refuses_when_no_bundle():
    """An LLM variant with no Genie bundle directory refuses by name (build / CW_REGISTRY)."""
    v = _llm_variant(run={"modality": "llm"})  # no local bundle dir
    with pytest.raises(RuntimeError, match="bundle|chipwright build|CW_REGISTRY"):
        runmod.run_variant(v)


def test_llm_run_refuses_when_no_pinned_reference(tmp_path):
    """A bundle present but no pinned reference → refuse: the generative gate can't run."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "genie_config.json").write_text("{}")
    v = _llm_variant(local=str(bundle), run={"modality": "llm"})  # bundle, but no `reference`
    with pytest.raises(RuntimeError, match="pinned reference|gate can't run"):
        runmod.run_variant(v)


def test_looks_refused_classifier():
    """The v1 behavioral classifier tells a refusal from a compliant answer."""
    assert runmod._looks_refused("I can't help with that.") is True
    assert runmod._looks_refused("I'm sorry, but I cannot assist.") is True
    assert runmod._looks_refused("The capital of France is Paris.") is False
    assert runmod._looks_refused("") is False
