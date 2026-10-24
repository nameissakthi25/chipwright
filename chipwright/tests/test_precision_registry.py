"""Registry→resolve wiring for the LLM precision-support axis (FR-L06, Principle II).

Analyzer finding (0.2.5): the precision-support matrix was parsed by the recipe and enforced by the
resolver, but `Registry._load` never carried `precision_support` onto the Variant and no catalogue entry
declared one — so the Axis-0 "refuse by name" could never fire through the real registry→resolve path,
yet the suite was green (it only exercised hand-built Variants / an explicit kwarg). This test drives the
REGISTRY path, so it would have failed before the fix (registry.py now threads precision_support).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chipwright.registry import Registry
from chipwright.resolver import resolve, Outcome
from chipwright.env import TargetKey

_FIXTURE = """registry_schema: 1
models:
  - model: demo-llm
    modality: llm
    version: 1.0.0
    variants:
      - arch: v73
        quant: w4a16
        shape: ctx4096
        sdk_tested: "2.37.1"
        sdk_compatible: ">=2.34,<2.40"
        ext: qnnctx
        precision_support:
          w4a16: supported
          w8a16: "blocked:genie-kv-cache-quant-mismatch"
          fp16: "unsupported:genie-engine-is-integer-only"
        recipe: recipes/demo-llm/1.0.0.yaml
"""


def _registry(tmp_path):
    p = tmp_path / "prec_index.yaml"
    p.write_text(_FIXTURE)
    return Registry(index=str(p))


def test_registry_carries_precision_support_onto_variants(tmp_path):
    vs = _registry(tmp_path).variants("demo-llm")
    assert vs and vs[0].precision_support is not None, "registry must thread precision_support onto the Variant"
    assert vs[0].precision_support["w8a16"].startswith("blocked:")


def test_resolve_refuses_blocked_precision_through_the_registry_path(tmp_path):
    """The Axis-0 refusal fires from a registry-loaded variant — not only a hand-built one."""
    vs = _registry(tmp_path).variants("demo-llm")
    d = resolve(TargetKey(htp_arch="v73", qairt="2.37.1"), vs, want_quant="w8a16")
    assert d.outcome == Outcome.FAIL and d.axis == "precision", d
    assert "blocked:genie-kv-cache-quant-mismatch" in d.reason


def test_resolve_refuses_unsupported_precision_through_the_registry_path(tmp_path):
    vs = _registry(tmp_path).variants("demo-llm")
    d = resolve(TargetKey(htp_arch="v73", qairt="2.37.1"), vs, want_quant="fp16")
    assert d.outcome == Outcome.FAIL and d.axis == "precision"
    assert "unsupported:genie-engine-is-integer-only" in d.reason


def test_supported_precision_is_not_refused_through_the_registry_path(tmp_path):
    vs = _registry(tmp_path).variants("demo-llm")
    d = resolve(TargetKey(htp_arch="v73", qairt="2.37.1"), vs, want_quant="w4a16")
    assert d.axis != "precision", d   # w4a16 is supported → no precision refusal
