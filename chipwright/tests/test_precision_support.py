"""Precision-support matrix (FR-L06) + arch v79/v81 as first-class hard axes (FR-L05).

Pure, no board: the recipe parse/validate and the resolver refusal are both plain functions. Also
runnable as `python3 test_precision_support.py`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chipwright.env import TargetKey
from chipwright.recipe import from_dict
from chipwright.resolver import Outcome, Variant, precision_refusal, resolve


# --- recipe schema: precision_support parses + validates -----------------------------------------
_BASE = {"name": "qwen2.5-1.5b", "version": "1.0.0",
         "source": {"format": "onnx", "sha256": "abc"},
         "target": {"arch": "v81", "quant": "w4a16", "shape": "ctx4096"}}


def _recipe(**extra):
    doc = dict(_BASE)
    doc.update(extra)
    return from_dict(doc)


def test_recipe_parses_precision_matrix():
    r = _recipe(precision_support={
        "w4a16": "supported",
        "w8a16": "blocked:kv-cache-encoding",
        "fp16": "unsupported:genie-engine"})
    assert r.precision_status("w4a16") == "supported"
    assert r.precision_status("w8a16") == "blocked:kv-cache-encoding"
    assert r.supported_precisions == ["w4a16"]


def test_recipe_without_matrix_is_empty_not_error():
    r = _recipe()
    assert r.precision_support == {} and r.supported_precisions == []


def test_recipe_rejects_unknown_status_kind():
    for bad in ({"w8a16": "maybe"}, {"w8a16": "nope:x"}):
        try:
            _recipe(precision_support=bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


def test_recipe_rejects_blocked_without_reason():
    for bad in ({"w8a16": "blocked"}, {"w8a16": "unsupported:"}, {"w8a16": "blocked: "}):
        try:
            _recipe(precision_support=bad)
        except ValueError:
            continue
        raise AssertionError(f"a blocked/unsupported status must NAME a reason: {bad!r}")


# --- the pure refusal helper ---------------------------------------------------------------------
def test_precision_refusal_helper():
    support = {"w4a16": "supported", "w8a16": "blocked:kv-cache-encoding"}
    assert precision_refusal("w4a16", support) is None          # supported -> no refusal
    assert precision_refusal("int4", support) is None           # not declared -> not refused here
    assert precision_refusal("w8a16", support) == "blocked:kv-cache-encoding"
    assert precision_refusal("w8a16", None) is None             # no matrix -> nothing to refuse


# --- resolver refuses a non-supported precision and NAMES the reason -----------------------------
_SUPPORT = {"w4a16": "supported", "w8a16": "blocked:kv-cache-encoding", "fp16": "unsupported:genie-engine"}


def _llm_variant(arch="v81", quant="w4a16", **kw):
    base = dict(model="qwen2.5-1.5b", version="1.0.0", arch=arch, quant=quant, shape="ctx4096",
                sdk_tested="2.38.0", sdk_compatible=">=2.34,<2.40", precision_support=_SUPPORT)
    base.update(kw)
    return Variant(**base)


def test_resolve_refuses_blocked_precision_by_name():
    t = TargetKey(htp_arch="v81", qairt="2.38.0")
    d = resolve(t, [_llm_variant(url="u", sha256="s")], want_quant="w8a16")
    assert d.outcome is Outcome.FAIL and d.axis == "precision"
    assert "kv-cache-encoding" in d.reason


def test_resolve_refuses_unsupported_precision_by_name():
    t = TargetKey(htp_arch="v81", qairt="2.38.0")
    d = resolve(t, [_llm_variant(url="u", sha256="s")], want_quant="fp16")
    assert d.outcome is Outcome.FAIL and d.axis == "precision"
    assert "genie-engine" in d.reason


def test_resolve_allows_supported_precision():
    t = TargetKey(htp_arch="v81", qairt="2.38.0")
    d = resolve(t, [_llm_variant(url="u", sha256="s")], want_quant="w4a16")
    assert d.ok and d.outcome is Outcome.USE


def test_resolve_takes_explicit_matrix_over_variants():
    # a caller may pass the matrix directly (e.g. straight from the recipe) — it wins
    t = TargetKey(htp_arch="v81", qairt="2.38.0")
    v = _llm_variant(url="u", sha256="s", precision_support=None)
    d = resolve(t, [v], want_quant="w8a16", precision_support=_SUPPORT)
    assert d.outcome is Outcome.FAIL and d.axis == "precision"


# --- arch v79 and v81 are first-class hard axes (not special-cased anywhere) ----------------------
def test_arch_v79_and_v81_resolve_as_hard_axis():
    for arch in ("v79", "v81"):
        t = TargetKey(htp_arch=arch, qairt="2.38.0")
        d = resolve(t, [_llm_variant(arch=arch, url="u", sha256="s")], want_quant="w4a16")
        assert d.ok, f"{arch} should USE its own artifact"


def test_arch_mismatch_v79_vs_v81_fails_on_arch_axis():
    t = TargetKey(htp_arch="v79", qairt="2.38.0")           # board is v79
    d = resolve(t, [_llm_variant(arch="v81", url="u", sha256="s")], want_quant="w4a16")
    assert d.outcome is Outcome.FAIL and d.axis == "htp_arch"
    assert "v79" in d.reason


if __name__ == "__main__":
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for f in fns:
        f(); print("ok", f.__name__)
    print(f"\n{len(fns)} passed")
