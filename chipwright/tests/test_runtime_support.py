"""Runtime axis (spec 004, FR-RT01/03/04/05, SC-RT01/02/04) — the OFFLINE subset.

The runtime axis is the precision axis a second time: a per-key "supported"|"<kind>:<reason>" matrix the
resolver refuses on, one level deeper (`{runtime: {arch: status}}`) because a runtime's support is per-arch.
Everything here is pure and board-free — the recipe parse/validate, the resolver priority + the axis=runtime
refusal, and the runtime-variant record threaded through the registry. Also runnable as
`python3 test_runtime_support.py`. (The on-device adapter run + dispatch check are Track 2 — they need silicon.)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chipwright.env import TargetKey
from chipwright.recipe import from_dict
from chipwright.registry import Registry
from chipwright.resolver import (Outcome, Variant, resolve, runnable_runtimes,
                                 runtime_refusal)


# --- recipe schema: runtime_support parses + validates (T-RT01) -----------------------------------
_BASE = {"name": "qwen2.5-1.5b", "version": "1.0.0",
         "source": {"format": "onnx", "sha256": "abc"},
         "target": {"arch": "v73", "quant": "w4a16", "shape": "ctx4096"}}

_MATRIX = {
    "genie": {"v68": "deprecated:geniex-qairt", "v73": "deprecated:geniex-qairt"},
    "geniex-qairt": {"v73": "supported"},
    "llamacpp-htp": {"v68": "unsupported:htp-backend-excludes-v68", "v73": "supported"},
}


def _recipe(**extra):
    doc = dict(_BASE)
    doc.update(extra)
    return from_dict(doc)


def test_recipe_parses_runtime_matrix():
    r = _recipe(runtime_support=_MATRIX)
    assert r.runtime_status("llamacpp-htp", "v73") == "supported"
    assert r.runtime_status("llamacpp-htp", "v68") == "unsupported:htp-backend-excludes-v68"
    assert r.runtime_status("genie", "v73") == "deprecated:geniex-qairt"
    # 'deprecated' still RUNS (so it's runnable), 'unsupported' does not
    assert r.runnable_runtimes("v73") == ["genie", "geniex-qairt", "llamacpp-htp"]
    assert r.runnable_runtimes("v68") == ["genie"]


def test_recipe_without_matrix_is_empty_not_error():
    r = _recipe()
    assert r.runtime_support == {} and r.runnable_runtimes("v73") == []


def test_recipe_rejects_unknown_status_kind():
    for bad in ({"llamacpp-htp": {"v73": "maybe"}}, {"genie": {"v73": "gone:x"}}):
        try:
            _recipe(runtime_support=bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


def test_recipe_rejects_deprecated_or_unsupported_without_reason():
    for bad in ({"genie": {"v73": "deprecated"}},
                {"llamacpp-htp": {"v68": "unsupported"}},
                {"llamacpp-htp": {"v68": "unsupported: "}}):
        try:
            _recipe(runtime_support=bad)
        except ValueError:
            continue
        raise AssertionError(f"a deprecated/unsupported status must NAME a reason/successor: {bad!r}")


def test_recipe_rejects_non_mapping_arch_entry():
    try:
        _recipe(runtime_support={"genie": "supported"})   # must be {arch: status}, not a bare status
    except ValueError:
        return
    raise AssertionError("expected ValueError for a non-mapping runtime entry")


# --- the pure refusal helpers --------------------------------------------------------------------
def test_runtime_refusal_helper():
    assert runtime_refusal("llamacpp-htp", "v68", _MATRIX) == "unsupported:htp-backend-excludes-v68"
    assert runtime_refusal("llamacpp-htp", "v73", _MATRIX) is None   # supported -> no refusal
    assert runtime_refusal("genie", "v73", _MATRIX) is None          # deprecated still RUNS -> no refusal
    assert runtime_refusal("llamacpp-htp", "v73", None) is None      # no matrix -> nothing to refuse
    assert runtime_refusal("nonesuch", "v73", _MATRIX) is None       # undeclared -> not refused here


def test_runnable_runtimes_helper():
    assert runnable_runtimes("v68", _MATRIX) == ["genie"]            # llamacpp unsupported; genie deprecated-but-runs
    assert runnable_runtimes("v73", _MATRIX) == ["genie", "geniex-qairt", "llamacpp-htp"]


# --- resolver: priority picks the runtime (SC-RT01) ----------------------------------------------
def _genie(arch="v73", **kw):
    base = dict(model="qwen2.5-1.5b", version="1.0.0", arch=arch, quant="w4a16", shape="ctx4096",
                sdk_tested="2.37.1", sdk_compatible=">=2.34,<2.40", runtime="genie",
                runtime_support=_MATRIX, url="u", sha256="s")
    base.update(kw)
    return Variant(**base)


def _htp(arch="v73", **kw):
    base = dict(model="qwen2.5-1.5b", version="1.0.0", arch=arch, quant="Q8_0", shape="ctx4096",
                sdk_tested="2.37.1", sdk_compatible=">=2.34,<2.40", runtime="llamacpp-htp",
                ext="gguf", runtime_support=_MATRIX, local="/tmp/q.gguf")
    base.update(kw)
    return Variant(**base)


def test_priority_quality_picks_llamacpp_htp():
    t = TargetKey(htp_arch="v73", qairt="2.37.1")
    d = resolve(t, [_genie(), _htp()], priority="quality")
    assert d.ok and d.variant.effective_runtime == "llamacpp-htp" and d.variant.quant == "Q8_0"
    assert "priority=quality" in (d.warning or "")        # the trade-off is recorded on the Decision


def test_priority_decode_latency_picks_genie():
    t = TargetKey(htp_arch="v73", qairt="2.37.1")
    d = resolve(t, [_genie(), _htp()], priority="decode-latency")
    assert d.ok and d.variant.effective_runtime == "genie" and d.variant.quant == "w4a16"


def test_default_picks_genie_floor_and_warns_to_choose():
    t = TargetKey(htp_arch="v73", qairt="2.37.1")
    d = resolve(t, [_genie(), _htp()])                    # no priority
    assert d.ok and d.variant.effective_runtime == "genie"
    assert "priority=quality|decode-latency" in (d.warning or "")   # tells the caller how to choose


def test_deprecated_runtime_is_flagged_with_successor():
    t = TargetKey(htp_arch="v73", qairt="2.37.1")
    d = resolve(t, [_genie(), _htp()], want_runtime="genie")
    assert d.ok and "deprecated:geniex-qairt" in (d.warning or "")  # resolve warns on a deprecated runtime


def test_single_runtime_model_is_unchanged_by_the_axis():
    # a model with one runtime resolves exactly as before — no priority machinery engages
    t = TargetKey(htp_arch="v73", qairt="2.37.1")
    d = resolve(t, [_htp()])
    assert d.ok and d.variant.effective_runtime == "llamacpp-htp"
    assert "priority" not in (d.warning or "")


# --- resolver: refuse a runtime the board can't host, BY NAME (SC-RT02) --------------------------
def test_resolve_refuses_llamacpp_htp_on_v68_by_name():
    t = TargetKey(htp_arch="v68", qairt="2.37.1")
    d = resolve(t, [_genie(arch="v68")], want_runtime="llamacpp-htp")
    assert d.outcome is Outcome.FAIL and d.axis == "runtime"
    # names the runtime, the arch, the reason, and lists what IS runnable there
    for token in ("llamacpp-htp", "v68", "htp-backend-excludes-v68", "genie"):
        assert token in d.reason, token


def test_resolve_refusal_reads_matrix_off_variants_or_explicit():
    t = TargetKey(htp_arch="v68", qairt="2.37.1")
    # matrix passed explicitly (e.g. straight from the recipe) also drives the refusal
    v = _genie(arch="v68", runtime_support=None)
    d = resolve(t, [v], want_runtime="llamacpp-htp", runtime_support=_MATRIX)
    assert d.outcome is Outcome.FAIL and d.axis == "runtime"


def test_supported_runtime_on_arch_is_not_refused():
    t = TargetKey(htp_arch="v73", qairt="2.37.1")
    d = resolve(t, [_genie(), _htp()], want_runtime="llamacpp-htp")
    assert d.ok and d.variant.effective_runtime == "llamacpp-htp"


# --- the runtime-variant RECORD carries runtime+precision+PPL+decode/prefill (SC-RT04) ----------
# A self-contained fixture registry (mirrors registry/index.dev.yaml, which is the author-local overlay)
# so the test is green in any checkout AND exercises registry._load threading runtime + runtime_support.
_FIXTURE = """registry_schema: 1
models:
  - model: qwen2.5-1.5b-instruct
    modality: llm
    version: 1.0.0
    variants:
      - arch: v73
        quant: w4a16
        shape: ctx4096
        sdk_tested: "2.37.1"
        sdk_compatible: ">=2.34,<2.40"
        ext: qnnctx
        runtime: genie
        runtime_support:
          genie: {v73: "deprecated:geniex-qairt"}
          llamacpp-htp: {v68: "unsupported:htp-backend-excludes-v68", v73: "supported"}
        recipe: recipes/qwen2.5-1.5b/1.0.0.yaml
      - arch: v73
        quant: Q8_0
        shape: ctx4096
        sdk_tested: "2.37.1"
        sdk_compatible: ">=2.34,<2.40"
        ext: gguf
        runtime: llamacpp-htp
        runtime_support:
          genie: {v73: "deprecated:geniex-qairt"}
          llamacpp-htp: {v68: "unsupported:htp-backend-excludes-v68", v73: "supported"}
        local: /tmp/qwen2.5-1.5b-instruct-q8_0.gguf
        verified:
          "on": "QCS8550 (Hexagon v73, soc_id 603)"
          runtime: llamacpp-htp
          precision: Q8_0
          loads: true
          quality: {metric: wikitext_ppl, value: 11.66, reference_f16: 11.64}
          latency: {decode_tok_s: 14.4, prefill_speedup_vs_cpu: 1.74, prefill_tok_s: 266.7}
          deferred: {genie_v73_comparison: "no w4a16-Genie-on-v73 bundle built"}
"""


def _fixture_registry(tmp_path):
    p = tmp_path / "runtime_index.yaml"
    p.write_text(_FIXTURE)
    return Registry(index=str(p))


def test_registry_threads_runtime_and_matrix_onto_variants(tmp_path):
    vs = _fixture_registry(tmp_path).variants("qwen2.5-1.5b-instruct")
    htp = next(v for v in vs if v.effective_runtime == "llamacpp-htp")
    assert htp.runtime == "llamacpp-htp" and htp.quant == "Q8_0" and htp.ext == "gguf"
    assert isinstance(htp.runtime_support, dict)
    assert htp.runtime_support["llamacpp-htp"]["v68"] == "unsupported:htp-backend-excludes-v68"


def test_record_carries_runtime_precision_ppl_and_latency(tmp_path):
    vs = _fixture_registry(tmp_path).variants("qwen2.5-1.5b-instruct")
    htp = next(v for v in vs if v.effective_runtime == "llamacpp-htp")
    rec = htp.verified
    assert rec["runtime"] == "llamacpp-htp"               # which runtime
    assert rec["precision"] == "Q8_0"                     # its precision
    assert rec["quality"]["value"] == 11.66              # on-device quality (PPL)
    assert rec["quality"]["reference_f16"] == 11.64
    assert rec["latency"]["decode_tok_s"] == 14.4         # decode tok/s
    assert rec["latency"]["prefill_speedup_vs_cpu"] == 1.74   # prefill (dispatch proof)
    assert "deferred" in rec                              # the Genie-v73 head-to-head is recorded as deferred


def test_resolve_over_fixture_registry_quality_vs_v68_refusal(tmp_path):
    vs = _fixture_registry(tmp_path).variants("qwen2.5-1.5b-instruct")
    # SC-RT01 on the registry: v73 + quality -> the Q8_0 HTP variant
    d = resolve(TargetKey(htp_arch="v73", qairt="2.37.1"), vs, priority="quality")
    assert d.ok and d.variant.effective_runtime == "llamacpp-htp"
    # SC-RT02 on the registry: v68 + llamacpp-htp -> FAIL axis=runtime
    d = resolve(TargetKey(htp_arch="v68", qairt="2.37.1"), vs, want_runtime="llamacpp-htp")
    assert d.outcome is Outcome.FAIL and d.axis == "runtime"


if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for f in fns:
        if "tmp_path" in f.__code__.co_varnames[:f.__code__.co_argcount]:
            with tempfile.TemporaryDirectory() as d:
                f(Path(d))
        else:
            f()
        print("ok", f.__name__)
    print(f"\n{len(fns)} passed")
