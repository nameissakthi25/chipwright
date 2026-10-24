"""The llama.cpp-HTP runtime's resolve + run path, end to end (spec 004 FR-RT07, public promotion).

Layer 1: the public index now carries a silicon-verified Q8_0 llamacpp-htp variant. These assert that
(a) it resolves USE on v73 when the runtime is chosen and FAILs by name on v68, and (b) `run_variant`
routes it to the runtime path — fetch/verify + surface the recipe + the record — instead of the Genie
transport, with no board and no network (a local artifact). The on-silicon proof is the record itself
(`llamacpp_htp.SILICON_RECORD`), re-verified 2026-10-24 on QCS9075/v73.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chipwright import cli, jsonout, run as runner
from chipwright.env import TargetKey
from chipwright.registry import Registry
from chipwright.resolver import Outcome, Variant, resolve


def _htp_variant(local_path):
    return Variant(
        model="qwen2.5-1.5b-instruct", version="1.0.0", arch="v73", quant="Q8_0", shape="ctx4096",
        sdk_tested="b3278e921", sdk_compatible="llama.cpp-HTP (no QAIRT dependency)", ext="gguf",
        local=local_path, runtime="llamacpp-htp",
        runtime_support={"llamacpp-htp": {"v68": "unsupported:needs-v73", "v73": "supported"},
                         "genie": {"v68": "supported", "v73": "supported"}})


# --- resolve (the public variant) -------------------------------------------------------------------
def test_public_index_has_silicon_verified_q8_0_llamacpp_htp_variant():
    reg = Registry(None)                              # the bundled public index
    vs = reg.variants("qwen2.5-1.5b-instruct")
    htp = [v for v in vs if v.effective_runtime == "llamacpp-htp"]
    assert htp, "public index must carry the llamacpp-htp variant"
    v = htp[0]
    assert v.quant == "Q8_0" and v.arch == "v73" and v.url and v.sha256
    assert v.verified and v.verified.get("runtime") == "llamacpp-htp"


def test_resolve_use_on_v73_when_runtime_chosen():
    reg = Registry(None)
    d = resolve(TargetKey("v73", None), reg.variants("qwen2.5-1.5b-instruct"),
                want_runtime="llamacpp-htp")
    assert d.outcome == Outcome.USE
    assert d.variant.effective_runtime == "llamacpp-htp" and d.variant.quant == "Q8_0"


def test_resolve_refuses_llamacpp_htp_on_v68_by_name():
    reg = Registry(None)
    d = resolve(TargetKey("v68", None), reg.variants("qwen2.5-1.5b-instruct"),
                want_runtime="llamacpp-htp")
    assert d.outcome == Outcome.FAIL and d.axis == "runtime"
    assert "v68" in d.reason and "genie" in d.reason


def test_default_runtime_is_the_genie_floor_not_llamacpp():
    """No --runtime, no priority: the conservative genie floor is chosen (and the user is told to choose)."""
    reg = Registry(None)
    d = resolve(TargetKey("v73", None), reg.variants("qwen2.5-1.5b-instruct"))
    assert d.variant.effective_runtime == "genie"     # Q8_0-HTP is opt-in, never the silent default


# --- run (the runtime path) -------------------------------------------------------------------------
def test_run_variant_routes_llamacpp_htp_to_record_and_recipe(tmp_path):
    gguf = tmp_path / "qwen2.5-1.5b-Q8_0.gguf"
    gguf.write_bytes(b"GGUF\x00dummy")
    rep = runner.run_variant(_htp_variant(str(gguf)))
    assert rep["runtime"] == "llamacpp-htp" and rep["pass"] is True
    assert rep["record"]["quality"]["ppl_wikitext_q8_0"] == 11.6631
    assert "--device" in " ".join(rep["recipe"]) and "HTP0" in " ".join(rep["recipe"])
    assert rep["cli_execution"] == "recipe" and rep.get("warn")


def test_run_variant_refuses_when_no_artifact():
    import pytest
    v = _htp_variant(None)
    v.local = None
    with pytest.raises(RuntimeError) as ei:
        runner.run_variant(v)
    assert "no GGUF artifact" in str(ei.value)


def test_human_render_returns_use_with_warn(tmp_path, capsys):
    from chipwright.resolver import Decision
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF\x00")
    rep = runner.run_variant(_htp_variant(str(gguf)))
    code = cli._render_run_report(rep, Decision(Outcome.USE))
    out = capsys.readouterr().out
    assert code == jsonout.EXIT_USE_WITH_WARN
    assert "llamacpp-htp" in out and "2.72" in out and "11.6631" in out
    assert "Traceback" not in out


def test_json_projection_carries_runtime_record(tmp_path):
    from chipwright.resolver import Decision
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF\x00")
    rep = runner.run_variant(_htp_variant(str(gguf)))
    doc = jsonout.run_to_dict("qwen2.5-1.5b-instruct", "v73", Decision(Outcome.USE), rep)
    assert doc["runtime"] == "llamacpp-htp"
    assert doc["record"]["dispatch"]["prefill_speedup"] == 2.72
    assert any("recipe" in w for w in doc.get("warnings", []))
