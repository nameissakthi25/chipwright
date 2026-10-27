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
def test_run_variant_routes_llamacpp_htp_to_record_and_recipe(tmp_path, monkeypatch):
    monkeypatch.delenv("CW_BOARD_HOST", raising=False)     # no board → the surface path (record + recipe)
    gguf = tmp_path / "qwen2.5-1.5b-Q8_0.gguf"
    gguf.write_bytes(b"GGUF\x00dummy")
    rep = runner.run_variant(_htp_variant(str(gguf)))
    assert rep["runtime"] == "llamacpp-htp" and rep["pass"] is True
    assert rep["record"]["quality"]["ppl_wikitext_q8_0"] == 11.6631
    assert "--device" in " ".join(rep["recipe"]) and "HTP0" in " ".join(rep["recipe"])
    assert rep["cli_execution"] == "recipe-or-replay" and rep.get("warn")


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


def test_json_projection_carries_runtime_record(tmp_path, monkeypatch):
    from chipwright.resolver import Decision
    monkeypatch.delenv("CW_BOARD_HOST", raising=False)     # no board → the surface path
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF\x00")
    rep = runner.run_variant(_htp_variant(str(gguf)))
    doc = jsonout.run_to_dict("qwen2.5-1.5b-instruct", "v73", Decision(Outcome.USE), rep)
    assert doc["runtime"] == "llamacpp-htp"
    assert doc["record"]["dispatch"]["prefill_speedup"] == 2.72
    assert any("board" in w for w in doc.get("warnings", []))


# --- the live AGENT gate on the board (T-AG07b) -----------------------------------------------------
class _FakeGenBoard:
    """A fake board for the agent-gate driver: records pushed prompts, returns canned HTP/CPU calls per
    prompt index (parsed from the tc_pN.txt filename). `generate` returns (call_or_None, stdout)."""
    def __init__(self, htp, cpu):
        import re
        self._re = re
        self.htp, self.cpu, self.pushed = htp, cpu, []

    def push_text(self, text, name):
        self.pushed.append(name)
        return f"/data/cw/{name}"

    def generate(self, gguf, pf, device="HTP0", ndev=None, n_predict=200, ctx=4096):
        i = int(self._re.search(r"tc_p(\d+)", pf).group(1))
        return (self.cpu[i] if ndev == 0 else self.htp[i]), "stdout"


# 3 relevant (>= the REF_ANSWERED_MIN=3 degenerate-guard floor) + 1 irrelevant, so a clean backend-faithful
# PASS is reachable and the degenerate-reference guard doesn't confound the backend check.
_DOC3 = {"tools": [{"name": "get_weather", "parameters": {}}],
         "prompts": [{"prompt": "weather?", "relevant": True},
                     {"prompt": "timer", "relevant": True},
                     {"prompt": "search", "relevant": True},
                     {"prompt": "poem", "relevant": False}]}
_REF4 = ['{"name":"get_weather","arguments":{"location":"Tokyo"}}',
         '{"name":"set_timer","arguments":{"s":60}}',
         '{"name":"search_web","arguments":{"q":"x"}}', None]


def test_live_tool_call_gate_driver_scores_from_board(monkeypatch):
    monkeypatch.setattr(runner, "_load_tool_prompts", lambda spec: _DOC3)
    # backend-faithful: HTP == CPU -> PASS, and the prompt set was pushed in order
    good = _FakeGenBoard(list(_REF4), _REF4)
    rep = runner._llamacpp_tool_call_gate(good, "/x/qwen.gguf", {"tool_prompts": "data/tool_prompts.yaml"})
    assert rep["pass"] is True and rep["tool_call_retention"] == 1.0 and rep["abstention_rate"] == 1.0
    assert rep["reference_degenerate"] is False
    assert good.pushed == ["tc_p0.txt", "tc_p1.txt", "tc_p2.txt", "tc_p3.txt"]
    # a diverging chip (hallucinated name on p0, dropped call on p1) FAILs by the tool_call axis
    bad = _FakeGenBoard(['{"name":"get_weather_v2","arguments":{"location":"Tokyo"}}', None,
                         '{"name":"search_web","arguments":{"q":"x"}}', None], _REF4)
    rep2 = runner._llamacpp_tool_call_gate(bad, "/x/qwen.gguf", {})
    assert rep2["pass"] is False and rep2["tool_call_retention"] < 0.9 and rep2["axis"] == "tool_call"


def test_live_gate_folds_agent_gate_into_overall_pass(monkeypatch):
    """ppl agrees (chip faithful on tensors) but the agent gate catches a hallucinated tool name -> the
    overall run FAILs. The agent gate rides the same staged board + proven dispatch as the ppl gate."""
    import chipwright.device_llamacpp as dl
    from chipwright.runtimes import llamacpp_htp as lh

    class _Proof:
        dispatched = True
        def as_record(self):
            return {"reason": "HTP0 ok", "dispatched": True}

    class _Board:
        def __init__(self, *a, **k):
            self.pushed = []
        def probe(self):
            return {}
        def ensure_pack(self, *a):
            pass
        def ensure_gguf(self, *a):
            return "/data/cw/qwen.gguf"
        def bench(self, *a, **k):
            return "ok"
        def perplexity(self, name, device="HTP0", ndev=None, chunks=16):
            return (11.66, "out")                 # HTP == CPU -> ppl agreement PASSES
        def push_text(self, text, name):
            self.pushed.append(name)
            return "/data/cw/" + name
        def generate(self, gguf, pf, device="HTP0", ndev=None, n_predict=200, ctx=4096):
            import re
            i = int(re.search(r"tc_p(\d+)", pf).group(1))
            # reference answers all 3 relevant (not degenerate); HTP hallucinates the name on p0 only
            htp = ['{"name":"HALLUCINATED","arguments":{"location":"Tokyo"}}', _REF4[1], _REF4[2], None]
            return (_REF4[i] if ndev == 0 else htp[i]), "out"

    monkeypatch.setattr(dl, "LlamaHtpBoard", _Board)
    monkeypatch.setattr(lh, "host_preflight", lambda probe: (True, None))
    monkeypatch.setattr(lh, "dispatch_proof", lambda out: _Proof())
    monkeypatch.setattr(runner, "_board_env", lambda: {"host": "h", "user": "root", "pw": "p", "port": 22})
    monkeypatch.setattr(runner, "_load_tool_prompts", lambda spec: _DOC3)
    res = runner._llamacpp_live_gate(_htp_variant("/x/qwen.gguf"), "/x/qwen.gguf",
                                     {"tool_prompts": "data/tool_prompts.yaml", "ppl_f16_baseline": 11.6388})
    assert res["gate"]["pass"] is True            # the tensor/ppl gate agrees
    assert res["tool_call"]["pass"] is False       # the agent gate caught the hallucinated name
    assert res["pass"] is False                    # overall FAILs — both gates must pass
    # and the --json projection carries the agent-gate block
    from chipwright.resolver import Decision
    doc = jsonout.run_to_dict("qwen2.5-1.5b-instruct", "v73", Decision(Outcome.USE), res)
    assert doc["tool_call"]["pass"] is False
