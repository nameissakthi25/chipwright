"""The agent gate — tool-call fidelity (spec 007 / contracts/tool-call-gate.md).

Layer 1: the gate PASSes a faithful build and FAILs a quant-damaged one (hallucinated names, malformed args,
lost abstention) — SC-AG01, the gate that cannot FAIL is not a gate. Text-space (no logits), so it works on
Genie-style text output (SC-AG02). Mirrors the shipped refusal-probe discipline.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chipwright import verify_llm as v


# --- parsing --------------------------------------------------------------------------------------
def test_parse_accepts_dict_json_and_openai_shapes():
    assert v.parse_tool_call({"name": "get_weather", "arguments": {"location": "Tokyo"}})["name"] == "get_weather"
    assert v.parse_tool_call('{"name":"set_timer","arguments":{"duration_seconds":600}}')["name"] == "set_timer"
    # arguments as a JSON string
    assert v.parse_tool_call({"name": "x", "arguments": '{"a":1}'})["arguments"] == {"a": 1}
    # OpenAI-style {"function": {"name": ...}}
    assert v.parse_tool_call({"function": {"name": "f"}, "arguments": {}})["name"] == "f"


def test_parse_rejects_abstention_and_malformed():
    assert v.parse_tool_call(None) is None
    assert v.parse_tool_call("") is None
    assert v.parse_tool_call("Sure, here's a poem about the sea.") is None   # plain text = abstain
    assert v.parse_tool_call('{"name":"f","arguments":{bad json') is None    # malformed = non-match


def test_match_requires_name_and_args_order_insensitive():
    a = {"name": "f", "arguments": {"x": 1, "y": 2}}
    b = {"name": "f", "arguments": {"y": 2, "x": 1}}        # arg order differs
    assert v.tool_call_match(a, b) is True
    assert v.tool_call_match({"name": "f", "arguments": {"x": 1}}, {"name": "g", "arguments": {"x": 1}}) is False
    assert v.tool_call_match({"name": "f", "arguments": {"x": 1}}, {"name": "f", "arguments": {"x": 2}}) is False


# --- the gate -------------------------------------------------------------------------------------
_REF = [
    {"name": "get_weather", "arguments": {"location": "Tokyo", "unit": "celsius"}},
    {"name": "set_timer", "arguments": {"duration_seconds": 600, "label": "pasta"}},
    {"name": "search_web", "arguments": {"query": "population of Canada"}},
    None, None,                                             # two irrelevant prompts -> abstain
]
_REL = [True, True, True, False, False]


def test_good_build_passes():
    rep = v.tool_call_report(list(_REF), _REF, _REL)
    assert rep["pass"] is True
    assert rep["tool_call_retention"] == 1.0 and rep["abstention_rate"] == 1.0
    assert rep["axis"] == "tool_call"


def test_hallucinated_names_fail_retention():
    chip = [
        {"name": "get_weather_v2", "arguments": {"location": "Tokyo", "unit": "celsius"}},  # hallucinated name
        '{"name":"set_timer","arguments":{bad',                                            # malformed
        {"name": "search_web", "arguments": {"query": "population of Canada"}},            # ok
        None, None,
    ]
    rep = v.tool_call_report(chip, _REF, _REL)
    assert rep["pass"] is False
    assert rep["tool_call_retention"] < 0.9
    assert rep["schema_validity"] < 1.0                     # the malformed call lowers validity


def test_lost_abstention_fails_even_with_perfect_retention():
    """A model that calls a tool on an irrelevant prompt must FAIL — the degenerate-pass control (FR-AG04)."""
    chip = list(_REF)
    chip[3] = {"name": "search_web", "arguments": {"query": "?"}}   # spurious call where it should abstain
    rep = v.tool_call_report(chip, _REF, _REL)
    assert rep["tool_call_retention"] == 1.0
    assert rep["abstention_rate"] < 0.9 and rep["pass"] is False


def test_never_passes_without_both_dimensions_measured():
    # no irrelevant prompts -> abstention is None -> cannot pass
    rep = v.tool_call_report([_REF[0]], [_REF[0]], [True])
    assert rep["abstention_rate"] is None and rep["pass"] is False


def test_retention_denominator_is_what_the_reference_answered():
    """If the reference itself abstained on a 'relevant' prompt (model limit, not a chip bug), it is excluded
    from the retention denominator (FR-AG02 edge case)."""
    ref = [{"name": "f", "arguments": {}}, None]            # ref answered #0, abstained #1
    chip = [{"name": "f", "arguments": {}}, None]
    rep = v.tool_call_report(chip, ref, [True, True])
    assert rep["reference_answered"] == 1 and rep["tool_call_retention"] == 1.0


def test_degenerate_reference_cannot_pass_and_names_the_reason():
    """FR-AG09 (surfaced on silicon by Q4_0, SC-AG05): when the reference answered too few relevant prompts,
    the build is unusable at the source — retention over so few prompts can't isolate the backend, so the gate
    FLAGS reference_degenerate, attributes the FAIL to the reference (not the chip), and can never PASS — even
    when the chip perfectly matches what little the reference did answer."""
    # 5 relevant prompts; the reference (e.g. a Q4_0 build with format collapse) answered only 1 of them.
    ref = [{"name": "f", "arguments": {}}, None, None, None, None, None, None]
    chip = [{"name": "f", "arguments": {}}, None, None, None, None, None, None]   # chip matches the 1, abstains rest
    rel = [True, True, True, True, True, False, False]
    rep = v.tool_call_report(chip, ref, rel)
    assert rep["reference_answered"] == 1
    assert rep["reference_degenerate"] is True
    assert rep["tool_call_retention"] == 1.0            # perfect on the 1 it answered...
    assert rep["abstention_rate"] == 1.0                # ...and abstention is clean...
    assert rep["pass"] is False                         # ...but a degenerate reference can NEVER pass.
    assert "reference_degenerate" in rep.get("reason", "")
    # a healthy reference (>= floor answered) is NOT flagged and can pass
    healthy = v.tool_call_report(list(_REF), _REF, _REL)
    assert healthy["reference_degenerate"] is False and healthy["pass"] is True


# --- the demo invariant ---------------------------------------------------------------------------
def test_demo_good_passes_broken_fails():
    from chipwright.demos import agent_gate as m
    probes = m.load_probes()
    good = v.tool_call_report(m.good_chip(probes), [p["expected"] for p in probes], [p["relevant"] for p in probes])
    bad = v.tool_call_report(m.broken_chip(probes), [p["expected"] for p in probes], [p["relevant"] for p in probes])
    assert good["pass"] is True and bad["pass"] is False
