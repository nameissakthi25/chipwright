"""The human run-report renderer is shape-safe (analyzer finding, 0.2.5).

`run_variant` returns a cosine-shaped report for CV/ASR but a tripwire-shaped one for the LLM generative
gate (`{tripwire, latency, pass}` — no `cosine_min`/`per_output`). The human door used to index the cosine
keys unconditionally → a bare KeyError traceback on an LLM success, while the `--json` door was already
shape-safe. These lock both shapes through the single `_render_run_report` helper (Principle IV: the two
front doors don't diverge).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chipwright import cli, jsonout
from chipwright.resolver import Decision, Outcome

_USE = Decision(Outcome.USE)


def test_llm_tripwire_report_renders_without_crash(capsys):
    rep = {"pass": True, "latency": {},
           "tripwire": {"token_match": 0.991, "thresholds": {"token_match": 0.98}}}
    code = cli._render_run_report(rep, _USE)
    out = capsys.readouterr().out
    assert "generative gate" in out and "token_match = 0.99100" in out
    assert "Traceback" not in out
    assert code == 0                               # USE pass → 0


def test_llm_tripwire_fail_exits_verify_mismatch(capsys):
    rep = {"pass": False, "tripwire": {"token_match": 0.50, "thresholds": {"token_match": 0.98}}}
    assert cli._render_run_report(rep, _USE) == jsonout.EXIT_VERIFY_MISMATCH


def test_llm_behavior_block_is_shown_when_present(capsys):
    rep = {"pass": True, "tripwire": {"token_match": 0.99, "thresholds": {"token_match": 0.98}},
           "behavior": {"refusal_retention": 1.0, "benign_answer_rate": 0.96}}
    cli._render_run_report(rep, _USE)
    assert "behavioral probe" in capsys.readouterr().out


def test_cosine_report_still_renders(capsys):
    rep = {"pass": True, "cosine_min": 0.95585, "threshold": 0.95, "per_output": {"class_logits": 0.95585}}
    code = cli._render_run_report(rep, _USE)
    out = capsys.readouterr().out
    assert "NPU vs CPU-reference" in out and "class_logits: 0.95585" in out
    assert code == 0
