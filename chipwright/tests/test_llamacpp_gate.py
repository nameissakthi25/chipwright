"""The llama.cpp-HTP generative gate — perplexity agreement (spec 004 T-RT11, contracts/llamacpp-htp-gate.md).

Layer 1 proves the LOGIC with a faked board: the gate PASSes when HTP0 and CPU perplexity of the same GGUF
agree, FAILs when they diverge (a chip bug — the whole point), refuses by name on a bad preflight, and
refuses when dispatch isn't proven (a PPL measured while silently on CPU is not an HTP gate). `llama-cli` is
never invoked. Layer 2 (the real numbers) is the on-silicon capture pinned into the record.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from chipwright import cli, jsonout, run as runner, verify_llm
from chipwright.resolver import Decision, Outcome, Variant
from chipwright.runtimes import llamacpp_htp as lh

_HTP_BENCH = ("ggml-hex: allocating new registry : ndev 1\nggml-hex: Hexagon Arch version v73\n"
              "ggml-hex: HTP0 new session : uri file:///libggml-htp-v73.so?_dom=cdsp&_session=0\n"
              "| model | backend | dev | test | t/s |\n| --- | --- | --- | --- | --- |\n"
              "| qwen2 1.5B Q8_0 | HTP | HTP0 | pp64 | 592.82 |\n")


class _FakeBoard:
    """Stands in for LlamaHtpBoard: canned probe + perplexity results, no SSH."""
    def __init__(self, ppl_htp=11.6631, ppl_cpu=11.6640, dispatched=True, preflight_ok=True, **_):
        self._htp, self._cpu, self._disp, self._pf = ppl_htp, ppl_cpu, dispatched, preflight_ok
        self.staged = False

    def probe(self):
        if self._pf:
            return {"has_fastrpc": True, "has_cdsprpc": True, "htp_arch": "v73",
                    "staged_htp_libs": {"v73"}, "staged_loader_libs": set(lh.REQUIRED_LOADER_LIBS)}
        return {"has_fastrpc": False, "has_cdsprpc": False, "htp_arch": "v73",
                "staged_htp_libs": set(), "staged_loader_libs": set()}

    def ensure_pack(self, url, sha): self.staged = True
    def ensure_gguf(self, local, name): return f"/data/cw/{name}"

    def bench(self, name, **k):
        return _HTP_BENCH if self._disp else "ggml-hex: ndev 0\nHexagon Arch version v73\n"

    def perplexity(self, name, device="HTP0", ndev=None, chunks=16, **k):
        ppl = self._cpu if ndev == 0 else self._htp
        return ppl, f"Final estimate: PPL = {ppl} +/- 0.49"


def _variant(local="/tmp/x.gguf"):
    v = Variant(model="qwen2.5-1.5b-instruct", version="1.0.0", arch="v73", quant="Q8_0", shape="ctx4096",
                sdk_tested="b3278e921", sdk_compatible="n/a", ext="gguf", local=local, runtime="llamacpp-htp",
                run={"pack_url": "https://h/pack.tgz", "pack_sha": "a"*64, "chunks": 16, "ppl_f16_baseline": 11.6388})
    return v


@pytest.fixture
def board_env(monkeypatch, tmp_path):
    g = tmp_path / "x.gguf"; g.write_bytes(b"GGUF\x00")
    monkeypatch.setenv("CW_BOARD_HOST", "127.0.0.1")
    monkeypatch.setenv("CW_BOARD_PASS", "oelinux123")
    monkeypatch.setenv("CW_BOARD_USER", "root")
    monkeypatch.setenv("CW_BOARD_PORT", "2222")
    return str(g)


# --- the gate math ----------------------------------------------------------------------------------
def test_gate_passes_when_backends_agree():
    g = verify_llm.ppl_agreement_gate(11.6631, 11.6640, 11.6388)
    assert g["pass"] is True
    assert g["backend_delta_pct"] < 1.0
    assert g["lossless_delta_pct"] is not None            # recorded, not gated


def test_gate_FAILS_when_htp_diverges_from_cpu():
    """A chip that computes Q8_0 wrong → HTP PPL ≠ CPU PPL of the identical weights → FAIL."""
    g = verify_llm.ppl_agreement_gate(13.9, 11.66, 11.64)
    assert g["pass"] is False and g["backend_delta_pct"] > 1.0


def test_gate_never_passes_without_both_measurements():
    assert verify_llm.ppl_agreement_gate(11.66, None)["pass"] is False
    assert verify_llm.ppl_agreement_gate(None, 11.66)["pass"] is False


def test_ppl_parse_and_no_cli_in_commands():
    assert lh.ppl_from_stdout("foo\nFinal estimate: PPL = 11.6631 +/- 0.49\n") == 11.6631
    assert lh.ppl_from_stdout("garbage / wedged") is None
    cmd = " ".join(lh.perplexity_command("m.gguf", "/d/pkg-snapdragon/lib", "/d/glibc238", "wiki.raw"))
    assert "llama-perplexity" in cmd and "llama-cli" not in cmd and "--device HTP0" in cmd
    cpu = " ".join(lh.perplexity_command("m.gguf", "/d/pkg-snapdragon/lib", "/d/glibc238", "wiki.raw", ndev=0))
    assert "--device" not in cpu                           # the CPU reference run


# --- the live orchestration (faked board) -----------------------------------------------------------
def test_live_gate_passes_on_agreeing_board(board_env, monkeypatch):
    monkeypatch.setattr("chipwright.device_llamacpp.LlamaHtpBoard",
                        lambda *a, **k: _FakeBoard(ppl_htp=11.6631, ppl_cpu=11.6640))
    rep = runner.run_variant(_variant(board_env))
    assert rep["live"] is True and rep["pass"] is True
    assert rep["gate"]["ppl_htp"] == 11.6631 and rep["dispatch"]["dispatched"] is True


def test_live_gate_FAILS_on_diverging_board(board_env, monkeypatch):
    monkeypatch.setattr("chipwright.device_llamacpp.LlamaHtpBoard",
                        lambda *a, **k: _FakeBoard(ppl_htp=13.9, ppl_cpu=11.66))
    rep = runner.run_variant(_variant(board_env))
    assert rep["pass"] is False and rep["gate"]["pass"] is False


def test_live_gate_refuses_bad_preflight_by_name(board_env, monkeypatch):
    monkeypatch.setattr("chipwright.device_llamacpp.LlamaHtpBoard",
                        lambda *a, **k: _FakeBoard(preflight_ok=False))
    with pytest.raises(RuntimeError) as ei:
        runner.run_variant(_variant(board_env))
    assert "libcdsprpc" in str(ei.value) or "fastrpc" in str(ei.value)


def test_live_gate_refuses_when_dispatch_not_proven(board_env, monkeypatch):
    monkeypatch.setattr("chipwright.device_llamacpp.LlamaHtpBoard",
                        lambda *a, **k: _FakeBoard(dispatched=False))
    with pytest.raises(RuntimeError) as ei:
        runner.run_variant(_variant(board_env))
    assert "dispatch not proven" in str(ei.value)


# --- replay (no board) ------------------------------------------------------------------------------
def test_replay_recomputes_verdict_from_pinned_record():
    v = _variant()
    v.verified = {"on": "QCS9075", "date": "2026-10-24",
                  "vs_reference": {"gate": {"ppl_htp": 11.6631, "ppl_cpu": 11.6640, "ppl_f16_baseline": 11.6388}}}
    rep = runner._llamacpp_replay(v)
    assert rep["replay"] is True and rep["pass"] is True


def test_replay_none_when_no_pinned_ppls():
    assert runner._llamacpp_replay(_variant()) is None


def test_replay_through_the_cli_path(capsys):
    """Drive cli._replay end to end (the integration path a unit test over _llamacpp_replay misses — it
    caught a NameError: Decision wasn't imported in cli)."""
    from chipwright.registry import Registry
    reg = Registry(None)                              # the bundled public index (has the pinned gate)
    code = cli._replay(reg, "qwen2.5-1.5b-instruct")
    out = capsys.readouterr().out
    assert code == 0 and "perplexity agreement" in out and "replay" in out
    assert "Traceback" not in out and "NameError" not in out


# --- render + json shape-safe -----------------------------------------------------------------------
def test_gate_render_and_json(board_env, monkeypatch, capsys):
    monkeypatch.setattr("chipwright.device_llamacpp.LlamaHtpBoard",
                        lambda *a, **k: _FakeBoard(ppl_htp=11.6631, ppl_cpu=11.6640))
    rep = runner.run_variant(_variant(board_env))
    code = cli._render_run_report(rep, Decision(Outcome.USE))
    out = capsys.readouterr().out
    assert code == 0 and "perplexity agreement" in out and "backend Δ" in out and "Traceback" not in out
    doc = jsonout.run_to_dict("qwen2.5-1.5b-instruct", "v73", Decision(Outcome.USE), rep)
    assert doc["gate"]["pass"] is True and doc["mode"] == "live"
