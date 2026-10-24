"""CLI UX refusals — the framework's thesis is `refuse rather than guess`, and a crash is a guess.

Two boundaries used to break it:
  1. a malformed `--target` (e.g. `v68`, the bare arch the tool itself prints) threw a Python
     traceback instead of a clean, axis-naming refusal;
  2. `doctor` said nothing about the per-modality extras the verify path needs, so a user on a base
     install only discovered the gap when a run failed.

These tests pin both. They monkeypatch the environment (no real board) and use a fixture registry,
matching the style in test_packaging.py / test_registry.py.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chipwright import cli

# A minimal portable registry so the resolve path reaches target parsing without touching the
# shipped index.yaml (owned elsewhere). One hosted variant is enough — resolution never runs because
# the bad --target is refused first.
_FIXTURE = """registry_schema: 1
models:
  - model: fixture-model
    variants:
      - arch: v68
        quant: w8a16
        shape: win30s
        sdk_tested: "2.37.1"
        sdk_compatible: ">=2.34,<2.40"
        ext: qnnctx
        url: https://example/fixture.qnnctx
        sha256: deadbeef
"""


@pytest.fixture
def registry_path(tmp_path):
    p = tmp_path / "index.yaml"
    p.write_text(_FIXTURE)
    return str(p)


@pytest.fixture(autouse=True)
def _no_board(monkeypatch):
    """No board in the environment — target comes only from --target, the path under test."""
    for var in ("CW_BOARD_HOST", "CW_BOARD_USER", "CW_BOARD_PASS", "CW_BOARD_PORT"):
        monkeypatch.delenv(var, raising=False)


# --- Fix 1: a malformed --target refuses cleanly, no traceback ----------------------------------

@pytest.mark.parametrize("bad", ["v68", "banana", "qnn2.37"])
def test_malformed_target_exits_2_without_traceback(bad, registry_path, capsys):
    with pytest.raises(SystemExit) as ei:
        cli.main(["--registry", registry_path, "--target", bad, "resolve", "fixture-model"])
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert "Traceback" not in err                 # no raw crash ever reaches the user
    assert f"target '{bad}'" in err               # the refusal names what was typed
    assert "htp" in err                           # and names the axis / expected form


def test_bare_arch_suggests_htp_prefixed_form(registry_path, capsys):
    """A bare arch the tool itself prints (`v68`) earns a specific `did you mean 'htpv68'` hint."""
    with pytest.raises(SystemExit) as ei:
        cli.main(["--registry", registry_path, "--target", "v68", "resolve", "fixture-model"])
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert "did you mean 'htpv68'?" in err
    assert "must start with 'htp'" in err


def test_unknown_token_gets_generic_hint(registry_path, capsys):
    """A token that is not an arch gets the generic example, not a bogus `htpbanana` suggestion."""
    with pytest.raises(SystemExit) as ei:
        cli.main(["--registry", registry_path, "--target", "banana", "resolve", "fixture-model"])
    err = capsys.readouterr().err
    assert "did you mean" not in err
    assert "htpv68,qnn2.37" in err


def test_malformed_target_on_list_also_refuses(registry_path, capsys):
    """`list` turns --target into a TargetKey too — the same boundary, the same clean refusal."""
    with pytest.raises(SystemExit) as ei:
        cli.main(["--registry", registry_path, "--target", "v73", "list"])
    assert ei.value.code == 2
    assert "Traceback" not in capsys.readouterr().err


def test_valid_target_is_unaffected(registry_path, capsys):
    """A well-formed --target still resolves normally — the guard only catches the malformed case."""
    rc = cli.main(["--registry", registry_path, "--target", "htpv68,qnn2.37", "resolve", "fixture-model"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "fixture-model" in out


# --- Fix 2: doctor reports the modality extras --------------------------------------------------

def test_doctor_reports_each_modality_extra_with_install_hint(monkeypatch, capsys):
    """With nothing installed, doctor lists every modality extra as missing and gives its pip hint."""
    monkeypatch.setattr(cli, "_can_import", lambda _m: False)
    rc = cli.main(["doctor"])
    assert rc == 0                                # read-only; absence of extras is not a failure
    out = capsys.readouterr().out
    for extra in ("cv", "asr", "tts", "llm"):
        assert f"pip install chipwright[{extra}]" in out
    assert "onnxruntime" in out and "tokenizers" in out   # the key imports are named
    assert "✗" in out


def test_doctor_marks_installed_extras(monkeypatch, capsys):
    """When the imports resolve, the same section marks the extra installed (✓), no pip hint needed."""
    monkeypatch.setattr(cli, "_can_import", lambda _m: True)
    cli.main(["doctor"])
    out = capsys.readouterr().out
    assert "✓" in out
    assert "pip install chipwright[cv]" not in out


def test_doctor_extras_safe_when_mixed(monkeypatch, capsys):
    """A partial install (onnxruntime present, pillow/tokenizers absent) reports per-extra correctly."""
    present = {"onnxruntime"}
    monkeypatch.setattr(cli, "_can_import", lambda m: m in present)
    cli.main(["doctor"])
    out = capsys.readouterr().out
    # asr/tts need only onnxruntime → installed; cv (PIL) and llm (tokenizers) → missing with a hint
    assert "pip install chipwright[cv]" in out
    assert "pip install chipwright[llm]" in out
    assert "pip install chipwright[asr]" not in out
