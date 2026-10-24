"""Machine interface (spec 006) — `--json` output + the documented exit-code contract.

The thesis of this feature is that a pipeline branches on *fields and codes*, never on regex-scraped
coloured text. These tests pin that: one valid JSON document on stdout with no ANSI, the outcome/axis as
first-class fields, the run verify fields exposed, and each outcome mapped to its published exit code.

Style matches test_cli_ux.py / test_packaging.py: a sys.path shim, a fixture registry passed via
--registry, monkeypatched environment (no real board), and a `_FakeDev` device double.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chipwright import cli

# Two models: one hosted+runnable v68 variant (resolves USE on the tested SDK point, has a run block so
# `run` reaches the verify gate), and one recipe-only variant (resolves BUILD). A wrong --target arch
# drives the FAIL path. No resolution logic is exercised beyond what the resolver already does.
_FIXTURE = """registry_schema: 1
models:
  - model: fixture-run
    variants:
      - arch: v68
        quant: w8a16
        shape: win30s
        sdk_tested: "2.37.1"
        sdk_compatible: ">=2.34,<2.40"
        ext: qnnctx
        url: https://example/fixture.qnnctx
        sha256: deadbeef
        run:
          modality: asr_encoder
          board_name: fx
          cpu_onnx: /tmp/ref.onnx
          input: /tmp/in.npy
          threshold: 0.99
  - model: fixture-build
    variants:
      - arch: v68
        quant: w8a16
        shape: win30s
        sdk_tested: "2.37.1"
        sdk_compatible: ">=2.34,<2.40"
        recipe: recipes/fixture.yaml
"""

_USE_TARGET = "htpv68,qnn2.37.1"       # exact SDK point → USE
_FAIL_TARGET = "htpv73,qnn2.37.1"      # wrong arch → FAIL on htp_arch


@pytest.fixture
def registry_path(tmp_path):
    p = tmp_path / "index.yaml"
    p.write_text(_FIXTURE)
    return str(p)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """No board in the environment (target comes from --target); no colour forcing unless a test sets it."""
    for var in ("CW_BOARD_HOST", "CW_BOARD_USER", "CW_BOARD_PASS", "CW_BOARD_PORT",
                "NO_COLOR", "FORCE_COLOR"):
        monkeypatch.delenv(var, raising=False)


def _mock_device(monkeypatch, npu, cpu):
    """Mock the device + artifact fetch so `run` reaches the real verify gate on crafted NPU/CPU outputs.

    Mirrors test_packaging.py's `_FakeDev`: the transport and the content-addressed fetch are doubled,
    and the modality runner returns the two output dicts the verifier compares. `verify.report` (the real
    one) then computes the cosine — so `verified`/`cosine_min`/`threshold` in the JSON are genuine.
    """
    from chipwright import run as runmod, cache
    from chipwright.modalities import asr_encoder

    class _FakeDev:
        def __init__(self, *a, **k):
            pass

        def ensure_ctx(self, _art, _name, ext=None):
            return "/board/ctx"

    monkeypatch.setattr(runmod, "_board_env", lambda: dict(host="h", user="root", pw="p", port=22))
    monkeypatch.setattr(runmod, "RemoteQNN", _FakeDev)
    monkeypatch.setattr(cache, "fetch", lambda url, sha: "/cache/art")
    monkeypatch.setattr(asr_encoder, "run", lambda dev, ctx, meta, cpu_onnx, inp: (npu, cpu))


# --- US1: structured resolve output -------------------------------------------------------------

def test_json_resolve_is_valid_json_with_no_ansi(registry_path, capsys):
    """SC-J01: `--json resolve` → exactly one valid JSON document, no ANSI, outcome as a field."""
    rc = cli.main(["--registry", registry_path, "--target", _USE_TARGET, "--json", "resolve", "fixture-run"])
    cap = capsys.readouterr()
    assert "\x1b[" not in cap.out                       # no ANSI escape on stdout
    doc = json.loads(cap.out)                           # parses as one document
    assert doc["model"] == "fixture-run"
    assert doc["outcome"] == "USE"
    assert (doc["arch"], doc["quant"], doc["shape"]) == ("v68", "w8a16", "win30s")
    assert doc["warnings"] == []
    assert rc == 0


def test_json_resolve_fail_exposes_outcome_and_axis(registry_path, capsys):
    """SC-J01/J02: on a wrong-arch FAIL, the outcome and the axis are data (not parsed prose); exit 20."""
    rc = cli.main(["--registry", registry_path, "--target", _FAIL_TARGET, "--json", "resolve", "fixture-run"])
    cap = capsys.readouterr()
    assert "\x1b[" not in cap.out
    doc = json.loads(cap.out)
    assert doc["outcome"] == "FAIL"
    assert doc["axis"] == "htp_arch"
    assert rc == 20


def test_json_unknown_model_is_a_json_error_object(registry_path, capsys):
    """An unknown model in --json mode still yields one JSON document (an error object), exit 2."""
    rc = cli.main(["--registry", registry_path, "--target", _USE_TARGET, "--json", "resolve", "nope"])
    doc = json.loads(capsys.readouterr().out)
    assert doc["exit_code"] == 2 and "nope" in doc["error"]
    assert rc == 2


# --- US1: structured run output -----------------------------------------------------------------

def test_json_run_exposes_verified_cosine_threshold(registry_path, capsys, monkeypatch):
    """SC-J03: `--json run` exposes verified/cosine_min/threshold/outputs; a matching pair verifies → 0."""
    _mock_device(monkeypatch, npu={"y": np.array([1.0, 2.0, 3.0])}, cpu={"y": np.array([1.0, 2.0, 3.0])})
    rc = cli.main(["--registry", registry_path, "--target", _USE_TARGET, "--json", "run", "fixture-run"])
    cap = capsys.readouterr()
    assert "\x1b[" not in cap.out                       # progress/human text went to stderr
    doc = json.loads(cap.out)
    assert doc["verified"] is True
    assert doc["cosine_min"] == pytest.approx(1.0)
    assert doc["threshold"] == 0.99
    assert "y" in doc["outputs"]
    assert rc == 0


def test_json_run_verify_mismatch_exits_30(registry_path, capsys, monkeypatch):
    """SC-J02: a mocked verify mismatch (orthogonal outputs → cosine 0) reports verified:false, exit 30."""
    _mock_device(monkeypatch, npu={"y": np.array([1.0, 0.0])}, cpu={"y": np.array([0.0, 1.0])})
    rc = cli.main(["--registry", registry_path, "--target", _USE_TARGET, "--json", "run", "fixture-run"])
    doc = json.loads(capsys.readouterr().out)
    assert doc["verified"] is False
    assert rc == 30


def test_json_run_unexpected_error_still_emits_one_json_doc(registry_path, capsys, monkeypatch):
    """Edge case: an unexpected internal error in --json mode still emits a JSON error object, exit 1 —
    a parser never gets a half-line."""
    from chipwright import run as runmod, cache
    monkeypatch.setattr(cache, "fetch", lambda url, sha: "/cache/art")
    monkeypatch.setattr(runmod, "_board_env", lambda: dict(host=None, user="root", pw=None, port=22))
    rc = cli.main(["--registry", registry_path, "--target", _USE_TARGET, "--json", "run", "fixture-run"])
    cap = capsys.readouterr()
    assert "\x1b[" not in cap.out
    doc = json.loads(cap.out)                           # still exactly one valid JSON document
    assert "error" in doc and doc["exit_code"] == 1
    assert rc == 1


# --- US2: the documented exit-code table --------------------------------------------------------

def test_exit_code_use_is_0(registry_path, capsys):
    rc = cli.main(["--registry", registry_path, "--target", _USE_TARGET, "--json", "resolve", "fixture-run"])
    assert json.loads(capsys.readouterr().out)["outcome"] == "USE"
    assert rc == 0


def test_exit_code_build_only_is_10(registry_path, capsys):
    rc = cli.main(["--registry", registry_path, "--target", _USE_TARGET, "--json", "resolve", "fixture-build"])
    doc = json.loads(capsys.readouterr().out)
    assert doc["outcome"] == "BUILD"
    assert rc == 10


def test_exit_codes_match_the_published_contract():
    """SC-J02: the exit-code map is the versioned contract — distinct, stable numbers."""
    from chipwright import jsonout
    from chipwright.resolver import Outcome
    assert jsonout.exit_code_for_outcome_table() == {
        Outcome.USE: 0, Outcome.USE_WITH_WARN: 11, Outcome.BUILD: 10, Outcome.FAIL: 20,
    }
    assert (jsonout.EXIT_VERIFY_MISMATCH, jsonout.EXIT_NOT_RUNNABLE,
            jsonout.EXIT_USAGE, jsonout.EXIT_UNEXPECTED) == (30, 40, 2, 1)


# --- US1: list is an array ----------------------------------------------------------------------

def test_json_list_emits_an_array_of_variant_records(registry_path, capsys):
    """FR-J02: `--json list` is a JSON array of per-variant records, no ANSI."""
    rc = cli.main(["--registry", registry_path, "--target", _USE_TARGET, "--json", "list"])
    cap = capsys.readouterr()
    assert "\x1b[" not in cap.out
    docs = json.loads(cap.out)
    assert isinstance(docs, list) and len(docs) == 2
    assert {d["model"] for d in docs} == {"fixture-run", "fixture-build"}
    by_model = {d["model"]: d for d in docs}
    assert by_model["fixture-run"]["outcome"] == "USE"
    assert by_model["fixture-build"]["outcome"] == "BUILD"


# --- US3: ANSI hygiene --------------------------------------------------------------------------

def test_piped_human_output_has_no_ansi(registry_path, capsys):
    """SC-J04: human output that is not a TTY (capsys) carries no raw ANSI, even without --json."""
    rc = cli.main(["--registry", registry_path, "--target", _USE_TARGET, "resolve", "fixture-run"])
    out = capsys.readouterr().out
    assert "\x1b[" not in out
    assert "fixture-run" in out
    assert rc == 0


def test_force_color_re_enables_ansi_when_not_a_tty(registry_path, capsys, monkeypatch):
    """SC-J04: FORCE_COLOR=1 re-enables colour even when stdout is not a TTY."""
    monkeypatch.setenv("FORCE_COLOR", "1")
    cli.main(["--registry", registry_path, "--target", _USE_TARGET, "resolve", "fixture-run"])
    assert "\x1b[" in capsys.readouterr().out


def test_no_color_disables_ansi_and_wins_over_force_color(registry_path, capsys, monkeypatch):
    """SC-J04: NO_COLOR disables colour; it takes precedence over FORCE_COLOR."""
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("FORCE_COLOR", "1")
    cli.main(["--registry", registry_path, "--target", _USE_TARGET, "resolve", "fixture-run"])
    assert "\x1b[" not in capsys.readouterr().out


def test_json_output_is_always_plain_even_with_force_color(registry_path, capsys, monkeypatch):
    """SC-J04: --json output is always plain, regardless of FORCE_COLOR."""
    monkeypatch.setenv("FORCE_COLOR", "1")
    cli.main(["--registry", registry_path, "--target", _USE_TARGET, "--json", "resolve", "fixture-run"])
    out = capsys.readouterr().out
    assert "\x1b[" not in out
    json.loads(out)


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-q"])
