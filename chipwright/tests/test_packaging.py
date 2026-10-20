"""Packaging guards — the public registry must be portable (runnable by a stranger).

The #1 release blocker was author-local absolute paths baked into the bundled registry, which make
`chipwright run` fail for everyone but the author. These tests fail the build if that regresses.
"""
import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLIC_INDEX = os.path.join(PKG, "data", "index.yaml")  # bundled public index (ships in the wheel)


def test_public_index_has_no_absolute_paths():
    """The bundled public registry must not carry author-local absolute paths."""
    text = open(PUBLIC_INDEX).read()
    offenders = [ln for ln in text.splitlines() if "/Users/" in ln or "/home/" in ln]
    assert not offenders, "author-local absolute paths leaked into the public registry:\n" + "\n".join(offenders)


def test_public_index_parses_and_keeps_records():
    """The public index still parses and every verified record survived the portability strip."""
    doc = yaml.safe_load(open(PUBLIC_INDEX))
    assert doc.get("registry_schema") == 1
    models = doc.get("models", [])
    assert models, "public registry has no models"
    for m in models:
        for v in m.get("variants", []):
            # a variant must not carry a local path; it is runnable via url, or build-only, or a record.
            assert "local" not in v or not str(v.get("local", "")).startswith(("/Users", "/home")), m["model"]


def test_every_variant_is_resolvable_or_buildable():
    """Each variant either hosts an artifact (url+sha256), or is build-only, or carries a record —
    never a dangling local path that only the author can reach."""
    doc = yaml.safe_load(open(PUBLIC_INDEX))
    for m in doc.get("models", []):
        for v in m.get("variants", []):
            has_hosted = bool(v.get("url"))
            has_record = bool(v.get("verified"))
            has_recipe = bool(v.get("recipe") or m.get("recipe") or v.get("source") or m.get("source"))
            assert has_hosted or has_record or has_recipe, (
                f"{m['model']} variant is neither hosted, recorded, nor buildable")


def test_default_index_is_bundled_inside_the_package():
    """The default index must live inside the `chipwright` package so it ships in the wheel and
    resolves after a clean `pip install` — not at a repo-relative path that vanishes once installed."""
    from chipwright.registry import DEFAULT_INDEX
    assert os.path.exists(DEFAULT_INDEX), DEFAULT_INDEX
    assert os.path.abspath(DEFAULT_INDEX).startswith(PKG + os.sep), (
        f"DEFAULT_INDEX {DEFAULT_INDEX} is outside the package dir {PKG}; it won't ship in the wheel")


def test_missing_sshpass_refuses_by_name(monkeypatch):
    """`sshpass` is an undeclared-by-pip system dep; a missing binary must refuse by name with the
    install hint, not raise a cryptic FileNotFoundError deep in subprocess (spec 003 FR-R05)."""
    from chipwright import device
    monkeypatch.setattr(device.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="sshpass"):
        device.RemoteQNN("1.2.3.4", "root", "pw", 22)


def _variant(**kw):
    from chipwright.resolver import Variant
    base = dict(model="x", version="1", arch="v68", quant="w8a16", shape="s",
                sdk_tested="2.37.1", sdk_compatible=">=2.34,<2.40")
    base.update(kw)
    return Variant(**base)


def test_run_refuses_when_no_artifact(monkeypatch):
    """Refusal path 1 (spec 003 FR-R02 / SC-R03): a record-only variant with no hosted url and no
    local path must refuse by name — pointing at `build` and CW_REGISTRY — not a raw traceback."""
    from chipwright import run as runmod
    v = _variant(run={"modality": "asr_encoder"})  # no url, no local
    with pytest.raises(RuntimeError, match="build .* chipwright build|CW_REGISTRY"):
        runmod.run_variant(v)


def test_run_refuses_when_no_cpu_reference(monkeypatch, tmp_path):
    """Refusal path 2 (spec 003 FR-R03 / SC-R03): an artifact present but no CPU reference in the
    active registry must refuse — the verification gate can't run, and a pass is never reported."""
    from chipwright import run as runmod
    art = tmp_path / "a.bin"
    art.write_bytes(b"x")
    v = _variant(local=str(art), run={"modality": "asr_encoder", "board_name": "fx"})  # no cpu_onnx

    class _FakeDev:
        def __init__(self, *a, **k):
            pass

        def ensure_ctx(self, _art, _name):
            return "/board/ctx"

    monkeypatch.setattr(runmod, "_board_env",
                        lambda: dict(host="h", user="root", pw="p", port=22))
    monkeypatch.setattr(runmod, "RemoteQNN", _FakeDev)
    with pytest.raises(RuntimeError, match="CPU reference|verification gate"):
        runmod.run_variant(v)
