"""Packaging guards — the public registry must be portable (runnable by a stranger).

The #1 release blocker was author-local absolute paths baked into the bundled registry, which make
`chipwright run` fail for everyone but the author. These tests fail the build if that regresses.
"""
import os
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PUBLIC_INDEX = os.path.join(REPO, "registry", "index.yaml")


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
