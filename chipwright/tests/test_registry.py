"""Registry — load the bundled registry/index.yaml (read-only) and check Variant parsing.

Uses the real seed index so the test fails if the shipped index stops parsing. Also runnable as
`python3 test_registry.py`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chipwright.registry import DEFAULT_INDEX, Registry


def test_bundled_index_exists_and_loads():
    assert os.path.exists(DEFAULT_INDEX), DEFAULT_INDEX
    reg = Registry()
    assert reg.schema == 1
    models = reg.models()
    for expected in ("laya-decision", "whisper-small-en", "zipformer-enin", "qwen2.5-1.5b-instruct"):
        assert expected in models, expected


def test_each_model_yields_variants():
    reg = Registry()
    for model in reg.models():
        vs = reg.variants(model)
        assert vs, f"{model} has no variants"
        for v in vs:
            assert v.model == model
            assert v.arch and v.quant and v.shape


def test_hosted_variant_has_artifact():
    reg = Registry()
    laya = reg.variants("laya-decision")[0]
    assert laya.url is not None
    assert laya.sha256 is not None
    assert laya.has_artifact is True


def test_recipe_only_variant_has_recipe_not_artifact():
    reg = Registry()
    whisper = reg.variants("whisper-small-en")[0]
    assert whisper.url is None and whisper.local is None
    assert whisper.has_artifact is False
    assert whisper.has_recipe is True
    assert whisper.recipe.endswith(".yaml")


def test_local_field_parses_as_on_disk_artifact():
    reg = Registry()
    zf = reg.variants("zipformer-enin")[0]
    # a `local:` path counts as a present artifact even with no url
    assert zf.url is None
    assert zf.local is not None and zf.local.endswith(".bin")
    assert zf.has_artifact is True


def test_run_field_parses_into_dict():
    reg = Registry()
    zf = reg.variants("zipformer-enin")[0]
    assert isinstance(zf.run, dict)
    assert zf.run["modality"] == "asr_encoder"
    assert zf.run["threshold"] == 0.95
    for key in ("board_name", "meta", "cpu_onnx", "input"):
        assert key in zf.run


def test_verified_record_parses():
    reg = Registry()
    laya = reg.variants("laya-decision")[0]
    assert isinstance(laya.verified, dict)
    assert laya.verified["loads"] is True


def test_unknown_model_returns_empty_list():
    reg = Registry()
    assert reg.variants("does-not-exist") == []


if __name__ == "__main__":
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for f in fns:
        f(); print("ok", f.__name__)
    print(f"\n{len(fns)} passed")
