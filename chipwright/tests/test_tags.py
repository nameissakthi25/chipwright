"""ArtifactTag — the wheel-tag grammar. Strict parse, round-trip, context-binary classification.

Pure string logic, no board. Also runnable as `python3 test_tags.py`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chipwright.tags import ArtifactTag


def test_parse_roundtrips_valid_tag():
    s = "whisper-small-en-1.2.0+qnn2.37-htpv68-w8a16-win30s.qnnctx"
    t = ArtifactTag.parse(s)
    assert t.name == "whisper-small-en"
    assert t.version == "1.2.0"
    assert t.sdk == "2.37"
    assert t.arch == "v68"
    assert t.quant == "w8a16"
    assert t.shape == "win30s"
    assert t.ext == "qnnctx"
    # str() is the exact inverse of parse()
    assert str(t) == s


def test_parse_handles_hyphenated_name_and_fp_quant():
    # name may contain hyphens (greedy up to the last -<semver>+); quant may be fp16/fp32.
    s = "laya-decision-tiny-10.20.30+qnn2.34-htpv73-fp16-ctx4096.dlc"
    t = ArtifactTag.parse(s)
    assert t.name == "laya-decision-tiny"
    assert t.version == "10.20.30"
    assert t.arch == "v73"
    assert t.quant == "fp16"
    assert t.ext == "dlc"
    assert str(t) == s


def test_parse_strips_surrounding_whitespace():
    s = "m-1.0.0+qnn2.37-htpv68-w8a16-win30s.bin"
    t = ArtifactTag.parse("  " + s + "\n")
    assert str(t) == s


def test_rejects_missing_version():
    for bad in [
        "whisper.qnnctx",                                  # no version at all
        "whisper-small+qnn2.37-htpv68-w8a16-win30s.qnnctx",  # name but no semver
        "whisper-1.0+qnn2.37-htpv68-w8a16-win30s.qnnctx",    # two-part version, not semver
    ]:
        _assert_rejected(bad)


def test_rejects_missing_arch():
    for bad in [
        "whisper-1.0.0+qnn2.37-w8a16-win30s.qnnctx",       # no htp arch segment
        "whisper-1.0.0+qnn2.37-htp68-w8a16-win30s.qnnctx",  # arch missing the 'v'
        "whisper-1.0.0+htpv68-w8a16-win30s.qnnctx",         # no +qnn sdk tag
    ]:
        _assert_rejected(bad)


def test_rejects_bad_extension():
    for bad in [
        "whisper-1.0.0+qnn2.37-htpv68-w8a16-win30s.onnx",   # not an allowed ext
        "whisper-1.0.0+qnn2.37-htpv68-w8a16-win30s",        # no ext
        "whisper-1.0.0+qnn2.37-htpv68-w8a16-win30s.qnnctx.bak",
    ]:
        _assert_rejected(bad)


def test_is_context_binary_true_for_qnnctx_and_bin():
    for ext in ("qnnctx", "bin"):
        s = f"m-1.0.0+qnn2.37-htpv68-w8a16-win30s.{ext}"
        assert ArtifactTag.parse(s).is_context_binary, ext


def test_is_context_binary_false_for_dlc():
    s = "m-1.0.0+qnn2.37-htpv68-w8a16-win30s.dlc"
    assert ArtifactTag.parse(s).is_context_binary is False


def _assert_rejected(bad: str):
    try:
        ArtifactTag.parse(bad)
    except ValueError:
        return
    raise AssertionError(f"expected ValueError for {bad!r}")


if __name__ == "__main__":
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for f in fns:
        f(); print("ok", f.__name__)
    print(f"\n{len(fns)} passed")
