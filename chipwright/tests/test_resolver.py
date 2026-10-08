"""The resolver is a pure function — the four outcomes are testable without a board."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chipwright.env import TargetKey
from chipwright.registry import Registry
from chipwright.resolver import Outcome, Variant, resolve
from chipwright.tags import ArtifactTag


def test_tag_roundtrip():
    s = "whisper-small-en-1.2.0+qnn2.37-htpv68-w8a16-win30s.qnnctx"
    t = ArtifactTag.parse(s)
    assert (t.name, t.version, t.sdk, t.arch, t.quant, t.shape, t.ext) == \
           ("whisper-small-en", "1.2.0", "2.37", "v68", "w8a16", "win30s", "qnnctx")
    assert str(t) == s
    assert t.is_context_binary


def test_tag_rejects_garbage():
    for bad in ["whisper.qnnctx", "x-1.0+htpv68.qnnctx", "x-1.0.0+qnn2.37-htpv68-w8a16.qnnctx"]:
        try:
            ArtifactTag.parse(bad); assert False, bad
        except ValueError:
            pass


def _v(**kw):
    base = dict(model="m", version="1.0.0", arch="v68", quant="w8a16", shape="win30s",
                sdk_tested="2.37.1", sdk_compatible=">=2.34,<2.40")
    base.update(kw)
    return Variant(**base)


def test_use_exact():
    t = TargetKey(htp_arch="v68", qairt="2.37.1")
    d = resolve(t, [_v(url="u", sha256="s")])
    assert d.outcome is Outcome.USE and d.ok


def test_use_with_warn_sdk_in_range():
    t = TargetKey(htp_arch="v68", qairt="2.38.0")        # in range, not the tested point
    d = resolve(t, [_v(url="u", sha256="s")])
    assert d.outcome is Outcome.USE_WITH_WARN and d.ok and d.warning


def test_offline_unknown_sdk_warns():
    t = TargetKey(htp_arch="v68", qairt=None)
    d = resolve(t, [_v(url="u", sha256="s")])
    assert d.outcome is Outcome.USE_WITH_WARN


def test_build_recipe_only():
    t = TargetKey(htp_arch="v68", qairt="2.37.1")
    d = resolve(t, [_v(recipe="r.yaml")])                 # no url
    assert d.outcome is Outcome.BUILD and not d.ok


def test_fail_arch_names_axis():
    t = TargetKey(htp_arch="v68", qairt="2.37.1")
    d = resolve(t, [_v(arch="v73", url="u", sha256="s")])
    assert d.outcome is Outcome.FAIL and d.axis == "htp_arch"


def test_fail_sdk_out_of_range_no_recipe():
    t = TargetKey(htp_arch="v68", qairt="2.50.0")        # outside the range, artifact but no recipe
    d = resolve(t, [_v(url="u", sha256="s")])
    assert d.outcome is Outcome.FAIL and d.axis == "qairt"


def test_sdk_out_of_range_but_recipe_builds():
    t = TargetKey(htp_arch="v68", qairt="2.50.0")
    d = resolve(t, [_v(url="u", sha256="s"), _v(recipe="r.yaml")])
    assert d.outcome is Outcome.BUILD


def test_seed_registry_loads():
    reg = Registry()
    assert "laya-decision" in reg.models() and "whisper-small-en" in reg.models()
    v68 = TargetKey(htp_arch="v68", qairt="2.37.1")
    assert resolve(v68, reg.variants("laya-decision")).outcome is Outcome.USE
    assert resolve(v68, reg.variants("whisper-small-en")).outcome is Outcome.BUILD
    assert resolve(v68, reg.variants("qwen2.5-1.5b-instruct")).axis == "htp_arch"


if __name__ == "__main__":
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for f in fns:
        f(); print("ok", f.__name__)
    print(f"\n{len(fns)} passed")
