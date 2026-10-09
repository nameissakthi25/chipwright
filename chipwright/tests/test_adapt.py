"""Graph-adaptation tests — pure, no board/QAIRT; skip cleanly if `onnx` isn't installed."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chipwright import adapt
from chipwright.recipe import Adaptation

try:
    import onnx
    from onnx import TensorProto, helper
    HAVE_ONNX = True
except ImportError:
    HAVE_ONNX = False


def _reshape_onnx(path, allowzero):
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [6])
    shape = helper.make_tensor("shape", TensorProto.INT64, [1], [6])
    node = helper.make_node("Reshape", ["x", "shape"], ["y"], allowzero=allowzero)
    g = helper.make_graph([node], "g", [inp], [out], [shape])
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)]), path)


def _allowzero_count(path):
    m = onnx.load(path)
    return sum(1 for n in m.graph.node if n.op_type == "Reshape"
               and any(a.name == "allowzero" and a.i == 1 for a in n.attribute))


def test_reshape_allowzero_cleared():
    if not HAVE_ONNX:
        return
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "m.onnx")
        _reshape_onnx(src, allowzero=1)
        assert _allowzero_count(src) == 1
        out, logs = adapt.apply(src, [Adaptation(kind="reshape_allowzero", params={})], d)
        assert _allowzero_count(out) == 0
        assert any("reshape_allowzero" in m for m in logs)


def test_noop_when_not_declared():
    # no supported adaptation declared -> original path returned unchanged, no logs
    out, logs = adapt.apply("/nonexistent.onnx", [], None)
    assert out == "/nonexistent.onnx" and logs == []


def test_missing_file_is_graceful():
    out, logs = adapt.apply("/nope.onnx", [Adaptation(kind="reshape_allowzero", params={})], None)
    assert out == "/nope.onnx" and logs and "skipped" in logs[0]


if __name__ == "__main__":
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for f in fns:
        f(); print("ok", f.__name__)
    print(f"\n{len(fns)} passed" + ("" if HAVE_ONNX else " (onnx-dependent test was a no-op)"))
