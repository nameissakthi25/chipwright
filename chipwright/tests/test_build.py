"""The build verb is testable with no QAIRT and no board: the recipe loader and the op-check rule
table are pure, and the executors are exercised with the external tools MOCKED (shutil.which +
subprocess.run monkeypatched). So this suite runs green on a laptop with nothing installed.
"""
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chipwright import build, config, recipe, sdk
from chipwright.errors import (ContextGenError, FidelityBelowThreshold,
                               OpUnsupported, PreflightError)

# --- a minimal valid recipe document -----------------------------------------------------------
RECIPE_YAML = """
name: whisper-small-en
version: 1.2.0
route: classic
source:
  format: onnx
  sha256: deadbeef
  url: hf://openai/whisper-small.en/encoder.onnx
target:
  arch: v68
  quant: w8a16
  shape: win30s
  sdk_range: ">=2.34,<2.40"
adaptations:
  - static_shape: {bucket: win30s, dims: {audio: [1, 80, 3000]}}
  - erf_to_tanh: {ops: [Erf]}
  - mask_clamp: {from: -3.4e38, to: -1e4}
quant:
  weights: 8
  activations: 16
  bias: 32
  per_channel: true
  calibration_ref: calib/whisper_encoder_30s.npz
  algorithm: min_max
host:
  pre: modalities.asr.frontend:log_mel
  post: modalities.asr.decoders.ctc:greedy
  deps: [numpy]
graphs:
  - {name: encoder, runs_on: npu}
  - {name: decode, runs_on: host}
"""


def _write(text, suffix=".yaml"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.write(fd, text.encode())
    os.close(fd)
    return path


def _recipe():
    return recipe.load(_write(RECIPE_YAML))


# --- recipe loader -----------------------------------------------------------------------------
def test_recipe_loads_full_schema():
    r = _recipe()
    assert r.name == "whisper-small-en" and r.version == "1.2.0" and r.route == "classic"
    assert r.source.format == "onnx" and r.source.sha256 == "deadbeef"
    assert r.target.arch == "v68" and r.target.sdk_range == ">=2.34,<2.40"
    assert r.quant.activations == 16 and r.quant.bias == 32 and r.quant.per_channel is True
    assert r.quant_tag == "w8a16"
    assert r.host.pre == "modalities.asr.frontend:log_mel"
    assert [g.runs_on for g in r.graphs] == ["npu", "host"]
    assert len(r.npu_graphs) == 1
    assert r.adaptation("erf_to_tanh") is not None and r.adaptation("nope") is None


def test_recipe_rejects_bad_runs_on():
    bad = RECIPE_YAML.replace("runs_on: host", "runs_on: gpu")
    try:
        recipe.load(_write(bad)); assert False, "expected ValueError"
    except ValueError:
        pass


def test_recipe_rejects_unknown_route_and_missing_field():
    for mut in [RECIPE_YAML.replace("route: classic", "route: magic"),
                RECIPE_YAML.replace("  arch: v68\n", "")]:
        try:
            recipe.load(_write(mut)); assert False, "expected ValueError"
        except ValueError:
            pass


# --- op-check rule table (fed a fake op list, no onnx file needed) ------------------------------
class _FakeTarget:
    def __init__(self, arch): self.htp_arch = arch


def _patch_op_scan(monkey_ops, fp16=False):
    """Patch sdk._onnx_op_types to return a fixed op list so the rule table is tested directly."""
    orig = sdk._onnx_op_types
    sdk._onnx_op_types = lambda path: (monkey_ops, {"fp16": fp16}, None)
    return orig


def test_will_it_run_flags_erf_and_gather():
    orig = _patch_op_scan(["Conv", "Erf", "Gather", "MatMul"])
    try:
        # point at a real existing file so the os.path.exists guard passes; content is ignored (scan patched)
        rep = sdk.will_it_run(__file__, _FakeTarget("v68"))
    finally:
        sdk._onnx_op_types = orig
    assert rep.supported is True                        # Erf/Gather are adaptable, not unsupported
    flagged = {f["op"] for f in rep.flags}
    assert "Erf" in flagged and "Gather" in flagged


def test_will_it_run_marks_truly_unsupported_op():
    orig = _patch_op_scan(["Conv", "NonMaxSuppression"])
    try:
        rep = sdk.will_it_run(__file__, _FakeTarget("v68"))
    finally:
        sdk._onnx_op_types = orig
    assert rep.supported is False and "NonMaxSuppression" in rep.unsupported


def test_will_it_run_flags_fp16_on_v6x():
    orig = _patch_op_scan(["Conv"], fp16=True)
    try:
        rep = sdk.will_it_run(__file__, _FakeTarget("v68"))
    finally:
        sdk._onnx_op_types = orig
    assert any(f["op"] == "<fp16-tensor>" for f in rep.flags)


def test_will_it_run_degrades_without_onnx_file():
    rep = sdk.will_it_run("/no/such/model.onnx", _FakeTarget("v68"))
    assert rep.supported is True and rep.note                 # graceful, not a crash


# --- tool mocking ------------------------------------------------------------------------------
class _FakeCompleted:
    def __init__(self, returncode=0, stdout="ok", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def _mock_tools(present=True, returncode=0):
    """Return (which, run) replacements. present=False ⇒ tool missing (PreflightError path)."""
    def which(name, path=None):
        return f"/opt/qairt/bin/{name}" if present else None
    def run(cmd, capture_output=True, text=True, timeout=None):
        return _FakeCompleted(returncode=returncode)
    return which, run


def _env(tmp):
    return config.Env(cache_dir=tmp)


# --- executor guards: missing tool raises PreflightError ---------------------------------------
def test_missing_tool_raises_preflight():
    which, run = _mock_tools(present=False)
    orig_which, orig_run = build.shutil.which, build.subprocess.run
    build.shutil.which, build.subprocess.run = which, run
    try:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                build.convert(_recipe(), _env(tmp)); assert False, "expected PreflightError"
            except PreflightError:
                pass
    finally:
        build.shutil.which, build.subprocess.run = orig_which, orig_run


def test_classic_route_reports_preflight_axis_when_tool_missing():
    which, run = _mock_tools(present=False)
    orig_which, orig_run = build.shutil.which, build.subprocess.run
    build.shutil.which, build.subprocess.run = which, run
    try:
        with tempfile.TemporaryDirectory() as tmp:
            res = build.classic(_recipe(), _env(tmp))
    finally:
        build.shutil.which, build.subprocess.run = orig_which, orig_run
    assert res.ok is False and res.failed_axis == "preflight" and res.artifact_path is None


# --- mocked successful chain returns a BuildResult(ok=True) ------------------------------------
def test_classic_chain_succeeds_mocked():
    which, run = _mock_tools(present=True, returncode=0)
    orig_which, orig_run = build.shutil.which, build.subprocess.run
    build.shutil.which, build.subprocess.run = which, run
    try:
        with tempfile.TemporaryDirectory() as tmp:
            res = build.classic(_recipe(), _env(tmp))
    finally:
        build.shutil.which, build.subprocess.run = orig_which, orig_run
    assert res.ok is True and res.failed_axis is None
    assert res.artifact_path and res.artifact_path.endswith(".bin")
    assert res.tag == "whisper-small-en-1.2.0+qnn2.34-htpv68-w8a16-win30s.bin"
    assert any("qairt-converter" in line for line in res.logs)
    assert any("qnn-context-binary-generator" in line for line in res.logs)


def test_dlc_route_ships_dlc_mocked():
    which, run = _mock_tools(present=True, returncode=0)
    orig_which, orig_run = build.shutil.which, build.subprocess.run
    build.shutil.which, build.subprocess.run = which, run
    try:
        r = _recipe(); r.route = "dlc_route"
        with tempfile.TemporaryDirectory() as tmp:
            res = build.build(r, _env(tmp))      # dispatch by recipe.route
    finally:
        build.shutil.which, build.subprocess.run = orig_which, orig_run
    assert res.ok is True and res.artifact_path.endswith(".dlc")
    assert res.tag.endswith(".dlc")
    assert any("on-board" in line for line in res.logs)


def test_context_tool_failure_raises_contextgen_axis():
    # converter/quantizer succeed, context generator returns nonzero → ContextGenError, failed_axis=context
    def which(name, path=None):
        return f"/opt/qairt/bin/{name}"
    def run(cmd, capture_output=True, text=True, timeout=None):
        rc = 1 if "context-binary-generator" in cmd[0] else 0
        return _FakeCompleted(returncode=rc, stderr="VTCM overflow")
    orig_which, orig_run = build.shutil.which, build.subprocess.run
    build.shutil.which, build.subprocess.run = which, run
    try:
        with tempfile.TemporaryDirectory() as tmp:
            res = build.classic(_recipe(), _env(tmp))
            # and the executor raises the typed error directly
            dlc = os.path.join(tmp, "x.dlc")
            raised = False
            try:
                build.context(dlc, _FakeTarget("v68"), _env(tmp))
            except ContextGenError:
                raised = True
    finally:
        build.shutil.which, build.subprocess.run = orig_which, orig_run
    assert res.ok is False and res.failed_axis == "context"
    assert raised


# --- host preflight + config roundtrip ---------------------------------------------------------
def test_host_preflight_reports_missing_tools():
    orig = config.shutil.which
    config.shutil.which = lambda name, path=None: None
    try:
        pf = config.host_preflight(config.Env())
    finally:
        config.shutil.which = orig
    assert pf.ok is False and set(config.REQUIRED_TOOLS).issubset(set(pf.missing))


def test_env_write_read_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, ".qualcomm-env")
        e = config.Env(hosts=[{"name": "rb3", "host": "10.0.0.5"}], sdk_roots=["/opt/qairt/2.37"],
                       keys=["~/.ssh/id_rsa"], cache_dir=os.path.join(tmp, "cache"))
        config.write_env(e, path)
        back = config.read_env(path)
    assert back.hosts[0]["name"] == "rb3" and back.sdk_root == "/opt/qairt/2.37"
    assert back.cache_dir.endswith("cache")


# --- fidelity gate (the publish gate, pure) ----------------------------------------------------
def test_fidelity_gate_rejects_low_cosine():
    try:
        build.fidelity_gate({"loads": True, "cosine_min": 0.80, "threshold": 0.99}); assert False
    except FidelityBelowThreshold:
        pass
    # a passing report does not raise
    build.fidelity_gate({"loads": True, "cosine_min": 0.995, "threshold": 0.99})


if __name__ == "__main__":
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_") and isinstance(f, types.FunctionType)]
    for f in fns:
        f(); print("ok", f.__name__)
    print(f"\n{len(fns)} passed")
