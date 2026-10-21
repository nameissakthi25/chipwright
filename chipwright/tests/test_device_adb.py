"""AdbDevice — the adb transport, with `adb` MOCKED (shutil.which + subprocess.run swapped).

No board and no `adb` binary needed: the suite runs green on a laptop with nothing installed. Also
runnable as `python3 test_device_adb.py`. Mirrors test_build.py's mock discipline (swap the module's
`shutil.which` / `subprocess.run`, restore in finally).
"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chipwright import device_adb
from chipwright.device_adb import AdbDevice


def _fake_run(returncode=0, stdout="", stderr=""):
    """A subprocess.run replacement that records argv and returns a canned CompletedProcess."""
    calls = []

    def run(argv, capture_output=True, text=True, timeout=None):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, returncode, stdout, stderr)

    return run, calls


def _patch(which=lambda n, path=None: "/usr/bin/adb", run=None):
    """Swap device_adb's shutil.which / subprocess.run / time.sleep; return a restore() callable."""
    orig = (device_adb.shutil.which, device_adb.subprocess.run, device_adb.time.sleep)
    device_adb.shutil.which = which
    if run is not None:
        device_adb.subprocess.run = run
    device_adb.time.sleep = lambda *a, **k: None  # never sleep on a retry in tests

    def restore():
        device_adb.shutil.which, device_adb.subprocess.run, device_adb.time.sleep = orig

    return restore


def test_require_adb_refuses_by_name_when_missing():
    restore = _patch(which=lambda n, path=None: None)
    try:
        AdbDevice()
    except RuntimeError as e:
        assert "adb" in str(e) and "platform-tools" in str(e)
    else:
        raise AssertionError("AdbDevice() must refuse when `adb` is absent")
    finally:
        restore()


def test_shell_returns_completed_process():
    run, calls = _fake_run(returncode=0, stdout="hello\n")
    restore = _patch(run=run)
    try:
        dev = AdbDevice(serial="ABC123")
        r = dev.shell("echo hello")
        assert r.returncode == 0 and r.stdout == "hello\n"
        # the serial is pinned into the argv and `shell` is the verb
        assert calls[-1][:4] == ["adb", "-s", "ABC123", "shell"]
    finally:
        restore()


def test_push_file_one_file_pushes():
    run, calls = _fake_run(returncode=0)
    restore = _patch(run=run)
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".bin", delete=False) as f:
            f.write("x")
            local = f.name
        dev = AdbDevice()
        remote = dev.push_file(local, "/data/local/tmp/chipwright/a.bin")
        assert remote == "/data/local/tmp/chipwright/a.bin"
        # exactly one `adb push` happened (the mkdir of the parent is a `shell`, not a push)
        pushes = [c for c in calls if "push" in c]
        assert len(pushes) == 1 and local in pushes[0]
        os.unlink(local)
    finally:
        restore()


def test_push_file_refuses_a_directory():
    run, _ = _fake_run(returncode=0)
    restore = _patch(run=run)
    try:
        dev = AdbDevice()
        with tempfile.TemporaryDirectory() as d:
            try:
                dev.push_file(d, "/data/local/tmp/chipwright/dir")
            except ValueError as e:
                assert "directory" in str(e) and "wedge" in str(e)
            else:
                raise AssertionError("push_file must refuse a directory (it wedges the adb daemon)")
    finally:
        restore()


def test_push_file_raises_on_adb_failure():
    run, _ = _fake_run(returncode=1, stderr="no space left")
    restore = _patch(run=run)
    try:
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("x")
            local = f.name
        dev = AdbDevice()
        try:
            dev.push_file(local, "/data/local/tmp/x.bin")
        except RuntimeError as e:
            assert "adb push failed" in str(e) and "no space left" in str(e)
        else:
            raise AssertionError("a non-zero adb push must raise, naming the failure")
        os.unlink(local)
    finally:
        restore()


def test_exists_parses_ok_marker():
    run_true, _ = _fake_run(returncode=0, stdout="OK\n")
    restore = _patch(run=run_true)
    try:
        assert AdbDevice().exists("/data/local/tmp/present") is True
    finally:
        restore()
    run_false, _ = _fake_run(returncode=0, stdout="")  # command ran, no OK => absent
    restore = _patch(run=run_false)
    try:
        assert AdbDevice().exists("/data/local/tmp/absent") is False
    finally:
        restore()


if __name__ == "__main__":
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for f in fns:
        f(); print("ok", f.__name__)
    print(f"\n{len(fns)} passed")
