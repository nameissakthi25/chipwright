"""env.py — the adb probe parse, the SM8750->v79 SoC mapping, and skel-load detection.

All offline: `probe_adb` is exercised with `subprocess.run` MOCKED (no board, no `adb`); the skel
detector takes a plain `shell` callable so it needs nothing at all. Also runnable as
`python3 test_env_adb.py`.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chipwright import env
from chipwright.env import SOC_ID, SkelCheck, detect_skel_load, probe_adb


# --- SoC table: SM8750 -> v79 (FR-L05, the one arch that was still missing) ----------------------
def test_soc_id_maps_sm8750_to_v79():
    name, arch = SOC_ID[618]
    assert "SM8750" in name
    assert arch == "v79"


def test_soc_table_now_covers_v73_v79_v81():
    archs = {arch for _name, arch in SOC_ID.values()}
    for a in ("v73", "v79", "v81"):
        assert a in archs, f"{a} missing from SOC_ID"


# --- probe_adb: parse the on-device echo block into a TargetKey ----------------------------------
def _fake_adb(stdout, returncode=0):
    def run(argv, capture_output=True, text=True, timeout=None):
        return subprocess.CompletedProcess(argv, returncode, stdout, "")
    return run


def _patch_run(run):
    orig = env.subprocess.run
    env.subprocess.run = run
    return lambda: setattr(env.subprocess, "run", orig)


def test_probe_adb_parses_sm8750_board():
    out = ("SOC=618\n"
           "NETRUN=/usr/bin/qnn-net-run\n"
           "LIBS=libQnnHtpV79.so libQnnHtpV79Stub.so\n"
           "QAIRT=2.38.0\n"
           "RAM=16219136\n"
           "TEMP=41200\n")
    restore = _patch_run(_fake_adb(out))
    try:
        t = probe_adb(serial="PHONE1")
        assert t.reachable is True
        assert t.soc_id == 618
        assert t.htp_arch == "v79"
        assert "SM8750" in t.soc_name
        assert t.qairt == "2.38.0"
        assert t.ram_kb == 16219136
        assert t.thermal_c == 41.2
    finally:
        restore()


def test_probe_adb_arch_falls_back_to_lib_name_for_unknown_soc():
    # an soc_id not in the table, but a single HTP lib on the board -> arch recovered from the lib name
    out = "SOC=99999\nLIBS=libQnnHtpV79.so\nQAIRT=2.38.0\nRAM=8000000\nTEMP=\n"
    restore = _patch_run(_fake_adb(out))
    try:
        t = probe_adb()
        assert t.htp_arch == "v79"
        assert t.soc_name is None       # unknown soc id -> no marketing name
    finally:
        restore()


def test_probe_adb_unreachable_is_offline_not_error():
    def boom(argv, capture_output=True, text=True, timeout=None):
        raise FileNotFoundError("adb not found")
    restore = _patch_run(boom)
    try:
        t = probe_adb()
        assert t.reachable is False and t.htp_arch == "unknown"
    finally:
        restore()


# --- skel-load detection: a locked image -> device-capability FAIL, not a hang -------------------
def _shell(stdout="", stderr="", returncode=0):
    def shell(cmd, timeout=60):
        return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)
    return shell


def test_detect_skel_load_flags_error_4000():
    sh = _shell(stderr="GraphPrepare: Failed to load skel, error 4000\n")
    r = detect_skel_load(sh)
    assert isinstance(r, SkelCheck)
    assert r.ok is False and bool(r) is False
    assert r.code == 4000
    assert "skel" in r.reason and "device-capability" in r.reason


def test_detect_skel_load_passes_on_clean_init():
    sh = _shell(stdout="qnn-net-run version 2.38.0\nbackend initialized\n")
    r = detect_skel_load(sh)
    assert r.ok is True and bool(r) is True and r.code is None


def test_detect_skel_load_reports_unreachable_shell():
    def shell(cmd, timeout=60):
        raise RuntimeError("device offline")
    r = detect_skel_load(shell)
    assert r.ok is False and "could not run" in r.reason


if __name__ == "__main__":
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for f in fns:
        f(); print("ok", f.__name__)
    print(f"\n{len(fns)} passed")
