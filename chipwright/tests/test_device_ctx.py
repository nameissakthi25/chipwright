"""ensure_ctx must decide DLC-vs-prebuilt by the registry-declared `ext`, not the filename.

Regression for a real-silicon bug: a HOSTED artifact is fetched into the content-addressed cache under
a hashed filename with NO extension, so `local_path.endswith('.dlc')` was False for a fetched DLC →
ensure_ctx pushed it as a prebuilt context binary and skipped context-gen → the board reported
"Could not create context from binary". Passing `ext` from the variant fixes it. No real SSH: _ssh/_scp
are monkeypatched to record the commands issued.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chipwright import device


class _R:
    def __init__(self, rc=0, out="", err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


def _fakes():
    """Return (calls, ssh, scp) — ssh returns 'OK' to a `test -f` only AFTER a context-gen ran, so the
    pre-gen presence check misses and the post-gen check passes (mirrors a fresh board)."""
    calls, gen = [], {"done": False}

    def ssh(pw, port, dest, *cmd, timeout=600, capture=True):
        c = cmd[0] if cmd else ""
        calls.append(c)
        if "qnn-context-binary-generator" in c:
            gen["done"] = True
        return _R(out="OK" if ("test -f" in c and gen["done"]) else "")

    def scp(pw, port, *args, timeout=1800):
        return _R()

    return calls, ssh, scp


def _dev(monkeypatch):
    monkeypatch.setattr(device.shutil, "which", lambda _: "/usr/bin/sshpass")
    return device.RemoteQNN("1.2.3.4", "root", "pw", 22)


def test_hosted_dlc_hashed_name_still_context_gens_with_ext(monkeypatch):
    """The bug: a fetched DLC has a hashed, extension-less path. With ext='dlc' it MUST context-gen."""
    calls, ssh, scp = _fakes()
    monkeypatch.setattr(device, "_ssh", ssh)
    monkeypatch.setattr(device, "_scp", scp)
    dev = _dev(monkeypatch)
    hashed = "/home/u/.cache/chipwright/blobs/ab/" + "a" * 64   # no .dlc suffix
    ctx = dev.ensure_ctx(hashed, "mobilenet", ext="dlc")
    assert any("qnn-context-binary-generator" in c for c in calls), "ext='dlc' must trigger context-gen"
    assert ctx.endswith("ctx_mobilenet/mobilenet_ctx.bin")


def test_without_ext_falls_back_to_suffix(monkeypatch):
    """Back-compat: no ext → infer from the filename. A .dlc path context-gens; a .bin is pushed as-is."""
    calls, ssh, scp = _fakes()
    monkeypatch.setattr(device, "_ssh", ssh)
    monkeypatch.setattr(device, "_scp", scp)
    dev = _dev(monkeypatch)
    dev.ensure_ctx("/x/model.dlc", "m", ext=None)
    assert any("qnn-context-binary-generator" in c for c in calls)

    calls2, ssh2, scp2 = _fakes()
    monkeypatch.setattr(device, "_ssh", ssh2)
    monkeypatch.setattr(device, "_scp", scp2)
    ctx = dev.ensure_ctx("/x/prebuilt.bin", "m", ext=None)
    assert not any("qnn-context-binary-generator" in c for c in calls2), "a .bin is pushed as-is"
    assert ctx.endswith("/m")


def test_ext_bin_is_pushed_as_is_even_if_path_says_dlc(monkeypatch):
    """ext is authoritative: ext='bin' must NOT context-gen even if the path happens to end .dlc."""
    calls, ssh, scp = _fakes()
    monkeypatch.setattr(device, "_ssh", ssh)
    monkeypatch.setattr(device, "_scp", scp)
    dev = _dev(monkeypatch)
    dev.ensure_ctx("/x/weird.dlc", "m", ext="bin")
    assert not any("qnn-context-binary-generator" in c for c in calls)
