"""Content-addressed artifact cache at ~/.cache/chipwright/.

Addressed by sha256 so two models that share a build never store it twice (brief §"The manifest and
the registry"). Verifies the hash on fetch and on read — a corrupted or truncated download is caught
here, not at load on the board where the error reads like file corruption.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request

CACHE_DIR = os.environ.get("CW_CACHE", os.path.expanduser("~/.cache/chipwright"))


def _sha256(path: str, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(buf), b""):
            h.update(b)
    return h.hexdigest()


def path_for(sha256: str) -> str:
    return os.path.join(CACHE_DIR, "blobs", sha256[:2], sha256)


def have(sha256: str) -> bool:
    p = path_for(sha256)
    return os.path.exists(p) and _sha256(p) == sha256


def _ensure(url: str, sha256: str, dst: str) -> str:
    """Download url to dst (verified by sha256), skipping if dst already matches. Returns dst."""
    if os.path.exists(dst) and _sha256(dst) == sha256:
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    urllib.request.urlretrieve(url, tmp)
    got = _sha256(tmp)
    if got != sha256:
        os.remove(tmp)
        raise RuntimeError(f"sha256 mismatch for {url}: got {got}, expected {sha256}")
    os.replace(tmp, dst)
    return dst


def fetch(url: str, sha256: str) -> str:
    """Download url into the content-addressed cache, verify the hash, return the local path."""
    return _ensure(url, sha256, path_for(sha256))


def fetch_named(url: str, sha256: str, name: str, sidecars=None) -> str:
    """Fetch a reference file (verified) into a per-reference dir, KEEPING its original filename, so an
    ONNX's external-data sidecar (`model.data` beside `model.onnx`) resolves. `sidecars` = an iterable of
    (url, sha256, name) fetched into the same dir. Returns the local path to `name`.

    The content-addressed `fetch` names a blob by its hash, which breaks a model that loads a neighbour
    by filename — so a run reference (cpu_onnx / input / meta) uses this instead.
    """
    d = os.path.join(CACHE_DIR, "refs", sha256[:12])
    main = _ensure(url, sha256, os.path.join(d, name))
    for s_url, s_sha, s_name in (sidecars or []):
        _ensure(s_url, s_sha, os.path.join(d, s_name))
    return main
