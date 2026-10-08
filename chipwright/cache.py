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


def fetch(url: str, sha256: str) -> str:
    """Download url into the content-addressed cache, verify the hash, return the local path."""
    dst = path_for(sha256)
    if have(sha256):
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
