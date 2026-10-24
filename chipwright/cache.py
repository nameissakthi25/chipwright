"""Content-addressed artifact cache at ~/.cache/chipwright/.

Addressed by sha256 so two models that share a build never store it twice (brief §"The manifest and
the registry"). Verifies the hash on fetch and on read — a corrupted or truncated download is caught
here, not at load on the board where the error reads like file corruption.

Hardening (spec 001, "refuse rather than guess", at the fetch boundary): every sha is validated as
64-hex before it is trusted (a url with no/!bad sha is refused by name, never fetched); a reference's
filename is confined to its cache dir (a registry-supplied `name` with a separator or `..` cannot write
outside the cache); and a remote url must be https unless CW_ALLOW_INSECURE=1 (so an http CW_REGISTRY /
artifact cannot be silently MITM'd — the sha only protects you over a trusted transport).
"""
from __future__ import annotations

import hashlib
import os
import re
import urllib.request
from urllib.parse import urlparse

CACHE_DIR = os.environ.get("CW_CACHE", os.path.expanduser("~/.cache/chipwright"))

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _require_sha(sha256, what: str = "artifact") -> str:
    """A fetch target MUST carry a 64-hex sha256. Refuse by name rather than fetch unverified bytes."""
    if not isinstance(sha256, str) or not _HEX64.match(sha256):
        raise RuntimeError(
            f"refusing to fetch an unverified {what}: expected a 64-hex sha256, got {sha256!r}")
    return sha256


def _safe_name(name: str) -> str:
    """A reference filename must be a plain basename inside its cache dir — no separator, no `..`, not
    absolute. A registry-supplied name that escapes the cache is refused (it could otherwise land
    sha-matching bytes at an arbitrary host path during a normal run)."""
    if (not isinstance(name, str) or name in ("", ".", "..")
            or os.path.isabs(name) or os.path.basename(name) != name):
        raise RuntimeError(f"unsafe reference filename {name!r}: must be a plain name with no path separator or '..'")
    return name


def _check_scheme(url: str) -> None:
    """A remote fetch must be https (the sha pins integrity only over a transport you trust; over http a
    MITM can serve a matching (url, sha) pair). Local paths pass; opt out with CW_ALLOW_INSECURE=1."""
    scheme = urlparse(url).scheme.lower()
    if scheme in ("", "file"):
        return
    if scheme != "https" and os.environ.get("CW_ALLOW_INSECURE") != "1":
        raise RuntimeError(
            f"refusing an insecure {scheme or 'no'}-scheme url ({url}): remote artifacts/references must be "
            f"https (set CW_ALLOW_INSECURE=1 to override)")


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
    _require_sha(sha256, "artifact")
    _check_scheme(url)
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
    _require_sha(sha256, "artifact")
    return _ensure(url, sha256, path_for(sha256))


def fetch_named(url: str, sha256: str, name: str, sidecars=None) -> str:
    """Fetch a reference file (verified) into a per-reference dir, KEEPING its original filename, so an
    ONNX's external-data sidecar (`model.data` beside `model.onnx`) resolves. `sidecars` = an iterable of
    (url, sha256, name) fetched into the same dir. Returns the local path to `name`.

    The content-addressed `fetch` names a blob by its hash, which breaks a model that loads a neighbour
    by filename — so a run reference (cpu_onnx / input / meta) uses this instead. Every `name` is confined
    to the cache dir (`_safe_name`) so a registry entry cannot write outside it.
    """
    _require_sha(sha256, "reference")
    d = os.path.join(CACHE_DIR, "refs", sha256[:12])
    main = _ensure(url, sha256, os.path.join(d, _safe_name(name)))
    for s_url, s_sha, s_name in (sidecars or []):
        _require_sha(s_sha, "reference sidecar")
        _ensure(s_url, s_sha, os.path.join(d, _safe_name(s_name)))
    return main


def _looks_url(s) -> bool:
    return isinstance(s, str) and s.startswith(("http://", "https://"))


def locate_ref(r: dict, key: str, override=None):
    """Locate a run-block reference (cpu_onnx / input / meta): a local path, or a HOSTED url fetched into
    the cache so a stranger's `run` works, not only the author's.

    A url value is sha-verified via a companion `<key>_sha256` (refusing by name if absent — never an
    unverified reference) and fetched with its original filename so an ONNX's external-data sidecar
    resolves; declare sidecars under `<key>_sidecars: [{url, sha256, name}]`.

    Lives here (device-free) rather than in `run.py` so both the live gate (`run.run_variant`) and the
    offline replay (`replay.rederive`) share one reference locator without either importing the board
    transport. `run._locate_ref` is an alias of this.
    """
    val = override if override is not None else r.get(key)
    if val is None:
        return None
    if _looks_url(val):
        sha = r.get(f"{key}_sha256")
        if not sha:
            raise RuntimeError(
                f"run block {key} is a url but carries no {key}_sha256 — refusing to fetch an unverified "
                f"reference (a reference the gate can't trust is worse than none).")
        sidecars = [(s["url"], s["sha256"], s["name"]) for s in (r.get(f"{key}_sidecars") or [])]
        name = val.rsplit("/", 1)[-1]
        return fetch_named(val, sha, name, sidecars)
    return os.path.expanduser(val)
