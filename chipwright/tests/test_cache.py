"""Content-addressed cache — sharded paths, hash of a file, membership, and fetch verification.

No network: the one fetch test replaces urllib.request.urlretrieve with a local writer. CACHE_DIR is
redirected to a temp dir so the real ~/.cache/chipwright is never touched. Runnable as
`python3 test_cache.py`.
"""
import hashlib
import os
import sys
import tempfile
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chipwright import cache

# a valid-looking sha256 (64 hex chars) that nothing in the cache will match
_RANDOM_SHA = "0" * 63 + "1"


def test_path_for_is_content_addressed_and_sharded():
    sha = "ab" + "c" * 62
    p = cache.path_for(sha)
    # sharded by the first two chars, then the full digest as the filename
    assert os.path.basename(p) == sha
    assert os.path.basename(os.path.dirname(p)) == "ab"
    assert os.path.dirname(os.path.dirname(p)).endswith(os.path.join("blobs"))
    # same input -> same path (deterministic, no randomness)
    assert cache.path_for(sha) == p


def test_sha256_matches_hashlib():
    payload = b"chipwright content-addressed blob \x00\x01\x02"
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(payload)
        tmp = f.name
    try:
        assert cache._sha256(tmp) == hashlib.sha256(payload).hexdigest()
    finally:
        os.remove(tmp)


def test_have_false_for_random_hash():
    with _temp_cache():
        assert cache.have(_RANDOM_SHA) is False


def test_fetch_with_wrong_sha_raises_and_leaves_no_file():
    payload = b"the actual downloaded bytes"
    wrong_sha = "f" * 64  # deliberately not the sha of `payload`
    assert hashlib.sha256(payload).hexdigest() != wrong_sha

    def _fake_urlretrieve(url, filename):
        with open(filename, "wb") as fh:
            fh.write(payload)
        return filename, None

    with _temp_cache():
        orig = urllib.request.urlretrieve
        urllib.request.urlretrieve = _fake_urlretrieve
        try:
            raised = False
            try:
                cache.fetch("http://example.invalid/blob", wrong_sha)
            except RuntimeError as e:
                raised = True
                assert "mismatch" in str(e)
            assert raised, "expected RuntimeError on sha mismatch"
            dst = cache.path_for(wrong_sha)
            # neither the final blob nor the .part temp survives a failed verify
            assert not os.path.exists(dst)
            assert not os.path.exists(dst + ".part")
        finally:
            urllib.request.urlretrieve = orig


def test_fetch_with_correct_sha_stores_and_is_have():
    payload = b"a correctly-hashed blob"
    good_sha = hashlib.sha256(payload).hexdigest()

    def _fake_urlretrieve(url, filename):
        with open(filename, "wb") as fh:
            fh.write(payload)
        return filename, None

    with _temp_cache():
        orig = urllib.request.urlretrieve
        urllib.request.urlretrieve = _fake_urlretrieve
        try:
            path = cache.fetch("http://example.invalid/blob", good_sha)
            assert path == cache.path_for(good_sha)
            assert os.path.exists(path)
            assert cache.have(good_sha) is True
            # a second fetch is a cache hit (no re-download needed)
            assert cache.fetch("http://example.invalid/blob", good_sha) == path
        finally:
            urllib.request.urlretrieve = orig


class _temp_cache:
    """Context manager: point cache.CACHE_DIR at a fresh temp dir, restore on exit."""

    def __enter__(self):
        self._orig = cache.CACHE_DIR
        self._dir = tempfile.mkdtemp(prefix="cw-cache-test-")
        cache.CACHE_DIR = self._dir
        return self._dir

    def __exit__(self, *exc):
        cache.CACHE_DIR = self._orig
        import shutil
        shutil.rmtree(self._dir, ignore_errors=True)
        return False


if __name__ == "__main__":
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for f in fns:
        f(); print("ok", f.__name__)
    print(f"\n{len(fns)} passed")
