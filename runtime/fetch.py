"""Download + sha256-verify bundle artifacts into <bundle>/.artifacts/. Idempotent."""
import os, hashlib, urllib.request


def _sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(buf), b""):
            h.update(b)
    return h.hexdigest()


def ensure(artifact, cache_dir):
    """artifact: {id, url, sha256, size}. Returns the verified local path."""
    os.makedirs(cache_dir, exist_ok=True)
    dst = os.path.join(cache_dir, os.path.basename(artifact["url"]))
    if os.path.exists(dst) and (not artifact.get("sha256") or _sha256(dst) == artifact["sha256"]):
        return dst
    print(f"  fetching {artifact['id']} ({artifact.get('size', '?')} B) ...", flush=True)
    tmp = dst + ".part"
    urllib.request.urlretrieve(artifact["url"], tmp)
    if artifact.get("sha256"):
        got = _sha256(tmp)
        if got != artifact["sha256"]:
            os.remove(tmp)
            raise RuntimeError(f"sha256 mismatch for {artifact['id']}: {got} != {artifact['sha256']}")
    os.replace(tmp, dst)
    return dst


def ensure_many(artifacts, ids, cache_dir):
    """Return {id: local_path} for the requested artifact ids."""
    by_id = {a["id"]: a for a in artifacts}
    return {i: ensure(by_id[i], cache_dir) for i in ids}
