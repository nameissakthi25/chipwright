"""Vendor-neutral bundle runner. Reads bundle.yaml, wires host.pre -> device graph(s) -> host.post.

Usage:
  LAYA_BOARD_HOST=192.168.31.41 LAYA_BOARD_USER=root LAYA_BOARD_PASS=... \
  python -m runtime.run --bundle bundles/laya-qcs6490 \
         --text "Your account is locked, verify at http://bit.ly/x" --question "Is this phishing?"
"""
import os, sys, argparse, importlib.util, subprocess, yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # repo root, so `runtime.*` imports work
from runtime import fetch                                    # noqa: E402
from runtime.devices.qualcomm import QualcommDevice          # noqa: E402

MASK_FOR = {128: "mask_128", 256: "mask_256"}


def _load_host(bundle_dir, mod_name):
    spec = importlib.util.spec_from_file_location(f"bundle_{mod_name}", os.path.join(bundle_dir, mod_name + ".py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _board_has(access, path):
    pw, host, user = os.environ[access["pass_env"]], os.environ[access["host_env"]], os.environ[access["user_env"]]
    r = subprocess.run(["sshpass", "-p", pw, "ssh", "-o", "StrictHostKeyChecking=no",
                        "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=20",
                        "-o", "PreferredAuthentications=password", "-o", "PubkeyAuthentication=no",
                        f"{user}@{host}", f"test -f {path} && echo OK"], capture_output=True, text=True, timeout=40)
    return "OK" in r.stdout


def _provision_board_ctx(access, board_ctx, local_ctx):
    """Push the context binary to the board if it isn't already there."""
    pw, host, user = os.environ[access["pass_env"]], os.environ[access["host_env"]], os.environ[access["user_env"]]
    subprocess.run(["sshpass", "-p", pw, "ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                    "-o", "PreferredAuthentications=password", "-o", "PubkeyAuthentication=no",
                    f"{user}@{host}", f"mkdir -p {os.path.dirname(board_ctx)}"], check=True, timeout=40)
    subprocess.run(["sshpass", "-p", pw, "scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                    "-o", "PreferredAuthentications=password", "-o", "PubkeyAuthentication=no",
                    local_ctx, f"{user}@{host}:{board_ctx}"], check=True, timeout=1200)


def run(bundle_dir, items, graph_id=None):
    b = yaml.safe_load(open(os.path.join(bundle_dir, "bundle.yaml")))
    graph = next(g for g in b["graphs"] if graph_id is None or g["id"] == graph_id)
    access = b["device_access"]
    L = int(graph["context_len"]); mask_id = MASK_FOR[L]
    cache = os.path.join(bundle_dir, ".artifacts")

    # host needs the mask helper locally; board needs the context binary
    arts = fetch.ensure_many(b["artifacts"], [mask_id], cache)
    board_ctx = graph["board_ctx"]
    if not _board_has(access, board_ctx):
        print("  provisioning context binary to board ...", flush=True)
        local_ctx = fetch.ensure_many(b["artifacts"], [graph["artifact"]], cache)[graph["artifact"]]
        _provision_board_ctx(access, board_ctx, local_ctx)

    dev = QualcommDevice(board_ctx, access)
    host = _load_host(bundle_dir, b["host"]["pre"]["module"])
    env = {"graph": graph, "artifacts": arts, "mask_id": mask_id, "bundle_dir": bundle_dir}

    ctx = {}
    feeds = getattr(host, b["host"]["pre"]["fn"])({"items": items}, ctx, env)
    outs = dev.run(feeds, graph["outputs"])
    return getattr(host, b["host"]["post"]["fn"])(outs, ctx, env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--question", default="Is this a phishing or scam attempt?")
    ap.add_argument("--qname", default="decision")
    ap.add_argument("--qtype", default="noul")
    ap.add_argument("--graph", default=None)
    a = ap.parse_args()
    q = {a.qname: {"type": a.qtype, "instructions": a.question}}
    res = run(a.bundle, [(a.text, q)], a.graph)
    import json
    print(json.dumps(res[0], indent=2, default=float))


if __name__ == "__main__":
    main()
