"""`chipwright` — the Chipwright CLI. Milestone 1: resolve-only, one model, one board.

    chipwright doctor                         # probe the board -> SoC, Hexagon arch, SDK, reachability
    chipwright resolve <model> [--target ...] # match a registry variant to the board; explain any miss
    chipwright run <model> --input <file>     # resolve -> fetch+verify -> (push+run+CPU cosine)

Board connection comes from env (CW_BOARD_HOST / _USER / _PASS / _PORT) or flags. With no board,
`resolve` still works against a hand-named --target, and `doctor`/`run` report the offline state
plainly rather than guessing.
"""
from __future__ import annotations

import argparse
import os
import sys

from . import cache
from .env import TargetKey, probe
from .registry import Registry
from .resolver import Outcome, resolve

GREEN, YELLOW, RED, DIM, RST = "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[0m"


def _board_from(args) -> dict:
    return {
        "host": args.host or os.environ.get("CW_BOARD_HOST"),
        "user": args.user or os.environ.get("CW_BOARD_USER", "root"),
        "pw": args.pw or os.environ.get("CW_BOARD_PASS"),
        "port": int(args.port or os.environ.get("CW_BOARD_PORT", 22)),
    }


def _get_target(args) -> TargetKey:
    """A TargetKey from the board if reachable, else from --target, else an error."""
    b = _board_from(args)
    if b["host"] and b["pw"]:
        t = probe(b["host"], b["user"], b["pw"], b["port"])
        if t.reachable:
            return t
        print(f"{YELLOW}board {b['host']} not reachable{RST}", file=sys.stderr)
    if args.target:
        return TargetKey.manual(args.target)
    print(f"{RED}no board reachable and no --target given{RST}", file=sys.stderr)
    sys.exit(2)


def cmd_doctor(args):
    b = _board_from(args)
    if not (b["host"] and b["pw"]):
        print(f"{YELLOW}no board configured{RST} — set CW_BOARD_HOST / CW_BOARD_PASS (or --host/--pw)")
        return 0
    t = probe(b["host"], b["user"], b["pw"], b["port"])
    if not t.reachable:
        print(f"{RED}● unreachable{RST}  {b['user']}@{b['host']}:{b['port']}")
        return 1
    print(f"{GREEN}● board reachable{RST}  {b['user']}@{b['host']}:{b['port']}")
    print(f"  SoC         : {t.soc_name or '?'}" + (f" (soc_id {t.soc_id})" if t.soc_id else ""))
    print(f"  Hexagon arch: {t.htp_arch}")
    print(f"  QNN libs    : {', '.join(t.qnn_libs) or '(none found)'}")
    if t.thermal_c is not None:
        print(f"  temperature : {t.thermal_c} °C")
    print(f"  target key  : {t}")
    return 0


def _print_decision(model, target, d):
    print(f"model   : {model}")
    print(f"target  : {target}")
    if d.outcome == Outcome.USE:
        v = d.variant
        print(f"{GREEN}✔ USE{RST}    {v.model}-{v.version}  htp{v.arch}/{v.quant}/{v.shape} "
              f"qnn{v.sdk_tested}  ({d.reason})")
        if v.verified:
            print(f"  {DIM}verified: {v.verified}{RST}")
    elif d.outcome == Outcome.USE_WITH_WARN:
        v = d.variant
        print(f"{YELLOW}▲ USE (warn){RST}  {v.model}-{v.version}  htp{v.arch}/{v.quant}/{v.shape}")
        print(f"  {YELLOW}warning:{RST} {d.warning}")
    elif d.outcome == Outcome.BUILD:
        v = d.variant
        print(f"{YELLOW}⚙ BUILD{RST}  no matching artifact, but a recipe exists: {v.recipe}")
        print(f"  {DIM}{d.reason}{RST}")
        if v.verified:
            print(f"  {DIM}prior verification on record: {v.verified}{RST}")
    else:
        print(f"{RED}✗ FAIL{RST}  axis={d.axis}: {d.reason}")
    return d


def cmd_resolve(args):
    reg = Registry(args.registry)
    if args.model not in reg.models():
        print(f"{RED}unknown model{RST} '{args.model}'. known: {', '.join(reg.models())}")
        return 2
    target = _get_target(args)
    d = resolve(target, reg.variants(args.model), args.quant, args.shape)
    _print_decision(args.model, target, d)
    return 0 if d.ok else 1


def cmd_run(args):
    reg = Registry(args.registry)
    if args.model not in reg.models():
        print(f"{RED}unknown model{RST} '{args.model}'. known: {', '.join(reg.models())}")
        return 2
    target = _get_target(args)
    d = resolve(target, reg.variants(args.model), args.quant, args.shape)
    _print_decision(args.model, target, d)
    if not d.ok:
        if d.outcome == Outcome.BUILD:
            print(f"{DIM}milestone 1 is resolve-only — building is out of scope; stopping.{RST}")
        return 1
    # USE / USE_WITH_WARN: execute on the board and verify against a CPU reference.
    v = d.variant
    if not (v.run and v.run.get("modality")):
        print(f"{DIM}artifact resolves, but this variant has no run block (modality/meta/cpu_onnx) — "
              f"resolve-only.{RST}")
        if v.url:
            print(f"fetching → cache ({cache.CACHE_DIR}) …")
            print(f"{GREEN}✔ cached + hash-verified{RST}  {cache.fetch(v.url, v.sha256)}")
        return 0
    from . import run as runner
    try:
        rep = runner.run_variant(v, args.input)
    except Exception as e:
        print(f"{RED}run failed:{RST} {e}")
        return 1
    mark = f"{GREEN}✔ VERIFIED{RST}" if rep["pass"] else f"{RED}✗ MISMATCH{RST}"
    print(f"\n{mark}  NPU vs CPU-reference  cosine_min = {rep['cosine_min']:.5f}  "
          f"(threshold {rep['threshold']})")
    for name, c in rep["per_output"].items():
        print(f"    {name}: {c:.5f}")
    return 0 if rep["pass"] else 1


def cmd_build(args):
    import os
    reg = Registry(args.registry)
    if args.model not in reg.models():
        print(f"{RED}unknown model{RST} '{args.model}'. known: {', '.join(reg.models())}")
        return 2
    target = _get_target(args)
    d = resolve(target, reg.variants(args.model), args.quant, args.shape)
    _print_decision(args.model, target, d)
    if d.ok:
        print(f"{DIM}already resolves to USE — nothing to build.{RST}")
        return 0
    if d.outcome != Outcome.BUILD:
        return 1
    v = d.variant
    from . import config, build as builder, errors, recipe as recipe_mod

    env = config.read_env()
    pf = config.host_preflight(env)
    if pf.ok:
        print(f"{GREEN}✔ build host ready{RST}")
    else:
        print(f"{YELLOW}▲ build host not ready{RST} — missing: {', '.join(pf.missing) or 'unknown'}")

    # recipes are repo-relative (sibling of registry/)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rp = v.recipe if os.path.isabs(v.recipe) else os.path.join(repo_root, v.recipe)
    if not os.path.exists(rp):
        print(f"{YELLOW}recipe not found{RST} at {v.recipe} — author it against the spec kit's Recipe "
              f"schema, then re-run. (Build needs an x86_64-linux host with QAIRT installed.)")
        return 1
    try:
        rec = recipe_mod.load(rp)
        res = builder.build(rec, env)
    except errors.BuildError as e:
        axis = getattr(e, "axis", None)
        print(f"{RED}build stopped{RST}: {e}" + (f"  (axis={axis})" if axis else ""))
        return 1
    except Exception as e:
        print(f"{RED}build failed{RST}: {e}")
        return 1
    if res.ok:
        print(f"{GREEN}✔ built{RST}  {res.artifact_path}  tag={res.tag}")
        print(f"{DIM}verify it on the board (chipwright run) before publishing.{RST}")
        return 0
    print(f"{RED}✗ build failed on axis {res.failed_axis}{RST}")
    return 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="chipwright", description="Chipwright — package manager for Qualcomm NPUs")
    ap.add_argument("--registry", default=None, help="index path/URL (default: bundled registry)")
    ap.add_argument("--host"); ap.add_argument("--user"); ap.add_argument("--pw"); ap.add_argument("--port")
    ap.add_argument("--target", help="offline target, e.g. 'htpv68,qnn2.37'")
    ap.add_argument("--quant", help="narrow to a quant, e.g. w8a16")
    ap.add_argument("--shape", help="narrow to a shape profile, e.g. win30s")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="probe the board")
    rp = sub.add_parser("resolve", help="resolve a model against the board"); rp.add_argument("model")
    rnp = sub.add_parser("run", help="resolve, fetch, run on the board, verify vs CPU"); rnp.add_argument("model"); rnp.add_argument("--input")
    bp = sub.add_parser("build", help="resolve; when it misses, build from a recipe"); bp.add_argument("model")
    args = ap.parse_args(argv)
    return {"doctor": cmd_doctor, "resolve": cmd_resolve, "run": cmd_run, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
