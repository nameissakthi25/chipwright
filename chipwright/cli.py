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
import contextlib
import importlib.util
import json
import os
import re
import sys

from . import cache, jsonout
from .env import TargetKey, probe
from .registry import Registry
from .resolver import Decision, Outcome, resolve

# ANSI palette. These are module globals so the whole file can keep printing with them; `_set_palette`
# blanks them when colour is off (not a TTY / NO_COLOR / --json). The literals below are the coloured
# default; the actual values for a given invocation are chosen in `main()`.
GREEN, YELLOW, RED, DIM, RST = "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[0m"


def _color_enabled(json_mode: bool) -> bool:
    """Should colour be emitted? (spec 006 FR-J05)

    `--json` is always plain. Otherwise honour `NO_COLOR` (set → off, any value, per the no-color.org
    convention), then `FORCE_COLOR=1` (on), else gate on whether stdout is a real TTY.
    """
    if json_mode:
        return False
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR") == "1":
        return True
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def _set_palette(enabled: bool):
    """Point the module colour globals at either the ANSI codes or empty strings."""
    global GREEN, YELLOW, RED, DIM, RST
    if enabled:
        GREEN, YELLOW, RED, DIM, RST = "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[0m"
    else:
        GREEN = YELLOW = RED = DIM = RST = ""


def _emit_json(obj):
    """Write exactly one JSON document to the real stdout, and nothing else (FR-J01).

    `default=str` keeps a verification record portable: YAML parses `date: 2026-10-08` into a
    `datetime.date`, which `json` can't serialize — stringify such leaves rather than crash.
    """
    sys.stdout.write(json.dumps(obj, default=str))
    sys.stdout.write("\n")


def _unknown_model(args, reg) -> int:
    """Report an unknown model consistently in both modes; exit code 2 (usage/unknown-model)."""
    if getattr(args, "json", False):
        _emit_json({"error": f"unknown model '{args.model}'. known: {', '.join(reg.models())}",
                    "exit_code": jsonout.EXIT_USAGE})
    else:
        print(f"{RED}unknown model{RST} '{args.model}'. known: {', '.join(reg.models())}")
    return jsonout.EXIT_USAGE

# A bare Hexagon arch (what `list` and FAIL messages print, e.g. "v68") looks like a target but is
# missing the mandatory `htp` prefix. Used to turn a malformed --target into a helpful suggestion.
_ARCH_LIKE = re.compile(r"^v\d+$", re.IGNORECASE)

# Modality extras → the key top-level imports that must resolve for that verify path to work. The
# package list behind each extra is in pyproject.toml ([project.optional-dependencies]); here we name
# the IMPORT each pip package provides (pillow → PIL). `doctor` reports these so a user learns an extra
# is missing up front, not when a run fails.
_MODALITY_EXTRAS = {
    "cv":  ["onnxruntime", "PIL"],          # image classifiers (onnxruntime + pillow)
    "asr": ["onnxruntime"],                 # streaming encoder/decoder/joiner
    "tts": ["onnxruntime"],                 # text-to-speech reference
    "llm": ["onnxruntime", "tokenizers"],   # CPU-reference decoder + genie retokenise
}


def _can_import(module: str) -> bool:
    """True if `module` is importable — a read-only probe (find_spec, never actually imports)."""
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        return False


def _manual_target(spec: str) -> TargetKey:
    """Turn --target into a TargetKey, refusing a malformed spec cleanly instead of a traceback.

    The tool prints bare archs (`v68`) in `list` and in FAIL messages, so a user naturally copies
    `v68` into --target. Rather than crash, we name the axis and — when the token looks like a known
    arch missing its `htp` prefix — suggest the corrected form. Exits 2, matching `refuse rather than
    guess`.
    """
    try:
        return TargetKey.manual(spec)
    except ValueError:
        suggestion = None
        for tok in spec.replace(" ", "").split(","):
            if _ARCH_LIKE.match(tok):
                suggestion = "htp" + tok.lower()
                break
        if suggestion:
            msg = (f"target {spec!r} — did you mean {suggestion!r}? "
                   f"(an arch must start with 'htp', e.g. htpv68,qnn2.37)")
        else:
            msg = (f"target {spec!r} — expected an arch starting with 'htp', "
                   f"e.g. htpv68,qnn2.37")
        print(f"{RED}✗ {msg}{RST}", file=sys.stderr)
        sys.exit(2)


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
        return _manual_target(args.target)
    print(f"{RED}no board reachable and no --target given{RST}", file=sys.stderr)
    sys.exit(2)


def _print_extras_report():
    """Report, per modality extra, whether its verify-path imports resolve. Read-only and safe with
    nothing installed: base `pip install chipwright` ships no onnxruntime/pillow/tokenizers, so a user
    told the product `proves it against a CPU reference` learns here — not mid-run — which extra to add.
    """
    print(f"\n{DIM}modality extras (the verify / CPU-reference path needs these){RST}")
    for extra in ("cv", "asr", "tts", "llm"):
        mods = _MODALITY_EXTRAS[extra]
        missing = [m for m in mods if not _can_import(m)]
        if missing:
            print(f"  {RED}✗{RST} {extra:3s}  missing {', '.join(missing)}  "
                  f"— pip install chipwright[{extra}]")
        else:
            print(f"  {GREEN}✓{RST} {extra:3s}  installed ({', '.join(mods)})")


def cmd_doctor(args):
    b = _board_from(args)
    rc = 0
    if not (b["host"] and b["pw"]):
        print(f"{YELLOW}no board configured{RST} — set CW_BOARD_HOST / CW_BOARD_PASS (or --host/--pw)")
    else:
        t = probe(b["host"], b["user"], b["pw"], b["port"])
        if not t.reachable:
            print(f"{RED}● unreachable{RST}  {b['user']}@{b['host']}:{b['port']}")
            rc = 1
        else:
            print(f"{GREEN}● board reachable{RST}  {b['user']}@{b['host']}:{b['port']}")
            print(f"  SoC         : {t.soc_name or '?'}" + (f" (soc_id {t.soc_id})" if t.soc_id else ""))
            print(f"  Hexagon arch: {t.htp_arch}")
            print(f"  QNN libs    : {', '.join(t.qnn_libs) or '(none found)'}")
            if t.thermal_c is not None:
                print(f"  temperature : {t.thermal_c} °C")
            print(f"  target key  : {t}")
    _print_extras_report()
    return rc


def _rederivable_note(v):
    """A small, non-breaking annotation (spec 005 FR-RR04): is this record's headline number
    independently re-derivable offline (NPU output pinned), or does it predate reproducible-record?
    Returns a colored line, or None when the variant carries no verification record at all.
    """
    from . import replay
    if not isinstance(v.verified, dict) or not isinstance(v.verified.get("vs_reference"), dict):
        return None
    if replay.is_rederivable(v):
        return f"{GREEN}↺ independently re-derivable{RST} — chipwright run {v.model} --replay"
    return f"{DIM}not independently re-derivable yet (record predates reproducible-record){RST}"


def _print_decision(model, target, d):
    print(f"model   : {model}")
    print(f"target  : {target}")
    if d.outcome == Outcome.USE:
        v = d.variant
        print(f"{GREEN}✔ USE{RST}    {v.model}-{v.version}  htp{v.arch}/{v.quant}/{v.shape} "
              f"qnn{v.sdk_tested}  ({d.reason})")
        if v.verified:
            print(f"  {DIM}verified: {v.verified}{RST}")
            note = _rederivable_note(v)
            if note:
                print(f"  {note}")
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
            note = _rederivable_note(v)
            if note:
                print(f"  {note}")
    else:
        print(f"{RED}✗ FAIL{RST}  axis={d.axis}: {d.reason}")
    return d


def cmd_resolve(args):
    reg = Registry(args.registry)
    if args.model not in reg.models():
        return _unknown_model(args, reg)
    target = _get_target(args)
    d = resolve(target, reg.variants(args.model), args.quant, args.shape,
                want_runtime=getattr(args, "runtime", None), priority=getattr(args, "priority", None))
    if getattr(args, "json", False):
        _emit_json(jsonout.decision_to_dict(args.model, target, d))
    else:
        _print_decision(args.model, target, d)
    return jsonout.exit_code_for(d)


def _replay(reg, model):
    """Offline replay of a recorded on-device verification (spec 005 US2 / FR-RR03).

    Picks a variant whose record pins the NPU output, re-derives the cosine from the pinned operands with
    the shared gate, and prints the result LABELLED as a replay of the recorded result (board + date) —
    never as a fresh on-silicon verification (Principle I). Exits 0 on a reproduced pass, non-zero on a
    fail / a recomputed≠recorded mismatch / a missing pin.
    """
    from . import replay as rp
    from . import run as runner
    variants = reg.variants(model)
    # llama.cpp-HTP generative gate: the pinned quantity is a PPL pair, not an NPU tensor — recompute the
    # perplexity-agreement verdict offline from the record and render it labelled as a replay.
    htp = [v for v in variants if v.effective_runtime == "llamacpp-htp"
           and (((v.verified or {}).get("vs_reference") or {}).get("gate") or {}).get("ppl_htp") is not None]
    if htp:
        rep = runner._llamacpp_replay(htp[0])
        if rep:
            return _render_run_report(rep, Decision(Outcome.USE))
    cand = [v for v in variants if rp.is_rederivable(v)]
    if not cand:
        print(f"{RED}✗ cannot replay{RST} {model}: no pinned on-device output to replay; this record "
              f"predates reproducible-record or was not captured.")
        # a record exists but was never captured for replay — point at the live gate as the way to verify.
        if any(v.verified for v in variants):
            print(f"  {DIM}the recorded number is still shown by `chipwright list` / `resolve`; to get a "
                  f"fresh VERIFIED line, run it on a board (chipwright run {model}).{RST}")
        return 2

    v = cand[0]
    try:
        res = rp.rederive(v)
    except ImportError as e:
        # The offline replay re-runs the CPU reference, which needs the modality's extra (onnxruntime, …).
        # Mirror `doctor`: name the extra to install instead of dead-ending on a bare ModuleNotFoundError —
        # this replay is the one place a skeptic re-derives the number cold, so it must not fail unhelpfully.
        miss = getattr(e, "name", None) or str(e)
        mod = (v.run or {}).get("modality")
        extra = mod if mod in _MODALITY_EXTRAS else next(
            (k for k, mods in _MODALITY_EXTRAS.items() if miss in mods), "cv")
        print(f"{RED}✗ cannot replay{RST} {model}: the offline CPU-reference re-derivation needs "
              f"`{miss}`, which the base install omits.")
        print(f"  {DIM}pip install chipwright[{extra}]{RST}  — then re-run "
              f"`chipwright verify {model} --offline`  (or run `chipwright doctor` to see every extra).")
        return 1
    except Exception as e:
        print(f"{RED}✗ replay failed:{RST} {e}")
        return 1

    rec = v.verified or {}
    board = rec.get("on", "?")
    date = rec.get("date", "?")
    cmin = res["cosine_min"]
    tag = f"{DIM}↺ REPLAY{RST} (recorded on {board}, {date})"

    if not res["matches"]:
        # recomputed ≠ recorded → refuse, never silently pass (FR-RR04 / Principle II).
        print(f"{tag}  recomputed cosine_min={cmin:.5f} {RED}≠{RST} recorded {res['recorded']} "
              f"(±{rp.TOLERANCE})  {RED}✗ REFUSING — record not reproducible{RST}")
        return 1
    if res["pass"]:
        print(f"{tag}  cosine_min={cmin:.5f}  {GREEN}VERIFIED{RST}")
        for name, c in res["outputs"].items():
            print(f"    {name}: {c:.5f}")
        return 0
    # reproduced faithfully, but the recorded number is itself below threshold → a recorded FAIL.
    print(f"{tag}  cosine_min={cmin:.5f}  {RED}FAILED{RST} (threshold {res['threshold']})")
    return 1


def cmd_verify(args):
    """`chipwright verify <model> --offline` — the replay under its own verb (spec 005 US2)."""
    reg = Registry(args.registry)
    if args.model not in reg.models():
        print(f"{RED}unknown model{RST} '{args.model}'. known: {', '.join(reg.models())}")
        return 2
    if not getattr(args, "offline", False):
        print(f"{YELLOW}verify currently supports only offline replay{RST} — pass --offline "
              f"(a live on-device verify is `chipwright run {args.model}`).")
        return 2
    return _replay(reg, args.model)


def _render_gate(model, tc):
    """Print a model's pinned ON-SILICON tool-call record (the agent gate, spec 007), labelled as a recorded
    result — board + date + the numbers, never re-badged as a fresh run. Returns 0 on a recorded PASS."""
    board, date = tc.get("on", "?"), tc.get("date", "?")
    thr = tc.get("thresholds") or {}
    mark = f"{GREEN}✔ VERIFIED{RST}" if tc.get("pass") else f"{RED}✗ FAIL  axis=tool_call{RST}"
    print(f"\n{DIM}↺ agent gate — tool-call fidelity, HTP0 vs the SAME GGUF's CPU reference "
          f"(recorded on {board}, {date}){RST}")
    print(f"  {model}  {mark}")
    print(f"    tool_call_retention = {tc.get('tool_call_retention'):.2f}  "
          f"(gate ≥ {thr.get('tool_call_retention')})   over {tc.get('reference_answered')} ref-answered prompts")
    print(f"    abstention_rate     = {tc.get('abstention_rate'):.2f}  (gate ≥ {thr.get('abstention_rate')})")
    print(f"    schema_validity     = {tc.get('schema_validity'):.2f}  {DIM}(recorded, not gated){RST}")
    if tc.get("note"):
        print(f"    {DIM}{tc['note']}{RST}")
    kb = tc.get("known_bad")
    if kb:
        kmark = f"{RED}✗ FAIL{RST}" if not kb.get("pass") else f"{GREEN}✔{RST}"
        print(f"  {DIM}the gate that cannot FAIL is not a gate — the SAME gate, SAME board, "
              f"a lossy {kb.get('quant')} build:{RST}")
        extra = ", reference_degenerate" if kb.get("reference_degenerate") else ""
        print(f"    {kmark}  retention = {kb.get('tool_call_retention'):.2f}, "
              f"abstention = {kb.get('abstention_rate'):.2f}{extra}")
        if kb.get("note"):
            print(f"    {DIM}{kb['note']}{RST}")
    print(f"  {DIM}backend-faithfulness vs the CPU reference, not an absolute safety claim; thresholds are "
          f"house rules. Run it yourself: python -m chipwright.demos.agent_gate{RST}")
    return 0 if tc.get("pass") else jsonout.EXIT_VERIFY_MISMATCH


def cmd_gate(args):
    """`chipwright gate` — the AGENT gate (tool-call fidelity, spec 007) on the front door: the gate ladder's
    third tier (cosine → perplexity-agreement → tool-call). `--demo` runs the shipped good-vs-broken
    demonstration; `gate <model>` surfaces the model's pinned on-silicon tool-call record."""
    if getattr(args, "demo", False):
        from chipwright.demos import agent_gate
        agent_gate.main()
        return 0
    model = getattr(args, "model", None)
    if not model:
        print(f"{YELLOW}usage:{RST} chipwright gate <model>   (or  chipwright gate --demo)")
        print(f"  {DIM}the agent gate proves a quantized tool-calling model still calls tools correctly on "
              f"the NPU vs its CPU reference — and FAILs by name if the chip silently diverged.{RST}")
        return 2
    reg = Registry(args.registry)
    if model not in reg.models():
        print(f"{RED}unknown model{RST} '{model}'. known: {', '.join(reg.models())}")
        return 2
    tc = None
    for v in reg.variants(model):
        rec = ((v.verified or {}).get("vs_reference") or {}).get("tool_call")
        if rec:
            tc = rec
            break
    if not tc:
        print(f"{RED}✗ no on-silicon tool-call record{RST} for '{model}'. The agent gate ships as a runnable "
              f"demonstration meanwhile:  {DIM}python -m chipwright.demos.agent_gate{RST}  "
              f"(or `chipwright gate --demo`).")
        return 2
    return _render_gate(model, tc)


def _render_run_report(rep, d):
    """Print a run's verification report (human door) and return its exit code. Shape-safe: the LLM
    generative gate reports a `tripwire` (token-match), CV/ASR a cosine — the human door must not diverge
    from the already-shape-safe `--json` projection (Principle IV, and never a KeyError traceback)."""
    if rep.get("runtime") == "llamacpp-htp" and "gate" in rep:   # the generative gate actually ran (live/replay)
        g = rep["gate"]
        mark = f"{GREEN}✔ VERIFIED{RST}" if rep["pass"] else f"{RED}✗ MISMATCH{RST}"
        how = "live on HTP0" if rep.get("live") else "replay (pinned record, no board)"
        thr = (g.get("thresholds") or {}).get("backend_delta_pct")
        print(f"\n{mark}  llama.cpp-HTP generative gate — perplexity agreement ({how})")
        print(f"    PPL  HTP0 = {g.get('ppl_htp')}   CPU = {g.get('ppl_cpu')}   "
              f"backend Δ = {g.get('backend_delta_pct'):.3f}%  (gate ≤ {thr}%)")
        if g.get("lossless_delta_pct") is not None:
            print(f"    quant cost vs f16: {g.get('lossless_delta_pct'):+.3f}%  (recorded, not gated)")
        if rep.get("dispatch"):
            print(f"    dispatch: {rep['dispatch'].get('reason')}")
        tc = rep.get("tool_call")                         # the agent gate ran on the same board (T-AG07b)
        if tc:
            tcmark = f"{GREEN}✔{RST}" if tc.get("pass") else f"{RED}✗ FAIL  axis=tool_call{RST}"
            tthr = tc.get("thresholds") or {}
            print(f"  {tcmark}  agent gate — tool-call fidelity, HTP0 vs CPU (live)")
            ret = tc.get("tool_call_retention")
            print(f"    tool_call_retention = {ret:.2f}  (gate ≥ {tthr.get('tool_call_retention')})   "
                  f"over {tc.get('reference_answered')} ref-answered prompts"
                  if ret is not None else "    tool_call_retention = n/a (no reference calls)")
            print(f"    abstention_rate     = {tc.get('abstention_rate'):.2f}  "
                  f"(gate ≥ {tthr.get('abstention_rate')})")
            if tc.get("reference_degenerate"):
                print(f"    {DIM}{tc.get('reason', 'reference_degenerate')}{RST}")
        return jsonout.exit_code_for(d) if rep["pass"] else jsonout.EXIT_VERIFY_MISMATCH
    if rep.get("runtime") == "llamacpp-htp":     # no board: surface the record + the host recipe (FR-RT07)
        rec = rep.get("record") or {}
        d_ = rec.get("dispatch") or {}
        q_ = rec.get("quality") or {}
        print(f"\n{GREEN}✔ cached + hash-verified{RST}  {rep.get('artifact')}")
        print(f"{YELLOW}runtime = llamacpp-htp{RST}  (configure a v73 board for the live gate, or --replay offline)")
        if d_:
            print(f"    on-silicon record ({rec.get('board')}, {rec.get('htp_arch')}, {rec.get('date')}): "
                  f"Q8_0 dispatched to HTP0 — prefill {d_.get('htp0_prefill_tps')} vs CPU "
                  f"{d_.get('cpu_prefill_tps')} t/s = {d_.get('prefill_speedup')}×")
        if q_:
            print(f"    quality: PPL(WikiText) = {q_.get('ppl_wikitext_q8_0')} (near-lossless)")
        print(f"    recipe: ADSP_LIBRARY_PATH=<lib> " + " ".join(rep.get("recipe") or []))
        if rep.get("warn"):
            print(f"{DIM}    note: {rep['warn']}{RST}")
        return jsonout.EXIT_USE_WITH_WARN
    mark = f"{GREEN}✔ VERIFIED{RST}" if rep["pass"] else f"{RED}✗ MISMATCH{RST}"
    if "tripwire" in rep:                        # LLM generative gate (token-match), not a cosine
        tw = rep["tripwire"]
        thr = (tw.get("thresholds") or {}).get("token_match")
        print(f"\n{mark}  generative gate  token_match = {float(tw.get('token_match', 0.0)):.5f}"
              + (f"  (threshold {thr})" if thr is not None else ""))
        if rep.get("behavior"):
            print(f"    behavioral probe: {rep['behavior']}")
    else:
        print(f"\n{mark}  NPU vs CPU-reference  cosine_min = {rep['cosine_min']:.5f}  "
              f"(threshold {rep['threshold']})")
        for name, c in rep["per_output"].items():
            print(f"    {name}: {c:.5f}")
    return jsonout.exit_code_for(d) if rep["pass"] else jsonout.EXIT_VERIFY_MISMATCH


def cmd_run(args):
    reg = Registry(args.registry)
    if args.model not in reg.models():
        return _unknown_model(args, reg)
    if getattr(args, "replay", False):
        return _replay(reg, args.model)
    target = _get_target(args)
    if getattr(args, "json", False):
        return _run_json(args, reg, target)
    d = resolve(target, reg.variants(args.model), args.quant, args.shape,
                want_runtime=getattr(args, "runtime", None), priority=getattr(args, "priority", None))
    _print_decision(args.model, target, d)
    if not d.ok:
        if d.outcome == Outcome.BUILD:
            print(f"{DIM}milestone 1 is resolve-only — building is out of scope; stopping.{RST}")
        return jsonout.exit_code_for(d)
    # USE / USE_WITH_WARN: execute on the board and verify against a CPU reference.
    v = d.variant
    # the llama.cpp-HTP runtime has its own run path (fetch + silicon record + recipe, FR-RT07) and no
    # cosine run block — route it through run_variant instead of the no-run-block short-circuit.
    if not (v.run and v.run.get("modality")) and v.effective_runtime != "llamacpp-htp":
        print(f"{DIM}artifact resolves, but this variant has no run block (modality/meta/cpu_onnx) — "
              f"resolve-only.{RST}")
        if v.url:
            print(f"fetching → cache ({cache.CACHE_DIR}) …")
            print(f"{GREEN}✔ cached + hash-verified{RST}  {cache.fetch(v.url, v.sha256)}")
        return jsonout.EXIT_NOT_RUNNABLE
    from . import run as runner
    try:
        rep = runner.run_variant(v, args.input, emit_record=getattr(args, "emit_record", False))
    except Exception as e:
        print(f"{RED}run failed:{RST} {e}")
        return jsonout.EXIT_UNEXPECTED
    return _render_run_report(rep, d)


def _run_json(args, reg, target) -> int:
    """`run` in `--json` mode: all human/progress text (including the device runner's own prints) goes to
    stderr, then exactly one JSON document goes to stdout. The resolve/run/verify logic is untouched —
    this only redirects the human stream and projects the result (Principle IV).
    """
    # Everything human — the decision render and the runner's progress lines — to stderr.
    with contextlib.redirect_stdout(sys.stderr):
        d = resolve(target, reg.variants(args.model), args.quant, args.shape,
                want_runtime=getattr(args, "runtime", None), priority=getattr(args, "priority", None))
        _print_decision(args.model, target, d)
        if not d.ok:
            doc, code = jsonout.decision_to_dict(args.model, target, d), jsonout.exit_code_for(d)
        else:
            v = d.variant
            if not (v.run and v.run.get("modality")) and v.effective_runtime != "llamacpp-htp":
                print(f"artifact resolves, but this variant has no run block — resolve-only.")
                doc = jsonout.decision_to_dict(args.model, target, d)
                doc["verified"] = None
                code = jsonout.EXIT_NOT_RUNNABLE
            else:
                from . import run as runner
                rep = runner.run_variant(v, args.input)   # may raise → caught by the --json wrapper in main
                doc = jsonout.run_to_dict(args.model, target, d, rep)
                if rep.get("runtime") == "llamacpp-htp":
                    code = jsonout.EXIT_USE_WITH_WARN     # usable, with the documented recipe caveat (FR-RT07)
                else:
                    code = jsonout.exit_code_for(d) if rep.get("pass") else jsonout.EXIT_VERIFY_MISMATCH
    _emit_json(doc)
    return code


def cmd_list(args):
    reg = Registry(args.registry)
    # a target is optional — list works offline, and resolves per-model when one is available
    target = None
    b = _board_from(args)
    if b["host"] and b["pw"]:
        t = probe(b["host"], b["user"], b["pw"], b["port"])
        if t.reachable:
            target = t
    if target is None and args.target:
        target = _manual_target(args.target)

    if getattr(args, "json", False):
        _emit_json(jsonout.list_to_dict(reg, target))
        return 0

    head = f"{target}" if target else "no target — pass --target or configure a board to resolve"
    print(f"{DIM}catalogue · {len(reg.models())} models · {head}{RST}")
    badge = {Outcome.USE: f"{GREEN}USE{RST}", Outcome.USE_WITH_WARN: f"{YELLOW}USE*{RST}",
             Outcome.BUILD: f"{YELLOW}BUILD{RST}", Outcome.FAIL: f"{RED}FAIL{RST}"}
    from . import replay as rp
    for m in reg.models():
        vs = reg.variants(m)
        archs = ",".join(sorted({v.arch for v in vs}))
        quants = ",".join(sorted({v.quant for v in vs}))
        out = ""
        if target:
            d = resolve(target, vs)
            out = badge.get(d.outcome, "")
            if d.outcome == Outcome.FAIL and d.axis:
                out += f"{DIM}({d.axis}){RST}"
        rec_v = next((v for v in vs if v.verified), None)
        rec = rec_v.verified if rec_v else None
        vr = ""
        if rec and isinstance(rec.get("vs_reference"), dict):
            val = rec["vs_reference"].get("value")
            met = rec["vs_reference"].get("metric", "")
            if val is not None:
                vr = f"{DIM}{met}={val}{RST}"
            # annotate re-derivability (spec 005 FR-RR04) — a small, non-breaking tag.
            if rp.is_rederivable(rec_v):
                vr += f" {GREEN}↺replay{RST}"
            else:
                vr += f" {DIM}(not independently re-derivable yet){RST}"
        print(f"  {m:20s} {archs:9s} {quants:9s} {out:16s} {vr}")
    return 0


def cmd_build(args):
    import os
    reg = Registry(args.registry)
    if args.model not in reg.models():
        return _unknown_model(args, reg)
    target = _get_target(args)
    d = resolve(target, reg.variants(args.model), args.quant, args.shape,
                want_runtime=getattr(args, "runtime", None), priority=getattr(args, "priority", None))
    _print_decision(args.model, target, d)
    if d.ok:
        print(f"{DIM}already resolves to USE — nothing to build.{RST}")
        return jsonout.exit_code_for(d)
    if d.outcome != Outcome.BUILD:
        return jsonout.exit_code_for(d)
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


# The outcome→exit-code contract, shown in `--help` so a pipeline author can read it without the README
# (spec 006 FR-J04). Kept in lockstep with chipwright.jsonout (the single source of the numbers).
_EXIT_CODE_HELP = f"""exit codes (stable contract, chipwright.jsonout.__api__={jsonout.__api__}):
  {jsonout.EXIT_USE:<3} USE               exact match — run it
  {jsonout.EXIT_USE_WITH_WARN:<3} USE_WITH_WARN     in range, outside the tested SDK point
  {jsonout.EXIT_BUILD:<3} BUILD             no artifact, but a recipe can build one
  {jsonout.EXIT_FAIL:<3} FAIL              nothing fits — the message names the axis
  {jsonout.EXIT_VERIFY_MISMATCH:<3} VERIFY_MISMATCH   ran, but the NPU vs CPU cosine fell below threshold
  {jsonout.EXIT_NOT_RUNNABLE:<3} NOT_RUNNABLE      resolved, but cannot execute here (no run block / no board)
  {jsonout.EXIT_USAGE:<3} usage             usage error / unknown model
  {jsonout.EXIT_UNEXPECTED:<3} unexpected        an unexpected internal error
"""


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="chipwright",
        description="Chipwright — a verified package manager for Qualcomm NPUs. The gate ladder proves "
                    "correctness on the chip: cosine (CV/ASR) -> perplexity-agreement (LLM) -> tool-call "
                    "fidelity (agents, `chipwright gate`).",
        epilog=_EXIT_CODE_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=None, help="index path/URL (default: bundled registry)")
    ap.add_argument("--host"); ap.add_argument("--user"); ap.add_argument("--pw"); ap.add_argument("--port")
    ap.add_argument("--target", help="offline target, e.g. 'htpv68,qnn2.37'")
    ap.add_argument("--quant", help="narrow to a quant, e.g. w8a16")
    ap.add_argument("--shape", help="narrow to a shape profile, e.g. win30s")
    ap.add_argument("--runtime", help="pick an inference runtime, e.g. llamacpp-htp (default: the genie floor)")
    ap.add_argument("--priority", choices=["quality", "decode-latency"],
                    help="when a model spans runtimes, prefer quality (llamacpp-htp) or decode-latency (genie)")
    ap.add_argument("--json", action="store_true",
                    help="emit one JSON document to stdout (resolve/run/list); human text → stderr, "
                         "always plain. Exit codes follow the table below.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="probe the board")
    sub.add_parser("list", help="browse the catalogue (with resolve outcome for your board)")
    rp = sub.add_parser("resolve", help="resolve a model against the board"); rp.add_argument("model")
    rnp = sub.add_parser("run", help="resolve, fetch, run on the board, verify vs CPU"); rnp.add_argument("model"); rnp.add_argument("--input")
    rnp.add_argument("--replay", action="store_true",
                     help="no board: replay the recorded on-device verification from its pinned tensors")
    rnp.add_argument("--emit-record", action="store_true",
                     help="capture the on-device NPU output tensor(s) + print a ready-to-host record fragment (spec 005 FR-RR05)")
    vp = sub.add_parser("verify", help="offline replay of a recorded verification (no board)")
    vp.add_argument("model"); vp.add_argument("--offline", action="store_true",
                                              help="re-derive the recorded result from pinned tensors")
    bp = sub.add_parser("build", help="resolve; when it misses, build from a recipe"); bp.add_argument("model")
    gp = sub.add_parser("gate", help="agent gate — on-silicon tool-call fidelity record, or --demo (spec 007)")
    gp.add_argument("model", nargs="?")
    gp.add_argument("--demo", action="store_true", help="run the shipped good-vs-broken demonstration")
    args = ap.parse_args(argv)
    _set_palette(_color_enabled(args.json))
    handler = {"doctor": cmd_doctor, "list": cmd_list, "resolve": cmd_resolve,
               "run": cmd_run, "verify": cmd_verify, "build": cmd_build, "gate": cmd_gate}[args.cmd]
    # In --json mode, an unexpected internal error must still leave exactly one JSON document on stdout
    # (never a half-line a parser chokes on) and exit 1 (FR-J01, edge case).
    if args.json and args.cmd in ("resolve", "run", "list"):
        try:
            return handler(args)
        except SystemExit:
            raise                       # clean usage refusals (exit 2) keep stdout empty — fine for a parser
        except Exception as e:
            _emit_json({"error": str(e), "exit_code": jsonout.EXIT_UNEXPECTED})
            return jsonout.EXIT_UNEXPECTED
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
