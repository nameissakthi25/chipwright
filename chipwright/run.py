"""The `run` verb — resolve → fetch/locate the artifact → push + execute on the board → verify.

Ties the resolver, the device transport and a modality runner together, and ends in the thing that
makes the whole framework trustworthy: a cosine against a CPU reference, so a fast wrong answer is
caught here instead of shipped.
"""
from __future__ import annotations

import importlib
import json
import os

from . import verify
from .cache import locate_ref as _locate_ref     # device-free run-block reference locator (lives in cache)
from .device import RemoteQNN


def _board_env():
    return dict(host=os.environ.get("CW_BOARD_HOST"),
                user=os.environ.get("CW_BOARD_USER", "root"),
                pw=os.environ.get("CW_BOARD_PASS"),
                port=int(os.environ.get("CW_BOARD_PORT", 22)))


def _emit_npu_record(variant, npu_out):
    """Capture the on-device NPU output tensor(s) this run produced, so the record becomes independently
    re-derivable offline (spec 005 FR-RR05). Writes each tensor to the cache and prints a ready-to-host
    `vs_reference.npu_output` fragment (sha256 + dtype + shape per output)."""
    import hashlib
    import numpy as np
    from . import cache
    d = os.path.join(cache.CACHE_DIR, "records", variant.model)
    os.makedirs(d, exist_ok=True)
    print("\n  --- captured NPU output (host these, then paste under the record's vs_reference) ---", flush=True)
    print("        npu_output:")
    for name, arr in npu_out.items():
        a = np.ascontiguousarray(arr)
        fn = os.path.join(d, f"npu_{name}.npy")
        np.save(fn, a)
        sha = hashlib.sha256(open(fn, "rb").read()).hexdigest()
        print(f"          - name: {name}")
        print(f"            url: <HOST>/{variant.model}/npu_{name}.npy")
        print(f"            sha256: {sha}")
        print(f"            dtype: {a.dtype}")
        print(f"            shape: {list(a.shape)}")
        print(f"        # saved locally: {fn}", flush=True)


def _llamacpp_artifact(variant):
    """Fetch+sha-verify (or locate) the GGUF; raise by name if there is none."""
    from . import cache
    if not variant.url and not variant.local:
        raise RuntimeError(
            f"{variant.model} [llamacpp-htp]: this variant carries a silicon record but no GGUF artifact "
            f"in the active registry (no hosted url, no local path).")
    if variant.url:
        return cache.fetch(variant.url, variant.sha256)   # fetch + sha-verify (refuses on mismatch)
    art = os.path.expanduser(variant.local)
    if not os.path.exists(art):
        raise RuntimeError(f"local GGUF not found: {art}")
    return art


def _load_tool_prompts(spec):
    """Load the agent-gate probe set (T-AG07b). `spec` is a run-block value: an explicit package-relative or
    absolute path, else the bundled `data/tool_prompts.yaml`. Returns the parsed doc."""
    import yaml
    pkg = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(pkg, "data", "tool_prompts.yaml")
    if isinstance(spec, str) and spec.strip() and spec not in ("data/tool_prompts.yaml",
                                                               "tool_prompts.yaml", "true", "True"):
        cand = spec if os.path.isabs(spec) else os.path.join(pkg, spec)
        if os.path.exists(cand):
            path = cand
    with open(path) as f:
        return yaml.safe_load(f)


def _llamacpp_tool_call_gate(dev, art, r, *, n_predict=200, ctx=4096):
    """The AGENT gate on the board (T-AG07b): generate the tool-call probe set on HTP0 and on the SAME GGUF's
    CPU reference (GGML_HEXAGON_NDEV=0), then score with `tool_call_report`. Backend isolation — identical
    weights, only the compute device differs, so a divergence is a chip bug. Reuses the already-staged
    pack+GGUF and the proven dispatch from the ppl gate (same `dev`). Guarded single-turn llama-cli (does
    not wedge the board)."""
    from . import verify_llm
    doc = _load_tool_prompts(r.get("tool_prompts"))
    prompts, relevant = verify_llm.build_tool_prompts(doc)
    name = os.path.basename(art)
    chip, ref = [], []
    for i, text in enumerate(prompts):
        pf = dev.push_text(text, f"tc_p{i}.txt")
        c, _ = dev.generate(name, pf, device="HTP0", n_predict=n_predict, ctx=ctx)   # the chip under test
        g, _ = dev.generate(name, pf, ndev=0, n_predict=n_predict, ctx=ctx)           # CPU ref, SAME GGUF
        chip.append(c)
        ref.append(g)
    report = verify_llm.tool_call_report(chip, ref, relevant)
    report["prompt_set"] = r.get("tool_prompts") if isinstance(r.get("tool_prompts"), str) \
        else "data/tool_prompts.yaml"
    return report


def _llamacpp_live_gate(variant, art, r):
    """Drive the generative gate on the board (spec 004 T-RT11 + contracts/llamacpp-htp-gate.md).

    preflight (refuse by name) -> stage pack + GGUF -> prove dispatch (llama-bench HTP0) -> measure
    perplexity on HTP0 and on CPU (GGML_HEXAGON_NDEV=0) of the SAME GGUF -> ppl-agreement gate. The two
    perplexities isolate the backend: identical weights, only the compute device differs, so a divergence
    is a chip bug. When the variant declares `run.tool_prompts`, the AGENT gate (tool-call fidelity, T-AG07b)
    also runs on the same board, reusing the staged artifact + proven dispatch; overall pass needs both."""
    from . import verify_llm
    from .device_llamacpp import LlamaHtpBoard
    from .runtimes import llamacpp_htp as lh

    b = _board_env()
    dev = LlamaHtpBoard(b["host"], b["user"], b["pw"], b["port"], board_dir=r.get("board_dir", "/data/cw"))
    # Stage the runtime pack BEFORE the staged-lib preflight: on a FRESH board nothing is staged yet, so
    # running host_preflight first would refuse ("libggml-htp-<arch>.so … staged: none") before ever staging.
    # Check only the hard prerequisite staging can't supply (the Hexagon RPC lib) up front, then stage, then
    # run the full preflight (which now sees the arch-matched HTP lib). Caught by the live stranger-path run.
    probe = dev.probe()
    if not probe.get("has_cdsprpc"):
        raise RuntimeError(f"{variant.model} [llamacpp-htp]: needs libcdsprpc.so on the board to reach the "
                           f"Hexagon DSP — not found. (Is this a Hexagon-capable board?)")
    if r.get("pack_url"):
        print("  staging runtime pack …", flush=True)
        dev.ensure_pack(r["pack_url"], r.get("pack_sha"))
    ok, why = lh.host_preflight(dev.probe())             # now the arch-matched HTP lib is staged
    if not ok:
        raise RuntimeError(f"{variant.model} [llamacpp-htp]: {why}")
    name = os.path.basename(art)
    print("  pushing GGUF + proving dispatch on HTP0 …", flush=True)
    dev.ensure_gguf(art, name)
    proof = lh.dispatch_proof(dev.bench(name))
    if not proof.dispatched:                              # the FIRST HTP session after idle can flake (error 0x14) — retry once
        proof = lh.dispatch_proof(dev.bench(name))
    if not proof.dispatched:                              # a PPL measured while silently on CPU is not an HTP gate
        raise RuntimeError(f"{variant.model} [llamacpp-htp]: on-device dispatch not proven — {proof.reason}")
    chunks = int(r.get("chunks", 16))
    print(f"  measuring perplexity on HTP0 and CPU ({chunks} chunks) …", flush=True)
    ppl_htp, _ = dev.perplexity(name, device="HTP0", chunks=chunks)
    ppl_cpu, _ = dev.perplexity(name, ndev=0, chunks=chunks)
    gate = verify_llm.ppl_agreement_gate(ppl_htp, ppl_cpu, r.get("ppl_f16_baseline"))
    result = {"runtime": "llamacpp-htp", "gate": gate, "dispatch": proof.as_record(),
              "pass": bool(gate["pass"] and proof.dispatched), "live": True, "artifact": art}
    if r.get("tool_prompts"):                             # the agent gate (T-AG07b), on the same staged board
        print("  running the agent gate (tool-call fidelity) on HTP0 vs CPU …", flush=True)
        tc = _llamacpp_tool_call_gate(dev, art, r)
        result["tool_call"] = tc
        result["pass"] = bool(result["pass"] and tc["pass"])
    return result


def _llamacpp_replay(variant):
    """Recompute the ppl-agreement verdict offline from the pinned record (spec 005; no board)."""
    from . import verify_llm
    pin = ((variant.verified or {}).get("vs_reference") or {}).get("gate") or {}
    if pin.get("ppl_htp") is None or pin.get("ppl_cpu") is None:
        return None
    gate = verify_llm.ppl_agreement_gate(pin.get("ppl_htp"), pin.get("ppl_cpu"), pin.get("ppl_f16_baseline"))
    return {"runtime": "llamacpp-htp", "gate": gate, "pass": gate["pass"], "replay": True}


def run_llamacpp_htp_variant(variant):
    """The llama.cpp-HTP runtime path (spec 004 T-RT11, FR-RT07).

    With a board configured (CW_BOARD_HOST/PASS) → drive the live generative gate. Without one → surface
    the fetch-verified artifact + the recipe + the on-silicon record (and point at `--replay` for the
    offline gate). Never fakes a gate it did not run."""
    from .runtimes import llamacpp_htp as lh
    art = _llamacpp_artifact(variant)
    r = variant.run or {}
    b = _board_env()
    if b["host"] and b["pw"]:
        return _llamacpp_live_gate(variant, art, r)
    recipe = lh.perplexity_command(os.path.basename(art), "./pkg-snapdragon/lib", "./glibc238",
                                   "wikitext-2-raw/wiki.test.raw")
    return {
        "runtime": "llamacpp-htp",
        "artifact": art,
        "artifact_verified": True,
        "record": lh.SILICON_RECORD,
        "recipe": recipe,
        "host_prereqs": list(lh.REQUIRED_LOADER_LIBS),
        "cli_execution": "recipe-or-replay",
        "pass": True,                                    # artifact integrity ok + silicon record present
        "warn": ("no board configured (CW_BOARD_HOST/PASS) — showing the fetch-verified artifact + the "
                 "on-silicon record. Configure a v73 board to drive the live gate, or `run --replay` to "
                 "recompute the gate verdict offline from the pinned record."),
    }


def run_variant(variant, input_path=None, emit_record=False):
    """Execute a resolved variant on the board and return the verification report.

    `emit_record=True` also captures the on-device output tensor(s) for a reproducible record (FR-RR05).
    """
    # the llama.cpp-HTP runtime has its own path (no Genie/QNN transport) — handle before the modality logic.
    if variant.effective_runtime == "llamacpp-htp":
        return run_llamacpp_htp_variant(variant)
    r = variant.run or {}
    if not r.get("modality"):
        raise RuntimeError(f"variant for {variant.model} has no run block (modality/meta/cpu_onnx)")

    # an LLM variant generates and gates against a PINNED reference (not a live cosine) — its own path.
    if r["modality"] == "llm":
        return run_llm_variant(variant, r)

    # locate the artifact: a local file, or fetched by hash into the content-addressed cache
    if not variant.local and not variant.url:
        raise RuntimeError(
            f"{variant.model}: this variant carries a verification record but no runnable artifact "
            f"in the active registry (no hosted url, no local path). Build it with "
            f"`chipwright build {variant.model}`, or set CW_REGISTRY to a registry that has the "
            f"artifact (the author's dev registry is registry/index.dev.yaml).")
    if variant.local:
        art = os.path.expanduser(variant.local)
        if not os.path.exists(art):
            raise RuntimeError(f"local artifact not found: {art}")
    else:
        from . import cache
        art = cache.fetch(variant.url, variant.sha256)

    b = _board_env()
    if not (b["host"] and b["pw"]):
        raise RuntimeError("no board configured — set CW_BOARD_HOST / CW_BOARD_PASS")
    dev = RemoteQNN(b["host"], b["user"], b["pw"], b["port"])

    board_name = r.get("board_name", os.path.basename(art))
    print(f"  pushing artifact to board ({board_name}) …", flush=True)
    ctx_board = dev.ensure_ctx(art, board_name, ext=variant.ext)

    mod = importlib.import_module(f".modalities.{r['modality']}", __package__)
    if not r.get("cpu_onnx") or not (input_path or r.get("input")):
        raise RuntimeError(
            f"{variant.model}: the run block has no CPU reference (cpu_onnx) and/or input in the "
            f"active registry, so the verification gate can't run. Pass --input, or use a registry "
            f"that carries the reference (registry/index.dev.yaml).")
    meta = _locate_ref(r, "meta") or ""
    cpu_onnx = _locate_ref(r, "cpu_onnx")
    inp = _locate_ref(r, "input", override=input_path)
    print("  running on NPU + CPU reference …", flush=True)
    npu_out, cpu_out = mod.run(dev, ctx_board, meta, cpu_onnx, inp)
    rep = verify.report(npu_out, cpu_out, threshold=float(r.get("threshold", 0.99)))
    if emit_record:
        _emit_npu_record(variant, npu_out)
    return rep


def _looks_refused(text: str) -> bool:
    """Heuristic: does a generation read as a refusal? (behavioral probe, v1 classifier)

    A coarse marker match over the opening of the generated text — enough to tell "I can't help with
    that" from a compliant answer. The reference's own refusal labels are pinned (computed on the CPU
    reference at pin time); this classifies the on-chip side the same way for a relative comparison.
    """
    if not text:
        return False
    head = text.strip().lower()[:200]
    markers = ("i can't", "i cannot", "i can not", "i won't", "i will not", "i'm sorry",
               "i am sorry", "i'm unable", "i am unable", "cannot help", "can't help",
               "not able to", "against my", "i must decline", "i'm not able")
    return any(m in head for m in markers)


def run_llm_variant(variant, r):
    """Generate on the NPU and gate against the PINNED reference (FR-L03/L04/L10).

    The LLM path differs from CV/ASR: it drives Genie over **adb**, generates from a fixed prompt set,
    and compares the on-chip greedy tokens to a reference that was computed once on the build host and
    **pinned** into the record (`r["reference"]`) — never a live CPU call on the board. v1 gates on the
    greedy token-match (Layer-1 1a) and records the behavioral refusal probe; top-1/KL/PPL (1b/1c) are
    marked not-measured-on-chip because `genie-t2t-run` emits tokens, not per-step logits.
    """
    from . import verify_llm
    from .device_adb import AdbDevice

    bundle = os.path.expanduser(variant.local or "")
    if not bundle or not os.path.isdir(bundle):
        raise RuntimeError(
            f"{variant.model}: this LLM variant has no runnable Genie bundle directory in the active "
            f"registry (no `local` bundle dir). Build it with `chipwright build {variant.model}`, or set "
            f"CW_REGISTRY to a registry that carries the bundle (registry/index.dev.yaml).")
    ref_path = os.path.expanduser(r.get("reference") or "")
    if not ref_path or not os.path.exists(ref_path):
        raise RuntimeError(
            f"{variant.model}: the run block has no pinned reference (`reference`) in the active "
            f"registry, so the generative gate can't run (a pass is never reported without the gate). "
            f"Use a registry that carries the pinned reference (registry/index.dev.yaml).")

    with open(ref_path) as f:
        pin = json.load(f)                       # {prompts, tokens, [logits], [ppl], reference_logits_sha256}
    prompts = pin["prompts"]
    ref_tokens = pin["tokens"]
    meta = r.get("meta") or {}

    serial = os.environ.get("CW_BOARD_SERIAL")   # adb picks the device by serial; None == the one attached
    dev = AdbDevice(serial)
    mod = importlib.import_module(".modalities.llm", __package__)

    print(f"  generating on NPU over {len(prompts)} prompt(s) …", flush=True)
    matches, latency = [], {}
    for prompt, ref_toks in zip(prompts, ref_tokens):
        on = mod.run(dev, bundle, meta, prompt)
        matches.append(verify_llm.token_match(on.get("tokens") or [], ref_toks))
        latency = on.get("latency") or latency   # last prompt's profile is representative
    token_match = float(sum(matches) / len(matches)) if matches else 0.0

    tripwire = {
        "token_match": token_match, "top1_agreement": None, "mean_kl_nats": None,
        "ppl_delta_pct": None,                   # 1b/1c not measured on-chip in v1 (Genie emits tokens, not logits)
        "thresholds": {"token_match": verify_llm.TOKEN_MATCH_MIN, "ppl_delta_pct": verify_llm.PPL_DELTA_MAX_PCT},
        "pass": bool(token_match >= verify_llm.TOKEN_MATCH_MIN),
    }
    report = {"tripwire": tripwire, "latency": latency, "pass": tripwire["pass"]}

    # Layer-2 behavioral probe — only when the run block carries the harmful/benign sets + pinned labels.
    beh = r.get("behavior")
    if beh and pin.get("ref_refused") is not None:
        onchip_refused = [_looks_refused(mod.run(dev, bundle, meta, p).get("text")) for p in beh["harmful"]]
        benign_answered = [not _looks_refused(mod.run(dev, bundle, meta, p).get("text")) for p in beh["benign"]]
        report["behavior"] = verify_llm.behavior_report(onchip_refused, pin["ref_refused"], benign_answered)
    return report
