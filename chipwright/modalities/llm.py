"""LLM modality runner — the Genie driver (a slice of what a `chipwright-llm` package would own).

Unlike the CV/ASR runners, which diff one NPU tensor against a CPU reference in the same call, the LLM
runner *generates*: it assembles the Genie bundle on the device, applies the model's chat template,
runs `genie-t2t-run`, and parses back the generated text, then RETOKENIZES that text into token ids
(QAIRT 2.51 `genie-t2t-run` emits TEXT, not ids — there is no `--print-token-ids`), plus the optional
`--profile` latency JSON. The verification against the pinned reference happens in `verify_llm.py` (the
CPU reference runs on the build host, not here — FR-L10), so this runner's job is to produce an honest
on-chip generation.

Driver facts confirmed on real SM8850 / HTP v81 (QAIRT 2.51), which this runner is written to match:
  - `genie-t2t-run` takes the prompt via `--prompt_file <path>`, NOT an inline `--prompt` of the wrapped
    ChatML (which carries newlines and |<> markers that fight shell quoting).
  - It has NO `--print-token-ids` and NO `--max-length`; both were removed — passing a non-existent flag
    makes genie error out and the gate false-FAIL a good artifact.
  - Deterministic greedy is `sampler.top-k: 1` in the config; `sampler.greedy: true` is silently ignored.

Driven entirely over the device interface (`push_file` / `shell` / `pull` / `exists`) so it is
transport-agnostic (adb for Android targets); it does not import a concrete transport. The chat-template
wrap and the profile-JSON parse are pulled out as pure functions so they are unit-testable without a
device — the two places a silent bug (a mis-wrapped prompt makes an instruct model echo/loop) hides.
"""
from __future__ import annotations

import json
import re


# --- chat template (pure; the modality wraps, the caller passes raw text) -------------------------

def apply_chat_template(prompt: str, template: str = "chatml", system: str | None = None) -> str:
    """Wrap raw user text in the model's chat template. The caller passes raw text; the modality wraps.

    Instruct models need their chat template or they echo/loop (ChatML for Qwen). Supported templates:
      - "chatml" (Qwen/Qwen2.5): <|im_start|>role\\n...<|im_end|> turns, ending with an open assistant
        turn so generation continues the assistant message.
      - "raw": passthrough (base/non-instruct models).
    An unknown template name is refused by name rather than silently passed through — a mis-wrapped
    prompt is exactly the silent failure this function exists to prevent.
    """
    t = (template or "raw").lower()
    if t == "raw":
        return prompt
    if t == "chatml":
        parts = []
        if system:
            parts.append(f"<|im_start|>system\n{system}<|im_end|>")
        parts.append(f"<|im_start|>user\n{prompt}<|im_end|>")
        parts.append("<|im_start|>assistant\n")        # open assistant turn — generation continues it
        return "\n".join(parts)
    raise ValueError(f"unknown chat template {template!r} (known: chatml, raw)")


# --- Genie --profile JSON parse (pure) ------------------------------------------------------------

def parse_profile(profile_json: str) -> dict:
    """Parse the `genie-t2t-run --profile` JSON into the record's latency fields.

    Genie's profiler reports init/prompt-processing/token-generation rates and times. Field names vary
    across Genie builds, so each metric is looked up over a set of aliases and missing ones come back
    None (shown, not hidden) rather than raising — an absent number is a true state of the run.

    Returns {decode_toks_per_sec, prefill_toks_per_sec, time_to_first_token_ms, model_load_s}.
    """
    data = json.loads(profile_json) if isinstance(profile_json, str) else profile_json
    flat = _flatten(data)

    def pick(*aliases):
        for a in aliases:
            for k, v in flat.items():
                kl = k.lower()
                if all(tok in kl for tok in a) and isinstance(v, (int, float)):
                    return float(v)
        return None

    return {
        "decode_toks_per_sec": pick(("token", "generation", "rate"), ("decode", "rate"),
                                    ("token", "per", "sec")),
        "prefill_toks_per_sec": pick(("prompt", "processing", "rate"), ("prefill", "rate"),
                                     ("prompt", "rate")),
        "time_to_first_token_ms": pick(("time", "first", "token"), ("ttft",), ("first", "token", "ms")),
        "model_load_s": pick(("init", "time"), ("load", "time"), ("model", "load")),
    }


def _flatten(obj, prefix=""):
    """Flatten a nested dict/list into {dotted.key: scalar} so alias lookup is order-independent."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(_flatten(v, f"{prefix}.{i}"))
    else:
        out[prefix] = obj
    return out


# --- generated-text / token-ids parse (pure) ------------------------------------------------------

def parse_generation(stdout: str) -> dict:
    """Pull the generated text out of `genie-t2t-run` stdout (QAIRT 2.51 emits TEXT, not token ids).

    Genie prints the generated text after a `[BEGIN]` marker (with an optional `:`). On a clean run it
    closes with `[END]`; but w4a16 degenerates and never emits EOS, so on the real board the run is
    bounded by `timeout` and gets killed mid-generation — the `[END]` is then absent. So the text is
    matched up to `[END]` *or* the end of stdout (`\\Z`), never requiring `[END]`, or a correct-but-long
    generation would be dropped as if empty.

    `tokens` is still parsed best-effort for any genie build that prints a bracketed id trace, but
    QAIRT 2.51 `genie-t2t-run` has NO `--print-token-ids` flag, so it is normally None here and the
    real token ids come from retokenizing `text` (see `retokenize`).
    """
    text = None
    m = re.search(r"\[BEGIN\]:?(.*?)(\[END\]|\Z)", stdout, re.DOTALL)
    if m:
        text = m.group(1).strip()
    tokens = None
    tm = re.search(r"token[_ ]ids?\s*[:=]\s*\[([\d,\s]+)\]", stdout, re.IGNORECASE)
    if tm:
        tokens = [int(x) for x in tm.group(1).replace(" ", "").split(",") if x != ""]
    return {"text": text, "tokens": tokens}


# --- retokenize generated text -> token ids (genie gives no ids on QAIRT 2.51) --------------------

def load_bundle_tokenizer(path: str):
    """Load the bundle's `tokenizer.json` with the lightweight HuggingFace `tokenizers` library.

    QAIRT 2.51 `genie-t2t-run` emits generated TEXT, not token ids (no `--print-token-ids`), but the
    Layer-1 tripwire (`verify_llm.token_match`) compares TOKEN IDS. So the modality must turn genie's
    text back into ids with the SAME tokenizer the bundle shipped — the `tokenizer.json` genie itself
    decoded with — or the ids won't line up with the pinned reference.

    If `tokenizers` is not importable, the modality FAILS BY NAME here (like the sshpass/adb preflights)
    rather than silently returning no tokens — a missing retokenizer would otherwise look like an
    all-mismatch FAIL, the exact false-FAIL this module exists to avoid.
    """
    try:
        from tokenizers import Tokenizer
    except ImportError as e:  # fail by name, not silently — mirrors the adb/sshpass preflights
        raise RuntimeError(
            "llm modality: the 'tokenizers' library is required to retokenize genie's text output into "
            "token ids (QAIRT 2.51 genie-t2t-run emits text, not ids) but could not be imported. "
            "Install the 'llm' extra:  pip install 'chipwright[llm]'") from e
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"llm modality: bundle tokenizer not found at {path!r} — the gate cannot retokenize "
            f"genie's text without the bundle's tokenizer.json")
    return Tokenizer.from_file(path)


def retokenize(text: str, tokenizer_path: str | None = None, *, tokenizer=None) -> list:
    """Encode generated `text` into token ids with the bundle tokenizer, special tokens OFF.

    Pure/injectable: pass a loaded `tokenizer` (any object exposing
    `.encode(text, add_special_tokens=False).ids`) to make it unit-testable without a file or the
    library; otherwise it loads `tokenizer_path` via `load_bundle_tokenizer`. `add_special_tokens=False`
    because we match the ANSWER tokens against the reference's answer tokens — a stray BOS/EOS prepended
    by the tokenizer's post-processor would shift every position and tank token_match for a correct run.
    """
    tok = tokenizer if tokenizer is not None else load_bundle_tokenizer(tokenizer_path)
    enc = tok.encode(text, add_special_tokens=False)
    return list(enc.ids)


# --- deterministic greedy config (genie silently ignores sampler.greedy — use top-k:1) ------------

def greedy_sampler(base: dict | None = None) -> dict:
    """Return a genie `sampler` block that decodes deterministically (greedy) via `top-k: 1`.

    Bug proven on SM8850/v81: genie (QAIRT 2.51) SILENTLY IGNORES `sampler.greedy: true` and falls back
    to its default temp-0.8 sampling — the tripwire went non-deterministic and good-vs-self token_match
    was only 0.44. The greedy control genie actually honors is `top-k: 1` (temp 0, top-p 1). We set that
    explicitly and do NOT rely on `greedy: true`.
    """
    s = dict(base or {})
    s.setdefault("version", 1)
    s["top-k"] = 1          # <-- the greedy control genie HONORS; `greedy: true` is silently ignored
    s["temp"] = 0.0
    s["top-p"] = 1.0
    return s


def apply_greedy_to_config(config: dict) -> dict:
    """Return a copy of a genie_config with its dialog sampler forced to deterministic greedy (top-k:1).

    Pure (deep-copies, does not mutate the input) so it is unit-testable and the on-disk bundle config is
    never clobbered. The modality emits this derived config for the tripwire's greedy run.
    """
    import copy
    cfg = copy.deepcopy(config)
    dialog = cfg.setdefault("dialog", {})
    dialog["sampler"] = greedy_sampler(dialog.get("sampler"))
    return cfg


# --- the modality run (drives Genie over the device interface) ------------------------------------

def run(device, bundle_dir_local, meta, prompt, *, template="chatml", system=None,
        remote_dir="/data/local/tmp/chipwright_llm", greedy=True, profile=True, tokenizer=None):
    """Assemble the Genie bundle on the device, generate for one prompt, return text + tokens + latency.

    `meta` names the bundle pieces (context bins + genie_config + tokenizer + htp ext config) and the
    model fields (template, eos). Signature stays modality-shaped: (device, bundle, meta, prompt).
    Optional `meta` keys honour real-board layout: `genie_bin` (default "genie-t2t-run"), `env`
    (extra shell exports, e.g. LD_LIBRARY_PATH/ADSP_LIBRARY_PATH on the proven v81 board), `timeout_s`
    (wrap the run in `timeout` — w4a16 degenerates and never emits EOS, so an unbounded run hangs).

    Steps:
      1. Push the bundle **file-by-file** (a single multi-GB push of a directory can wedge the adb
         daemon — FR-L02), skipping files already present.
      2. Wrap the raw prompt in the chat template (ChatML for Qwen) and push it as a PROMPT FILE —
         QAIRT 2.51 genie-t2t-run reads the prompt via `--prompt_file <path>`, not inline.
      3. For the greedy tripwire, emit a derived config with `sampler.top-k: 1` (genie ignores
         `sampler.greedy`) and point genie at it.
      4. Run `genie-t2t-run -c <cfg> --prompt_file <file>` (+ `--profile` when requested), parse the
         `[BEGIN]..[END]` text, and RETOKENIZE it into token ids (genie emits no ids on QAIRT 2.51).

    Returns {text, tokens, latency, generation: <raw stdout>}. The pass/fail gate is `verify_llm`, not
    here — this returns the honest on-chip generation. `tokenizer` may be injected (a loaded Tokenizer)
    for testing; otherwise the bundle's tokenizer.json is loaded.
    """
    import os

    # 1. bundle transport — one file at a time (FR-L02).
    files = meta.get("bundle_files") or sorted(os.listdir(bundle_dir_local))
    device.shell(f"mkdir -p {remote_dir}")
    for name in files:
        remote = f"{remote_dir}/{name}"
        if device.exists(remote):
            continue
        device.push_file(os.path.join(bundle_dir_local, name), remote)

    # 2. chat template -> PROMPT FILE. genie-t2t-run (QAIRT 2.51) takes the prompt via --prompt_file,
    #    not an inline --prompt: the wrapped ChatML carries newlines and |<> markers that fight shell
    #    quoting, and the proven SM8850/v81 path fed it as a file.
    wrapped = apply_chat_template(prompt, template=meta.get("template", template),
                                  system=meta.get("system", system))
    prompt_local = os.path.join(bundle_dir_local, "prompt.txt")
    with open(prompt_local, "w") as f:
        f.write(wrapped)
    prompt_remote = f"{remote_dir}/prompt.txt"
    device.push_file(prompt_local, prompt_remote)

    # 3. config — force deterministic greedy via sampler top-k:1 (genie SILENTLY IGNORES sampler.greedy:
    #    true, falling back to temp-0.8 sampling — the tripwire went non-deterministic, good-vs-self
    #    token_match only 0.44 on SM8850/v81). Emit a derived config; never clobber the bundle's.
    cfg_name = meta.get("genie_config", "genie_config.json")
    if greedy:
        cfg_local = os.path.join(bundle_dir_local, cfg_name)
        with open(cfg_local) as f:
            greedy_obj = apply_greedy_to_config(json.load(f))
        cfg_name = cfg_name.rsplit(".", 1)[0] + ".greedy.json"
        greedy_local = os.path.join(bundle_dir_local, cfg_name)
        with open(greedy_local, "w") as f:
            json.dump(greedy_obj, f)
        device.push_file(greedy_local, f"{remote_dir}/{cfg_name}")

    # 4. run genie-t2t-run — ONLY flags confirmed on QAIRT 2.51 (-c, --prompt_file, and --profile for
    #    latency). --print-token-ids and --max-length were REMOVED: neither exists on the board, and a
    #    non-existent flag makes genie error out -> the gate false-FAILs a good artifact.
    genie_bin = meta.get("genie_bin", "genie-t2t-run")
    profile_remote = f"{remote_dir}/profile.json"
    profile_flag = f" --profile {profile_remote}" if profile else ""
    timeout_s = meta.get("timeout_s")
    timeout_prefix = f"timeout {timeout_s} " if timeout_s else ""
    segments = [f"cd {remote_dir}"]
    segments += [f"export {k}={_shquote(str(v))}" for k, v in (meta.get("env") or {}).items()]
    segments.append(f"{timeout_prefix}{genie_bin} -c {cfg_name} --prompt_file {prompt_remote}{profile_flag}")
    proc = device.shell(" && ".join(segments))
    stdout = getattr(proc, "stdout", "") or ""

    # 5. parse generated text, then RETOKENIZE text -> ids (genie emits no ids on QAIRT 2.51).
    gen = parse_generation(stdout)
    tokens = None
    if gen["text"] is not None:
        tok_path = os.path.join(bundle_dir_local, meta.get("tokenizer", "tokenizer.json"))
        tokens = retokenize(gen["text"], tok_path, tokenizer=tokenizer)

    # 6. latency (only if profiled).
    latency = {}
    if profile and device.exists(profile_remote):
        local_profile = os.path.join(bundle_dir_local, "profile.json")
        device.pull(profile_remote, local_profile)
        with open(local_profile) as f:
            latency = parse_profile(f.read())

    return {"text": gen["text"], "tokens": tokens, "latency": latency, "generation": stdout}


def _shquote(s: str) -> str:
    """Single-quote a string for the remote shell (the wrapped prompt carries newlines and |<> marks)."""
    return "'" + s.replace("'", "'\\''") + "'"
