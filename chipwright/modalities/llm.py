"""LLM modality runner — the Genie driver (a slice of what a `chipwright-llm` package would own).

Unlike the CV/ASR runners, which diff one NPU tensor against a CPU reference in the same call, the LLM
runner *generates*: it assembles the Genie bundle on the device, applies the model's chat template,
runs `genie-t2t-run`, and parses back the generated text, the token ids, and the `--profile` latency
JSON. The verification against the pinned reference happens in `verify_llm.py` (the CPU reference runs
on the build host, not here — FR-L10), so this runner's job is to produce an honest on-chip generation.

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
    """Pull the generated text and, if present, the token-id trace out of `genie-t2t-run` stdout.

    Genie prints the generated text after a `[BEGIN]`/`[END]` delimiter on most builds; token ids,
    when `--print-token-ids` is on, appear as a bracketed int list. Both are best-effort and either may
    be absent — the tripwire needs the ids, so a run without them is a caller/flag error surfaced here
    (ids == None) rather than a fake empty pass.
    """
    text = None
    m = re.search(r"\[BEGIN\](.*?)\[END\]", stdout, re.DOTALL)
    if m:
        text = m.group(1).strip()
    tokens = None
    tm = re.search(r"token[_ ]ids?\s*[:=]\s*\[([\d,\s]+)\]", stdout, re.IGNORECASE)
    if tm:
        tokens = [int(x) for x in tm.group(1).replace(" ", "").split(",") if x != ""]
    return {"text": text, "tokens": tokens}


# --- the modality run (drives Genie over the device interface) ------------------------------------

def run(device, bundle_dir_local, meta, prompt, *, template="chatml", system=None,
        remote_dir="/data/local/tmp/chipwright_llm", max_tokens=64, greedy=True):
    """Assemble the Genie bundle on the device, generate for one prompt, return text + tokens + latency.

    `meta` names the bundle pieces (context bins + genie_config + tokenizer + htp ext config) and the
    model fields (template, eos). Signature stays modality-shaped: (device, bundle, meta, prompt).

    Steps:
      1. Push the bundle **file-by-file** (a single multi-GB push of a directory can wedge the adb
         daemon — FR-L02), skipping files already present.
      2. Wrap the raw prompt in the chat template (ChatML for Qwen).
      3. Run `genie-t2t-run` with `--profile` (and greedy sampling off for determinism).
      4. Parse generated text + token ids + the profile JSON.

    Returns {text, tokens, latency, generation: <raw stdout>}. The pass/fail gate is `verify_llm`, not
    here — this returns the honest on-chip generation.
    """
    import os

    # 1. bundle transport — one file at a time.
    files = meta.get("bundle_files") or sorted(os.listdir(bundle_dir_local))
    device.shell(f"mkdir -p {remote_dir}")
    for name in files:
        remote = f"{remote_dir}/{name}"
        if device.exists(remote):
            continue
        device.push_file(os.path.join(bundle_dir_local, name), remote)

    # 2. chat template — the modality wraps; the caller passed raw text.
    wrapped = apply_chat_template(prompt, template=meta.get("template", template),
                                  system=meta.get("system", system))

    # 3. run genie-t2t-run (greedy for the tripwire; --profile for latency).
    cfg = meta.get("genie_config", "genie_config.json")
    profile_remote = f"{remote_dir}/profile.json"
    sample = "" if greedy else "--use-sampler "
    cmd = (f"cd {remote_dir} && genie-t2t-run -c {cfg} {sample}"
           f"--prompt {_shquote(wrapped)} --profile {profile_remote} --print-token-ids "
           f"--max-length {max_tokens}")
    proc = device.shell(cmd)
    stdout = getattr(proc, "stdout", "") or ""

    # 4. parse generation + latency.
    gen = parse_generation(stdout)
    latency = {}
    if device.exists(profile_remote):
        local_profile = os.path.join(bundle_dir_local, "profile.json")
        device.pull(profile_remote, local_profile)
        with open(local_profile) as f:
            latency = parse_profile(f.read())

    return {"text": gen["text"], "tokens": gen["tokens"], "latency": latency, "generation": stdout}


def _shquote(s: str) -> str:
    """Single-quote a string for the remote shell (the wrapped prompt carries newlines and |<> marks)."""
    return "'" + s.replace("'", "'\\''") + "'"
