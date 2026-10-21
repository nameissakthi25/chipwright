"""The CPU reference decoder — what "correct" means for every LLM run, and the pinned reference.

The Layer-1 tripwire (`verify_llm.py`) needs something to compare the chip against. That something is
the **same quantized artifact** (the w4a16 ONNX) run greedily on CPU through onnxruntime — the qai-hub
`onnxruntime_genai` path. Because it is the same quantized model, a token mismatch isolates the
backend (the chip), not the quantization. This file is the reference the whole gate trusts, so it must
match the Genie decode semantics exactly: same tokenizer, same chat template, same KV layout, greedy.

**Why it is pinned (FR-L10).** The CPU reference runs on the x86-linux *build host*, not on the host
that drives the board over adb. So v1 computes the reference greedy tokens/logits **once** and records
them (with a `sha256`) in the artifact's provenance block; the on-device run compares against that pin
rather than re-running the CPU reference live over the board transport. `pin_reference` below emits
that pin and its hash.

The ORT-GenAI model load is validated on hardware later (checkpoint C2). Here the decode-loop
*structure* is written cleanly, and the pin/hash emit is made unit-testable by injecting the decode fn
(`decode` parameter) — so the provenance shape can be tested without onnxruntime or a model present.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


# --- the greedy decode loop (structure; model load validated on hardware) -------------------------

def greedy_decode(session, tokenizer, prompt_ids, max_new_tokens=64, eos_id=None):
    """Greedy-decode one prompt through the w4a16 prefill/decode split with KV-cache + host embedding.

    This is the structure of the qai-hub `onnxruntime_genai` path:
      1. **Prefill** — run the prompt token ids through the prefill graph (seq-N, the VTCM-fitting
         length the artifact was exported at), seeding the KV-cache. The embedding lookup is done on
         the host (host embedding) and the hidden states fed in.
      2. **Decode** — autoregressively run the decode graph one token at a time: pick argmax (greedy,
         deterministic — sampling OFF so a mismatch is signal, not sampler noise), append to the
         KV-cache, feed back, until EOS or `max_new_tokens`.

    Returns (tokens, logits): the generated token ids [T] and the per-step next-token logits [T, vocab].
    `session` is an onnxruntime-genai-style handle exposing `.prefill(ids)->(logits, kv)` and
    `.decode(token, kv)->(logits, kv)`; it is injected so the loop is testable with a fake.
    """
    logits0, kv = session.prefill(prompt_ids)          # [vocab], seeded KV-cache
    tokens, logits = [], []
    nxt = int(np.argmax(logits0))                      # greedy: argmax, no sampling
    for _ in range(max_new_tokens):
        tokens.append(nxt)
        if eos_id is not None and nxt == eos_id:       # terminate cleanly on EOS
            break
        step_logits, kv = session.decode(nxt, kv)
        logits.append(np.asarray(step_logits, dtype=np.float64))
        nxt = int(np.argmax(step_logits))
    return tokens, (np.stack(logits) if logits else np.zeros((0, 0)))


# --- the pinned reference + its hash (provenance; unit-testable with a fake decoder) --------------

def _canonical_bytes(tokens_per_prompt, logits_per_prompt) -> bytes:
    """Deterministic byte serialization of the reference, so the sha256 is stable across runs/hosts.

    Logits are rounded to float32 and dumped in a fixed, sorted-key JSON envelope; token ids are plain
    ints. Rounding to float32 keeps the hash reproducible across numpy/BLAS builds that differ only in
    float64 noise below the precision the chip could ever match.
    """
    env = {
        "tokens": [[int(t) for t in toks] for toks in tokens_per_prompt],
        "logits": [np.asarray(lg, dtype=np.float32).round(6).tolist() for lg in logits_per_prompt],
    }
    return json.dumps(env, sort_keys=True, separators=(",", ":")).encode("utf-8")


def pin_reference(prompt_set, decode, max_new_tokens=64, eos_id=None) -> dict:
    """Compute the pinned reference for a fixed prompt set and emit it with its sha256 (FR-L10 / T-L08b).

    `prompt_set` is the fixed list of (already tokenized) prompts. `decode` is the decode function —
    `greedy_decode` bound to a real session on the build host, or a fake in tests; injecting it makes
    the pin/hash emit testable without onnxruntime or a model. The returned dict is the provenance
    payload the record carries: the greedy tokens + per-step logits per prompt and the hash the board
    compares against.

      {
        "prompts": <N>,
        "decode": "greedy",
        "tokens": [[...], ...],              # per prompt
        "logits": [<np.ndarray [T, vocab]>, ...],
        "reference_logits_sha256": "<hex>",  # goes into provenance.reference_logits_sha256
      }
    """
    tokens_per_prompt, logits_per_prompt = [], []
    for ids in prompt_set:
        toks, lg = decode(ids, max_new_tokens=max_new_tokens, eos_id=eos_id)
        tokens_per_prompt.append([int(t) for t in toks])
        logits_per_prompt.append(np.asarray(lg))

    digest = hashlib.sha256(_canonical_bytes(tokens_per_prompt, logits_per_prompt)).hexdigest()
    return {
        "prompts": len(prompt_set),
        "decode": "greedy",
        "tokens": tokens_per_prompt,
        "logits": logits_per_prompt,
        "reference_logits_sha256": digest,
    }


def verify_pin_hash(pin: dict) -> bool:
    """Recompute the sha256 from a pin's tokens/logits and check it matches the recorded hash.

    Lets a reader confirm the board compared against the reference the provenance block names — the
    reproducibility guarantee of rule 5 (the record is reproducible from its provenance block).
    """
    expect = pin.get("reference_logits_sha256")
    actual = hashlib.sha256(_canonical_bytes(pin["tokens"], pin["logits"])).hexdigest()
    return expect == actual
