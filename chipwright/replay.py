"""Offline replay — re-derive a recorded verification from its pinned operands, with no board.

Spec 005 (reproducible-record). A verification record ships a scalar cosine and the CPU reference
(ONNX + input, hosted + sha-pinned), but historically NOT the on-device tensor the cosine was computed
against — so the headline number rested on the author's board run and a boardless user could never
reach a `VERIFIED` line. This module closes that: when a record pins the NPU output
(`vs_reference.npu_output: [{name, url, sha256, dtype, shape}]`), the cosine becomes
``f(NPU_out, CPU_out)`` with both operands public, so a skeptic recomputes it offline and a boardless
user sees the real recorded result — explicitly as a **replay of a recorded on-device result**, never
a fresh on-silicon verification (Principle I).

Design notes:
  - It reuses the SAME gate math as a live run (`verify.report`) — one implementation, two front doors
    (Principle IV). The CPU half is recomputed through the SAME modality reference the live gate uses
    (`modalities.<m>.cpu_reference`), or loaded directly when the record pins the CPU output as a tensor.
  - It is a *read of recorded evidence*, not a device transport: it never imports `device`/`device_adb`
    and never touches the board. `_locate_ref` lives in the device-free `cache` module for exactly this.
"""
from __future__ import annotations

import importlib

from . import cache, verify

TOLERANCE = 1e-3    # ±0.001: the recomputed cosine must equal the recorded value within this (FR-RR02)


def pinned_npu_output(variant):
    """Return the record's pinned NPU-output spec list (`vs_reference.npu_output`), or None.

    Read-only parsing of the optional field (FR-RR01): a list of ``{name, url, sha256, dtype, shape}``.
    Returns None when the variant has no record, no `vs_reference`, or no `npu_output` — i.e. a record
    that predates reproducible-record and is therefore not independently re-derivable.
    """
    rec = variant.verified
    if not isinstance(rec, dict):
        return None
    vsref = rec.get("vs_reference")
    if not isinstance(vsref, dict):
        return None
    specs = vsref.get("npu_output")
    if not specs or not isinstance(specs, list):
        return None
    return specs


def is_rederivable(variant) -> bool:
    """True iff the record pins the NPU output (so the headline cosine can be recomputed offline)."""
    return pinned_npu_output(variant) is not None


def _load_tensor(path):
    import numpy as np
    return np.load(path, allow_pickle=False)


def _fetch_pinned_npu(specs):
    """Fetch every pinned NPU output into the content-addressed cache (sha-verified) and load it.

    Each spec is ``{name, url, sha256, ...}``; `cache.fetch` validates the sha as 64-hex and refuses an
    insecure scheme before the bytes are trusted (0.2.2 hardening). A tampered tensor whose bytes do not
    hash to the pinned sha raises here — the replay refuses, it never reports a recomputed pass (FR-RR04).
    """
    out = {}
    for s in specs:
        try:
            name, url, sha = s["name"], s["url"], s["sha256"]
        except (TypeError, KeyError) as e:
            raise RuntimeError(f"malformed npu_output spec {s!r}: needs name/url/sha256") from e
        out[name] = _load_tensor(cache.fetch(url, sha))
    return out


def _cpu_reference(r, names):
    """The CPU reference output(s) for the gate, keyed by output name — recomputed offline, no board.

    Two sources, both device-free:
      - the record pins the CPU reference as a tensor (`cpu_onnx` resolving to a ``.npy`` / ``.npz``) —
        load it directly with numpy (fully offline, no onnxruntime needed); or
      - the usual float ONNX reference — run it through the SAME modality `cpu_reference` the live gate
        uses (`onnxruntime`, CPU). This is offline (no board), though it needs the modality's verify
        extra installed.

    `cache.locate_ref` fetches a hosted reference (sha-verified, filename-confined) or expands a local
    path — the same locator the live gate uses.
    """
    ref = cache.locate_ref(r, "cpu_onnx")
    if ref is None:
        raise RuntimeError(
            "the record has no CPU reference (cpu_onnx) to recompute the cosine against — cannot replay")
    if ref.endswith(".npz"):
        import numpy as np
        data = np.load(ref, allow_pickle=False)
        return {k: data[k] for k in data.files}
    if ref.endswith(".npy"):
        arr = _load_tensor(ref)
        if len(names) == 1:
            return {names[0]: arr}
        raise RuntimeError(
            "pinned CPU reference is a single .npy but the record has multiple outputs; "
            "pin it as a .npz keyed by output name")
    # a float ONNX reference → recompute via the modality's shared CPU-reference path.
    inp = cache.locate_ref(r, "input")
    if inp is None:
        raise RuntimeError("the record's CPU reference is an ONNX model but carries no input to run it on")
    modality = (r.get("modality") or "cv")
    mod = importlib.import_module(f".modalities.{modality}", __package__)
    cpu_out, _x, _in = mod.cpu_reference(ref, inp)
    return cpu_out


def rederive(variant):
    """Re-derive a recorded verification offline and report whether it matches the record.

    Fetches the pinned NPU output tensor(s) and the CPU reference (both via the hardened cache), runs
    them through the SAME `verify.report` a live run uses, and compares the recomputed `cosine_min` to
    the record's `vs_reference.value` within ±0.001. No board, no device import.

    Returns ``{cosine_min, recorded, matches, outputs, pass, threshold}``. Raises (refuses by name) when
    the record pins no NPU output, or when a tensor fails its sha / a reference is missing.
    """
    specs = pinned_npu_output(variant)
    if specs is None:
        raise RuntimeError(
            "no pinned on-device output to replay; this record predates reproducible-record or was not "
            "captured")

    rec = variant.verified
    vsref = rec["vs_reference"]
    recorded = vsref.get("value")
    r = variant.run or {}
    threshold = float(r.get("threshold", 0.99))          # same default the live gate uses (run_variant)

    npu_out = _fetch_pinned_npu(specs)
    cpu_out = _cpu_reference(r, list(npu_out.keys()))

    rep = verify.report(npu_out, cpu_out, threshold=threshold)   # the SAME gate the live run calls
    cosine_min = rep["cosine_min"]
    matches = (recorded is not None
               and abs(cosine_min - float(recorded)) <= TOLERANCE + 1e-12)
    return {
        "cosine_min": cosine_min,
        "recorded": recorded,
        "matches": matches,
        "outputs": rep["per_output"],
        "pass": rep["pass"],
        "threshold": threshold,
    }
