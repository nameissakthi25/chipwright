# Kiln — consolidated research findings & the decisions they force

Five parallel research threads (Oct 2026). Raw reports are long; this is the decision-focused synthesis.
Specific version numbers/dates below are "verify at integration time" (some came from forward-dated
search snippets). Sources are in the per-thread reports; key links kept inline.

---

## 0. The one-paragraph answer to "the SDKs update whenever they want"
**They do — Qualcomm ships ~10–12 QAIRT minors/year and actively *removes* old versions from AI Hub
within months.** You cannot stop that. What you *can* do is make the vendor-SDK version an **explicit,
declared axis of every bundle**, ship the **durable artifact (DLC) not the brittle one (context binary)**
as primary, **pin + containerize** the toolchain so old bundles stay rebuildable, and **auto-re-verify**
with a golden fidelity gate on every SDK bump. That discipline *is* Kiln's product — churn turns from a
liability into the reason Kiln exists.

---

## 1. Artifact durability — the single most important technical fact
| artifact | durability | what breaks it |
|---|---|---|
| **`.dlc` (not finalized)** | **GOOD — forward-compatible by Qualcomm guarantee** (new QAIRT loads old DLC) | only backward moves (new DLC on old runtime); custom op-package API changes; pays device-side JIT at load |
| **HTP context binary (`.bin`)** | **BRITTLE** | pinned to **SoC × HTP-arch (V68/73/75…) × QAIRT version × backend lib**; *any* QAIRT minor can invalidate it; **fails at load, not creation** |
| **custom Op Packages** | poor | no backward compat; rebuild on HTP API bumps |
| **quantization reproducibility** | medium-poor | converter/quantizer behavior drifts between minors → **silent accuracy regressions**, not hard errors |

**Decision → ship DLC as the primary, portable artifact; treat the context binary as an optional,
device-pinned, regenerate-on-provision optimization. Never ship a context binary alone.**
(Our Laya bundle already does the right thing: it ships the DLC and *finalizes/provisions the context
binary on the board*. Keep that pattern.)

## 2. Architecture — build on ExecuTorch, or on vendor SDKs directly? → **HYBRID**
Every vendor-neutral layer (ExecuTorch, LiteRT, ONNX Runtime QNN EP) reaches NPUs by **wrapping the
vendor SDK and pinning a version** (ExecuTorch pins QNN 2.37.0). So they insulate the **authoring**
surface, **not** the SDK-churn/verification timing — the churn just moves down a level into a dependency
you wait on Meta/Google to bump.

**Decision → Kiln owns the vendor-neutral IR + bundle format + verification spine (the durable promise);
backends are pluggable and *reuse* upstream compilers:**
- **Qualcomm + Arm Ethos-U → delegate to ExecuTorch** (GA Oct 2025, best NPU-LLM quant: 16a4w + SpinQuant; PyTorch-native). Optionally ONNX Runtime QNN EP as a second Qualcomm path (best context-binary story).
- **Rockchip → direct RKNN toolchain** (neutral layers don't cover it; see §3).
- **MediaTek → ExecuTorch-where-mature + direct NeuroPilot fallback** (ExecuTorch MediaTek still in dev).
- Do **not** bet the whole framework on one upstream; do **not** hand-roll every vendor SDK either.
- Dead ends confirmed: **IREE has no NPU backend; Android NNAPI is deprecated (Android 15).**

## 3. Backend #2 → **Rockchip RK3588** (not MediaTek)
| | Rockchip | MediaTek Genio |
|---|---|---|
| toolchain | **open, pip-installable** (RKNN-Toolkit2) | partner/**NDA-gated** compiler |
| on-device LLM (1–3B) | **verified**: Qwen2.5-3B ~7 tok/s, TinyLlama ~24 tok/s (rknn-llm) | unproven on Genio 510/700 (marketing only) |
| hardware | **$100–300** (Orange Pi 5 / Radxa Rock 5) | **$1,200+** EVK, distributor-only |
| cross-vendor proof | very different from QNN (`.rknn`, own runtime/quant) | also different |

Rockchip wins 4/5. Its artifact is **version-coupled to `librknnrt.so` + NPU driver** (loud
"model version X not match driver Y" errors) — a *good* concrete test surface for Kiln's packaging
contract. MediaTek stays a *later* backend if a design win demands it.

## 4. LLM on our QCS6490 (V68) — HARD, YES, but NOT via Qualcomm's stack
- **Qualcomm dropped V68 for LLMs.** LLM-on-Genie requires **Hexagon v73+**; the "QCS6490 (Proxy)"
  device was **removed from AI Hub (Jan 2026)**; the AI Hub Llama-3.2-1B export for `qcs6490` produces
  **degenerate output** (rope-scaling config bug 8.0 vs 32.0), issue stale/unanswered.
- **But it's proven in the wild on the exact chip:** Radxa runs **Llama-3.2-1B coherent ~12 tok/s**;
  ENERZAI ran **1.7B ~32 tok/s *without QNN*** (custom Hexagon kernels + 1.58-bit QAT); karusrus
  **integer-HMX** gives byte-identical-to-CPU w8a16 on no-FP16 NPUs.
- **Mechanics:** separate **prefill + decode** graphs, **KV cache as explicit int8 I/O**, static/bucketed
  shapes, model sharding. **Bandwidth-bound** (QCS6490 LPDDR5 ~25.6 GB/s is the wall) → realistic
  **~10–15 decode tok/s for a 1B**.
- **Decision →** first LLM = **Llama-3.2-1B-Instruct** (or Qwen2.5-1.5B — *and Qwen is Apache-2.0, far
  simpler licensing, see §6*), w4a16 / int8-KV / ctx 2048. Proof-of-life via **llama.cpp Hexagon +
  integer-HMX**, evaluate **ExecuTorch-QNN** in parallel, Adreno-GPU fallback. First milestone: reproduce
  Radxa's coherent ~12 tok/s on our board. **This is a DIY integration, not a packaging job** — scope it so.
- **Strategic upside:** Qualcomm *abandoned* LLMs on a huge mid-tier/robotics install base. "Verified
  small-LLM bundles on low-tier Qualcomm" is a real, unmet gap — squarely Kiln's lane.

## 5. Churn-resilience engineering (how Kiln survives the updates)
1. **Pin + containerize the whole toolchain** per SDK version: Docker image pinned by `sha256` digest,
   vendor SDK tarball vendored in by hash (QAIRT is x86-64 Ubuntu 22.04 + Python 3.10 + onnx 1.16.1 /
   ort 1.17.1 — narrow). **Mirror the SDK yourself; AI Hub removes old versions.**
2. **Bundle identity = {model@hash, quant-recipe, soc_arch (V68/73/…), precision, SDK+version, format}.**
   Put it in the filename AND manifest, with `min_runtime_version`; host-glue selects by probing
   runtime+SoC and **refuses/falls back on mismatch rather than crashing**.
3. **Golden fidelity gate re-run on every SDK bump** (CI matrix keyed on SDK × SoC × precision):
   re-convert → run golden inputs on real hardware/emulator → diff vs stored golden outputs with a
   metric gate (hidden-state/PSNR for encoders; logit-KL / task-acc for LLMs) → **block release on
   regression.** Catches the *silent* quant regressions "does it load" tests miss. (qai-hub-models ships
   exactly this as PSNR comparison — reference impl.)  **Our `fid256` is already this gate.**
4. **Decouple Kiln's adoption cadence from Qualcomm's** — pick a baseline, validate hard, re-baseline
   deliberately (~quarterly); keep old containers reproducible for already-shipped bundles.

## 6. Licensing — DO / DON'T (verify with counsel where flagged)
**DO**
- Redistribute compiled **DLC / context binary / `.rknn`** — the output **tracks the *source model's*
  license**, not Qualcomm IP (AI Hub FAQ; `qualcomm/` HF org is precedent). HF is a fine, precedented host.
- Host **Apache-2.0 / MIT** model artifacts freely (keep NOTICE). *(Our Laya DLCs are Apache-2.0 → the
  HF publish we already did is clean. Qwen being Apache-2.0 makes it the licensing-simplest first LLM.)*
- Host **Llama**-derived artifacts only with the **Llama Community License** carried, name starting
  **"Llama"**, **"Built with Llama"** attribution.
- Embed vendor **runtime `.so`** only as object code inside Kiln's host-glue app (object-code grant).

**DON'T**
- **Don't redistribute the QAIRT/QNN SDK, converters, or compilers standalone** — forbidden by the QuIC
  license. *(Gating legal question for a "turnkey bundle" that bundles `libQnn*.so` — verify with Qualcomm.)*
- **Don't** publicly host MediaTek NeuroPilot NDA binaries / `.dla` (gate them).
- **Don't** treat Llama as OSI-open or assume 3.1/3.2 are Apache (a search snippet claimed this — wrong).
- **Don't** mix differently-licensed artifacts in one HF repo.

**Flag for counsel:** (i) whether bundling Qualcomm runtime `.so` qualifies as the object-code grant;
(ii) MediaTek `.dla` redistribution; (iii) exact Llama version terms + 700M-MAU threshold.

---

## What this changes about our current state
- **Laya bundle:** already aligned (DLC primary + on-device context-binary provision; Apache-2.0; fidelity
  gate = fid256). **Add to `bundle.yaml`: a `soc_arch: v68` + `sdk:{qairt: 2.37.1}` axis** on each artifact,
  and a `min_runtime_version`, so the identity/version discipline from §1/§5 is encoded from day one.
- **Kiln architecture:** adopt the hybrid (own IR + spine; ExecuTorch/RKNN backends). Our
  `runtime/devices/qualcomm.py` is a thin direct-QNN adapter — fine for the encoder bundle; the LLM and
  future models should go through ExecuTorch-QNN, so plan the backend interface to allow both
  "direct-QNN" and "via-ExecuTorch" Qualcomm paths.
- **Roadmap:** backend #2 = Rockchip RK3588 (buy an Orange Pi 5 / Radxa Rock 5); first LLM = Qwen2.5-1.5B
  or Llama-3.2-1B on V68 as a **DIY** integration (llama.cpp Hexagon + integer-HMX first).
