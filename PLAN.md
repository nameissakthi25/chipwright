# Kiln — plan

> Vision: a **quality-verified, turnkey optimization + distribution layer** for running modern models
> (incl. small LLMs and typed-decision models) on **edge NPUs across vendors** — starting with
> Qualcomm, then MediaTek / Rockchip / others. Not "HuggingFace for Qualcomm" (redundant with AI Hub),
> but **"the Ollama for edge-agentic inference"**: clone → runs on *your* board, proven not to collapse.

## Why this exists (the market pain)
Every edge-NPU vendor ships its own microarchitecture, quantizer, compiled format, runtime, and its
own pile of landmines. There is no cross-vendor path that also **verifies quantization quality** and
covers **LLMs on low/mid-tier parts**. Vendors' own hubs (Qualcomm AI Hub) are vision/speech-heavy,
flagship-centric for LLMs, and ship DLCs — not ready-to-run, quality-checked device bundles.

## Landscape

| Vendor | NPU | Toolchain / quantizer | Format | Notes |
|---|---|---|---|---|
| **Qualcomm** | Hexagon V68/69/73/75 | QAIRT/QNN + AIMET | DLC → context binary | no FP16 on V68; runtime-Gather collapse; sqnr fix — **proven (Laya)** |
| **MediaTek** | APU (Dimensity/**Genio**) | NeuroPilot / Neuron SDK | TFLite+Neuron / DLA | "Helio" ≈ weak/no NPU; **Genio** is the QCS6490 analog for IoT |
| **Rockchip** | RKNN (RK3588 ~6 TOPS) | RKNN-Toolkit2, rknn-llm | .rknn | cheap SBCs; small-LLM support |
| **NVIDIA Jetson** | GPU + DLA | TensorRT, CUDA | .engine | best LLM story; robotics default (not an NPU) |
| **Intel** | NPU (Core Ultra), Myriad | OpenVINO + POT | OpenVINO IR | strong tooling |
| **Hailo** | Hailo-8 (dataflow) | Dataflow Compiler | .hef | unusual arch; int8 |
| **NXP / Arm** | i.MX NPU / Ethos-U | eIQ / Vela + TFLite-Micro | TFLite int8 | TinyML/MCU |
| **Google** | Edge TPU | TFLite (int8 only) | .tflite | int8-only |

## Architecture — one frontend, N backend adapters, one shared quality spine

```
PyTorch / ONNX  ─►  [ per-backend adapter ]  ─►  device-ready bundle
                      Qualcomm: QAIRT + sqnr + HTP-emulator
                      MediaTek: NeuroPilot + Neuron
                      Rockchip: RKNN-Toolkit
                      ...each: convert → quantize → compile
                             │
                     ┌───────┴────────┐
                     │  SHARED SPINE  │  ← the reusable moat (vendor-neutral)
                     │  fidelity gate: float-vs-quant compare, collapse detection,
                     │  honest benchmark, host-split glue, bundle packer
                     └────────────────┘
```

- **Per-backend adapters** = vendor-specific grunt work (gated SDKs, x86-Linux-only, version-fragile).
- **Shared spine** = the real asset; generalizes across all vendors ("did quantization break the model?"
  + honest metrics + clone-and-run bundle). Already built for Qualcomm via the Laya work.
- **A100 = build farm**: batch convert+quantize+emulate for every backend with an x86 toolchain
  (Qualcomm HTP emulator, RKNN sim, OpenVINO, TensorRT), fidelity gate on each.

## Differentiation vs prior art
ExecuTorch (Meta, multi-backend: Qualcomm/MediaTek/Arm/Apple/Cadence), LiteRT (Google), ONNX Runtime
EPs, Apache TVM, IREE/MLIR — all solve **compilation portability**. None own: **quality-verified
quantization**, **LLMs on low-tier parts**, **turnkey device bundles**, **honest benchmarks**.
→ That gap is the Laya story. That's the lane.

## The moat
Not "supports many chips" (ExecuTorch already claims that). It's:
**every catalog model is proven not to silently collapse, and runs turnkey on your exact board.**

## Sequencing
1. **Qualcomm first (proving ground — Laya done).** Productize the manual pipeline into a reusable
   Qualcomm backend + the vendor-neutral fidelity spine. Add the first LLM catalog entry
   (1–3B, prefill/decode + KV-cache graph on QCS6490).
2. **Build the spine properly** so adding a backend = "write an adapter," not rebuild.
3. **Backend #2 = Rockchip RK3588 or MediaTek Genio** — proves the abstraction holds across two very
   different NPUs. *This is the moment it becomes a framework, not a repo.*
4. **Lead with** verified quality + small LLMs/typed-decision models on low-tier edge NPUs.

## Honest risks
- **Platform risk:** Qualcomm/others can extend their hubs and erase the wedge → stay the turnkey/
  verified/vertical layer, not the raw-conversion layer.
- **Breadth is a trap:** N vendors ≈ N× ops burden; quirks are per-backend and don't transfer.
  Prove on TWO backends before scaling.
- **Licensing:** per-model weight/DLC redistribution constraints.
- **Buyers:** robotics & IoT integrators who want on-device agents but lack NPU expertise (underserved).

## Proven so far (Qualcomm / Laya)
- Repos: github.com/nameissakthi25/laya-qcs6490 (public), …-lab (private).
- Model artifacts: huggingface.co/nameissakthi/laya-qcs6490-encoder (DLCs + mask helpers + context bins).
- The pipeline prototype: ONNX surgery → convert → **sqnr** quantize → fidelity gate (hidden-state r
  0.82–0.88) → HTP x86 emulate → device bundle. Plus the host/NPU split and wedge-safe batched runs.
