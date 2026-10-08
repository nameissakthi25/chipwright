# Chipwright

**Fire a raw float model into a hardened, *verified*, device-ready bundle that runs turnkey on a
specific edge NPU — and refuse to ship one that broke in quantization.**

A quality-verified, turnkey optimization + distribution layer for modern models on **edge NPUs** —
Qualcomm first, then Rockchip and others. Not a HuggingFace/AI-Hub clone: the differentiator is
**every model is proven not to silently collapse, and runs turnkey on your exact board.**

- **Plan & landscape:** [`PLAN.md`](PLAN.md)
- **Research findings (SDK churn, hybrid architecture, licensing):** [`RESEARCH.md`](RESEARCH.md)
- **Bundle contract:** [`SPEC.md`](SPEC.md)
- **Runtime:** [`runtime/`](runtime/) — vendor-neutral runner + `devices/` backend adapters
- **Bundles:** [`bundles/`](bundles/) — clone-and-run packages with verified metrics

## Status (Qualcomm first — Rockchip deferred until the Qualcomm LLM works)
- ✅ `bundles/laya-qcs6490` — reference bundle #1: single-graph typed-decision encoder on QCS6490, verified live.
- 🔨 **Qwen2.5-1.5B on QCS6490** — bundle #2: prefill/decode + KV-cache graph. A **DIY integration**
  (Qualcomm dropped LLMs on Hexagon V68 — see `RESEARCH.md` §4); llama.cpp-Hexagon + integer-HMX first.
- ⬜ Factor the Qualcomm pipeline + fidelity gate into the reusable `kiln build` engine.
- ⬜ Backend #2 = **Rockchip RK3588** (designed-for in `SPEC.md`; deferred).

## Durability discipline (why Kiln exists — see `RESEARCH.md`)
- **Ship DLCs (forward-compatible), not context binaries alone** (brittle: pinned to SoC × HTP-arch × QAIRT version).
- **Bundle identity carries the SDK version + SoC arch**; re-run the fidelity gate on every SDK bump.
- Backends reuse upstream compilers (ExecuTorch for Qualcomm/Arm, direct RKNN for Rockchip); Kiln owns the IR + verification spine.

## Layout
```
runtime/
  run.py            vendor-neutral: reads bundle.yaml, wires host.pre -> graphs -> host.post
  fetch.py          download + sha256-verify artifacts (not vendored)
  devices/
    qualcomm.py     backend adapter: QNN HTP context binary over SSH to the board
bundles/
  laya-qcs6490/     bundle.yaml + host.py + bench.json + README
```
