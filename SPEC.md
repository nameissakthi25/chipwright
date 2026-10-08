# Bundle spec (v0)

A **bundle** is a self-contained, clone-and-run package that takes a model + a target device and
produces a working inference on that device — with honest, verified metrics attached. One `bundle.yaml`
is the contract; a vendor-neutral **runtime** reads it and drives the device via a **backend adapter**.

Design goal: the same contract must express a **single-graph encoder** (Laya, today) *and* a
**multi-graph autoregressive LLM** (prefill + decode with a KV cache, next) — so `graphs:` is a list,
and the host pre/post and the KV-cache wiring are declared, not hard-coded.

## `bundle.yaml` schema

```yaml
name: <bundle id>                 # e.g. laya-qcs6490
model: <source model>            # e.g. convaiinnovations/laya
task: <kind>                     # typed-decision | text-generation | ...
backend: qualcomm                # which adapter drives the device
device: qcs6490                  # target device id (within the backend)
precision: w8a16                 # declared quantization
toolchain: { qairt: "2.37.1.250807" }

artifacts:                       # pulled + hash-verified by the runtime (not vendored in git)
  - id: encoder_ctx              # logical name graphs refer to
    url: https://huggingface.co/<repo>/resolve/main/<path>
    sha256: <lfs oid>
    size: <bytes>

graphs:                          # ordered; one for an encoder, two (prefill/decode) for an LLM
  - id: encoder
    runs_on: npu
    artifact: encoder_ctx        # a QNN context binary (or dlc)
    inputs:  [inputs_embeds, gmask, smask]
    outputs: [hidden]
    # for LLM decode: state_in/state_out declare the KV-cache ping-pong tensors
    # state: { in: [k_cache_in, v_cache_in], out: [k_cache_out, v_cache_out], loop: per_token }

host:                            # host-side stages (float), run on CPU around the NPU graphs
  pre:  { module: host, fn: encode_inputs }    # text -> graph inputs
  post: { module: host, fn: decode_outputs }   # graph outputs -> result
  deps: ["laya", "onnxruntime"]

entrypoint: { module: runtime.run, args: ["--bundle", "."] }

metrics:                         # the honest, verified numbers that ship with the bundle
  verified_on: physical-qcs6490
  fidelity: { kind: hidden_state_pearson, markers: 0.82, all_tokens: 0.88 }
  latency_ms_per_decision: 84
  notes: "phishing AUROC is length-confounded on PhishNChips; see bench.json"
```

## Contract rules
- **Artifacts are referenced, never vendored.** The runtime downloads by `url` and verifies `sha256`
  (HF LFS oid) before use. Keeps the repo small; keeps provenance honest.
- **`graphs` is ordered and typed.** `runs_on: npu|host`. An encoder bundle has one; an LLM bundle has
  a `prefill` graph and a `decode` graph with a declared `state` (KV cache) block and `loop: per_token`.
- **Host stages are declared**, not implied — `pre` maps text→graph inputs, `post` maps graph
  outputs→result. The runtime wires `host.pre -> graphs -> host.post`.
- **Metrics are part of the bundle.** A bundle that can't state a verified fidelity number is not done.
- **Backend adapter** (`backend:`) implements a small interface: `load(artifact) -> session`,
  `run(session, feeds) -> outs`. The Qualcomm adapter wraps QNN `qnn-net-run --retrieve_context`.

## Status
- `bundles/laya-qcs6490/` — reference bundle #1 (single-graph encoder, typed decision).
- LLM bundle (prefill/decode + KV cache) — next; will exercise the `graphs`/`state` parts above.
