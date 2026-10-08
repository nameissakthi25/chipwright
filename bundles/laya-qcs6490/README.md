# Bundle: laya-qcs6490

Turnkey **typed-decision** inference with [Laya](https://huggingface.co/convaiinnovations/laya) on a
**Qualcomm QCS6490** (Hexagon V68) NPU at w8a16. Clone → point at your board → get a calibrated
decision. Artifacts are pulled and hash-verified from the Hub, not vendored.

## Run

```bash
pip install laya onnxruntime torch numpy pyyaml          # host deps (see bundle.yaml: host.deps)
# sshpass must be installed; the board is reached over the LAN.

export LAYA_BOARD_HOST=192.168.31.41
export LAYA_BOARD_USER=root
export LAYA_BOARD_PASS=********

python -m runtime.run --bundle bundles/laya-qcs6490 \
  --text "Your account is locked. Verify now at http://bit.ly/secure-login" \
  --question "Is this a phishing or scam attempt?"
```

On first run the runtime fetches the mask helper (and, if the board doesn't already have it, provisions
the ~344 MB context binary to the board once). Output is Laya's calibrated typed decision
(`noul`/`choice`/`score` with probabilities).

## What it does (the host/NPU split)

```
text + question
  → tokenize + embedding lookup + attention masks   (host, float)   ← host.encode_inputs
  → encoder                                           (QCS6490 NPU, w8a16, context binary)
  → marker/scorer head + temperature calibration      (host, float)   ← host.decode_outputs
  → calibrated typed decision
```

The head runs through Laya's own `ag.predict` via a capture→replay encoder, so the decision is exact
(no head re-implementation).

## Verified numbers
See [`bench.json`](bench.json). Headlines: ~84 ms/decision at ~6 W (~5× better energy than a T4);
@256 encoder fidelity vs float **hidden-state Pearson 0.82 (markers) / 0.88 (all tokens)**; the min-max
→ sqnr marker-collapse fix (0.00 → 23.37 separation). The PhishNChips AUROC is length-confounded and is
**not** used as a capability claim — fidelity is judged on hidden states.

## Config
`bundle.yaml` declares the artifacts (with sha256), the single NPU `encoder` graph (256-token sqnr by
default; a 128-token min-max path is also provided), the host pre/post stages, and the board access
(env vars). To use the 128 path, point the graph at `encoder_ctx_128` / `mask_128` / `context_len: 128`.
