# Hosting the verified artifacts

To make a stranger's `run` work (not just `resolve`), the verified CV/ASR artifacts need to be hosted and
referenced by `url`+`sha256` in the public index — the same way `laya-decision` already is
(`https://huggingface.co/nameissakthi/laya-qcs6490-encoder/resolve/main/…`). This file is the manifest + the
exact commands (the HF upload is the one step that needs your account).

## Honest gap this surfaced (a small build task, flagged for the PM)
Hosting the *artifact* lets `resolve` return **USE** and the DLC fetch into the cache. But the generative/CV
gate also reads the run block's `cpu_onnx` and `input`, which `run.py` currently resolves as **local paths**
only (it fetches the *artifact* by url, not the reference/input). So full public `run` needs a small change:
`run.py` / `modalities/cv.py` MUST also fetch `cpu_onnx`+`input` by url (add `cpu_onnx_url`/`input_url`, or a
bundled fetch). Until then, hosting makes `resolve`→USE + artifact-fetch work; the on-device gate still needs
the reference locally. **Track as a follow-on build task** (fold into `003-chipwright-release` or a new spec).

## Lead target — `mobilenet-v2` (proven 0.956 on v68, w8a16)
Files to upload (the artifact + the CPU reference, which has an external-data sidecar, + the input):

| file | size | sha256 |
|---|---|---|
| `mobilenet_w8a16.dlc` | 4.5 M | `674e275ab45e27d90f0493216717f8fae4de4a864b0ff463573a2bbe7b01cff3` |
| `mobilenet_v2.onnx` | 343 K | `8d0b30c659317596eedd46e30b65eac658a293c181bc0d8dff27c5d29ae5f284` |
| `mobilenet_v2.data` | 13 M | *(external-data sidecar — must sit next to the .onnx)* |
| `input.npy` | 588 K | `93f8247982c540dc1bda3942ec3671cefef9a26921221e53aba0da9fba6918f2` |

Public-index entry once uploaded (artifact url+sha256; the reference/input await the fetch change above):
```yaml
# chipwright/data/index.yaml — mobilenet-v2 variant
url: https://huggingface.co/nameissakthi/chipwright-cv/resolve/main/mobilenet-v2/mobilenet_w8a16.dlc
sha256: 674e275ab45e27d90f0493216717f8fae4de4a864b0ff463573a2bbe7b01cff3
```

## Upload commands (your account — HF namespace `nameissakthi`)
```bash
pip install -U huggingface_hub   # if needed
huggingface-cli login            # your HF token (write)
# one model repo for the CV set:
huggingface-cli repo create chipwright-cv --type model -y
M=~/Desktop/personal/chipwright-models/mobilenet-v2
huggingface-cli upload nameissakthi/chipwright-cv "$M/mobilenet_w8a16.dlc" mobilenet-v2/mobilenet_w8a16.dlc
huggingface-cli upload nameissakthi/chipwright-cv "$M/mobilenet_v2.onnx"   mobilenet-v2/mobilenet_v2.onnx
huggingface-cli upload nameissakthi/chipwright-cv "$M/mobilenet_v2.data"   mobilenet-v2/mobilenet_v2.data
huggingface-cli upload nameissakthi/chipwright-cv "$M/input.npy"           mobilenet-v2/input.npy
```

## The rest of the set (same pattern, under `chipwright-models/<model>/`)
`zipformer-enin` (encoder ctx + encoder.onnx + sv00000.npy) · `quicksrnet-small` (w8a8) · `fcn-resnet50`
(w8a8) · `depth-anything-v2` (w8a16, ViT) · `movenet` (w8a16, has a `.data` sidecar). Compute each file's
sha256 with `shasum -a 256`, upload under `nameissakthi/chipwright-cv/<model>/…`, and add its `url`+`sha256`
to the matching variant in `chipwright/data/index.yaml`.

## Order of operations
1. Land the `run.py` fetch-by-url change for `cpu_onnx`/`input` (the gap above) — so hosting fully unlocks `run`.
2. Upload `mobilenet-v2` first; wire its `url`+`sha256`; `chipwright run mobilenet-v2` from a clean install on a board → the first stranger-runnable model.
3. Repeat for the rest; each becomes publicly runnable as its reference is hosted.
