# Hosting the verified artifacts

To make a stranger's `run` work (not just `resolve`), the verified CV/ASR artifacts need to be hosted and
referenced by `url`+`sha256` in the public index — the same way `laya-decision` already is
(`https://huggingface.co/nameissakthi/laya-qcs6490-encoder/resolve/main/…`). This file is the manifest + the
exact commands (the HF upload is the one step that needs your account).

## The run-by-url fetch is now landed (FR-R08)
`run` fetches the run block's `cpu_onnx`/`input`/`meta` by url (sha-verified, keeping the filename so an
ONNX external-data sidecar resolves — see `cache.fetch_named` + `run._locate_ref`). So a **fully-hosted
variant runs end-to-end from a clean `pip install`** with no author assets: upload the files below, paste
the registry entry, done.

## Lead target — `mobilenet-v2` (proven 0.956 on v68, w8a16)
Files to upload (the artifact + the CPU reference, which has an external-data sidecar, + the input):

| file | size | sha256 |
|---|---|---|
| `mobilenet_w8a16.dlc` | 4.5 M | `674e275ab45e27d90f0493216717f8fae4de4a864b0ff463573a2bbe7b01cff3` |
| `mobilenet_v2.onnx` | 343 K | `8d0b30c659317596eedd46e30b65eac658a293c181bc0d8dff27c5d29ae5f284` |
| `mobilenet_v2.data` | 13 M | *(external-data sidecar — must sit next to the .onnx)* |
| `input.npy` | 588 K | `93f8247982c540dc1bda3942ec3671cefef9a26921221e53aba0da9fba6918f2` |

Complete public-index entry once uploaded — artifact AND reference all by url (paste into the
`mobilenet-v2` variant in `chipwright/data/index.yaml`; `HF=https://huggingface.co/nameissakthi/chipwright-cv/resolve/main`):
```yaml
      ext: dlc
      url: {HF}/mobilenet-v2/mobilenet_w8a16.dlc
      sha256: 674e275ab45e27d90f0493216717f8fae4de4a864b0ff463573a2bbe7b01cff3
      run:
        modality: cv
        board_name: mobilenet_w8a16.dlc
        threshold: 0.95
        cpu_onnx: {HF}/mobilenet-v2/mobilenet_v2.onnx
        cpu_onnx_sha256: 8d0b30c659317596eedd46e30b65eac658a293c181bc0d8dff27c5d29ae5f284
        cpu_onnx_sidecars:
          - {url: "{HF}/mobilenet-v2/mobilenet_v2.data", sha256: "da61b048c0d696bb8bc0260e644f71f998c6c5b203746258d1e7fc41b329a36f", name: mobilenet_v2.data}
        input: {HF}/mobilenet-v2/input.npy
        input_sha256: 93f8247982c540dc1bda3942ec3671cefef9a26921221e53aba0da9fba6918f2
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
