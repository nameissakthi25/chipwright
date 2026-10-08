# Chipwright

[![python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![verified on QCS6490](https://img.shields.io/badge/verified%20on-QCS6490-brightgreen.svg)](#tested-is-a-record-not-a-badge)

**A verified package manager and model hub for NPUs.** Chipwright reads your board, hands you an artifact that will actually load on your exact chip, and proves it against a CPU reference before you trust the output.

`resolve · run · build` — Qualcomm first, cross-vendor by design.

---

## The problem

On an NPU, a model isn't a file. It's a context binary pinned to a precise tuple: **architecture × SDK × quantization × shape × backend**. Load it on the wrong chip, or the wrong SDK, and it fails at load — or worse, it loads and quietly returns a fast wrong answer.

A model hub tells you *which model*. It can't tell you *which chip*. The gap between "here is a checkpoint" and "here is something that runs correctly on the board in your hand" is where edge ML projects stall.

Chipwright closes that gap. Think of it as **PyPI wheel tags for NPUs**: the chip is the platform tag. You don't hand-pick a binary and hope — you ask the resolver, and it either gives you one that matches or tells you, by name, the axis that doesn't.

---

## Quickstart

```bash
pip install -e .
```

Point Chipwright at a board (over SSH) and probe it:

```bash
export CW_BOARD_HOST=192.168.1.50
export CW_BOARD_PASS=…

chipwright doctor
```

```text
● board reachable  root@192.168.1.50:22
  SoC         : QCS6490 (soc_id 498)
  Hexagon arch: v68
  QNN libs    : libQnnHtp.so  (QAIRT 2.38.0)
  target key  : htpv68,qnn2.38
```

Resolve a model against that exact board — no guessing:

```bash
chipwright resolve whisper-small-en
```

```text
model   : whisper-small-en
target  : htpv68,qnn2.38
▲ USE (warn)  whisper-small-en-1.2.0  htpv68/w8a16/win30s
  warning: artifact tested at qnn2.37.1; board runs 2.38.0 — inside compatible range >=2.34,<2.40
```

Then run it on the board and verify the NPU output against a CPU reference:

```bash
chipwright run zipformer-enin --input utterance.npy
```

```text
model   : zipformer-enin
target  : htpv68,qnn2.38
✔ USE    zipformer-enin-7.0.0  htpv68/w8a16/win39  qnn2.37.1  (exact match)

✔ VERIFIED  NPU vs CPU-reference  cosine_min = 0.98800  (threshold 0.95)
    encoder: 0.98800
```

That `0.98800` is not a claim in a README. It is the number Chipwright measured on a live QCS6490 against an ONNX-Runtime reference, on the input you gave it, this run.

---

## The three verbs

| Verb | What it does |
| --- | --- |
| **resolve** | Match a registry variant to your board's target key. If nothing matches, name the axis that failed — never silently downgrade to a near-miss. |
| **run** | `resolve` → fetch (or locate) the artifact → push and execute on the board → compare the NPU output to a CPU reference and report the cosine. |
| **build** | Produce the artifact from a recipe when no hosted variant matches your chip — op pre-check, host preflight, convert → quantize → on-board context, then gate on the verification before publishing. |

---

## The four resolve outcomes

The resolver is deliberately honest. Every resolution lands in exactly one of four outcomes:

- **USE** — an exact match on arch, quant, and shape, with the board's SDK inside the compatible range. Run it.
- **USE-with-warning** — a match, but the board's SDK sits outside the *tested* point while staying inside the compatible range. You get the artifact and the caveat, both.
- **BUILD** — no hosted artifact for your chip, but a recipe exists that can produce one.
- **FAIL** — nothing fits, and the message names the axis:

  ```text
  ✗ FAIL  axis=htp_arch: no variant for v68; registry has [v73]
  ```

The arch axis is hard — a mismatch fails at load, so the resolver refuses. The SDK axis is a range, not a point — so outside the tested point the resolver warns rather than refuses. That distinction is the whole discipline: **never downgrade to a near-miss artifact silently.**

---

## Artifact tags

An artifact's identity is its name. Chipwright names the five axes a resolver matches on in the filename and expands the rest in the manifest — exactly how a PyPI wheel names `cp312-manylinux_x86_64` and leaves the rest to metadata:

```text
whisper-small-en-1.2.0+qnn2.37-htpv68-w8a16-win30s.qnnctx
└──────┬───────┘ └─┬─┘ └──┬──┘ └──┬──┘ └─┬─┘ └─┬─┘ └─┬──┘
     name       version  sdk    arch   quant  shape  ext
```

Parsing is strict: a tag that doesn't parse is a bug, not a near-miss to paper over.

---

## Tested is a record, not a badge

Chipwright does not stamp a green checkmark and ask you to believe it. Every registry variant carries a **verification record** — which board it ran on, what it was compared to, and the numbers — so a reader who distrusts the claim can check the inputs.

The record for `zipformer-enin`, from a live run:

```yaml
verified:
  on: "QCS6490 (soc_id 498)"
  date: 2026-10-08
  loads: true
  vs_reference:
    backend: onnxruntime-cpu
    metric: encoder_cosine
    value: 0.988           # threshold 0.95 — VERIFIED
```

A high cosine is necessary, not sufficient — a cosine of 0.99 can still cost task accuracy — so the record keeps the numerical check separate from the task metric. The point is not the badge. The point is that the evidence travels with the artifact.

---

## Architecture

Three front doors, one core, three targets.

- **Three front doors** — the `chipwright` CLI, a desktop app, and Claude Code skills. Each is a thin surface.
- **One core** — [`chipwright-core`](chipwright/) holds all the logic: tags, board probe, resolver, registry client, cache, device transport, verification, and the run loop. The front doors call it; they don't reimplement it.
- **Three targets** — the core reaches the **registry** (the index of verified variants), the **board** (probe and execute), and the **build-host** (produce an artifact from a recipe).

See [`docs/architecture.md`](docs/architecture.md) for the diagram and the module map.

---

## Repo layout

```text
chipwright/
  cli.py          the `chipwright` CLI — doctor · resolve · run · build
  tags.py         wheel-tag grammar for NPU artifacts (parse / format)
  env.py          board probe → TargetKey (arch × SDK × SoC)
  resolver.py     match variants to a target; the four outcomes
  registry.py     read the index (a git-repo file), list a model's variants
  cache.py        content-addressed artifact cache (~/.cache/chipwright)
  device.py       remote QNN runner — push + execute over SSH
  verify.py       NPU-vs-CPU cosine; the check that catches a fast wrong answer
  run.py          the run verb — ties resolve · device · modality · verify
  modalities/     per-task glue (frontend, decoders, CPU reference)
registry/
  index.yaml      the registry index — one reviewable file, variants by hash
```

---

## Status

- ✅ **resolve** — done, with the four honest outcomes, offline targets, and axis-named failures.
- ✅ **run** — done and **verified on hardware**, four real models pulled from Hugging Face, on a live QCS6490 / RB3 Gen2:

  | model | task | quant | cosine |
  |---|---|---|---|
  | `zipformer-enin` | ASR (streaming RNN-T) | w8a16 | **0.988** |
  | `mobilenet-v2` | image classification | w8a16 | **0.956** |
  | `quicksrnet-small` | super-resolution | w8a8 | **0.9999** |
  | `fcn-resnet50` | semantic segmentation | w8a8 | **0.998** |

  Two modalities, three CV tasks, two quant profiles — the quant axis is real, and every run ends in a measured number. The gate earns its keep: a random-noise input to MobileNet correctly **FAILs** at 0.776, and a MediaPipe face detector surfaced a genuine w8a8 fidelity loss on its secondary score head (0.80 vs 0.98 elsewhere) — caught, not shipped. A Depth-Anything ViT was **refused at convert** (an unsupported reshape attribute) rather than built into garbage — the honest boundary, handled by a recipe adaptation.
- ✅ **build** — implemented: op pre-check, host preflight, `convert → quantize → on-board context`, and a fidelity gate before publish. (A real build needs an x86_64-linux host with QAIRT installed; the CLI says so when it isn't there.)

**SDK-range in practice:** that same board runs QAIRT **2.38.0** while the artifact was tested at **2.37.1**, both inside the compatible range `>=2.34,<2.40`. It ran correctly — and `resolve` says so plainly (USE-with-warning), rather than pretending the tested point and the board agree.

### Modality roadmap

```text
ASR  →  TTS  →  VLM  →  LLM  →  VLA
 ▲
 verified on hardware today
```

---

## Vendor-neutrality

Chipwright is Qualcomm first because that is where the hardware and the proof are today. It is cross-vendor by design: the tag grammar, the resolver's axis model, the verification record, and the device transport are not tied to one silicon vendor. Adding a backend means adding a target adapter and its arch axis — not reworking the framework.

The specification lives in a separate repo, the **spec kit** (`chipwright-speckit`, in GitHub Spec Kit format). Changes are spec-first: the spec kit moves before the code.

---

## License

MIT — see [LICENSE](LICENSE). Contributions welcome; start with [CONTRIBUTING.md](CONTRIBUTING.md).
