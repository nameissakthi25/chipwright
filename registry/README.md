# Chipwright model catalogue

This directory is the Chipwright registry. The catalogue lives in a single reviewable file,
[`index.yaml`](index.yaml); artifacts themselves live by content hash in object storage, not here.
There is no server to run — the client reads the index as a plain file (local, or a git URL once
published), which is what makes a verification record auditable in a pull request.

## The unit is a build *variant*, not a model

A model name (e.g. `laya-intent`) is just a heading. The thing the resolver actually picks is a
**variant**: one artifact for one point in the target space —

```
arch × quant × shape × SDK-range
```

- **arch** — the Hexagon HTP architecture (`v68`, `v73`, `v75`). This is the *hard* axis: a binary
  built for the wrong arch fails at load, so a mismatch is never downgraded.
- **quant** — `w8a16` for most graphs on v68 (the HTP is w8a16-only — no fp16), `w4a16` for LLMs,
  which must be all-integer on v68.
- **shape** — the input/context geometry the artifact was compiled for (`win256`, `win39`,
  `win30s`, `ctx4096`, `multigraph`).
- **SDK range** — `sdk_tested` is the exact QAIRT point the artifact was built and verified at;
  `sdk_compatible` is the range it is expected to load across. The SDK axis is a *range*, not a
  point: a board inside the range but off the tested point warns, it does not refuse.

One model fans out to many variants; the catalogue stores each one separately so the resolver can
answer a specific board honestly.

## USE / BUILD / FAIL

Given a board's target key, the resolver returns exactly one decision:

| Outcome | When | What the user gets |
|---|---|---|
| **USE** | a built artifact matches arch/quant/shape and the board's QAIRT is the tested point | run it |
| **USE (warn)** | same, but the board's QAIRT is in-range yet off the tested point (e.g. board 2.38.0 vs tested 2.37.1) | run it, with a compatibility warning |
| **BUILD** | arch matches, no usable artifact, but a `recipe:` can produce one | an offer to build, with an estimate |
| **FAIL** | nothing satisfies a required axis | a refusal that *names the axis* (e.g. `htp_arch`) |

Target board for this catalogue: QCS6490 / RB3 Gen2, Hexagon **v68**, QAIRT **2.38.0** (in-field),
artifacts built at **2.37.1**. On that board: `laya-decision` resolves **USE**, the recipe-only
entries resolve **BUILD**, and `qwen2.5-7b-instruct` (v73/v75 only) **FAILs on `htp_arch`**.

## The verification record

`verified:` is evidence, not a badge. It records the run that validated a variant so a skeptical
reader can check the inputs:

- `on` — the device it ran on (e.g. `QCS6490 (soc_id 498)`).
- `loads` — whether the artifact loaded on the HTP at all.
- `vs_reference` — the numeric agreement with a trusted reference (CPU float ONNX / fp32 pipeline),
  with the metric named (`hidden_state_pearson`, `wer_delta`, `logmel_corr_dtw`, `top1_vs_fp32`).
- `task_metric` — an end-task number where we have one, with a note on what it does and does not prove.
- `latency_ms` — timing, as a point (`p50`) or a note.

A record describes one measured run. Read the notes: a high agreement number on a synthetic
held-out set is honest about not being a real-world accuracy claim.

## Entry kinds

- **Hosted** — carries a `url` + `sha256`; the artifact is downloadable and resolves **USE**.
  (`laya-decision`.)
- **Recipe-only** — no hosted artifact; a `recipe:` can build one on the board, so it resolves
  **BUILD**. The DLC may live off-repo. Several of these still carry a real on-device `verified:`
  record from a build we ran. (`laya-intent`, `supertonic-tts`, `whisper-small-en`,
  `zipformer-enin` also has a built artifact on disk + a `run:` block that re-verifies it.)
- **Experimental** — recipe-only and explicitly not accuracy-claimed; the path works but the
  numbers aren't settled. (`qwen2.5-1.5b-instruct` on v68.)

## The models

All verified records below are w8a16 on Hexagon v68, on a live QCS6490, unless noted.

| Model | Modality | Kind | Key verified number | Status |
|---|---|---|---|---|
| `laya-decision` | intent typed-decision (ModernBERT) | hosted | hidden-state Pearson **0.88** vs CPU; p50 **84 ms** | USE on v68 |
| `laya-intent` | joint 12-way intent + 17-tag BIO slots (ModernBERT) | recipe-only + record | QAT w8a16 **100% top-1** on-device (120/120 vs fp32 and gold, 0 missing; PTQ baseline 42.5%) | verified, BUILD on v68 |
| `zipformer-enin` | streaming RNN-T STT, Indian English | recipe + local artifact + `run` | encoder cosine **0.988** vs CPU float; WER **0.270** (htp) vs **0.287** (cpu-float); enc ~16.4 ms/chunk, RTF ~0.05 | verified live, BUILD on v68 |
| `supertonic-tts` | 4-model flow-matching TTS | recipe-only + record | log-mel correlation **0.77** vs fp32 (DTW-aligned); ~3.8 s wall / 3.07 s audio | verified, BUILD on v68 |
| `whisper-small-en` | ASR | recipe-only | — (canonical resolve → BUILD) | BUILD on v68 |
| `qwen2.5-1.5b-instruct` | small LLM | recipe-only, experimental | — (w4a16; on-board context-gen ~30–45 min/split) | EXPERIMENTAL, BUILD on v68 |
| `qwen2.5-7b-instruct` | LLM | recipe-only, v73/v75 only | — | FAIL on `htp_arch` for a v68 board |

Notes:
- `laya-intent`'s 100% is on a synthetic held-out set where fp32 also scores ~100%; it proves QAT
  removed the on-device quantization loss (vs the 42.5% PTQ baseline), not a real-world accuracy
  ceiling.
- `supertonic-tts` has a small residual buzz from the w8a16 flow-matching vocoder; the vocoder uses
  per-channel weights + CLE.
- `qwen2.5-1.5b-instruct` on v68 needs all-integer w4a16 (the fp16 context-gen path fails); the
  build is the slow part. It carries no task metric on purpose.
