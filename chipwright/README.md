# qcf-core

The shared core of Chipwright: **the resolver, registry client, artifact cache, and board probe** that
the CLI, the desktop app, and a Claude Code session all call, so they always agree about what will
run on your board. (Design: Devansh's *Qualcomm AI Model Framework — Design Brief*.)

## Milestone 1 — resolve-only, one model, one board

```
chipwright doctor                              # probe board -> SoC, Hexagon arch, QNN libs, reachability
chipwright resolve whisper-small-en            # match a registry variant to the board; explain any miss
chipwright run whisper-small-en --input x.wav  # resolve -> fetch+hash-verify -> (push+run+CPU cosine)
```

Board connection from env (`CW_BOARD_HOST` / `_USER` / `_PASS` / `_PORT`) or flags. With no board,
`resolve` still works against a hand-named `--target htpv68,qnn2.37`.

## The pieces

| Module | Role |
|---|---|
| `tags.py` | The wheel-tag grammar: `name-ver+qnn2.37-htpv68-w8a16-win30s.qnnctx` (parse/format, strict) |
| `env.py` | Board probe → `TargetKey` (Hexagon arch, QAIRT, SoC, reachability). Offline is first-class |
| `registry.py` | Reads the index (a git-repo YAML file, cached); the unit is a **build variant**, not a model |
| `resolver.py` | `resolve(target, variants)` → one `Decision`. The **explicit 4-outcome miss path** below |
| `cache.py` | Content-addressed `~/.cache/chipwright/`; hash-verified on fetch and read |
| `cli.py` | `chipwright doctor` / `resolve` / `run` |

## The resolution contract (never silently downgrade)

1. **USE** — exact match on arch/quant/shape; SDK inside the compatible range.
2. **USE (warn)** — in range, but outside the SDK point the artifact was tested at (warn once).
3. **BUILD** — no artifact, but a recipe can produce one for this target (milestone 1 stops here).
4. **FAIL** — nothing; the Decision **names the axis** (arch / quant / shape / qairt) that failed.

The arch axis is hard (a mismatch fails at load); the SDK axis is a range, not a point.

## Why "tested" is a record, not a badge

Each registry variant carries a `verified:` block — which board, what it was compared against, the
numbers — so a reader who distrusts the claim can check it. The seed registry already carries real
on-device records (e.g. `zipformer-enin`: HTP w8a16 WER 0.270 vs CPU-float 0.287).

Run the tests: `python chipwright/tests/test_resolver.py` (10 cases, no board needed).
