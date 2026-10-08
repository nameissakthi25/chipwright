# Contributing to Chipwright

Thanks for helping build a package manager that edge ML developers can actually trust. Two principles keep it trustworthy, and both are non-negotiable: **changes are spec-first**, and **every published artifact is verified on hardware**.

## Spec-first

The specification lives in a separate repo — the **spec kit** (`chipwright-speckit`, in GitHub Spec Kit format). It is the source of truth for the tag grammar, the resolver's four outcomes and axis model, the verification record, and the registry schema.

A change that alters any of that moves the spec kit **before** the code:

1. Open the change against the spec kit and get it reviewed there.
2. Implement it in `chipwright-core` to match.
3. Reference the spec change in your pull request.

A bug fix that doesn't change the contract doesn't need a spec change — but if you find yourself arguing about intended behavior, that argument belongs in the spec kit.

## Running the tests

```bash
pip install -e .
python -m pytest chipwright/tests
```

The unit tests cover the resolver's outcomes and the tag grammar — the parts that must stay correct without a board attached. Run them before every pull request. If you change resolver behavior, add or update a test that pins the new outcome.

## The verification gate

The registry does not accept an artifact on trust. Before a variant is published, it must:

1. **Load** on the target board, and
2. **pass a CPU-reference cosine** — the NPU output is compared to an ONNX-Runtime (or equivalent) reference on the same input, above the variant's threshold.

The numbers from that run — the board, the reference backend, the metric, the value — are recorded in the variant's `verified:` block. The record is the evidence, not a badge: a reviewer must be able to check the inputs. A pull request that adds or updates a hosted variant without a verification record will not be merged.

## Board-CI

Numerical fidelity on real silicon can't be proven in a cloud runner with no NPU. Hardware-touching checks — probe, push, execute, verify — run on **board-CI**: a runner with a physical board attached (today a QCS6490 / RB3 Gen2). Tests that need the board are marked so they're skipped off board-CI and run there. If your change touches `device.py`, `run.py`, a modality runner, or a hosted variant, expect it to be gated on board-CI before merge.

## Scope hygiene

Chipwright is Qualcomm first but cross-vendor by design. Keep vendor-specific logic inside the device/modality adapters; keep the tag grammar, resolver, registry, cache, and verification spine vendor-neutral. New backends arrive as adapters plus their arch axis — not as changes to the framework's shape.
