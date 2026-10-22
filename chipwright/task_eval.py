"""Layer-2 task evaluation — WikiText perplexity + MMLU accuracy, reference-side and REPORTED-not-gated.

The generative gate has two layers kept separate (see `verify_llm.py`):

  - **Layer 1** (`verify_llm.tripwire_report`) is the on-chip backend tripwire: the same quantized
    artifact, run greedily on the chip, must match a pinned CPU reference. It can only rule degradation
    *out* — near the FP baseline the correlation between its cheap signals and task quality collapses
    (the "silent zone", arXiv 2606.19558). That is *why* Layer 2 exists.
  - **Layer 2 — task** (this file) answers the other question: *is the quantized model still good enough?*
    That is a property of the **artifact**, not of a particular chip, so it is measured **reference-side
    on the build host** (perplexity and MMLU both need per-token logits over a corpus, which the chip's
    Genie path does not emit — Genie returns text, not logits). FR-L04: task metrics are
    necessary-not-sufficient and are REPORTED, never gated. The gate is Layer 1 + the behavioral flag.

This module is deliberately thin. It does NOT re-implement perplexity or MMLU — it wraps qai-hub's own
eval harness (`qai_hub_models.datasets` + `qai_hub_models.evaluators.{ppl,mmlu}_evaluator`, the same
`add_from_dataset` / `get_accuracy_score` path the qwen reference test drives). Chipwright's job here is
the **record shape + named provenance** (dataset / split / samples) so a reader can reproduce the number,
not the eval kernel.

Two honest constraints shape the design:
  - `qai-hub-models` is a heavy dependency (torch + transformers + the dataset downloaders). It is
    **lazy-imported inside the scorer** and, if absent, raised **by name** with the install to run — the
    same preflight discipline as the sshpass/adb/tokenizers checks elsewhere. There is NO top-level
    import of it, so `import chipwright.task_eval` stays cheap and the record-assembly stays testable.
  - Record-assembly is kept **pure and injectable**: `run_task_eval` takes an optional `scorer` seam,
    so the record shape can be unit-tested with a fake scorer and no model, dataset, or heavy dep.

Honest ceiling: this is a reference-side **quality** characterization of the artifact, not an on-silicon
measurement — the record must say "build host, ORT-GenAI / FP16 torch," not imply the chip was measured.
Report WikiText as ΔPPL% vs the model's own FP baseline (PPL can paradoxically *drop* under quantization
and hide errors); use >=256 WikiText samples for a believable number (tiny_mmlu / 256-512 MMLU for CI vs
release). It is also why task metrics alone are not enough: a quantized LLM can keep its perplexity
(1.03x) while silently losing 15.2% of its refusals (arXiv 2606.09864) — that failure is invisible here
and is caught by the separate behavioral probe (`verify_llm.behavior_report`).
"""
from __future__ import annotations

from collections import namedtuple

# A task's identity: the chipwright task name, the metric key written into the record, the qai-hub
# dataset name, and which evaluator computes it. Pure data — no heavy import to define this.
TaskSpec = namedtuple("TaskSpec", "name metric dataset evaluator")

# The task registry. Datasets confirmed present in qai_hub_models.datasets (wikitext, mmlu, tiny_mmlu).
#   - "wikitext"  -> WikiText perplexity (PerplexityEvaluator.get_accuracy_score() == exp(mean NLL))
#   - "mmlu"      -> MMLU accuracy (MMLUEvaluator.get_accuracy_score() == correct / total)
#   - "tiny_mmlu" -> the ~100-question CI-grade MMLU subset (same accuracy metric, cheaper)
TASKS = {
    "wikitext": TaskSpec("wikitext", "wikitext_perplexity", "wikitext", "perplexity"),
    "mmlu": TaskSpec("mmlu", "mmlu_accuracy", "mmlu", "mmlu"),
    "tiny_mmlu": TaskSpec("tiny_mmlu", "mmlu_accuracy", "tiny_mmlu", "mmlu"),
}


def resolve_task(task) -> TaskSpec:
    """Resolve a task name (or a ready TaskSpec) to its TaskSpec; refuse an unknown name BY NAME.

    An unknown task is refused with the known set rather than silently skipped — a typo must not quietly
    drop a metric a reader expects in the record.
    """
    if isinstance(task, TaskSpec):
        return task
    try:
        return TASKS[task]
    except KeyError:
        raise ValueError(f"unknown task {task!r} (known: {', '.join(sorted(TASKS))})") from None


def task_record(metric, value, dataset, split, samples) -> dict:
    """Assemble one `task:` record (the contract block shape). Pure — the unit-testable core.

    Mirrors the contract's Layer-2 `task:` list entry: `{metric, value, dataset, split, samples}`. The
    value is stored as a float so the record serializes cleanly; everything a reader needs to reproduce
    the number (which dataset, which split, how many samples) travels with it (FR-L04 / FR-L09).
    """
    return {
        "metric": metric,
        "value": float(value),
        "dataset": dataset,
        "split": split,
        "samples": int(samples),
    }


def run_task_eval(generator=None, tasks=("wikitext",), samples=256, split="test",
                  *, scorer=None, tokenizer=None, context_length=4096, device="cpu") -> list[dict]:
    """Reference-side Layer-2 task metrics for an LLM artifact (build host, NOT on-device).

    Returns one `task:` record per requested metric — each naming dataset/split/samples so a reader can
    reproduce it (FR-L04, contract `task:` block). Perplexity/MMLU are REPORTED here, never gated; the
    gate is Layer 1 (`verify_llm.tripwire_report`) plus the behavioral flag. Call this at build/publish
    time to populate the record's `task:` block — not on the hot `run` path that drives the board.

    The scoring seam is injectable for testing: pass `scorer(spec, split, samples) -> float` to compute
    the metric without the heavy qai-hub dependency, a model, or a dataset. When `scorer` is None, the
    real qai-hub harness is wired against `generator` (an `LLM_Generator` wrapping the quantized/FP model)
    + `tokenizer`, lazy-importing qai-hub and raising BY NAME if it is absent.

    Example return:
      [{"metric": "wikitext_perplexity", "value": 10.58, "dataset": "wikitext",
        "split": "test", "samples": 256},
       {"metric": "mmlu_accuracy", "value": 0.554, "dataset": "mmlu", "split": "test", "samples": 256}]
    """
    specs = [resolve_task(t) for t in tasks]
    score = scorer if scorer is not None else _qai_hub_scorer(
        generator, tokenizer=tokenizer, context_length=context_length, device=device)
    return [task_record(s.metric, score(s, split, samples), s.dataset, split, samples) for s in specs]


# --- the qai-hub harness scorer (lazy, heavy; raises BY NAME if the dep is absent) ----------------

def _require_qai_hub():
    """Import the qai-hub eval harness lazily, or raise BY NAME naming the install (preflight style).

    `qai-hub-models` pulls in torch + transformers + the dataset downloaders — far too heavy to import at
    module load. Like the sshpass/adb/tokenizers preflights, its absence is reported by name with the
    exact install to run, never as a bare ImportError deep in a traceback. Returns the handful of symbols
    the real scorer needs.
    """
    try:
        from qai_hub_models.datasets import get_dataset_from_name
        from qai_hub_models.datasets.common import DatasetSplit
        from qai_hub_models.evaluators.ppl_evaluator import PerplexityEvaluator
        from qai_hub_models.evaluators.mmlu_evaluator import MMLUEvaluator
    except ImportError as e:  # fail by name, not silently — mirrors the adb/sshpass/tokenizers preflights
        raise RuntimeError(
            "task_eval: the 'qai-hub-models' package is required to compute reference-side WikiText "
            "perplexity / MMLU accuracy (it supplies the datasets and the eval kernels) but could not be "
            "imported. Install the 'eval' extra:  pip install 'chipwright[eval]'") from e
    return {
        "get_dataset_from_name": get_dataset_from_name,
        "DatasetSplit": DatasetSplit,
        "PerplexityEvaluator": PerplexityEvaluator,
        "MMLUEvaluator": MMLUEvaluator,
    }


def _split_enum(DatasetSplit, split):
    """Map a split string ('test'/'train') to the qai-hub `DatasetSplit` enum; refuse an unknown split."""
    key = str(split).upper()
    try:
        return getattr(DatasetSplit, key)
    except AttributeError:
        raise ValueError(f"unknown split {split!r} (known: test, train)") from None


def _qai_hub_scorer(generator, *, tokenizer=None, context_length=4096, device="cpu"):
    """Build the real reference-side scorer over qai-hub's eval harness.

    Returns `score(spec, split, samples) -> float`. For each task it builds the named dataset
    (`get_dataset_from_name(spec.dataset, split=<enum>, num_samples=samples)`), constructs the matching
    evaluator (`PerplexityEvaluator` / `MMLUEvaluator`, each `(context_length, device, tokenizer)`),
    drives it over the generator (`evaluator.add_from_dataset(generator, dataset, eval_iterations=samples)`
    — the same path qai-hub's own model tests use), and reads `evaluator.get_accuracy_score()`
    (PPL == exp(mean NLL); MMLU == correct / total). Exactly the qai-hub harness, no re-implemented math.

    `generator` is an `LLM_Generator` wrapping the quantized (ORT-GenAI) or FP16-torch model, built on the
    build host (model/recipe-specific, so it is passed in, not constructed here). It and `tokenizer` are
    required for the real path; their absence is refused by name before any dataset is downloaded.
    """
    mods = _require_qai_hub()

    if generator is None:
        raise RuntimeError(
            "task_eval: run_task_eval needs an `LLM_Generator` (the quantized/FP model wrapped for the "
            "qai-hub harness, built on the build host) to score tasks, but none was passed. Pass "
            "`generator=` (and `tokenizer=`), or inject a `scorer=` for a dry run. Never on-device — "
            "perplexity/MMLU need logits the chip's Genie path does not emit.")
    if tokenizer is None:
        raise RuntimeError(
            "task_eval: the qai-hub PerplexityEvaluator/MMLUEvaluator require the model's `tokenizer`; "
            "pass `tokenizer=` alongside `generator=`.")

    def _evaluator(kind):
        if kind == "perplexity":
            return mods["PerplexityEvaluator"](context_length, device, tokenizer)
        if kind == "mmlu":
            return mods["MMLUEvaluator"](context_length, device, tokenizer)
        raise ValueError(f"unknown evaluator {kind!r}")

    def score(spec, split, samples):
        spec = resolve_task(spec)
        dataset = mods["get_dataset_from_name"](
            spec.dataset, split=_split_enum(mods["DatasetSplit"], split), num_samples=int(samples))
        ev = _evaluator(spec.evaluator)
        ev.add_from_dataset(generator, dataset, eval_iterations=int(samples))
        return float(ev.get_accuracy_score())

    return score
