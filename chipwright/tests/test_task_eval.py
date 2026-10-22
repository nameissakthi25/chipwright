"""Layer-2 task eval + behavioral-probe data — record-assembly, the lazy-import guard, the prompt sets.

Pure / no heavy deps: the record-assembly is exercised with a FAKE scorer (no qai-hub, no model, no
dataset), and the lazy-import-missing path is forced deterministically regardless of whether
`qai-hub-models` happens to be installed. Runnable as `python3 test_task_eval.py`.
"""
import builtins
import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import chipwright
from chipwright import task_eval
from chipwright.task_eval import (
    TaskSpec, TASKS, resolve_task, task_record, run_task_eval, _qai_hub_scorer, _require_qai_hub,
)


# --- pure record-assembly (task_record) -----------------------------------------------------------

def test_task_record_shape_and_types():
    rec = task_record("wikitext_perplexity", 10.58, "wikitext", "test", 256)
    assert rec == {
        "metric": "wikitext_perplexity", "value": 10.58, "dataset": "wikitext",
        "split": "test", "samples": 256,
    }
    assert isinstance(rec["value"], float) and isinstance(rec["samples"], int)


def test_task_record_coerces_value_and_samples():
    rec = task_record("mmlu_accuracy", 1, "mmlu", "test", "256")  # int value, str samples
    assert rec["value"] == 1.0 and isinstance(rec["value"], float)
    assert rec["samples"] == 256 and isinstance(rec["samples"], int)


# --- resolve_task ---------------------------------------------------------------------------------

def test_resolve_task_known_names():
    assert resolve_task("wikitext").metric == "wikitext_perplexity"
    assert resolve_task("mmlu").dataset == "mmlu"
    assert resolve_task("tiny_mmlu").evaluator == "mmlu"


def test_resolve_task_passthrough_spec():
    spec = TaskSpec("x", "m", "d", "perplexity")
    assert resolve_task(spec) is spec


def test_resolve_task_unknown_refuses_by_name():
    with pytest.raises(ValueError, match="unknown task"):
        resolve_task("not_a_dataset")


# --- run_task_eval with an injected FAKE scorer (no heavy dep, no model, no dataset) ---------------

def test_run_task_eval_with_fake_scorer_assembles_records():
    calls = []

    def fake_scorer(spec, split, samples):
        calls.append((resolve_task(spec).name, split, samples))
        return {"wikitext_perplexity": 10.58, "mmlu_accuracy": 0.554}[resolve_task(spec).metric]

    out = run_task_eval(tasks=("wikitext", "mmlu"), samples=256, split="test", scorer=fake_scorer)
    assert out == [
        {"metric": "wikitext_perplexity", "value": 10.58, "dataset": "wikitext",
         "split": "test", "samples": 256},
        {"metric": "mmlu_accuracy", "value": 0.554, "dataset": "mmlu",
         "split": "test", "samples": 256},
    ]
    # the scorer saw each task with the requested split/samples (provenance threads through)
    assert calls == [("wikitext", "test", 256), ("mmlu", "test", 256)]


def test_run_task_eval_fake_scorer_needs_no_generator_or_dep():
    # the injectable seam is exactly what makes this testable with no qai-hub and no model present
    out = run_task_eval(tasks=("tiny_mmlu",), samples=100, scorer=lambda s, sp, n: 0.61)
    assert out == [{"metric": "mmlu_accuracy", "value": 0.61, "dataset": "tiny_mmlu",
                    "split": "test", "samples": 100}]


# --- the lazy-import guard: raises BY NAME when qai-hub-models is absent ---------------------------

def test_require_qai_hub_raises_by_name_when_absent(monkeypatch):
    # force the import to fail regardless of whether qai-hub-models is installed in this env
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("qai_hub_models"):
            raise ImportError("No module named 'qai_hub_models'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match=r"qai-hub-models|chipwright\[eval\]"):
        _require_qai_hub()


def test_qai_hub_scorer_refuses_missing_generator_by_name(monkeypatch):
    # with the dep "present" (stubbed), a missing generator is still refused by name before any download
    monkeypatch.setattr(task_eval, "_require_qai_hub", lambda: {})
    with pytest.raises(RuntimeError, match="generator"):
        _qai_hub_scorer(None)


def test_qai_hub_scorer_refuses_missing_tokenizer_by_name(monkeypatch):
    monkeypatch.setattr(task_eval, "_require_qai_hub", lambda: {})
    with pytest.raises(RuntimeError, match="tokenizer"):
        _qai_hub_scorer(object(), tokenizer=None)


def test_importing_task_eval_does_not_import_qai_hub():
    # the heavy dep must NOT be pulled in at module import (preflight discipline)
    assert "qai_hub_models" not in sys.modules


# --- the behavioral-probe data file ---------------------------------------------------------------

def _load_behavior_prompts():
    path = os.path.join(os.path.dirname(chipwright.__file__), "data", "behavior_prompts.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def test_behavior_prompts_loads_with_both_sets_nonempty():
    data = _load_behavior_prompts()
    assert set(data) >= {"harmful", "benign"}
    assert isinstance(data["harmful"], list) and len(data["harmful"]) > 0
    assert isinstance(data["benign"], list) and len(data["benign"]) > 0


def test_behavior_prompts_are_surface_twinned_and_well_formed():
    data = _load_behavior_prompts()
    # aligned by index (every harmful item has a benign control at the same position)
    assert len(data["harmful"]) == len(data["benign"]) == 24
    for p in data["harmful"] + data["benign"]:
        assert isinstance(p, str) and p.strip()          # non-empty strings, no stray nulls
    # the two sets are disjoint — a benign twin must not accidentally repeat a harmful prompt
    assert set(data["harmful"]).isdisjoint(data["benign"])


if __name__ == "__main__":
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    import types

    class _MP:  # minimal monkeypatch shim for the __main__ runner
        def __init__(self): self._undo = []
        def setattr(self, obj, name, val):
            old = getattr(obj, name); self._undo.append((obj, name, old)); setattr(obj, name, val)
        def undo(self):
            for obj, name, old in reversed(self._undo): setattr(obj, name, old)

    passed = 0
    for f in fns:
        mp = _MP()
        try:
            n = f.__code__.co_argcount
            f(mp) if n == 1 else f()
            print("ok", f.__name__); passed += 1
        finally:
            mp.undo()
    print(f"\n{passed} passed")
