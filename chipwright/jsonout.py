"""Machine interface (spec 006) — a JSON projection + a documented exit-code contract.

This is a *presentation* layer, not a second brain (Principle IV: one implementation). Every function
here only reads the typed `Decision` the resolver already produced and the dict `verify.report()`
already returned — it never re-resolves, re-runs, or re-verifies. `--json` and the exit codes are a
second front door over the same objects, so a CI/CD pipeline can branch on fields and codes instead of
regex-scraping coloured human text.

The outcome→code table is a versioned contract (FR-J03): the codes are distinct and stable, and a
change of meaning needs a MAJOR bump.
"""
from __future__ import annotations

from typing import List, Optional

from .resolver import Decision, Outcome, Variant, resolve

# --- the versioned exit-code contract ---------------------------------------------------------
__api__ = "1.0"

EXIT_USE = 0             # exact match — run it
EXIT_BUILD = 10         # no artifact, but a recipe can build one
EXIT_USE_WITH_WARN = 11  # in range, outside the tested SDK point
EXIT_FAIL = 20          # nothing fits — the named axis is why
EXIT_VERIFY_MISMATCH = 30  # ran, but the NPU vs CPU cosine fell below threshold
EXIT_NOT_RUNNABLE = 40   # resolved, but cannot execute here (no run block / no board)
EXIT_USAGE = 2          # usage error / unknown model
EXIT_UNEXPECTED = 1     # an unexpected internal error

_OUTCOME_EXIT = {
    Outcome.USE: EXIT_USE,
    Outcome.USE_WITH_WARN: EXIT_USE_WITH_WARN,
    Outcome.BUILD: EXIT_BUILD,
    Outcome.FAIL: EXIT_FAIL,
}

# The stable UPPERCASE outcome name carried in JSON — data, not a glyph.
_OUTCOME_NAME = {
    Outcome.USE: "USE",
    Outcome.USE_WITH_WARN: "USE_WITH_WARN",
    Outcome.BUILD: "BUILD",
    Outcome.FAIL: "FAIL",
}


def outcome_name(outcome: Outcome) -> str:
    return _OUTCOME_NAME[outcome]


def exit_code_for_outcome_table() -> dict:
    """The outcome→code map as a plain dict — the versioned contract, for tests and introspection."""
    return dict(_OUTCOME_EXIT)


def exit_code_for(decision: Decision) -> int:
    """The documented exit code for a resolve `Decision`.

    `run` layers VERIFY_MISMATCH / NOT_RUNNABLE on top of this (a run that resolves USE but fails the
    cosine gate exits 30, not 0); this is the resolve-outcome base.
    """
    return _OUTCOME_EXIT[decision.outcome]


# --- projections (pure: dict out, no logic fork) ----------------------------------------------
def decision_to_dict(model: str, target, d: Decision) -> dict:
    """Project a resolve `Decision` to the result document (FR-J02).

    Keys: model, target, outcome, [axis], variant, arch, quant, shape, warnings[]. `reason` is carried
    too (the same human string the text renderer prints) — additive and handy, but the branchable signal
    is `outcome`/`axis`, which are data, never parsed prose.
    """
    v: Optional[Variant] = d.variant
    rec = {
        "model": model,
        "target": str(target),
        "outcome": outcome_name(d.outcome),
        "variant": f"{v.model}-{v.version}" if v else None,
        "arch": v.arch if v else None,
        "quant": v.quant if v else None,
        "shape": v.shape if v else None,
        "warnings": [d.warning] if d.warning else [],
        "reason": d.reason or None,
    }
    if d.axis:
        rec["axis"] = d.axis           # present only when an axis could not be satisfied
    return rec


def run_to_dict(model: str, target, d: Decision, report: dict) -> dict:
    """Project a `run` to its result document: the resolve fields plus the verify record (FR-J02).

    Adds `verified` (bool), `cosine_min`, `threshold`, and `outputs{name:value}` so a deploy can gate on
    `.verified == true`. Defensive over the report shape: the LLM generative gate reports a `tripwire`
    instead of a cosine, which is passed through verbatim when present.
    """
    rec = decision_to_dict(model, target, d)
    rec["verified"] = bool(report.get("pass"))
    rec["cosine_min"] = report.get("cosine_min")
    rec["threshold"] = report.get("threshold")
    rec["outputs"] = dict(report.get("per_output") or {})
    if "tripwire" in report:           # LLM path: a generative tripwire, not a cosine
        rec["tripwire"] = report["tripwire"]
    if report.get("runtime") == "llamacpp-htp":   # runtime path: gate verdict (live/replay) or record+recipe
        rec["runtime"] = report["runtime"]
        if "gate" in report:                       # the generative gate actually ran
            rec["gate"] = report["gate"]
            rec["dispatch"] = report.get("dispatch")
            rec["mode"] = "live" if report.get("live") else ("replay" if report.get("replay") else None)
        else:                                      # no board: surfaced the record + recipe
            rec["record"] = report.get("record")
            rec["recipe"] = report.get("recipe")
            rec["cli_execution"] = report.get("cli_execution")
            if report.get("warn"):
                rec.setdefault("warnings", []).append(report["warn"])
    return rec


def list_to_dict(reg, target=None) -> List[dict]:
    """Project the catalogue to an array of per-variant records (FR-J02).

    Each record carries the variant's axes and its verification record (the raw dict, already
    JSON-serializable). When a target is in hand the per-model resolve outcome (and failing axis) is
    attached — resolve is the SAME function the human `list` calls, no logic fork.
    """
    out: List[dict] = []
    for m in reg.models():
        vs = reg.variants(m)
        decision = resolve(target, vs) if target else None
        for v in vs:
            rec = {
                "model": m,
                "version": v.version,
                "arch": v.arch,
                "quant": v.quant,
                "shape": v.shape,
                "has_artifact": v.has_artifact,
                "has_recipe": v.has_recipe,
                "verified": v.verified,
            }
            if decision is not None:
                rec["outcome"] = outcome_name(decision.outcome)
                if decision.axis:
                    rec["axis"] = decision.axis
            out.append(rec)
    return out
