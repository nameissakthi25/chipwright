"""chipwright agent-gate demo — catch a quantized agent that silently stopped calling tools correctly.

Run (works from a pip install):  python -m chipwright.demos.agent_gate

The story (spec 007): a tool-calling SLM is quantized for a Qualcomm NPU. Its perplexity looks fine and its
aggregate benchmark score barely moves — but arXiv 2607.27275 measured INT4 DOUBLING tool-name hallucination
(19.5% -> 38.3%) at a flat task score. That is the fast-wrong-answer an agent commits, invisible to PPL/MMLU.

This runs chipwright's tool-call fidelity gate over the SAME probe set for two "chips":
  - a GOOD build   (calls match the CPU reference, abstains when it should)        -> VERIFIED
  - a BROKEN build (simulated quant damage: hallucinated names, malformed args,
    and a spurious tool call where it should have abstained)                        -> FAIL, by name

The gate compares the chip's calls to the SAME artifact's CPU-reference calls (backend isolation), so a
divergence means the chip computed the agent worse — not that the quant choice was lossy. It parses calls
from TEXT, so it needs no logits and runs even on classic Genie.
"""
from __future__ import annotations

import os

import yaml

from .. import verify_llm

G, R, Y, DIM, RST = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

_PROBES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tool_prompts.yaml")


def load_probes():
    with open(_PROBES) as f:
        return yaml.safe_load(f)["prompts"]


def good_chip(probes):
    """A faithful build: emits exactly the reference call on relevant prompts, abstains on irrelevant ones."""
    return [p["expected"] for p in probes]


def broken_chip(probes):
    """A quant-damaged build mirroring the real failure modes (arXiv 2607.27275 + malformed JSON + lost
    abstention). Deterministic, so the demo is reproducible."""
    out, rel_i = [], 0
    for p in probes:
        if p["relevant"]:
            exp = dict(p["expected"])
            if rel_i % 2 == 0:                         # ~half the tool calls go wrong (doubled hallucination)
                if rel_i % 4 == 0:
                    exp = {"name": exp["name"] + "_v2", "arguments": exp["arguments"]}  # hallucinated tool name
                else:
                    exp = '{"name": "' + exp["name"] + '", "arguments": {bad json'       # malformed JSON
            out.append(exp)
            rel_i += 1
        else:
            out.append({"name": "search_web", "arguments": {"query": "?"}} if len(out) % 2 == 0 else None)
    return out


def _show(title, probes, chip):
    ref = [p["expected"] for p in probes]
    rel = [p["relevant"] for p in probes]
    rep = verify_llm.tool_call_report(chip, ref, rel)
    mark = f"{G}✔ VERIFIED{RST}" if rep["pass"] else f"{R}✗ FAIL  axis=tool_call{RST}"
    thr = rep["thresholds"]
    print(f"\n{title}\n  {mark}")
    print(f"    tool_call_retention = {rep['tool_call_retention']:.2f}  (gate ≥ {thr['tool_call_retention']})"
          f"   [{rep['reference_answered']} reference calls]")
    print(f"    abstention_rate     = {rep['abstention_rate']:.2f}  (gate ≥ {thr['abstention_rate']})"
          f"   [{rep['irrelevant_prompts']} irrelevant prompts]")
    print(f"    schema_validity     = {rep['schema_validity']:.2f}  {DIM}(recorded, not gated){RST}")
    return rep


def main():
    probes = load_probes()
    print(f"{DIM}chipwright agent gate — tool-call fidelity (spec 007). "
          f"{len(probes)} prompts; backend-isolation vs the CPU reference.{RST}")
    good = _show("GOOD build (faithful on-chip agent):", probes, good_chip(probes))
    bad = _show("BROKEN build (quant-damaged — hallucinated names, malformed args, lost abstention):",
                probes, broken_chip(probes))
    print()
    if not (good["pass"] and not bad["pass"]):
        raise SystemExit("demo invariant broken: good must PASS and broken must FAIL")
    print(f"{G}The gate did its job:{RST} the broken agent kept a passing perplexity but its tool calls")
    print("silently diverged from the CPU reference — chipwright caught it on the bench and refused by name.")
    print(f"{DIM}Backend-faithfulness, not an absolute safety claim. Thresholds are house rules.{RST}")


if __name__ == "__main__":
    main()
