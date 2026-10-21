"""Generative-verify math — Layer-1 tripwire, the behavioral probe, and the pinned reference.

Pure numpy / hashlib, no board and no onnxruntime (the decoder is injected). Runnable as
`python3 test_verify_llm.py`.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chipwright.verify_llm import (
    token_match, top1_agreement, mean_kl, ppl_delta_pct, tripwire_report,
    refusal_retention, benign_answer_rate, behavior_report,
    TOKEN_MATCH_MIN, PPL_DELTA_MAX_PCT,
)
from chipwright.llm_reference import pin_reference, verify_pin_hash, greedy_decode


# --- 1a: token-match ------------------------------------------------------------------------------

def test_token_match_identical_is_one():
    toks = [10, 20, 30, 40]
    assert token_match(toks, toks) == 1.0


def test_token_match_divergent_is_low():
    assert token_match([1, 2, 3, 4], [9, 9, 9, 9]) == 0.0


def test_token_match_partial():
    # 2 of 4 match
    assert math.isclose(token_match([1, 2, 3, 4], [1, 2, 9, 9]), 0.5, rel_tol=1e-12)


def test_token_match_length_mismatch_counts_missing_as_misses():
    # a truncated/looping decode cannot score a false 1.0: denominator is the longer length
    assert math.isclose(token_match([1, 2], [1, 2, 3, 4]), 0.5, rel_tol=1e-12)


def test_token_match_empty_is_zero():
    assert token_match([], []) == 0.0


# --- 1b: top-1 agreement + mean KL ----------------------------------------------------------------

def test_top1_agreement_identical_logits():
    lg = np.array([[0.1, 5.0, 0.2], [3.0, 0.0, 0.1]])
    assert top1_agreement(lg, lg) == 1.0


def test_top1_agreement_divergent():
    a = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]])  # argmax 0, 1
    b = np.array([[0.0, 0.0, 5.0], [5.0, 0.0, 0.0]])  # argmax 2, 0
    assert top1_agreement(a, b) == 0.0


def test_mean_kl_identical_is_zero():
    lg = np.array([[0.1, 5.0, 0.2], [3.0, 0.0, 0.1]])
    assert math.isclose(mean_kl(lg, lg), 0.0, abs_tol=1e-12)


def test_mean_kl_is_nonnegative_and_positive_when_divergent():
    a = np.array([[5.0, 0.0, 0.0]])
    b = np.array([[0.0, 0.0, 5.0]])
    kl = mean_kl(a, b)
    assert kl >= 0.0
    assert kl > 0.1


def test_mean_kl_numerically_safe_on_large_logits():
    # no overflow at large magnitudes (float64 + max-subtract)
    a = np.array([[1000.0, 0.0, -1000.0]])
    b = np.array([[0.0, 1000.0, -1000.0]])
    kl = mean_kl(a, b)
    assert math.isfinite(kl) and kl >= 0.0


def test_single_step_logits_accepted_as_1d():
    assert top1_agreement([0.0, 9.0, 0.0], [0.0, 9.0, 0.0]) == 1.0
    assert math.isclose(mean_kl([0.0, 9.0, 0.0], [0.0, 9.0, 0.0]), 0.0, abs_tol=1e-12)


# --- 1c: PPL delta --------------------------------------------------------------------------------

def test_ppl_delta_positive_when_chip_worse():
    assert math.isclose(ppl_delta_pct(10.5, 10.0), 5.0, rel_tol=1e-9)


def test_ppl_delta_negative_when_chip_better():
    assert ppl_delta_pct(9.5, 10.0) < 0.0


def test_ppl_delta_guards_zero_reference():
    assert ppl_delta_pct(10.0, 0.0) == 0.0


# --- Layer-1 report -------------------------------------------------------------------------------

def test_tripwire_passes_on_good_artifact():
    lg = np.array([[0.1, 5.0, 0.2], [3.0, 0.0, 0.1]])
    r = tripwire_report([1, 2], [1, 2], lg, lg, npu_ppl=10.2, ref_ppl=10.0)
    assert r["pass"] is True
    assert r["token_match"] == 1.0
    assert r["top1_agreement"] == 1.0
    assert math.isclose(r["mean_kl_nats"], 0.0, abs_tol=1e-12)
    assert r["thresholds"] == {"token_match": TOKEN_MATCH_MIN, "ppl_delta_pct": PPL_DELTA_MAX_PCT}


def test_tripwire_fails_known_bad_artifact_on_token_match():
    # a broken artifact (wrong precision/config): tokens far off -> FAIL, not a false pass
    a = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    b = np.array([[0.0, 0.0, 5.0], [5.0, 0.0, 0.0]])
    r = tripwire_report([9, 9, 9, 9], [1, 2, 3, 4], a, b)
    assert r["pass"] is False
    assert r["token_match"] == 0.0


def test_tripwire_fails_on_ppl_delta_over_threshold():
    lg = np.array([[0.1, 5.0]])
    r = tripwire_report([1], [1], lg, lg, npu_ppl=11.0, ref_ppl=10.0)  # +10% > 5%
    assert r["ppl_delta_pct"] > PPL_DELTA_MAX_PCT
    assert r["pass"] is False


def test_tripwire_skips_ppl_when_not_measured():
    lg = np.array([[0.1, 5.0]])
    r = tripwire_report([1], [1], lg, lg)
    assert r["ppl_delta_pct"] is None
    assert r["pass"] is True  # rests on token-match


# --- Layer-2 behavioral probe ---------------------------------------------------------------------

def test_refusal_retention_full():
    ref = [True, True, True, True]
    chip = [True, True, True, True]
    assert refusal_retention(chip, ref) == 1.0


def test_refusal_retention_drop_detected():
    # reference refused 4; chip keeps 3 -> 0.75 (the arXiv 2606.09864 failure mode)
    ref = [True, True, True, True]
    chip = [True, True, True, False]
    assert math.isclose(refusal_retention(chip, ref), 0.75, rel_tol=1e-12)


def test_refusal_retention_denominator_is_reference_refusals():
    # prompts the reference did not refuse do not dilute the score
    ref = [True, True, False, False]
    chip = [True, False, False, False]  # of the 2 refused, keeps 1
    assert math.isclose(refusal_retention(chip, ref), 0.5, rel_tol=1e-12)


def test_refusal_retention_nothing_to_retain_is_one():
    assert refusal_retention([False, False], [False, False]) == 1.0


def test_benign_answer_rate():
    assert benign_answer_rate([True, True, True, False]) == 0.75
    assert benign_answer_rate([]) == 0.0


def test_behavior_report_flags_below_threshold_but_does_not_fail():
    ref = [True] * 10
    chip = [True] * 8 + [False, False]            # retention 0.8 < 0.95
    benign = [True] * 10
    b = behavior_report(chip, ref, benign)
    assert math.isclose(b["refusal_retention"], 0.8, rel_tol=1e-12)
    assert b["benign_answer_rate"] == 1.0
    assert b["flagged"] is True
    assert "pass" not in b                        # recorded, never a pass/fail gate (FR-L11)
    assert b["harmful_prompts"] == 10 and b["benign_prompts"] == 10


def test_behavior_report_not_flagged_when_retained():
    ref = [True] * 10
    b = behavior_report([True] * 10, ref, [True] * 10)
    assert b["flagged"] is False


# --- pinned reference (injected fake decoder) -----------------------------------------------------

def _fake_decode(ids, max_new_tokens=64, eos_id=None):
    """A deterministic fake decoder: tokens are the prompt ids + 1; logits one-hot on each token."""
    toks = [int(i) + 1 for i in ids][:max_new_tokens]
    vocab = 8
    lg = np.zeros((len(toks), vocab))
    for r, t in enumerate(toks):
        lg[r, t % vocab] = 10.0
    return toks, lg


def test_pin_reference_shape_and_hash():
    prompts = [[1, 2, 3], [4, 5]]
    pin = pin_reference(prompts, _fake_decode)
    assert pin["prompts"] == 2
    assert pin["decode"] == "greedy"
    assert pin["tokens"] == [[2, 3, 4], [5, 6]]
    assert len(pin["reference_logits_sha256"]) == 64
    assert verify_pin_hash(pin) is True


def test_pin_reference_hash_is_deterministic():
    prompts = [[1, 2, 3]]
    h1 = pin_reference(prompts, _fake_decode)["reference_logits_sha256"]
    h2 = pin_reference(prompts, _fake_decode)["reference_logits_sha256"]
    assert h1 == h2


def test_pin_reference_hash_changes_with_content():
    a = pin_reference([[1, 2, 3]], _fake_decode)["reference_logits_sha256"]
    b = pin_reference([[1, 2, 4]], _fake_decode)["reference_logits_sha256"]
    assert a != b


def test_verify_pin_hash_detects_tamper():
    pin = pin_reference([[1, 2, 3]], _fake_decode)
    pin["tokens"] = [[9, 9, 9]]                   # tamper after pinning
    assert verify_pin_hash(pin) is False


def test_greedy_decode_terminates_on_eos():
    # a fake session whose decode always points at token 0; eos_id=0 stops it after one step
    class FakeSession:
        def prefill(self, ids):
            lg = np.zeros(4); lg[2] = 5.0         # first token = 2
            return lg, None
        def decode(self, tok, kv):
            lg = np.zeros(4); lg[0] = 5.0         # next token = 0 (eos)
            return lg, None
    toks, logits = greedy_decode(FakeSession(), None, [1, 2], max_new_tokens=10, eos_id=0)
    assert toks[0] == 2
    assert toks[-1] == 0                          # terminated on eos
    assert len(toks) == 2


if __name__ == "__main__":
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for f in fns:
        f(); print("ok", f.__name__)
    print(f"\n{len(fns)} passed")
