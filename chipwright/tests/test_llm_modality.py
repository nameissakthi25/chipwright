"""LLM modality pure helpers — chat-template wrap, profile-JSON parse, generation parse, mock run.

No device and no onnxruntime: the device is a tiny fake exposing push_file/shell/pull/exists.
Runnable as `python3 test_llm_modality.py`.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chipwright.modalities.llm import (
    apply_chat_template, parse_profile, parse_generation, run,
    retokenize, load_bundle_tokenizer, greedy_sampler, apply_greedy_to_config,
)


# --- chat template --------------------------------------------------------------------------------

def test_chatml_wrap_has_im_markers_and_open_assistant_turn():
    out = apply_chat_template("What is the capital of France?", template="chatml")
    assert "<|im_start|>user\nWhat is the capital of France?<|im_end|>" in out
    assert out.endswith("<|im_start|>assistant\n")      # open turn so generation continues


def test_chatml_wrap_includes_system_when_given():
    out = apply_chat_template("hi", template="chatml", system="You are terse.")
    assert "<|im_start|>system\nYou are terse.<|im_end|>" in out


def test_raw_template_is_passthrough():
    assert apply_chat_template("just text", template="raw") == "just text"


def test_unknown_template_is_refused_by_name():
    try:
        apply_chat_template("x", template="llama-chat")
    except ValueError as e:
        assert "llama-chat" in str(e)
    else:
        assert False, "expected ValueError for unknown template"


# --- profile JSON parse ---------------------------------------------------------------------------

def test_parse_profile_maps_latency_fields():
    prof = json.dumps({
        "token_generation_rate": 33.0,
        "prompt_processing_rate": 292.0,
        "time_to_first_token_ms": 55.0,
        "init_time": 0.88,
    })
    lat = parse_profile(prof)
    assert lat["decode_toks_per_sec"] == 33.0
    assert lat["prefill_toks_per_sec"] == 292.0
    assert lat["time_to_first_token_ms"] == 55.0
    assert lat["model_load_s"] == 0.88


def test_parse_profile_handles_nested_and_missing_fields():
    prof = {"profile": {"decode": {"rate": 30.0}}}   # nested; others absent
    lat = parse_profile(prof)
    assert lat["decode_toks_per_sec"] == 30.0
    assert lat["prefill_toks_per_sec"] is None       # absent -> shown as None, not hidden


# --- generation parse -----------------------------------------------------------------------------

def test_parse_generation_extracts_text_and_token_ids():
    stdout = "loading...\n[BEGIN]Paris.[END]\ntoken_ids: [40, 1138, 13]\ndone"
    g = parse_generation(stdout)
    assert g["text"] == "Paris."
    assert g["tokens"] == [40, 1138, 13]


def test_parse_generation_missing_tokens_is_none():
    g = parse_generation("[BEGIN]hello[END]")
    assert g["text"] == "hello"
    assert g["tokens"] is None


def test_parse_generation_without_end_marker_reads_to_eof():
    # w4a16 degenerates and never emits EOS; the board kills the run on `timeout` before [END],
    # so the text must be read up to EOF, not dropped as empty.
    g = parse_generation("noise\n[BEGIN]: Paris is the capital of France and it keeps going")
    assert g["text"] == "Paris is the capital of France and it keeps going"


# --- retokenize (text -> token ids; genie emits no ids on QAIRT 2.51) -----------------------------

class _Enc:
    def __init__(self, ids): self.ids = ids


class FakeTokenizer:
    """Duck-typed stand-in for a HuggingFace Tokenizer: encode(text, add_special_tokens).ids."""
    def __init__(self, mapping=None): self.mapping = mapping or {}; self.calls = []
    def encode(self, text, add_special_tokens=False):
        self.calls.append((text, add_special_tokens))
        return _Enc(self.mapping.get(text, [len(text)]))


def test_retokenize_uses_injected_tokenizer_with_special_tokens_off():
    tok = FakeTokenizer({"Paris.": [40, 1138, 13]})
    ids = retokenize("Paris.", tokenizer=tok)
    assert ids == [40, 1138, 13]
    assert tok.calls == [("Paris.", False)]        # special tokens OFF — we match answer tokens


def test_retokenize_with_real_tokenizers_library_roundtrips():
    # Build a tiny real tokenizer.json via the `tokenizers` library and prove the file path loads.
    pytest.importorskip("tokenizers")  # needs the `llm` extra; skip (not fail) on a minimal install
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    import tempfile
    vocab = {"paris": 0, "london": 1, "is": 2, "the": 3, "capital": 4, "[UNK]": 5}
    t = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    t.pre_tokenizer = Whitespace()
    p = os.path.join(tempfile.mkdtemp(), "tokenizer.json")
    t.save(p)
    ids = retokenize("paris is the capital", p)
    assert ids == [0, 2, 3, 4]


def test_load_bundle_tokenizer_missing_file_fails_by_name():
    pytest.importorskip("tokenizers")  # without the lib, load raises the extra-missing RuntimeError first
    try:
        load_bundle_tokenizer("/no/such/tokenizer.json")
    except FileNotFoundError as e:
        assert "tokenizer" in str(e).lower()
    else:
        assert False, "expected FileNotFoundError for a missing bundle tokenizer"


def test_load_bundle_tokenizer_fails_by_name_without_library(monkeypatch=None):
    # If `tokenizers` cannot be imported, the modality must fail BY NAME, not silently return no tokens.
    import builtins
    real_import = builtins.__import__

    def blocked(name, *a, **k):
        if name == "tokenizers" or name.startswith("tokenizers."):
            raise ImportError("blocked for test")
        return real_import(name, *a, **k)

    builtins.__import__ = blocked
    try:
        load_bundle_tokenizer("/tmp/whatever.json")
        assert False, "expected RuntimeError naming the tokenizers library"
    except RuntimeError as e:
        assert "tokenizers" in str(e)
    finally:
        builtins.__import__ = real_import


# --- greedy sampler: top-k:1, NOT sampler.greedy (genie silently ignores greedy) ------------------

def test_greedy_sampler_sets_topk_1_not_relying_on_greedy_flag():
    s = greedy_sampler()
    assert s["top-k"] == 1          # the control genie HONORS
    assert s["temp"] == 0.0
    assert s["top-p"] == 1.0


def test_apply_greedy_to_config_forces_topk1_without_mutating_input():
    cfg = {"dialog": {"sampler": {"version": 1, "greedy": True, "temp": 0.8, "top-k": 40}}}
    out = apply_greedy_to_config(cfg)
    assert out["dialog"]["sampler"]["top-k"] == 1
    assert out["dialog"]["sampler"]["temp"] == 0.0
    # input is not mutated (pure) — the on-disk bundle config is never clobbered
    assert cfg["dialog"]["sampler"]["top-k"] == 40


# --- mock-device run ------------------------------------------------------------------------------

class _Proc:
    def __init__(self, stdout): self.returncode = 0; self.stdout = stdout


class FakeDevice:
    """Minimal device fake over the (push_file/shell/pull/exists) interface."""
    def __init__(self, stdout, profile=None):
        self._stdout = stdout
        self._profile = profile
        self.pushed = []
        self.shelled = []
        self._present = set()
    def push_file(self, local, remote):
        self.pushed.append((os.path.basename(local), remote))
        self._present.add(remote)
    def shell(self, cmd):
        self.shelled.append(cmd)
        return _Proc(self._stdout)
    def exists(self, remote):
        return remote.endswith("profile.json") and self._profile is not None or remote in self._present
    def pull(self, remote, local):
        with open(local, "w") as f:
            f.write(self._profile or "{}")


def test_run_pushes_bundle_file_by_file_and_parses(tmp_path=None):
    import tempfile
    d = tempfile.mkdtemp()
    # genie_config.json must be valid JSON so the greedy patch (top-k:1) can read it.
    with open(os.path.join(d, "genie_config.json"), "w") as f:
        json.dump({"dialog": {"sampler": {"version": 1, "greedy": True, "temp": 0.8}}}, f)
    for name in ("tokenizer.json", "context_0.bin"):
        with open(os.path.join(d, name), "w") as f:
            f.write("x")
    stdout = "loading...\n[BEGIN]Paris.[END]"            # genie emits TEXT, no token ids on QAIRT 2.51
    profile = json.dumps({"token_generation_rate": 33.0, "prompt_processing_rate": 292.0})
    dev = FakeDevice(stdout, profile=profile)
    meta = {"template": "chatml", "genie_config": "genie_config.json",
            "bundle_files": ["genie_config.json", "tokenizer.json", "context_0.bin"]}
    tok = FakeTokenizer({"Paris.": [40, 1138]})         # inject: no real tokenizer.json needed

    out = run(dev, d, meta, "capital of France?", tokenizer=tok, profile=True)  # exercise the profile-on path

    pushed = [remote for _, remote in dev.pushed]
    # bundle pushed one file at a time (FR-L02: never a single multi-GB directory push)
    assert sum(1 for p in pushed if p.endswith(("genie_config.json", "tokenizer.json", "context_0.bin"))) == 3
    # prompt went over as a FILE (--prompt_file), not inline on the shell command line
    assert any(p.endswith("prompt.txt") for p in pushed)
    with open(os.path.join(d, "prompt.txt")) as f:
        assert "im_start" in f.read()                   # chat template applied, written to the prompt file
    cmd = " ".join(dev.shelled)
    assert "--prompt_file" in cmd
    assert "--print-token-ids" not in cmd and "--max-length" not in cmd   # Bug 1: non-existent flags gone
    # greedy run drives genie at a derived top-k:1 config (Bug 2), not sampler.greedy
    assert any(p.endswith("genie_config.greedy.json") for p in pushed)
    assert "-c genie_config.greedy.json" in cmd
    with open(os.path.join(d, "genie_config.greedy.json")) as f:
        assert json.load(f)["dialog"]["sampler"]["top-k"] == 1
    # text parsed, then RETOKENIZED into ids for the gate
    assert out["text"] == "Paris."
    assert out["tokens"] == [40, 1138]
    assert out["latency"]["decode_toks_per_sec"] == 33.0


def test_run_omits_profile_by_default():
    """`--profile` defaults OFF: on a real SM8850/v81 board it broke genie 2.51's output and
    false-FAILed a known-good artifact (token_match 0.0). Default run must not pass the flag."""
    import tempfile
    d = tempfile.mkdtemp()
    for name in ("tokenizer.json", "genie_config.json"):
        with open(os.path.join(d, name), "w") as f:
            f.write("{}" if name.endswith(".json") else "x")
    dev = FakeDevice("[BEGIN]Paris.[END]")
    meta = {"template": "chatml", "genie_config": "genie_config.json",
            "bundle_files": ["genie_config.json", "tokenizer.json"]}
    out = run(dev, d, meta, "capital of France?", tokenizer=FakeTokenizer({"Paris.": [40]}))
    cmd = " ".join(dev.shelled)
    assert "--profile" not in cmd                 # default off (silicon finding)
    assert out["latency"] == {}                   # no profile pulled/parsed when off


if __name__ == "__main__":
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for f in fns:
        f(); print("ok", f.__name__)
    print(f"\n{len(fns)} passed")
