"""LLM modality pure helpers — chat-template wrap, profile-JSON parse, generation parse, mock run.

No device and no onnxruntime: the device is a tiny fake exposing push_file/shell/pull/exists.
Runnable as `python3 test_llm_modality.py`.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chipwright.modalities.llm import (
    apply_chat_template, parse_profile, parse_generation, run,
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
    for name in ("genie_config.json", "tokenizer.json", "context_0.bin"):
        with open(os.path.join(d, name), "w") as f:
            f.write("x")
    stdout = "[BEGIN]Paris.[END]\ntoken_ids: [40, 1138]"
    profile = json.dumps({"token_generation_rate": 33.0, "prompt_processing_rate": 292.0})
    dev = FakeDevice(stdout, profile=profile)
    meta = {"template": "chatml", "genie_config": "genie_config.json",
            "bundle_files": ["genie_config.json", "tokenizer.json", "context_0.bin"]}

    out = run(dev, d, meta, "capital of France?")

    # bundle pushed one file at a time (FR-L02: never a single multi-GB directory push)
    assert len(dev.pushed) == 3
    # chat template was applied before the genie call
    assert any("im_start" in c for c in dev.shelled)
    assert out["text"] == "Paris."
    assert out["tokens"] == [40, 1138]
    assert out["latency"]["decode_toks_per_sec"] == 33.0


if __name__ == "__main__":
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for f in fns:
        f(); print("ok", f.__name__)
    print(f"\n{len(fns)} passed")
