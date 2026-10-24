"""The llama.cpp-HTP adapter: the dispatch gate + host preflight (spec 004 T-RT07 / SC-RT03).

Layer 1 (these tests) proves the LOGIC: the dispatch gate must separate a real HTP0 dispatch from a silent
CPU fallback, and preflight must refuse-by-name. The fixtures below are the ACTUAL stdout captured on
silicon (QCS9075 / v73, 2026-10-24) — the HTP run and the `GGML_HEXAGON_NDEV=0` CPU run of the same Q8_0
model. Layer 2 (the record the fixtures come from) is `llamacpp_htp.SILICON_RECORD`. A gate that can't FAIL
the CPU-fallback case isn't a gate — so that negative is the central assertion, mirroring the CV gate on
random noise.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chipwright import runtimes
from chipwright.runtimes import llamacpp_htp as lh

# --- real captured stdout (QCS9075 / v73, 2026-10-24) ------------------------------------------------
HTP_STDOUT = """\
ggml-hex: Loading driver libcdsprpc.so
ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev 1
ggml-hex: Hexagon Arch version v73
ggml-hex: HTP0 allocating new session
ggml-hex: HTP0 hwinfo: threads 4, hvx 4, hmx 1, vtcm 8 MB
ggml-hex: HTP0 new session : session-id 0 domain-id 3 uri file:///libggml-htp-v73.so?htp_iface_skel_handle_invoke&_modver=1.0&_dom=cdsp&_session=0 handle 0x556115f200
ggml-hex: HTP0 op batching: n-bufs 16 n-tensors 7168 n-ops 1024 vmem 3145728000
| model                          |       size |     params | backend    | ngl | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------ | --------------: | -------------------: |
| qwen2 1.5B Q8_0                |   1.53 GiB |     1.54 B | HTP        |  -1 | HTP0         |            pp64 |       592.82 ± 20.43 |
| qwen2 1.5B Q8_0                |   1.53 GiB |     1.54 B | HTP        |  -1 | HTP0         |            tg32 |         23.51 ± 0.04 |

build: 3278e921b (10049)
"""

CPU_STDOUT = """\
ggml-hex: Loading driver libcdsprpc.so
ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev 0
ggml-hex: Hexagon Arch version v73
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen2 1.5B Q8_0                |   1.53 GiB |     1.54 B | HTP        |  -1 |            pp64 |        217.55 ± 0.02 |
| qwen2 1.5B Q8_0                |   1.53 GiB |     1.54 B | HTP        |  -1 |            tg32 |         28.99 ± 0.01 |

build: 3278e921b (10049)
"""


def test_dispatch_gate_confirms_real_htp_run():
    proof = lh.dispatch_proof(HTP_STDOUT)
    assert proof.dispatched is True
    assert proof.arch == "v73"
    assert proof.ndev == 1
    assert proof.on_cdsp is True
    # the pp/tg rows actually executed on device HTP0
    tests = {r["test"] for r in proof.device_rows}
    assert "pp64" in tests and "tg32" in tests


def test_dispatch_gate_FAILS_cpu_fallback():
    """The whole point: NDEV=0 ran on CPU. The gate must NOT call that an NPU dispatch."""
    proof = lh.dispatch_proof(CPU_STDOUT)
    assert proof.dispatched is False
    assert proof.ndev == 0
    assert "CPU" in proof.reason or "ndev 0" in proof.reason
    assert proof.device_rows == []          # no row ran on HTP0


def test_dispatch_gate_FAILS_session_without_device_row():
    """A session can open yet nothing dispatch — still not a pass."""
    stdout = ("ggml-hex: allocating new registry : ndev 1\n"
              "ggml-hex: Hexagon Arch version v73\n"
              "ggml-hex: HTP0 new session : uri file:///libggml-htp-v73.so?_dom=cdsp&_session=0\n"
              "| model | backend | test | t/s |\n| --- | --- | --- | --- |\n")
    proof = lh.dispatch_proof(stdout)
    assert proof.dispatched is False
    assert "no bench row" in proof.reason


def test_silicon_prefill_speedup_is_a_real_win():
    d = lh.SILICON_RECORD["dispatch"]
    assert d["htp0_prefill_tps"] > d["cpu_prefill_tps"]      # the Q8_0 route Genie lacks, faster on-chip
    assert round(d["htp0_prefill_tps"] / d["cpu_prefill_tps"], 2) == d["prefill_speedup"]


def test_quality_is_near_lossless():
    assert abs(lh.SILICON_RECORD["quality"]["ppl_wikitext_q8_0"] - 11.6631) < 1e-4


# --- host preflight: refuse-by-name -----------------------------------------------------------------
_OK_PROBE = {"has_fastrpc": True, "has_cdsprpc": True, "htp_arch": "v73",
             "staged_htp_libs": {"v73", "v75", "v79", "v81"},
             "staged_loader_libs": set(lh.REQUIRED_LOADER_LIBS)}


def test_preflight_passes_on_the_verified_board():
    ok, reason = lh.host_preflight(_OK_PROBE)
    assert ok is True and reason is None


def test_preflight_refuses_missing_fastrpc_by_name():
    probe = dict(_OK_PROBE, has_cdsprpc=False)
    ok, reason = lh.host_preflight(probe)
    assert ok is False and "libcdsprpc" in reason


def test_preflight_refuses_wrong_arch_lib_by_name():
    probe = dict(_OK_PROBE, htp_arch="v68", staged_htp_libs={"v73"})
    ok, reason = lh.host_preflight(probe)
    assert ok is False and "v68" in reason and "libggml-htp-v68.so" in reason


def test_preflight_refuses_missing_loader_by_name():
    probe = dict(_OK_PROBE, staged_loader_libs={"libc.so.6"})
    ok, reason = lh.host_preflight(probe)
    assert ok is False and "loader" in reason


# --- adapter registry seam (T-RT03) -----------------------------------------------------------------
def test_llamacpp_htp_adapter_is_wired():
    assert "llamacpp-htp" in runtimes.wired_runtimes()
    assert runtimes.get_adapter("llamacpp-htp") is lh


def test_unwired_runtime_refuses_by_name_not_keyerror():
    import pytest
    with pytest.raises(NotImplementedError) as ei:
        runtimes.get_adapter("genie")
    assert "genie" in str(ei.value)


def test_bench_command_matches_the_verified_recipe():
    argv = lh.bench_command("qwen2.5-1.5b-Q8_0.gguf",
                            "./pkg-snapdragon/lib", "./glibc238", p=64, n=32, r=2)
    joined = " ".join(argv)
    assert argv[0] == "./glibc238/ld-linux-aarch64.so.1"
    assert "--device HTP0" in joined
    assert "./pkg-snapdragon/bin/llama-bench" in joined
    assert "-m qwen2.5-1.5b-Q8_0.gguf" in joined


def test_bench_command_cpu_reference_drops_device_flag():
    argv = lh.bench_command("m.gguf", "./pkg-snapdragon/lib", "./glibc238", ndev=0)
    assert "--device" not in argv          # the CPU reference run (GGML_HEXAGON_NDEV=0) has no HTP device
