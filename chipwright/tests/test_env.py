"""TargetKey — the pure parts of the board descriptor (no probe(), no board).

Covers manual() spec parsing, str() formatting, and the SOC_ID lookup table. probe() is deliberately
NOT exercised here; it needs a board over SSH/ADB. Also runnable as `python3 test_env.py`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chipwright.env import SOC_ID, TargetKey


def test_manual_parses_arch_and_qairt():
    t = TargetKey.manual("htpv68,qnn2.37")
    assert t.htp_arch == "v68"
    assert t.qairt == "2.37"
    # a hand-typed target is offline by construction
    assert t.reachable is False


def test_manual_tolerates_whitespace_and_order():
    t = TargetKey.manual(" qnn2.38.0 , htpv73 ")
    assert t.htp_arch == "v73"
    assert t.qairt == "2.38.0"


def test_manual_arch_only_leaves_qairt_none():
    t = TargetKey.manual("htpv66")
    assert t.htp_arch == "v66"
    assert t.qairt is None


def test_manual_without_arch_raises():
    for bad in ["qnn2.37", "", "soc498"]:
        try:
            TargetKey.manual(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


def test_str_formats_arch_sdk_and_soc():
    # arch only
    assert str(TargetKey(htp_arch="v68")) == "htpv68"
    # arch + qairt
    assert str(TargetKey(htp_arch="v68", qairt="2.37.1")) == "htpv68, qnn2.37.1"
    # arch + qairt + soc name
    t = TargetKey(htp_arch="v68", qairt="2.37.1", soc_name="QCS6490 / RB3 Gen2")
    assert str(t) == "htpv68, qnn2.37.1 [QCS6490 / RB3 Gen2]"


def test_soc_id_map_resolves_498_to_qcs6490_v68():
    name, arch = SOC_ID[498]
    assert "QCS6490" in name
    assert arch == "v68"


def test_soc_id_map_covers_known_boards():
    # the lookup returns a (name, arch) pair for each known SoC id
    for soc_id in (498, 35, 30, 57, 69):
        name, arch = SOC_ID[soc_id]
        assert isinstance(name, str) and name
        assert arch.startswith("v")
    # an unknown id yields the caller's default via .get()
    assert SOC_ID.get(9999, (None, "unknown")) == (None, "unknown")


if __name__ == "__main__":
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for f in fns:
        f(); print("ok", f.__name__)
    print(f"\n{len(fns)} passed")
