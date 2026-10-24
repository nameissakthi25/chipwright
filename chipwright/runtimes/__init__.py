"""Runtime adapters — the seam the runtime axis resolves onto (spec 004, T-RT03).

The resolver (`resolver.py`) picks a *runtime* ({genie, geniex-qairt, llamacpp-htp}) as one axis of a
Decision; an adapter is the thing that then actually stages and runs a variant on that runtime and proves
it ran. The resolver stays pure and offline — adapters are where the on-device, runtime-specific work lives
(Principle IV: one implementation; the CLI/app call the adapter, they never fork its logic).

Only the adapters that are *silicon-verified* are registered here. `llamacpp-htp` is registered because its
Q8_0-on-HTP route is proven on v73 (see `llamacpp_htp.SILICON_RECORD`). `genie` / `geniex-qairt` are named
by the resolver but have no adapter yet — `get_adapter` refuses them by name rather than pretending.
"""
from __future__ import annotations

from typing import Callable, Dict

from . import llamacpp_htp

# name -> the adapter module (each exposes dispatch_proof / host_preflight / bench_command / SILICON_RECORD)
_ADAPTERS: Dict[str, object] = {
    "llamacpp-htp": llamacpp_htp,
}


def get_adapter(runtime: str):
    """Return the adapter module for `runtime`, or raise with a named reason if none is wired yet.

    Refuse-by-name, not a KeyError traceback — the same discipline the resolver uses (Principle II).
    """
    adapter = _ADAPTERS.get(runtime)
    if adapter is None:
        known = sorted(_ADAPTERS)
        raise NotImplementedError(
            f"no silicon-verified adapter for runtime {runtime!r}; wired runtimes: {known}. "
            f"(the resolver may still *select* {runtime!r} offline, but running it needs an adapter)")
    return adapter


def wired_runtimes():
    """The runtimes that have a real, silicon-verified adapter."""
    return sorted(_ADAPTERS)
