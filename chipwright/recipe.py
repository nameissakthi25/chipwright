"""Recipe — the reproducible build input, loaded from a YAML document.

A recipe is the *only* input to a build: the same recipe plus the same SDK produces the same
artifact, which is what makes the resulting verification record trustworthy (Constitution III). It is
a pure dataclass — `load(path)` reads and validates the YAML and nothing else; both build routes
(`build.classic`, `build.dlc_route`) read the same schema, only the orchestration differs.

See `specs/001-chipwright-core/data-model.md` (Recipe ✅) for the authored schema.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

ROUTES = ("classic", "dlc_route")
RUNS_ON = ("npu", "host")


@dataclass
class Source:
    """The float model this recipe converts — content-addressed so a build is pinned to one graph."""
    format: str = "onnx"
    sha256: Optional[str] = None
    url: Optional[str] = None


@dataclass
class Target:
    """The single target key this recipe builds for (the axes an artifact tag carries)."""
    arch: str = ""
    quant: str = ""
    shape: str = ""
    sdk_range: str = ""


@dataclass
class Adaptation:
    """One declared, auditable graph edit. `kind` is the YAML key; `params` is its value."""
    kind: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Quant:
    """How the DLC is quantized — maps to qairt-quantizer flags (act/bias bitwidth, per-channel)."""
    weights: int = 8
    activations: int = 16
    bias: int = 32
    per_channel: bool = True
    calibration_ref: Optional[str] = None
    algorithm: str = "min_max"


@dataclass
class Host:
    """The modality adapter — the CPU stages around the NPU graph(s). `pre`/`post` are 'module:callable'."""
    pre: Optional[str] = None
    post: Optional[str] = None
    deps: List[str] = field(default_factory=list)


@dataclass
class Graph:
    """One ordered, typed stage. `runs_on` ∈ {npu, host}; list order is execution order."""
    name: str
    runs_on: str = "npu"


@dataclass
class Recipe:
    name: str
    version: str
    route: str
    source: Source
    target: Target
    adaptations: List[Adaptation] = field(default_factory=list)
    quant: Quant = field(default_factory=Quant)
    host: Host = field(default_factory=Host)
    graphs: List[Graph] = field(default_factory=list)

    @property
    def quant_tag(self) -> str:
        """The 'w8a16'-style tag this recipe's quant block realises."""
        return f"w{self.quant.weights}a{self.quant.activations}"

    def adaptation(self, kind: str) -> Optional[Adaptation]:
        """The declared adaptation of a given kind, or None (e.g. recipe.adaptation('erf_to_tanh'))."""
        return next((a for a in self.adaptations if a.kind == kind), None)

    @property
    def npu_graphs(self) -> List[Graph]:
        return [g for g in self.graphs if g.runs_on == "npu"]


def _parse_adaptations(items: Any) -> List[Adaptation]:
    """`adaptations:` is a list of single-key maps ({erf_to_tanh: {...}}) — keep key as kind, value as params."""
    out: List[Adaptation] = []
    for item in items or []:
        if isinstance(item, dict):
            for kind, params in item.items():
                out.append(Adaptation(kind=str(kind), params=params if isinstance(params, dict) else {"value": params}))
        else:
            out.append(Adaptation(kind=str(item), params={}))
    return out


def from_dict(doc: Dict[str, Any]) -> Recipe:
    """Build a Recipe from an already-parsed YAML mapping. Pure — no I/O. Raises ValueError if malformed."""
    if not isinstance(doc, dict):
        raise ValueError(f"recipe must be a YAML mapping, got {type(doc).__name__}")
    for req in ("name", "version", "source", "target"):
        if req not in doc:
            raise ValueError(f"recipe missing required field {req!r}")

    route = str(doc.get("route", "classic"))
    if route not in ROUTES:
        raise ValueError(f"unknown route {route!r}; expected one of {ROUTES}")

    src = doc["source"] or {}
    source = Source(format=str(src.get("format", "onnx")), sha256=src.get("sha256"), url=src.get("url"))
    if source.format != "onnx":
        raise ValueError(f"source.format must be 'onnx' (the only supported source); got {source.format!r}")

    tgt = doc["target"] or {}
    target = Target(arch=str(tgt.get("arch", "")), quant=str(tgt.get("quant", "")),
                    shape=str(tgt.get("shape", "")), sdk_range=str(tgt.get("sdk_range", "")))
    if not target.arch:
        raise ValueError("target.arch is required (the hard axis, e.g. 'v68')")

    q = doc.get("quant") or {}
    quant = Quant(weights=int(q.get("weights", 8)), activations=int(q.get("activations", 16)),
                  bias=int(q.get("bias", 32)), per_channel=bool(q.get("per_channel", True)),
                  calibration_ref=q.get("calibration_ref"), algorithm=str(q.get("algorithm", "min_max")))

    h = doc.get("host") or {}
    host = Host(pre=h.get("pre"), post=h.get("post"), deps=list(h.get("deps", []) or []))

    graphs: List[Graph] = []
    for g in doc.get("graphs", []) or []:
        name = str(g.get("name", "")) if isinstance(g, dict) else str(g)
        runs_on = str(g.get("runs_on", "npu")) if isinstance(g, dict) else "npu"
        if runs_on not in RUNS_ON:
            raise ValueError(f"graph {name!r} has runs_on={runs_on!r}; expected one of {RUNS_ON}")
        graphs.append(Graph(name=name, runs_on=runs_on))

    return Recipe(name=str(doc["name"]), version=str(doc["version"]), route=route,
                  source=source, target=target, adaptations=_parse_adaptations(doc.get("adaptations")),
                  quant=quant, host=host, graphs=graphs)


def load(path: str) -> Recipe:
    """Load and validate a recipe YAML file into a `Recipe`. Raises ValueError on a malformed recipe."""
    with open(path) as f:
        doc = yaml.safe_load(f)
    return from_dict(doc)
