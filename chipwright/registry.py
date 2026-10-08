"""Registry client — read the index (a git-repo file, cached) and list a model's variants.

Per the brief: start the registry as a git repository, not a service. The index is a single YAML
file the client reads; artifacts live in object storage addressed by hash. No server to run, and the
whole index is reviewable in pull requests — which is what makes a verification record trustworthy.
"""
from __future__ import annotations

import os
import urllib.request
from typing import Dict, List

import yaml

from .resolver import Variant

# Default index: the local registry/ file in this repo (becomes a git URL once published).
DEFAULT_INDEX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "registry", "index.yaml")


def _load_yaml(src: str) -> dict:
    if src.startswith(("http://", "https://")):
        with urllib.request.urlopen(src, timeout=30) as r:
            return yaml.safe_load(r.read())
    with open(src) as f:
        return yaml.safe_load(f)


class Registry:
    def __init__(self, index: str = None):
        self.index_src = index or os.environ.get("CW_REGISTRY", DEFAULT_INDEX)
        self._models: Dict[str, List[Variant]] = {}
        self._schema = 0
        self._load()

    def _load(self):
        doc = _load_yaml(self.index_src) or {}
        self._schema = int(doc.get("registry_schema", 1))
        for entry in doc.get("models", []):
            model = entry["model"]
            vs = []
            for v in entry.get("variants", []):
                vs.append(Variant(
                    model=model,
                    version=str(v.get("version", entry.get("version", "0.0.0"))),
                    arch=v["arch"], quant=v["quant"], shape=v["shape"],
                    sdk_tested=str(v["sdk_tested"]), sdk_compatible=str(v["sdk_compatible"]),
                    ext=v.get("ext", "qnnctx"),
                    url=v.get("url"), sha256=v.get("sha256"),
                    local=v.get("local"), run=v.get("run"),
                    recipe=v.get("recipe"), verified=v.get("verified"),
                ))
            self._models[model] = vs

    @property
    def schema(self) -> int:
        return self._schema

    def models(self) -> List[str]:
        return sorted(self._models)

    def variants(self, model: str) -> List[Variant]:
        return self._models.get(model, [])
