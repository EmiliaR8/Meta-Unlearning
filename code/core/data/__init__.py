"""Dataset registry.

Adding a corpus means writing one module with a `load(paths, **kw) -> Corpus`
function and adding one line here. Nothing downstream is dataset-aware: family
selection, scaling, task construction and evaluation all operate on the Corpus.
"""

from __future__ import annotations

from .base import (Corpus, apply_family_selection, cap_per_family,
                   scale_features, select_families)

# name -> (module path, default task setup, default scaling)
REGISTRY = {
    "ember2018":     ("core.data.ember2018", "ember"),
    "ember2024":     ("core.data.ember2024", "ember"),
    "lamda_classil": ("core.data.lamda",     "lamda"),
    "synthetic":     ("core.data.synthetic", "30+5x2"),
}


def available() -> list[str]:
    return sorted(REGISTRY)


def default_task_setup(name: str) -> str:
    _require(name)
    return REGISTRY[name][1]


def _require(name: str) -> None:
    if name not in REGISTRY:
        raise SystemExit(
            f"unknown dataset {name!r}. Available: {', '.join(available())}.")


def load_corpus(name: str, paths, **kw) -> Corpus:
    _require(name)
    import importlib

    module = importlib.import_module(REGISTRY[name][0])
    return module.load(paths, **kw)


__all__ = ["Corpus", "REGISTRY", "available", "default_task_setup", "load_corpus",
           "apply_family_selection", "cap_per_family", "scale_features",
           "select_families"]
