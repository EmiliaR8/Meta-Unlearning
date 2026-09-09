"""Forget-set selectors: which samples the unlearning step acts on.

Every selector receives the same candidate pool and the same budget (a fixed
fraction `ratio` of the pool) and returns a partition into forget and retain.
Holding the budget fixed is what makes a comparison attributable to WHICH
samples are named rather than to how many.

  random  -- control. Uniform sample at the same budget.
  donut   -- fit an Isolation Forest jointly over the task in latent space and
             forget the MIDDLE band of the score distribution, keeping both the
             most anomalous and the most prototypical. This is the prior MADAR
             uses to CHOOSE buffer contents, applied to deletion.
  leftover-- per-family Isolation Forest in RAW feature space; forget everything
             a MADAR-style selection would not have kept. Note this ignores
             `ratio`: the forget set is whatever the buffer budget leaves over,
             which is typically far larger than a 10% slab.

`register` exists so a new selector is one decorated function, not an edit in
three places.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.ensemble import IsolationForest

_REGISTRY: dict[str, Callable] = {}


def register(name: str):
    def wrap(fn):
        _REGISTRY[name] = fn
        return fn
    return wrap


def available() -> list[str]:
    return sorted(_REGISTRY)


def split(name: str, **kw):
    """-> (forget_idx, retain_idx), indices into the candidate pool."""
    if name not in _REGISTRY:
        raise SystemExit(
            f"unknown selector {name!r}. Available: {', '.join(available())}")
    return _REGISTRY[name](**kw)


def _budget(n: int, ratio: float) -> int:
    return int(n * float(ratio))


@register("random")
def _random(*, n, ratio, seed, **_):
    k = _budget(n, ratio)
    if k == 0:
        return np.zeros(0, np.int64), np.arange(n)
    perm = np.random.default_rng(seed).permutation(n)
    return perm[:k], perm[k:]


@register("donut")
def _donut(*, n, ratio, latents, seed, contamination=0.1, **_):
    k = _budget(n, ratio)
    if k == 0 or n < 2:
        return np.zeros(0, np.int64), np.arange(n)
    iso = IsolationForest(contamination=contamination, n_jobs=-1,
                          random_state=seed).fit(latents)
    order = np.argsort(iso.decision_function(latents))
    start = (n // 2) - (k // 2)
    forget = order[start:start + k]
    retain = np.concatenate([order[:start], order[start + k:]])
    return forget, retain


@register("leftover")
def _leftover(*, n, features, labels, budget_per_family, seed,
              contamination=0.1, **_):
    forget, retain = [], []
    for fam in np.unique(labels):
        idx = np.flatnonzero(labels == fam)
        n_keep = min(int(budget_per_family), len(idx))
        if n_keep >= len(idx) or len(idx) < 2:
            retain.extend(idx)
            continue
        iso = IsolationForest(contamination=contamination, n_jobs=-1,
                              random_state=seed).fit(features[idx])
        order = np.argsort(iso.decision_function(features[idx]))
        half = n_keep // 2
        keep_local = np.concatenate([order[:half], order[-(n_keep - half):]])
        keep = set(idx[keep_local].tolist())
        retain.extend(sorted(keep))
        forget.extend([i for i in idx.tolist() if i not in keep])
    return np.asarray(forget, np.int64), np.asarray(retain, np.int64)
