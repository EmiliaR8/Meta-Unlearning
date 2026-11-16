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
  nn1     -- the learned mode-A scorer. Takes `scores` ALREADY COMPUTED by the
             caller and forgets the top `ratio` fraction. The scoring needs the
             model, a device and a saved .npz; keeping all of that on the
             caller's side is what lets this module stay free of torch, so a
             selector can be reasoned about and tested without a GPU.

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


@register("nn1")
def _nn1(*, n, ratio, scores, **_):
    k = _budget(n, ratio)
    if k == 0 or n < 2:
        return np.zeros(0, np.int64), np.arange(n)
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if scores.shape != (n,):
        raise ValueError(f"nn1 selector: got {scores.shape[0]} scores for "
                         f"{n} candidates")
    # Highest score first: the scorer is trained so that a larger score means
    # "more worth forgetting", the same convention mode B ranks under.
    order = np.argsort(scores)[::-1]
    return np.sort(order[:k]), np.sort(order[k:])


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


# --------------------------------------------------------------------------
# MADAR's own two groups, as forget sets.
#
# core/buffer.py selects PER FAMILY: it fits an Isolation Forest over each
# family and keeps `half` most-anomalous plus the rest most-prototypical,
# discarding the middle. `donut` applies that prior to deletion but fits
# JOINTLY over the whole task, which is a different notion -- "anomalous within
# the task" rather than "anomalous for its family" -- and the two diverge
# whenever one family sits further from the model than another.
#
# These forget the two bands the buffer KEEPS, per family, so that "the half
# MADAR keeps as anomalous" means here exactly what it means there. Together
# with `leftover` (forget what MADAR discards) they cover the partition:
#
#     anomalous      | middle            | representative
#     kept by MADAR  | dropped by MADAR  | kept by MADAR
#     <- anomalous   | <- leftover       | <- representative
#
# `-joint` variants fit one forest over the whole task, matching `donut`'s
# geometry exactly, so the three bands form a strict three-way slice of one
# score distribution. Use those for a partition of `donut`; use the per-family
# forms to test MADAR's prior.
#
# BUDGET. `donut` forgets exactly int(n * ratio). Summing int(n_f * ratio) over
# families falls a little short of that, so the per-family budget is apportioned
# by largest remainder to hit the same total. A comparison is only attributable
# to WHICH samples are named if the count matches.


def _apportion(sizes, total):
    """Split `total` across families in proportion to `sizes`, exactly.

    Largest-remainder: floor each share, then hand the leftover one at a time
    to the largest fractional parts. Plain truncation loses up to one sample per
    family, which at 50 families is a budget the headline selector does not pay.
    """
    sizes = np.asarray(sizes, dtype=np.float64)
    if sizes.sum() <= 0 or total <= 0:
        return np.zeros(len(sizes), dtype=np.int64)
    exact = sizes / sizes.sum() * total
    base = np.floor(exact).astype(np.int64)
    short = int(total - base.sum())
    if short > 0:
        for i in np.argsort(-(exact - base))[:short]:
            base[i] += 1
    return np.minimum(base, sizes.astype(np.int64))


def _iso_order(X, seed, contamination):
    """Indices of X sorted by decision_function, ASCENDING.

    Ascending means most anomalous first and most prototypical last, which is
    the convention core/buffer.py uses when it takes order[:half] as anomalies
    and order[-(n-half):] as inliers.
    """
    iso = IsolationForest(contamination=contamination, n_jobs=-1,
                          random_state=seed).fit(X)
    return np.argsort(iso.decision_function(X))


def _band_per_family(n, ratio, latents, labels, seed, contamination, end):
    """Forget one tail of each family's own score distribution.

    `end` is 'low' for the most anomalous or 'high' for the most prototypical.
    """
    k_total = _budget(n, ratio)
    if k_total == 0 or n < 2:
        return np.zeros(0, np.int64), np.arange(n)
    labels = np.asarray(labels)
    fams = np.unique(labels)
    groups = [np.flatnonzero(labels == f) for f in fams]
    quota = _apportion([len(g) for g in groups], k_total)

    forget = []
    for idx, k in zip(groups, quota):
        # A family too small to rank, or allotted nothing, keeps everything.
        # Taking a tail of a 1-sample family would forget it outright.
        if k <= 0 or len(idx) < 2:
            continue
        k = int(min(k, len(idx) - 1))      # never empty a family
        order = _iso_order(latents[idx], seed, contamination)
        picked = order[:k] if end == "low" else order[-k:]
        forget.extend(idx[picked].tolist())

    forget = np.sort(np.asarray(forget, np.int64))
    retain = np.setdiff1d(np.arange(n), forget)
    return forget, retain


def _band_joint(n, ratio, latents, seed, contamination, end):
    """Forget one tail of the task's score distribution, fitted jointly.

    Same geometry as `donut`, which takes the middle of this same ordering.
    """
    k = _budget(n, ratio)
    if k == 0 or n < 2:
        return np.zeros(0, np.int64), np.arange(n)
    order = _iso_order(latents, seed, contamination)
    forget = order[:k] if end == "low" else order[-k:]
    retain = np.setdiff1d(np.arange(n), forget)
    return np.sort(forget), retain


@register("anomalous")
def _anomalous(*, n, ratio, latents, labels, seed, contamination=0.1, **_):
    """Forget the most anomalous fraction of each family -- one of the two
    halves MADAR keeps."""
    return _band_per_family(n, ratio, latents, labels, seed, contamination, "low")


@register("representative")
def _representative(*, n, ratio, latents, labels, seed, contamination=0.1, **_):
    """Forget the most prototypical fraction of each family -- the other half
    MADAR keeps."""
    return _band_per_family(n, ratio, latents, labels, seed, contamination, "high")


@register("anomalous-joint")
def _anomalous_joint(*, n, ratio, latents, seed, contamination=0.1, **_):
    """The low tail of `donut`'s own ordering, fitted jointly over the task."""
    return _band_joint(n, ratio, latents, seed, contamination, "low")


@register("representative-joint")
def _representative_joint(*, n, ratio, latents, seed, contamination=0.1, **_):
    """The high tail of `donut`'s own ordering, fitted jointly over the task."""
    return _band_joint(n, ratio, latents, seed, contamination, "high")
