"""GradApprox: a weighted gradient-matching coreset (GCR, Alg. 2).

Pick a subset X of a pool D, and weights w >= 0, minimising

    || sum_{d in D} wD_d * g_d  -  sum_{i in X} w_i * g_i ||^2 + lambda * ||w||^2

where g is a per-sample gradient of the replay loss. The subset's WEIGHTED
gradient stands in for the whole pool's, which is what lets a small buffer
reproduce the training signal of everything seen so far rather than merely
resembling it.

PER CLASS, not globally. GCR solves this separately for each label with a
budget of K/Y, so the buffer cannot drift class-imbalanced -- the same reason
core/buffer.py budgets per family. On a corpus with a 90:1 family imbalance a
global solve would spend the whole budget on the head.

LAST-LAYER GRADIENTS. Eq. (4) is written over all of theta. Computing that per
sample is a Jacobian the size of the model per example. GRAD-MATCH, the paper's
own reference for this machinery, uses the last layer instead, and so does this:
for a cross-entropy head the per-sample last-layer gradient is (p - onehot(y))
outer the penultimate feature, which is exact, cheap, and the standard stand-in.
It is an approximation of the stated objective and is named as one.

WHY ORTHOGONAL MATCHING PURSUIT. The objective is weakly submodular, so greedy
selection carries an approximation guarantee. Each step adds the sample whose
gradient best explains what the current subset leaves unexplained, then refits
ALL the chosen weights -- refitting is what distinguishes OMP from plain greedy
and is why the weights are usable as loss coefficients afterwards.

NON-NEGATIVITY is in the objective (Eq. 4 constrains WX >= 0) and is enforced
rather than clipped after the fact: a negative weight would mean a replayed
sample pushing the model AWAY from its own label, which is not a coreset.

THE EMPTY-CLASS GUARD. A class the model already fits has per-sample gradients
near zero, so the empty subset matches its target and the solve legitimately
selects nothing. In supervised coreset selection that is correct -- there is no
signal to preserve. In CONTINUAL learning it is a trap: a class absent from the
buffer is never replayed, drifts as later tasks arrive, and is forgotten, at
which point its gradients are large again and it is too late. Such a class is
therefore backfilled by gradient norm at unit weight, and the backfill is
reported rather than done quietly.
"""

from __future__ import annotations

import numpy as np

try:                                   # NNLS is the only scipy call here
    from scipy.optimize import nnls
    _HAVE_NNLS = True
except ImportError:                    # pragma: no cover
    _HAVE_NNLS = False


def _solve(gram: np.ndarray, rhs: np.ndarray, lam: float) -> np.ndarray:
    """argmin_{w >= 0} w' (G + lam I) w - 2 w' rhs.

    Solved in the |S|-dimensional normal-equation space, not against the full
    (|S| x d) design: |S| is at most the per-class budget while d is
    active_count * latent, so this keeps a 50x50 solve from becoming a
    12800x50 one on every one of thousands of OMP steps.
    """
    k = gram.shape[0]
    A = gram + lam * np.eye(k)
    if not _HAVE_NNLS:
        w = np.linalg.solve(A, rhs)
        return np.maximum(w, 0.0)
    # NNLS needs a design matrix, so factor: with A = L L', minimising
    # ||L' w - L^-1 rhs||^2 has the same minimiser as the quadratic above.
    try:
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        A = A + 1e-8 * np.eye(k)
        L = np.linalg.cholesky(A)
    target = np.linalg.solve(L, rhs)
    w, _ = nnls(L.T, target)
    return w


def omp(grads: np.ndarray, target: np.ndarray, budget: int, lam: float = 1.0,
        tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """One class's solve. -> (chosen row indices, their weights).

    `grads` is (n, d) per-sample gradients; `target` is the (d,) weighted sum
    this subset must reproduce.
    """
    n = len(grads)
    budget = int(min(budget, n))
    if budget <= 0 or n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0)

    chosen: list[int] = []
    gram = np.zeros((0, 0))
    rhs = np.zeros(0)
    w = np.zeros(0)
    residual = target.astype(np.float64).copy()

    for _ in range(budget):
        corr = grads @ residual
        corr[chosen] = -np.inf          # never pick the same sample twice
        j = int(np.argmax(corr)) if np.isfinite(corr).any() else -1
        if j < 0 or not np.isfinite(corr[j]):
            break
        gj = grads[j]
        # Extend the Gram matrix and right-hand side by one row/column.
        cross = grads[chosen] @ gj if chosen else np.zeros(0)
        k = len(chosen)
        new = np.zeros((k + 1, k + 1))
        new[:k, :k] = gram
        new[:k, k] = new[k, :k] = cross
        new[k, k] = gj @ gj
        gram = new
        rhs = np.concatenate([rhs, [gj @ target]])
        chosen.append(j)

        w = _solve(gram, rhs, lam)
        residual = target - (w @ grads[chosen])
        if float(residual @ residual) <= tol:
            break

    keep = np.asarray(w) > 0
    idx = np.asarray(chosen, dtype=np.int64)[keep]
    return idx, np.asarray(w)[keep]


def last_layer_grads(probs: np.ndarray, feats: np.ndarray, labels: np.ndarray,
                     rows: np.ndarray) -> np.ndarray:
    """Per-sample last-layer gradients for `rows`, as a dense (len(rows), C*F).

    For a softmax head the gradient of cross-entropy w.r.t. the final weight
    matrix is exactly outer(p - onehot(y), feature), so this needs no backward
    pass -- only the probabilities and the penultimate features, both of which
    one forward pass already produced.

    MATERIALISED PER CLASS, deliberately. The full matrix is
    (pool x active_count x latent): at 10k pool, 100 classes and a 128-wide
    latent that is half a gigabyte, and the solve never needs more than one
    class at a time. Building it per class keeps the peak at a few megabytes.
    """
    p = probs[rows].astype(np.float64).copy()
    p[np.arange(len(rows)), labels[rows]] -= 1.0
    f = feats[rows].astype(np.float64)
    return (p[:, :, None] * f[:, None, :]).reshape(len(rows), -1)


def grad_approx_lowrank(probs, feats, labels, weights, budget: int,
                        lam: float = 1.0, tol: float = 1e-6) -> dict:
    """grad_approx from (probabilities, features) without ever forming the
    whole gradient matrix. Same contract, same return."""
    probs, feats = np.asarray(probs), np.asarray(feats)
    labels = np.asarray(labels)
    weights = np.asarray(weights, dtype=np.float64)
    classes = np.unique(labels)
    per_class = max(1, int(budget) // max(1, len(classes)))

    idx_out, w_out, rel, backfilled = [], [], [], {}
    for c in classes:
        rows = np.flatnonzero(labels == c)
        g = last_layer_grads(probs, feats, labels, rows)
        target = weights[rows] @ g
        sel, w = omp(g, target, per_class, lam=lam, tol=tol)
        if len(sel) == 0 and len(rows):
            take = min(per_class, len(rows))
            sel = np.argsort(np.linalg.norm(g, axis=1))[::-1][:take]
            w = np.ones(len(sel))
            backfilled[int(c)] = int(take)
        idx_out.append(rows[sel]); w_out.append(w)
        denom = float(target @ target)
        if denom > 0:
            r = target - (w @ g[sel]) if len(sel) else target
            rel.append(float(r @ r) / denom)
        del g

    idx = np.concatenate(idx_out) if idx_out else np.zeros(0, dtype=np.int64)
    ws = np.concatenate(w_out) if w_out else np.zeros(0)
    order = np.argsort(idx)
    return {"index": idx[order].astype(np.int64), "weights": ws[order],
            "n_classes": int(len(classes)), "per_class_budget": per_class,
            "selected": int(len(idx)),
            "mean_residual": float(np.mean(rel)) if rel else None,
            "max_residual": float(np.max(rel)) if rel else None,
            "backfilled_classes": backfilled}


def grad_approx(grads: np.ndarray, labels: np.ndarray, weights: np.ndarray,
                budget: int, lam: float = 1.0, tol: float = 1e-6) -> dict:
    """Per-class GradApprox over a pool. -> indices, weights, diagnostics.

    `weights` are the pool's existing weights (1 for freshly added candidates,
    the fitted values for entries already in the buffer), and they enter the
    target: the thing being approximated is the pool's WEIGHTED gradient, since
    that is what the previous round's selection made the training signal.
    """
    grads = np.asarray(grads, dtype=np.float64)
    labels = np.asarray(labels)
    weights = np.asarray(weights, dtype=np.float64)
    classes = np.unique(labels)
    per_class = max(1, int(budget) // max(1, len(classes)))

    idx_out, w_out = [], []
    rel, backfilled = [], {}
    for c in classes:
        rows = np.flatnonzero(labels == c)
        g = grads[rows]
        target = weights[rows] @ g
        sel, w = omp(g, target, per_class, lam=lam, tol=tol)
        if len(sel) == 0 and len(rows):
            # See THE EMPTY-CLASS GUARD above.
            take = min(per_class, len(rows))
            sel = np.argsort(np.linalg.norm(g, axis=1))[::-1][:take]
            w = np.ones(len(sel))
            backfilled[int(c)] = int(take)
        idx_out.append(rows[sel])
        w_out.append(w)
        denom = float(target @ target)
        if denom > 0:
            r = target - (w @ g[sel]) if len(sel) else target
            rel.append(float(r @ r) / denom)

    idx = np.concatenate(idx_out) if idx_out else np.zeros(0, dtype=np.int64)
    ws = np.concatenate(w_out) if w_out else np.zeros(0)
    order = np.argsort(idx)
    return {"index": idx[order].astype(np.int64), "weights": ws[order],
            "n_classes": int(len(classes)), "per_class_budget": per_class,
            "selected": int(len(idx)),
            "mean_residual": float(np.mean(rel)) if rel else None,
            "max_residual": float(np.max(rel)) if rel else None,
            "backfilled_classes": backfilled}
