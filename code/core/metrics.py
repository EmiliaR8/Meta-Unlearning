"""Evaluation metrics, all derived from one confusion matrix.

Everything reported -- accuracy, precision, recall, F1 -- is computed from the
same confusion matrix, so the numbers in a log are mutually consistent by
construction rather than by three separate code paths agreeing.

AVERAGING. For single-label multiclass, micro-precision == micro-recall ==
micro-F1 == accuracy. Reporting "micro F1" as though it were a separate quantity
is a common way to pad a results table with one number four times, so it is not
emitted; `accuracy` is that number, named honestly.

  * macro  -- unweighted mean over classes WITH SUPPORT in y_true. Classes the
              evaluation slice contains no examples of are excluded, not scored
              as zero: a class absent from a slice has no recall to measure, and
              counting it as 0 would make the metric a function of which classes
              happen to be absent.
  * weighted -- mean over the same classes, weighted by support.

Precision has a second subtlety: a class can be PREDICTED without being present.
Such a class contributes false positives that belong in no per-class precision
under the support rule above. Those are visible in the confusion matrix and in
`n_pred_outside_support`, rather than being silently dropped.

THREE SCOPES are evaluated per task, matching what the runs need to distinguish:
  seen    -- all families seen so far        (the headline number)
  recent  -- the families just added         (acquisition / plasticity)
  task0   -- the first task's families       (retention / forgetting)
`seen` averages the other two, so a method trading one for the other is
invisible in it alone.
"""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    """Rows = true class, columns = predicted class. Counts, int64."""
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true {y_true.shape} != y_pred {y_pred.shape}")
    if y_true.size == 0:
        return np.zeros((n_classes, n_classes), dtype=np.int64)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    if lo < 0 or hi >= n_classes:
        raise ValueError(f"labels outside [0,{n_classes}): saw [{lo},{hi}]")
    flat = np.bincount(y_true * n_classes + y_pred, minlength=n_classes ** 2)
    return flat.reshape(n_classes, n_classes).astype(np.int64)


def metrics_from_confusion(cm: np.ndarray) -> dict:
    """Accuracy / precision / recall / F1 from a confusion matrix."""
    cm = np.asarray(cm, dtype=np.float64)
    support = cm.sum(axis=1)                 # true count per class
    predicted = cm.sum(axis=0)               # predicted count per class
    tp = np.diag(cm)
    total = cm.sum()

    present = support > 0
    n_present = int(present.sum())

    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, tp / np.maximum(support, 1), np.nan)
        precision = np.where(predicted > 0, tp / np.maximum(predicted, 1), np.nan)
        denom = precision + recall
        f1 = np.where((denom > 0) & np.isfinite(denom),
                      2 * precision * recall / np.maximum(denom, 1e-12), 0.0)

    # Restrict averaging to classes with support (see module docstring).
    p_s = np.where(present, np.nan_to_num(precision, nan=0.0), 0.0)
    r_s = np.where(present, np.nan_to_num(recall, nan=0.0), 0.0)
    f_s = np.where(present, np.nan_to_num(f1, nan=0.0), 0.0)

    accuracy = float(tp.sum() / total * 100) if total else 0.0
    if n_present:
        w = support / support.sum()
        macro = dict(precision=float(p_s[present].mean() * 100),
                     recall=float(r_s[present].mean() * 100),
                     f1=float(f_s[present].mean() * 100))
        weighted = dict(precision=float((p_s * w).sum() * 100),
                        recall=float((r_s * w).sum() * 100),
                        f1=float((f_s * w).sum() * 100))
    else:
        macro = weighted = dict(precision=0.0, recall=0.0, f1=0.0)

    # Classes predicted but never present: their false positives are real but
    # belong to no scored class. Surfaced rather than dropped in silence.
    outside = int(predicted[~present].sum())

    return {
        "accuracy": accuracy,
        "macro_precision": macro["precision"],
        "macro_recall": macro["recall"],
        "macro_f1": macro["f1"],
        "weighted_precision": weighted["precision"],
        "weighted_recall": weighted["recall"],
        "weighted_f1": weighted["f1"],
        "n_samples": int(total),
        "n_classes_present": n_present,
        "n_pred_outside_support": outside,
        "per_class": {
            "classes": np.flatnonzero(present).tolist(),
            "support": support[present].astype(np.int64).tolist(),
            "precision": (p_s[present] * 100).round(6).tolist(),
            "recall": (r_s[present] * 100).round(6).tolist(),
            "f1": (f_s[present] * 100).round(6).tolist(),
        },
    }


def sparse_confusion(cm: np.ndarray) -> dict:
    """Confusion matrix as COO triplets.

    A dense 100x100 matrix per task per run is 10k integers, nearly all zero. The
    sparse form keeps the matrix inside the run's JSON -- readable, greppable, no
    second file to keep in sync -- at a few percent of the size.
    """
    cm = np.asarray(cm, dtype=np.int64)
    rows, cols = np.nonzero(cm)
    return {"format": "coo", "shape": list(cm.shape),
            "true": rows.tolist(), "pred": cols.tolist(),
            "count": cm[rows, cols].tolist()}


def dense_confusion(entry: dict) -> np.ndarray:
    """Inverse of sparse_confusion, for the aggregator."""
    if entry.get("format") != "coo":
        return np.asarray(entry, dtype=np.int64)
    cm = np.zeros(entry["shape"], dtype=np.int64)
    cm[np.asarray(entry["true"], dtype=int), np.asarray(entry["pred"], dtype=int)] = \
        np.asarray(entry["count"], dtype=np.int64)
    return cm
