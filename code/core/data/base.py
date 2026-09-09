"""Corpus container and the family-selection / scaling steps shared by all
datasets.

Family selection and scaling live here rather than in each loader because they
are protocol, not corpus detail: every dataset takes families with at least
`min_family_samples` training examples, keeps the top `n_classes` by frequency,
and remaps ids to presentation order. A loader's only job is to return raw
features and integer family labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Corpus:
    """Raw arrays for one dataset. Unscaled; labels are original family ids."""

    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaling: str = "standard"        # 'standard' | 'none'
    preselected: bool = False        # loader already selected + remapped families
    meta: dict = field(default_factory=dict)

    @property
    def input_shape(self) -> tuple:
        """Per-example shape, batch dimension dropped: (d,) or (C, H, W)."""
        return tuple(int(v) for v in self.X_train.shape[1:])

    @property
    def input_dim(self) -> int:
        """Total features per example.

        For a tabular corpus this is the feature count, unchanged. For images it
        is C*H*W, which is not what any convolutional layer consumes but IS what
        the parameter-count bookkeeping and the log's `input_dim` field mean.
        Anything that needs the real geometry asks for `input_shape`.
        """
        n = 1
        for v in self.input_shape:
            n *= int(v)
        return int(n)

    @property
    def is_image(self) -> bool:
        return len(self.input_shape) == 3

    def summary(self) -> dict:
        return {"dataset": self.name, "input_dim": self.input_dim,
                "input_shape": list(self.input_shape),
                "n_train": int(len(self.y_train)), "n_test": int(len(self.y_test)),
                "scaling": self.scaling, **self.meta}


def select_families(y_train, min_family_samples: int, n_classes: int):
    """Top `n_classes` families with >= min_family_samples training examples.

    Sorted by descending frequency; ties break on ascending original id, which
    is what the existing pipeline does and what makes a longer schedule's first
    N families identical to a shorter one's.
    """
    y = np.asarray(y_train)
    ids, counts = np.unique(y, return_counts=True)
    sizes = {int(i): int(c) for i, c in zip(ids, counts)}
    eligible = sorted([f for f in sorted(sizes) if sizes[f] >= min_family_samples],
                      key=lambda f: sizes[f], reverse=True)
    if len(eligible) < n_classes:
        raise SystemExit(
            f"only {len(eligible)} families have >= {min_family_samples} training "
            f"samples, but the task setup needs {n_classes}. Lower --min-family-samples "
            f"or choose a shorter task setup.")
    chosen = eligible[:n_classes]
    id_map = {old: new for new, old in enumerate(chosen)}
    return id_map, chosen, sizes


def apply_family_selection(corpus: Corpus, min_family_samples: int,
                           n_classes: int) -> Corpus:
    """Filter to the selected families and remap labels to presentation order.

    A corpus flagged `preselected` has already had this applied by its loader --
    LAMDA selects on family STRINGS via its own vendored module, which this
    integer-keyed path cannot reproduce. Re-running selection there would re-sort
    an already-ordered label space, so it is refused rather than silently redone.
    The label range is verified instead.
    """
    if corpus.preselected:
        for split, y in (("train", corpus.y_train), ("test", corpus.y_test)):
            if len(y) and (y.min() < 0 or y.max() >= n_classes):
                raise SystemExit(
                    f"{corpus.name}: loader reported preselected labels but {split} "
                    f"labels span [{y.min()}, {y.max()}], outside [0, {n_classes}).")
        return corpus

    id_map, chosen, sizes = select_families(
        corpus.y_train, min_family_samples, n_classes)

    def remap(X, y):
        lut = np.full(int(max(id_map)) + 1, -1, dtype=np.int64)
        for old, new in id_map.items():
            lut[old] = new
        y = np.asarray(y, dtype=np.int64)
        safe = np.where(y <= int(max(id_map)), y, 0)
        mapped = np.where(y <= int(max(id_map)), lut[safe], -1)
        keep = mapped >= 0
        return np.ascontiguousarray(X[keep]), mapped[keep]

    Xtr, ytr = remap(corpus.X_train, corpus.y_train)
    Xte, yte = remap(corpus.X_test, corpus.y_test)
    sel_sizes = [sizes[f] for f in chosen]
    meta = dict(corpus.meta)
    meta.update({
        "n_families_eligible": len([f for f in sizes if sizes[f] >= min_family_samples]),
        "n_families_selected": len(chosen),
        "min_family_samples": int(min_family_samples),
        "selected_family_size_max": int(max(sel_sizes)),
        "selected_family_size_min": int(min(sel_sizes)),
    })
    return Corpus(name=corpus.name, X_train=Xtr, y_train=ytr, X_test=Xte,
                  y_test=yte, scaling=corpus.scaling,
                  preselected=corpus.preselected, meta=meta)


def cap_per_family(X, y, cap: int, cap_seed: int):
    """Subsample each TRAINING family down to `cap`. Test data is never capped.

    The seed is deliberately independent of the run seed so that every condition
    and every seed trains on the identical subset -- otherwise the cap becomes a
    second, uncontrolled source of variation between conditions.
    """
    if not cap or cap <= 0:
        return X, y, 0
    rng = np.random.default_rng(cap_seed)
    y = np.asarray(y)
    keep = []
    for fam in np.unique(y):
        idx = np.flatnonzero(y == fam)
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    return np.ascontiguousarray(X[keep]), y[keep], int(len(y) - len(keep))


def scale_features(corpus: Corpus, task0_classes, clip: float = 10.0):
    """Standardise on TASK 0 ONLY, then clip.

    Fitting on the full training set would leak the statistics of families the
    continual learner has not met yet, which is not a continual protocol.
    Returns (X_train, X_test, info).
    """
    if corpus.scaling == "none":
        return (corpus.X_train.astype(np.float32),
                corpus.X_test.astype(np.float32),
                {"scaling": "none", "reason": "features are binary indicators"})

    if corpus.scaling == "image":
        # Images keep their uint8 storage and are normalised inside the model's
        # forward pass -- see ResNet18's docstring for why the float32 copy is
        # not made here. What IS computed now is the per-channel mean and
        # standard deviation, and it is computed on TASK 0 ONLY, for exactly the
        # reason the tabular branch below fits its scaler on task 0: statistics
        # taken over all 200 classes would describe classes the learner has not
        # been shown yet. The values are in 0-255 units, matching the input.
        mask = np.isin(corpus.y_train, np.asarray(task0_classes))
        if not mask.any():
            raise SystemExit("no task-0 training samples; cannot fit the "
                             "channel statistics")
        sub = np.asarray(corpus.X_train[mask], dtype=np.float32)
        mean = sub.mean(axis=(0, 2, 3))
        std = sub.std(axis=(0, 2, 3))
        if (std <= 0).any():
            raise SystemExit(
                f"channel(s) {np.flatnonzero(std <= 0).tolist()} are constant "
                f"across task 0; cannot normalise")
        return (corpus.X_train, corpus.X_test,
                {"scaling": "image", "fit_on": "task0",
                 "n_fit_samples": int(mask.sum()),
                 "channel_mean": [round(float(v), 4) for v in mean],
                 "channel_std": [round(float(v), 4) for v in std],
                 "note": "uint8 retained; normalisation applied in the model"})

    from sklearn.preprocessing import StandardScaler

    mask = np.isin(corpus.y_train, np.asarray(task0_classes))
    if not mask.any():
        raise SystemExit("no task-0 training samples; cannot fit the scaler")
    scaler = StandardScaler().fit(corpus.X_train[mask])
    Xtr = np.clip(scaler.transform(corpus.X_train), -clip, clip).astype(np.float32)
    Xte = np.clip(scaler.transform(corpus.X_test), -clip, clip).astype(np.float32)
    return Xtr, Xte, {"scaling": "standard", "fit_on": "task0",
                      "n_fit_samples": int(mask.sum()), "clip": float(clip)}
