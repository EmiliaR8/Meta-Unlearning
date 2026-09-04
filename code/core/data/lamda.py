"""LAMDA Class-IL loader.

Delegates entirely to the project's own `lamda_data.build_arrays_lamda`, vendored
alongside this file. That module is the established single source of truth for
LAMDA family selection and task construction, and reusing it verbatim is what
makes runs from this runner comparable with the prior LAMDA results.

Two consequences follow, both deliberate:

  * PRESELECTED. lamda_data selects families by their STRING name and returns
    labels already remapped to 0..N-1 in descending train frequency. The runner's
    generic integer-keyed selection cannot reproduce that and must not re-run on
    top of it, so the Corpus is flagged `preselected` and that step is skipped.

  * NO SCALING. LAMDA's 4561 Baseline features are binary indicators.
    Standardising them is not meaningful, so scaling is 'none'.

The schedule lamda_data builds must agree with the runner's --task-setup; a
mismatch is checked rather than assumed, since both sides independently believe
they know the task layout.
"""

from __future__ import annotations

import numpy as np

from .base import Corpus

DEFAULT_CACHE = "lamda_class_il_cache.npz"


def load(paths, *, split_mode: str = "random", temporal_cut: int = 2020,
         min_family_samples: int = 200, num_classes: int = 80,
         task0_classes: int = 30, step_classes: int = 5,
         min_test_samples: int = 0, verbose: bool = True, **_) -> Corpus:
    cache_path = paths.dataset_dir(DEFAULT_CACHE)
    if not cache_path.is_file():
        raise SystemExit(
            f"LAMDA feature cache not found at {cache_path}.\n"
            f"Build it with L2U_Code/build_lamda_cache.py, or point data_root at "
            f"the directory that holds it (`python -m runner.paths --show`).")

    from . import lamda_data

    r = lamda_data.build_arrays_lamda(
        str(cache_path), split_mode=split_mode, temporal_cut=temporal_cut,
        min_family_samples=min_family_samples, num_classes=num_classes,
        task0_classes=task0_classes, step_classes=step_classes,
        min_test_samples=min_test_samples, verbose=verbose)

    return Corpus(
        name="lamda_classil",
        X_train=np.ascontiguousarray(r["X_train"]).astype(np.float32),
        y_train=np.asarray(r["y_train"], dtype=np.int64),
        X_test=np.ascontiguousarray(r["X_test"]).astype(np.float32),
        y_test=np.asarray(r["y_test"], dtype=np.int64),
        scaling="none",
        preselected=True,
        meta={"split_mode": split_mode,
              "temporal_cut": temporal_cut if split_mode == "temporal" else None,
              "n_families_selected": len(r["ordered"]),
              "task_sizes": r["task_sizes"],
              "cache_file": str(cache_path),
              "selection": "lamda_data.build_arrays_lamda (vendored verbatim)"},
    )
