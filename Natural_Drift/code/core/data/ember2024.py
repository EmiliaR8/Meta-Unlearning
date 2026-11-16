"""EMBER 2024 loader.

Simpler than 2018: thrember exposes family labels natively, so there is no
sha256->AVClass join. Two corpus-specific steps instead:

  * NEGATIVE SENTINEL LABELS. thrember marks benign/unlabelled samples with
    negative ids. They are dropped before family selection, because they are not
    families and would otherwise be counted as one.

  * PER-FAMILY CAP. This corpus is far more imbalanced than 2018 (largest family
    ~94x the smallest). The cap is applied by the shared `cap_per_family` with a
    seed deliberately independent of the run seed, so every condition and every
    seed trains on the identical subset -- otherwise the cap becomes a second,
    uncontrolled source of variation between conditions.

Selection itself is the shared integer-keyed path, identical to 2018.
"""

from __future__ import annotations

import numpy as np

from .base import Corpus

LOADER_VERSION = 1
DEFAULT_SUBDIR = "ember2024"


def load(paths, *, cache: bool = True, **_) -> Corpus:
    cache_file = paths.feature_cache_dir() / "ember2024_arrays.npz"
    if cache and cache_file.is_file():
        z = np.load(cache_file, allow_pickle=False)
        stored = str(z["dataset"]) if "dataset" in z else "?"
        version = int(z["loader_version"]) if "loader_version" in z else -1
        if stored != "ember2024" or version != LOADER_VERSION:
            raise SystemExit(
                f"{cache_file} holds dataset={stored!r} loader_version={version}, "
                f"expected 'ember2024' v{LOADER_VERSION}. Delete it and re-run.")
        print(f"  loaded EMBER 2024 from cache {cache_file}")
        return Corpus(name="ember2024", X_train=z["Xtr"], y_train=z["ytr"],
                      X_test=z["Xte"], y_test=z["yte"], scaling="standard",
                      meta={"loader_version": LOADER_VERSION,
                            "cache_file": str(cache_file)})

    data_dir = paths.dataset_dir(DEFAULT_SUBDIR)
    if not data_dir.is_dir():
        raise SystemExit(
            f"EMBER 2024 not found at {data_dir}.\n"
            f"Check `python -m runner.paths --show`.")
    try:
        import thrember
    except ImportError:
        raise SystemExit(
            "the `thrember` package is required to build the EMBER 2024 arrays")

    if not (data_dir / "X_train.dat").exists():
        print(f"  vectorising EMBER 2024 features in {data_dir} (slow, one-off)...")
        thrember.create_vectorized_features(str(data_dir), label_type="family")
    Xtr, ytr = thrember.read_vectorized_features(str(data_dir), subset="train")
    Xte, yte = thrember.read_vectorized_features(str(data_dir), subset="test")

    ytr, yte = np.asarray(ytr), np.asarray(yte)
    n_neg = int((ytr < 0).sum()) + int((yte < 0).sum())
    keep_tr, keep_te = ytr >= 0, yte >= 0
    Xtr, ytr = np.ascontiguousarray(Xtr[keep_tr]), ytr[keep_tr].astype(np.int64)
    Xte, yte = np.ascontiguousarray(Xte[keep_te]), yte[keep_te].astype(np.int64)
    if n_neg:
        print(f"  dropped {n_neg:,} samples with negative (sentinel) labels")

    corpus = Corpus(name="ember2024", X_train=Xtr.astype(np.float32), y_train=ytr,
                    X_test=Xte.astype(np.float32), y_test=yte, scaling="standard",
                    meta={"loader_version": LOADER_VERSION,
                          "n_sentinel_dropped": n_neg})

    if cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, Xtr=corpus.X_train, ytr=corpus.y_train,
                 Xte=corpus.X_test, yte=corpus.y_test,
                 dataset="ember2024", loader_version=LOADER_VERSION)
        print(f"  cached EMBER 2024 arrays to {cache_file}")
        corpus.meta["cache_file"] = str(cache_file)
    return corpus
