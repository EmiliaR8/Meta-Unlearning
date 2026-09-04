"""EMBER 2018 loader.

EMBER ships vectorised features but no family labels; the EMBERSim databank
ships AVClass family strings keyed by sha256. Joining them means walking the
feature jsonl files in the same order the vectorised matrix was built, which is
slow (minutes), so the joined arrays are cached under cache_root as one .npz.

The cache is keyed by content, not by filename alone: the stored record carries
the dataset name and the loader version, and a mismatch is a hard failure. A
cache silently reused across corpora is the failure this guards against -- it
produces plausible numbers for the wrong dataset.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from .base import Corpus

LOADER_VERSION = 1
DEFAULT_SUBDIR = "ember2018"
EMBERSIM_SUBDIR = "embersim-databank"
METADATA_REL = os.path.join("data", "raw", "ember_original_metadata.csv")


def _hash_to_family(embersim_dir: Path) -> dict:
    label_file = embersim_dir / METADATA_REL
    if not label_file.is_file():
        raise SystemExit(
            f"EMBER 2018 family labels not found at {label_file}.\n"
            f"Expected the EMBERSim databank under <data_root>/{EMBERSIM_SUBDIR}/.\n"
            f"Check `python -m code.runner.paths --show`.")
    mapping = {}
    with label_file.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fam = (row.get("avclass") or "").strip()
            if fam and fam != "SINGLETON":
                mapping[row["sha256"]] = fam
    if not mapping:
        raise SystemExit(f"{label_file} yielded no usable avclass labels")
    return mapping


def _join_subset(data_dir: Path, subset: str, hash_to_fam: dict,
                 family_to_id: dict, X_raw):
    """Walk the jsonl shards in sorted order, keeping rows with a family label.

    Row order here must match the order `ember.read_vectorized_features` built
    its matrix in -- that is the contract the whole join rests on.
    """
    shards = sorted(p for p in data_dir.iterdir()
                    if p.name.startswith(f"{subset}_features") and p.suffix == ".jsonl")
    if not shards:
        raise SystemExit(f"no {subset}_features*.jsonl found in {data_dir}")
    keep_rows, labels, row_idx = [], [], 0
    for shard in shards:
        with shard.open() as f:
            for line in f:
                h = json.loads(line)["sha256"]
                fam = hash_to_fam.get(h)
                if fam is not None:
                    family_to_id.setdefault(fam, len(family_to_id))
                    keep_rows.append(row_idx)
                    labels.append(family_to_id[fam])
                row_idx += 1
    if row_idx != len(X_raw):
        raise SystemExit(
            f"{subset}: walked {row_idx} jsonl rows but the vectorised matrix has "
            f"{len(X_raw)}. The features and the jsonl shards are out of sync; "
            f"delete X_{subset}.dat and let the ember package rebuild it.")
    idx = np.asarray(keep_rows, dtype=np.int64)
    return np.ascontiguousarray(X_raw[idx]), np.asarray(labels, dtype=np.int64)


def _build(data_dir: Path, embersim_dir: Path) -> Corpus:
    try:
        import ember
    except ImportError:
        raise SystemExit(
            "the `ember` package is required to build the EMBER 2018 arrays "
            "(only for the first run; afterwards the cache is used)")

    if not (data_dir / "X_train.dat").exists():
        print(f"  vectorising EMBER 2018 features in {data_dir} (slow, one-off)...")
        ember.create_vectorized_features(str(data_dir), feature_version=2)
    X_train_raw, _, X_test_raw, _ = ember.read_vectorized_features(
        str(data_dir), feature_version=2)

    hash_to_fam = _hash_to_family(embersim_dir)
    family_to_id: dict = {}
    Xtr, ytr = _join_subset(data_dir, "train", hash_to_fam, family_to_id, X_train_raw)
    Xte, yte = _join_subset(data_dir, "test", hash_to_fam, family_to_id, X_test_raw)
    return Corpus(name="ember2018", X_train=Xtr.astype(np.float32), y_train=ytr,
                  X_test=Xte.astype(np.float32), y_test=yte, scaling="standard",
                  meta={"n_families_labelled": len(family_to_id),
                        "loader_version": LOADER_VERSION})


def load(paths, *, cache: bool = True, **_) -> Corpus:
    cache_file = paths.feature_cache_dir() / "ember2018_joined.npz"
    if cache and cache_file.is_file():
        z = np.load(cache_file, allow_pickle=False)
        stored = str(z["dataset"]) if "dataset" in z else "?"
        version = int(z["loader_version"]) if "loader_version" in z else -1
        if stored != "ember2018" or version != LOADER_VERSION:
            raise SystemExit(
                f"{cache_file} holds dataset={stored!r} loader_version={version}, "
                f"expected 'ember2018' v{LOADER_VERSION}. Delete it and re-run.")
        print(f"  loaded EMBER 2018 from cache {cache_file}")
        return Corpus(name="ember2018", X_train=z["Xtr"], y_train=z["ytr"],
                      X_test=z["Xte"], y_test=z["yte"], scaling="standard",
                      meta={"loader_version": LOADER_VERSION,
                            "cache_file": str(cache_file)})

    data_dir = paths.dataset_dir(DEFAULT_SUBDIR)
    embersim_dir = paths.dataset_dir(EMBERSIM_SUBDIR)
    if not data_dir.is_dir():
        raise SystemExit(
            f"EMBER 2018 not found at {data_dir}.\n"
            f"Set data_root in paths.local.json or ICLR_DATA_ROOT; see "
            f"`python -m code.runner.paths --show`.")
    corpus = _build(data_dir, embersim_dir)

    if cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, Xtr=corpus.X_train, ytr=corpus.y_train,
                 Xte=corpus.X_test, yte=corpus.y_test,
                 dataset="ember2018", loader_version=LOADER_VERSION)
        print(f"  cached EMBER 2018 arrays to {cache_file}")
        corpus.meta["cache_file"] = str(cache_file)
    return corpus
