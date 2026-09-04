"""A small synthetic corpus with the same shape as the real ones.

Not a scientific artifact. It exists so the runner, the metrics, the buffer, the
logging schema and the aggregator can be exercised end to end in seconds without
EMBER on disk -- the failure mode this guards against is a module-level bug that
passes py_compile and only surfaces an hour into a real run.

Families are Gaussian blobs of deliberately unequal size, so the imbalance that
makes macro and micro diverge on the real corpora is present here too.
"""

from __future__ import annotations

import numpy as np

from .base import Corpus


def load(paths=None, *, n_families: int = 30, input_dim: int = 64,
         min_size: int = 40, max_size: int = 400, seed: int = 0,
         test_fraction: float = 0.25, **_) -> Corpus:
    rng = np.random.default_rng(seed)
    sizes = np.geomspace(max_size, min_size, n_families).round().astype(int)
    centres = rng.standard_normal((n_families, input_dim)) * 2.5

    Xtr, ytr, Xte, yte = [], [], [], []
    for fam in range(n_families):
        n = int(sizes[fam])
        n_test = max(2, int(n * test_fraction))
        pts = centres[fam] + rng.standard_normal((n + n_test, input_dim))
        Xtr.append(pts[:n]); ytr.append(np.full(n, fam))
        Xte.append(pts[n:]); yte.append(np.full(n_test, fam))

    return Corpus(
        name="synthetic",
        X_train=np.vstack(Xtr).astype(np.float32),
        y_train=np.concatenate(ytr).astype(np.int64),
        X_test=np.vstack(Xte).astype(np.float32),
        y_test=np.concatenate(yte).astype(np.int64),
        scaling="standard",
        meta={"synthetic": True, "n_families_generated": n_families,
              "note": "smoke-test corpus; not a scientific result"},
    )
