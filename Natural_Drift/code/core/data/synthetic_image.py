"""A small synthetic IMAGE corpus, the convolutional counterpart of `synthetic`.

Not a scientific artifact. `synthetic` exercises the tabular path in seconds;
this does the same for the path Tiny ImageNet opened up -- uint8 storage, a
(C, H, W) Corpus, per-channel task-0 normalisation, the ResNet backbone, GPU
augmentation, latent-space buffer selection and DEDUCE's convolutional neuron
reinitialisation. Without it, the first thing to ever run the image path would
be a real sweep on a machine with 100k JPEGs on disk.

Each class is a coloured blob at a class-specific position and hue over noise,
which is enough structure for a convolutional model to actually fit -- a corpus
of pure noise would make every method score at chance and hide exactly the bugs
this is meant to catch.
"""

from __future__ import annotations

import numpy as np

from .base import Corpus


def load(paths=None, *, n_classes: int = 20, size: int = 32, min_size: int = 30,
         max_size: int = 120, seed: int = 0, test_fraction: float = 0.25,
         **_) -> Corpus:
    rng = np.random.default_rng(seed)
    counts = np.geomspace(max_size, min_size, n_classes).round().astype(int)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)

    Xtr, ytr, Xte, yte = [], [], [], []
    for cls in range(n_classes):
        n = int(counts[cls])
        n_test = max(2, int(n * test_fraction))
        cy, cx = rng.uniform(size * 0.25, size * 0.75, 2)
        hue = rng.uniform(0.3, 1.0, 3).astype(np.float32)
        radius = rng.uniform(size * 0.12, size * 0.3)
        total = n + n_test
        jitter = rng.uniform(-2.0, 2.0, (total, 2)).astype(np.float32)
        imgs = np.empty((total, 3, size, size), dtype=np.uint8)
        for i in range(total):
            d = np.sqrt((yy - cy - jitter[i, 0]) ** 2 + (xx - cx - jitter[i, 1]) ** 2)
            blob = np.exp(-(d ** 2) / (2 * radius ** 2))
            noise = rng.normal(0, 0.08, (3, size, size)).astype(np.float32)
            frame = hue[:, None, None] * blob[None] * 0.8 + 0.1 + noise
            imgs[i] = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        Xtr.append(imgs[:n]); ytr.append(np.full(n, cls))
        Xte.append(imgs[n:]); yte.append(np.full(n_test, cls))

    return Corpus(
        name="synthetic_image",
        X_train=np.concatenate(Xtr), y_train=np.concatenate(ytr).astype(np.int64),
        X_test=np.concatenate(Xte), y_test=np.concatenate(yte).astype(np.int64),
        scaling="image",
        meta={"synthetic": True, "n_classes_generated": n_classes,
              "note": "smoke-test corpus; not a scientific result"},
    )
