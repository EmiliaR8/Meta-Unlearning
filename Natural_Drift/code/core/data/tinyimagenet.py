"""Tiny ImageNet 200 -- the project's one non-malware corpus.

200 classes, 500 training images each (100,000), 50 labelled validation images
each (10,000), 64x64 RGB. The distributed `test/` split has no labels, so VAL IS
USED AS THE TEST SET, which is what every continual-learning paper on this
benchmark does; calling it "test" in the logs keeps one vocabulary across
corpora, and this note is the record that it is really val.

WHY A CACHE. The source is 110,000 individual JPEGs. Decoding them takes minutes
and would be repeated by every run in a sweep -- 27 runs, three seeds, two
schedules -- for a result that never changes. The first load decodes once and
writes a uint8 .npy per split to the cache root; later loads read that back
directly. Delete the cache files to force a re-decode.

WHY uint8. As uint8 the training split is 100000*3*64*64 = 1.2 GB. As float32 it
is 4.9 GB, and the replay buffer, the teacher's inputs and each batch would all
be taken from that copy. The arrays therefore stay uint8 from disk to the GPU
and are converted per batch inside the model's forward, where the cost is one
kernel over a 32-image batch instead of a 4.9 GB allocation.

GREYSCALE IMAGES. A handful of Tiny ImageNet files are single-channel. They are
converted with PIL's "RGB" mode, which replicates the channel; the alternative
-- dropping them -- would silently change the per-class counts that the task
schedule depends on.
"""

from __future__ import annotations

import numpy as np

from .base import Corpus

DIRNAME = "tiny-imagenet-200"
IMAGE_SHAPE = (3, 64, 64)


def _root(paths):
    root = paths.data_root / DIRNAME
    if not root.exists():
        raise SystemExit(
            f"Tiny ImageNet not found at {root}.\n"
            f"Download and unzip it there:\n"
            f"  cd {paths.data_root}\n"
            f"  wget http://cs231n.stanford.edu/tiny-imagenet-200.zip\n"
            f"  unzip -q tiny-imagenet-200.zip")
    return root


def _wnids(root) -> list[str]:
    """Class order is wnids.txt as distributed, NOT directory listing order.

    Directory order varies by filesystem, and a label space that depends on the
    machine would make two servers' runs silently incomparable while every
    recorded config claimed they matched.
    """
    f = root / "wnids.txt"
    if not f.exists():
        raise SystemExit(f"{f} missing; the archive looks incomplete")
    ids = [ln.strip() for ln in f.read_text().splitlines() if ln.strip()]
    if len(ids) != 200:
        raise SystemExit(f"{f} lists {len(ids)} classes, expected 200")
    return ids


def _decode(files) -> np.ndarray:
    from PIL import Image

    out = np.empty((len(files),) + IMAGE_SHAPE, dtype=np.uint8)
    for i, f in enumerate(files):
        with Image.open(f) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        out[i] = arr.transpose(2, 0, 1)
    return out


def _build_train(root, wnids):
    files, labels = [], []
    for idx, wnid in enumerate(wnids):
        d = root / "train" / wnid / "images"
        if not d.is_dir():                       # some mirrors omit images/
            d = root / "train" / wnid
        got = sorted(p for p in d.iterdir() if p.suffix.upper() == ".JPEG")
        if not got:
            raise SystemExit(f"no JPEGs for class {wnid} under {d}")
        files.extend(got)
        labels.extend([idx] * len(got))
    return _decode(files), np.asarray(labels, dtype=np.int64)


def _build_val(root, wnids):
    ann = root / "val" / "val_annotations.txt"
    if not ann.exists():
        raise SystemExit(f"{ann} missing; cannot label the validation split")
    index = {w: i for i, w in enumerate(wnids)}
    files, labels = [], []
    for line in ann.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        name, wnid = parts[0], parts[1]
        if wnid not in index:
            raise SystemExit(f"{ann} references unknown class {wnid}")
        files.append(root / "val" / "images" / name)
        labels.append(index[wnid])
    if len(files) != 10000:
        raise SystemExit(f"{ann} lists {len(files)} images, expected 10,000")
    return _decode(files), np.asarray(labels, dtype=np.int64)


def _cached(paths, split: str, build):
    """Decode once, then read the .npy back into memory on later runs.

    NOT memory-mapped, deliberately, though it was at first. Two reasons, and
    the first is a correctness one: `np.load(mmap_mode="r")` returns a read-only
    array, and `torch.from_numpy` on a read-only array warns that writes to the
    resulting tensor are undefined behaviour. Nothing here writes to it today,
    but that is a property of the current call sites rather than of the data, and
    it is not a warning to leave standing in a training pipeline.

    The second is speed: training and replay both index this array in random
    order, which is the access pattern a memory map is worst at -- every batch
    would fault in scattered pages. Resident, the training split is 1.2 GB, which
    a machine that can hold a ResNet-18's activations can spare.
    """
    xs = paths.cache_root / f"tinyimagenet_{split}_x.npy"
    ys = paths.cache_root / f"tinyimagenet_{split}_y.npy"
    if xs.exists() and ys.exists():
        return np.load(xs), np.load(ys)
    X, y = build()
    paths.cache_root.mkdir(parents=True, exist_ok=True)
    np.save(xs, X)
    np.save(ys, y)
    return np.asarray(X), y


def load(paths, **kw) -> Corpus:
    root = _root(paths)
    wnids = _wnids(root)
    X_train, y_train = _cached(paths, "train", lambda: _build_train(root, wnids))
    X_test, y_test = _cached(paths, "val", lambda: _build_val(root, wnids))
    return Corpus(
        name="tinyimagenet",
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        scaling="image",
        # NOT preselected, despite wnids.txt already fixing the class order.
        # Marking it preselected skips the shared selection step and only checks
        # the label range, which pins the corpus to all 200 classes: any shorter
        # schedule -- a 60-class pilot, say -- was rejected outright. The shared
        # path reproduces wnids order here anyway, because every class has
        # exactly 500 training images and `select_families` breaks ties on
        # ascending original id, so the top N are classes 0..N-1 in wnids order.
        # Letting it run costs nothing and buys every sub-schedule.
        preselected=False,
        meta={"classes": 200, "image_shape": list(IMAGE_SHAPE),
              "test_split": "val (the distributed test split is unlabelled)",
              "source": str(root)})
