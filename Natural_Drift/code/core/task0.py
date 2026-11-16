"""A task-0 checkpoint shared across runs that differ only after task 0.

This is what makes meta-training affordable. Evolution Strategies evaluates a
population per generation -- 16 candidates x 40 generations is 640 inner runs --
and every one of them begins by training exactly the same task 0, because the
scorer under test does not act until task 1. On EMBER 2018 task 0 was about 80%
of a run's wall clock. Paying it 640 times to learn nothing is the difference
between a meta-run that fits in a week and one that does not.

`--protocol meta` has named this arrangement since the runner was written; until
now it was a label with nothing behind it. This is the implementation.

WHAT KEYS THE CACHE. Only the inputs task 0 actually has: the corpus, the task
schedule, the model, the seed, and the hyperparameters that shape either the
task-0 fit or the buffer built from it. Deliberately NOT the continual or
unlearning knobs, and emphatically not `scorer_weights` -- that is the whole
point, since those are what differ between candidates. A knob that changes task
0 and is missing from this list would silently share a checkpoint between two
different task-0 fits, so the list errs toward including things.

WHY ITS OWN KEY AND NOT core/checkpoint.py's. A resume checkpoint is keyed to
one run's exact configuration, which is correct for resuming and useless here:
every candidate has a different `scorer_weights` and therefore a different hash.
The two mechanisms answer different questions and share only the state format.

PROTOCOL IS A HARD BOUNDARY, still. A stage1 run trains its own task 0 per seed;
a meta run shares one. Their accuracies are not comparable and the aggregator
already refuses to difference across them. Nothing here weakens that -- the
cache is only consulted under `--protocol meta`.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

import torch

from core import checkpoint as ckpt

SCHEMA = 1

# Hyperparameters that reach task 0: the fit itself, the data preparation, and
# the buffer construction that follows it.
TASK0_KEYS = ("task0_epochs", "task0_lr", "batch_size", "eval_batch_size",
              "feature_clip", "min_family_samples", "family_cap", "cap_seed",
              "mem_size", "contamination", "buffer_space", "buffer_policy",
              "logit_buffer_policy",
              # Carving a validation slice removes rows from TRAIN, so task 0
              # is fitted on different data. Omit these and two runs at
              # different val fractions would share one cached task 0.
              "val_fraction", "val_seed")

# Configuration fields that reach task 0. `method` is included because methods
# do not all train task 0 the same way -- MalCL runs its GAN from the start, and
# joint retrains from scratch -- so a checkpoint is shareable only within one.
# `perm_seed` is here because a permutation changes which families land in task
# 0 -- a genuinely different task-0 fit, not the same one relabelled. Omit it
# and every permutation of a corpus would silently share one cached checkpoint,
# which is precisely the failure this cache is otherwise designed to avoid.
CONFIG_KEYS = ("dataset", "task_setup", "model", "seed", "method", "perm_seed")


def key_for(config: dict, hyperparams: dict, length: int = 12) -> str:
    payload = {
        "config": {k: config.get(k) for k in CONFIG_KEYS},
        "hyperparams": {k: hyperparams.get(k) for k in TASK0_KEYS},
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:length]


def path_for(store, key: str) -> Path:
    return store.cache_root / "task0" / f"{key}.pt"


def load(store, key: str, device="cpu"):
    """The cached task 0, or None. A damaged file is reported and ignored."""
    target = path_for(store, key)
    if not target.exists():
        return None
    try:
        payload = torch.load(target, map_location=device, weights_only=False)
    except Exception as e:
        print(f"  ! task-0 cache {target.name} unreadable ({e}); training it")
        return None
    if payload.get("schema") != SCHEMA:
        return None
    return payload


def save(store, key: str, *, experiment, task_entry: dict) -> Path:
    """Write atomically: a half-written cache is worse than none, because every
    later candidate would read it."""
    target = path_for(store, key)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": SCHEMA, "key": key,
               "experiment": experiment.state_dict(),
               "task_entry": task_entry,
               # The RNG matters as much as the weights. Training task 0 draws
               # from the global streams -- dataloader shuffling, optimiser
               # init -- and a run that LOADS task 0 makes none of those draws.
               # Without restoring the state, tasks 1..N begin from a different
               # random position than they would have, and a cached run stops
               # being the run it is standing in for. Measured: 75.164 vs
               # 75.311 on a five-task smoke before this was saved.
               "rng": ckpt.rng_state()}
    # A UNIQUE temp name per writer, not a shared `<key>.pt.tmp`. Sharing the
    # key is the entire purpose of this cache, so concurrent writers to one key
    # are the normal case, not a corner: the first meta-test generation launches
    # five runs at a new perm_seed at once, all miss, all train task 0, and all
    # save. With one temp name they overwrite each other's partial file and
    # whichever renames second dies on a path that is no longer there. The
    # payloads are equivalent -- same config, same seed, same deterministic fit
    # -- so last-writer-wins is correct once each writer has its own file.
    tmp = target.with_name(f"{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return target
