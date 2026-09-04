"""Determinism. One implementation, so conditions cannot drift apart.

The stage-1 scripts each carry a copy of this block; identical today, but
identical-by-copy is how the rest of that duplication decayed.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def child_seed(seed: int, *parts: int) -> int:
    """A derived seed that is stable in its inputs.

    Used where a sub-decision must not consume the global RNG stream -- e.g. the
    random selector at task t. Deriving instead of drawing means adding a
    selector cannot shift every later draw in the run, which would silently
    de-pair two conditions that are supposed to share a seed.
    """
    h = seed & 0xFFFFFFFF
    for p in parts:
        h = (h * 1_000_003 + int(p)) & 0xFFFFFFFF
    return h
