"""Per-task checkpoints, so an interrupted run resumes instead of restarting.

A run is a sequence of tasks, and the expensive, irreplaceable thing is the
tasks already finished. Task 0 on Tiny ImageNet under 100+10x10 is 100 classes
x 500 images x 50 epochs -- the majority of a run's wall clock -- and MalCL on
EMBER 2018 spends over eight hours on it. Before this, a run killed at task 7 of
11 restarted at task 0 and did all of that again.

WHAT IS SAVED. Everything the driver loop needs to continue: the model, the
frozen teacher, the replay buffer, the Synaptic Intelligence accumulators, each
method's own state (a GAN and its optimisers, a Fisher diagonal, a reservoir),
the record of tasks already evaluated, and the RNG streams. A resumed run is
meant to be the run that would have happened, not a similar one.

WHAT IS NOT. Mid-task progress. The unit is the task boundary, because that is
where every method already has a consistent state and where the record gains an
entry; checkpointing inside a task would mean capturing an optimiser mid-epoch
and a dataloader mid-shuffle for a fraction of a task's worth of savings.

CORRECTNESS OVER CONVENIENCE. A checkpoint is only ever loaded into the run that
wrote it. The file records the run id AND the full config hash, and a mismatch
is refused rather than adapted: silently resuming a run whose hyperparameters
changed would produce a result that is half one configuration and half another,
reported under a single id, and no amount of disk saving is worth that.

ATOMICITY. Written to a temporary file in the same directory and renamed over
the target. A checkpoint half-written when the machine died -- which is exactly
when checkpoints matter -- would otherwise be indistinguishable from a good one
and would poison the resume it exists to enable.
"""

from __future__ import annotations

import os
import random
import uuid
from pathlib import Path

import numpy as np
import torch

SCHEMA = 1


def path_for(store, run_id: str) -> Path:
    return store.checkpoint_dir() / f"{run_id}.pt"


# ------------------------------------------------------------------ RNG
def rng_state() -> dict:
    """The three streams this project draws from.

    CUDA state is captured only when CUDA has been initialised; asking for it
    otherwise initialises the context as a side effect, which turns a
    checkpoint write on a CPU run into a GPU allocation.
    """
    state = {"python": random.getstate(),
             "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> str | None:
    """Restore the streams. Returns a note when the restore was partial.

    THE DEVICE COUNT IS NOT PORTABLE. `get_rng_state_all` returns one generator
    state per visible CUDA device, and `set_rng_state_all` assumes the machine
    reading it has at least as many. It does not: a checkpoint written on a
    two-GPU host and resumed on a one-GPU host raised IndexError deep inside
    torch, killing a run that was otherwise perfectly resumable. The same
    happens with CUDA_VISIBLE_DEVICES narrowed between the two runs, which the
    meta driver does routinely when it pins candidates to GPUs.

    So restore what this machine can hold and say so. Cross-machine bit
    reproducibility was never on offer anyway -- different GPUs run different
    kernels -- and a partial RNG restore is a far smaller departure than
    refusing to resume at all. The note goes in the run record rather than
    being swallowed, because a resumed run that drew from a different random
    position is not quite the run it stands in for.
    """
    if not state:
        return None
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_cpu_byte(state["torch"]))
    saved = state.get("cuda")
    if not saved or not torch.cuda.is_available():
        return None
    have = torch.cuda.device_count()
    if have == 0:
        return (f"checkpoint carried {len(saved)} CUDA RNG state(s); this run "
                f"sees no CUDA device, so none were restored")
    n = min(len(saved), have)
    for i in range(n):
        torch.cuda.set_rng_state(_cpu_byte(saved[i]), i)
    if len(saved) != have:
        return (f"checkpoint carried {len(saved)} CUDA RNG state(s) but this "
                f"machine has {have}; restored {n}. Results will not match the "
                f"machine that wrote it bit-for-bit")
    return None


def _cpu_byte(t) -> torch.Tensor:
    """torch.set_rng_state demands a CPU ByteTensor; a checkpoint loaded with
    map_location can hand back something else."""
    return torch.as_tensor(t, dtype=torch.uint8, device="cpu")


# ------------------------------------------------------------------ io
def save(store, run_id: str, config_hash: str, *, tid: int, experiment,
         record) -> Path:
    target = path_for(store, run_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA,
        "run_id": run_id,
        "config_hash": config_hash,
        "last_task": int(tid),
        "experiment": experiment.state_dict(),
        # `elapsed` carries the wall clock accumulated so far. Without it a
        # resumed run restarts its timer, and every per-task elapsed_s after the
        # resume -- plus the run's wall_time_s -- silently under-reports by the
        # duration of everything before the interruption. Timing is a reported
        # quantity here, so it has to survive the resume like any other.
        "record": {"per_task": record.per_task, "notes": record.notes,
                   "elapsed": record.elapsed_s()},
        "rng": rng_state(),
    }
    # Unique per writer, for the reason spelled out in core/task0.py's save():
    # a shared temp name turns two concurrent writers into one corrupt file and
    # one FileNotFoundError. Resume checkpoints are keyed per run so the race is
    # rarer here, but "rarer" is not a property worth relying on.
    tmp = target.with_name(f"{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, target)      # atomic within a filesystem
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return target


def load(store, run_id: str, config_hash: str, device="cpu"):
    """The checkpoint for this run, or None. Refuses a mismatched one loudly."""
    target = path_for(store, run_id)
    if not target.exists():
        return None
    try:
        payload = torch.load(target, map_location=device, weights_only=False)
    except Exception as e:                                   # truncated, etc.
        print(f"  ! checkpoint {target.name} unreadable ({e}); starting fresh")
        return None
    if payload.get("schema") != SCHEMA:
        print(f"  ! checkpoint {target.name} has schema {payload.get('schema')}, "
              f"expected {SCHEMA}; starting fresh")
        return None
    if payload.get("config_hash") != config_hash:
        raise SystemExit(
            f"checkpoint {target} was written by a run with a different "
            f"configuration ({payload.get('config_hash')} vs {config_hash}).\n"
            f"Resuming it would splice two configurations into one result. "
            f"Delete the file to start this configuration fresh.")
    return payload


def clear(store, run_id: str) -> None:
    """Drop the checkpoint once the run has written its record."""
    target = path_for(store, run_id)
    # The glob catches temp files left by a writer that was killed mid-save,
    # which now carry a pid and a token rather than one fixed name.
    for p in [target, *target.parent.glob(f"{target.name}.*.tmp")]:
        try:
            p.unlink()
        except FileNotFoundError:
            pass
