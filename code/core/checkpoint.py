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


def set_rng_state(state: dict) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_cpu_byte(state["torch"]))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_cpu_byte(s) for s in state["cuda"]])


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
    tmp = target.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, target)          # atomic within a filesystem
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
    for p in (path_for(store, run_id), path_for(store, run_id).with_suffix(".pt.tmp")):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
