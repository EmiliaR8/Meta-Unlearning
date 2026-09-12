"""The run record: one JSON per run, written to Logs/<group>/jsons/.

Design rules, each of which exists because of a specific way results go wrong:

  * SELF-DESCRIBING. Every record carries the full configuration that produced
    it. The aggregator never infers anything from a filename, so renaming a file
    cannot change what it means.

  * RUN ID FROM CONFIG. The filename is derived from the configuration, so two
    runs differing in any recorded parameter cannot collide, and a rerun of the
    same configuration lands on the same name. That is what makes a sweep
    resumable without a cache silently serving one dataset's result for another.

  * PROTOCOL IS RECORDED AND LOAD-BEARING. `protocol` distinguishes runs that
    retrain task 0 per seed ('stage1') from runs that share one task-0
    checkpoint ('meta'). Accuracies from the two are NOT comparable, and the
    aggregator refuses to difference across the boundary. This is the single
    error most likely to inflate a headline number.

  * ENVIRONMENT AND GIT STATE. A number you cannot attribute to a commit is a
    number you cannot defend. `git_dirty` records uncommitted changes rather
    than pretending the commit describes the code that ran.

Schema version is bumped when a field changes meaning, never when one is added.
"""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1


def _git(*args, cwd) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def environment(repo_root) -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform(),
           "hostname": socket.gethostname()}
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["gpu_count"] = torch.cuda.device_count()
    except Exception:
        env["torch"] = None
    for mod in ("numpy", "sklearn"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None
    commit = _git("rev-parse", "HEAD", cwd=repo_root)
    status = _git("status", "--porcelain", cwd=repo_root)
    env["git_commit"] = commit
    env["git_branch"] = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo_root)
    env["git_dirty"] = bool(status) if status is not None else None
    return env


def config_hash(config: dict, length: int = 8) -> str:
    """Stable hash over the configuration, excluding presentational fields."""
    ignore = {"group", "notes", "tag", "out", "quiet", "paths"}
    payload = {k: v for k, v in sorted(config.items()) if k not in ignore}
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:length]


def make_run_id(config: dict, overrides: dict | None = None) -> str:
    """<dataset>_<setup>_<method>_<model>_s<seed>_<hash>

    The readable prefix is for humans scanning a directory; the hash covers the
    config plus any hyperparameter that DIFFERS from the dataset's defaults.

    Hashing overrides rather than the full parameter set is what makes the id
    stable when a new knob is added: a parameter left at its default does not
    participate, so runs that predate the knob keep their identity. The cost is
    that changing a DEFAULT would let an old and a new run collide on one name --
    which run.py catches by comparing the stored record before skipping.
    """
    def clean(v):
        return "".join(c if c.isalnum() else "-" for c in str(v)).strip("-")

    parts = [clean(config.get("dataset", "ds")),
             clean(config.get("task_setup", "setup")),
             clean(config.get("method", "method")),
             clean(config.get("model", "model")),
             f"s{config.get('seed', 0)}"]
    if config.get("tag"):
        parts.append(clean(config["tag"]))
    payload = dict(config)
    if overrides:
        payload["_hp"] = {k: overrides[k] for k in sorted(overrides)}
    return "_".join(parts) + "_" + config_hash(payload)


def _jsonable(obj):
    """numpy scalars/arrays are not JSON-serialisable; convert rather than fail."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON-serialisable: {type(obj)}")


@dataclass
class RunRecord:
    config: dict
    paths: dict
    model_info: dict
    dataset_info: dict
    schedule_info: dict
    hyperparams: dict
    run_id: str = ""
    schema_version: int = SCHEMA_VERSION
    per_task: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    env: dict = field(default_factory=dict)
    started_utc: str = ""
    _t0: float = field(default_factory=time.time, repr=False)

    def __post_init__(self):
        if not self.run_id:
            self.run_id = make_run_id(self.config)
        if not self.started_utc:
            self.started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # -- accumulation -----------------------------------------------------
    def add_task(self, tid: int, *, active_count: int, n_train: int,
                 grad_steps: int, scopes: dict, confusion=None,
                 extra: dict | None = None) -> None:
        """`scopes` maps scope name -> metrics dict from metrics_from_confusion."""
        entry = {"task": int(tid), "active_count": int(active_count),
                 "n_train": int(n_train), "grad_steps": int(grad_steps),
                 "elapsed_s": round(time.time() - self._t0, 2),
                 "metrics": scopes}
        if confusion is not None:
            entry["confusion_matrix"] = confusion
        if extra:
            entry.update(extra)
        self.per_task.append(entry)

    def note(self, message: str) -> None:
        self.notes.append(message)

    # -- wall clock -------------------------------------------------------
    def elapsed_s(self) -> float:
        return time.time() - self._t0

    def adopt_elapsed(self, seconds: float) -> None:
        """Continue an earlier run's clock rather than starting a new one.

        Used when resuming from a checkpoint: the timer is wound back so that
        elapsed_s() counts from the original start, which is what the per-task
        elapsed_s column and wall_time_s are documented to mean.
        """
        self._t0 = time.time() - float(seconds)

    # -- derived ----------------------------------------------------------
    def _series(self, scope: str, key: str = "accuracy") -> list:
        return [t["metrics"].get(scope, {}).get(key) for t in self.per_task]

    def summary(self) -> dict:
        """Flat, aggregation-friendly view. The scalars a table would quote."""
        def stats(scope, key="accuracy"):
            vals = [v for v in self._series(scope, key) if v is not None]
            if not vals:
                return {}
            return {"mean": float(np.mean(vals)), "final": float(vals[-1])}

        out = {}
        for scope in ("seen", "recent", "task0"):
            for key in ("accuracy", "macro_f1", "macro_precision", "macro_recall",
                        "weighted_f1"):
                s = stats(scope, key)
                if s:
                    out[f"{scope}_{key}_mean"] = s["mean"]
                    out[f"{scope}_{key}_final"] = s["final"]
        out["n_tasks_completed"] = len(self.per_task)
        out["total_grad_steps"] = int(sum(t["grad_steps"] for t in self.per_task))
        out["wall_time_s"] = round(time.time() - self._t0, 2)
        return out

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "started_utc": self.started_utc,
            "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": self.config,
            "hyperparams": self.hyperparams,
            "dataset": self.dataset_info,
            "task_setup": self.schedule_info,
            "model": self.model_info,
            "environment": self.env,
            "paths": self.paths,
            # accuracy_per_task / on the most recent task / on task 0, hoisted
            # out of per_task so a plot does not have to walk the nested form.
            "curves": {
                "accuracy_seen": self._series("seen"),
                "accuracy_recent": self._series("recent"),
                "accuracy_task0": self._series("task0"),
                "macro_f1_seen": self._series("seen", "macro_f1"),
                "macro_precision_seen": self._series("seen", "macro_precision"),
                "macro_recall_seen": self._series("seen", "macro_recall"),
            },
            "summary": self.summary(),
            "per_task": self.per_task,
            "notes": self.notes,
        }

    def write(self, directory) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        out = directory / f"{self.run_id}.json"
        tmp = out.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(self.to_dict(), f, indent=2, default=_jsonable)
        tmp.replace(out)          # atomic: a killed run leaves no half-file
        return out
