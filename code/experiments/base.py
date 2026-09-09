"""Experiment base class.

A method differs from the others in exactly one place: how it trains a task.
Everything else -- data, schedule, evaluation over three scopes, confusion
matrices, logging -- is identical by construction and lives here, so a change to
the evaluation protocol cannot land in three conditions and miss the fourth.

Subclasses implement `train_task(tid, loader)` and may override `after_task`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from core import training
from core.buffer import ReplayBuffer
from core.metrics import sparse_confusion
from core.models import build_model, model_info
from core.records import RunRecord

_REGISTRY: dict[str, type] = {}


def register(name: str):
    def wrap(cls):
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return wrap


def available() -> list[str]:
    return sorted(_REGISTRY)


def build_experiment(name: str, ctx):
    if name not in _REGISTRY:
        raise SystemExit(
            f"unknown experiment/method {name!r}. Available: {', '.join(available())}")
    return _REGISTRY[name](ctx)


class Experiment(ABC):
    """One continual-learning run under one method."""

    name = "base"
    uses_buffer = False
    uses_si = False

    def __init__(self, ctx):
        self.ctx = ctx
        self.cfg = ctx.config
        self.hp = ctx.hyperparams
        self.device = ctx.device
        self.schedule = ctx.schedule
        self.model = self.build_model()
        self.teacher = None
        self.si = None
        self.buffer = None
        if self.uses_si:
            self.si = training.SynapticIntelligence(
                self.model, si_c=self.hp["si_c"], eps=self.hp["si_eps"])
        if self.uses_buffer:
            self.buffer = ReplayBuffer(mem_size=self.hp["mem_size"],
                                       contamination=self.hp["contamination"],
                                       seed=self.cfg["seed"],
                                       space=self.hp["buffer_space"])

    # -- interface --------------------------------------------------------
    def build_model(self):
        """The network this method trains. Overridden by methods that do not use
        the shared backbone (MalCL brings a GAN and its own classifier)."""
        net = build_model(self.cfg["model"], self.ctx.input_dim,
                          self.schedule.n_classes, self.device,
                          getattr(self.ctx, "input_shape", None))
        stats = getattr(self.ctx, "channel_stats", None)
        if stats is not None:
            # Task-0 channel statistics, computed in scale_features. Installed
            # here rather than at construction so that a teacher clone and a
            # STAR perturbation -- both of which copy the state_dict -- carry
            # the same normalisation as the live model.
            net.set_norm(*stats)
        return net

    def augment(self, x):
        """The batch as the training step should see it.

        Identity for tabular corpora. For images this is the random crop and
        flip, applied ONCE here so that every forward pass a method runs over
        this batch -- STAR's clean and perturbed pair, X-DER's implant, DEDUCE's
        two detector gradients -- sees the same pixels. Re-drawing inside those
        would turn each comparison into a measurement of augmentation noise.
        """
        aug = getattr(self.model, "augment", None)
        if aug is None or not self.model.training:
            return x
        return aug(x)

    @abstractmethod
    def train_task(self, tid: int, loader) -> dict:
        """Train on task `tid`. Returns at least {'grad_steps': int}."""

    def after_task(self, tid: int, loader, info: dict) -> None:
        """Post-task bookkeeping: teacher refresh, SI boundary, buffer update."""

    # -- shared machinery -------------------------------------------------
    def task_loader(self, tid: int, drop_last: bool = True):
        return training.make_loader(
            self.ctx.X_train, self.ctx.y_train, self.schedule.classes_for(tid),
            batch_size=self.hp["batch_size"], drop_last=drop_last)

    def model_info(self) -> dict:
        return model_info(self.cfg["model"], self.ctx.input_dim,
                          self.schedule.n_classes,
                          getattr(self.ctx, "input_shape", None))

    def refresh_teacher(self) -> None:
        self.teacher = training.clone_teacher(self.model)

    def update_buffer(self, loader) -> dict:
        """Fold a loader's dataset into the replay buffer."""
        if self.buffer is None or loader is None:
            return {}
        X, y = loader.dataset.tensors
        latents = training.latents_for(self.model, X, y, self.device,
                                       self.hp["eval_batch_size"])
        return self.buffer.update(X, y, latents)

    def evaluate_all(self, tid: int) -> tuple[dict, dict]:
        """The three scopes. Returns (scope -> metrics, sparse confusion of seen)."""
        active = self.schedule.active_count(tid)
        scopes = {
            "seen": self.schedule.seen_classes(tid),
            "recent": self.schedule.classes_for(tid),
            "task0": self.schedule.classes_for(0),
        }
        out, seen_cm = {}, None
        for label, classes in scopes.items():
            metrics, cm = training.evaluate(
                self.model, self.ctx.X_test, self.ctx.y_test, classes, active,
                self.device, self.schedule.n_classes, self.hp["eval_batch_size"])
            out[label] = metrics
            if label == "seen":
                seen_cm = cm
        return out, seen_cm

    # -- driver -----------------------------------------------------------
    def run(self, record: RunRecord) -> RunRecord:
        record.model_info = self.model_info()
        n_tasks = self.ctx.n_tasks
        for tid in range(n_tasks):
            loader = self.task_loader(tid)
            if loader is None:
                record.note(f"task {tid}: no training samples; skipped")
                continue
            n_train = len(loader.dataset)
            info = self.train_task(tid, loader)
            self.after_task(tid, loader, info)

            scopes, cm = self.evaluate_all(tid)
            confusion = None
            if self.cfg.get("confusion", "sparse") == "sparse" and cm is not None:
                confusion = sparse_confusion(cm)

            extra = {k: v for k, v in info.items() if k != "grad_steps"}
            if self.buffer is not None:
                extra["buffer"] = {"size": len(self.buffer),
                                   "n_families": len(self.buffer.family_buffers)}
            record.add_task(tid, active_count=self.schedule.active_count(tid),
                            n_train=n_train, grad_steps=info["grad_steps"],
                            scopes=scopes, confusion=confusion, extra=extra)

            if not self.cfg.get("quiet"):
                m = scopes["seen"]
                print(f"  task {tid:>2} | acc {m['accuracy']:6.2f} "
                      f"macro-F1 {m['macro_f1']:6.2f} | "
                      f"recent {scopes['recent']['accuracy']:6.2f} | "
                      f"task0 {scopes['task0']['accuracy']:6.2f}", flush=True)
        return record


class ContinualExperiment(Experiment):
    """Shared shape of the replay-based methods: task 0 by epochs, then a
    fixed-iteration continual phase against a frozen teacher."""

    uses_buffer = True
    uses_si = True

    def train_task(self, tid: int, loader) -> dict:
        if tid == 0:
            steps = training.train_epochs(
                self.model, loader, epochs=self.hp["task0_epochs"],
                lr=self.hp["task0_lr"], active_count=self.schedule.active_count(0),
                device=self.device)
            return {"grad_steps": steps, "phase": "task0"}
        return self.train_continual_phase(tid, loader)

    def train_continual_phase(self, tid: int, loader) -> dict:
        info = training.train_continual(
            self.model, self.teacher, loader, self.buffer,
            iters=self.hp["cl_iters"],
            active_count=self.schedule.active_count(tid),
            prev_active=self.schedule.prev_active_count(tid),
            si=self.si, tid=tid, device=self.device, lr=self.hp["cl_lr"],
            momentum=self.hp["momentum"], weight_decay=self.hp["weight_decay"],
            kd_temp=self.hp["kd_temp"], grad_clip=self.hp["grad_clip"],
            batch_size=self.hp["batch_size"], rnt_mode=self.hp["rnt_mode"],
            rnt_floor=self.hp["rnt_floor"], rnt_value=self.hp["rnt_value"])
        info["phase"] = "continual"
        return info

    def after_task(self, tid: int, loader, info: dict) -> None:
        if tid > 0:
            info["si_penalty"] = self.si.end_task(self.model)
        else:
            self.si.advance_anchor(self.model)
        self.refresh_teacher()
        buf = self.update_buffer(loader)
        if buf:
            info.setdefault("buffer_update", buf)
