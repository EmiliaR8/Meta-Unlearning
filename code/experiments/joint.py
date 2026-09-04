"""Joint retraining. The oracle upper bound.

At every task boundary the model is discarded and retrained from scratch on all
data seen so far. Not a continual method -- it violates the premise by keeping
every sample -- but it is the ceiling, and the denominator of any
headroom-normalised statistic.
"""

from __future__ import annotations

from core import training
from core.models import build_model

from .base import Experiment, register


@register("joint")
class JointExperiment(Experiment):
    uses_buffer = False
    uses_si = False

    def task_loader(self, tid: int, drop_last: bool = True):
        # Cumulative, not per-task: everything seen up to and including `tid`.
        return training.make_loader(
            self.ctx.X_train, self.ctx.y_train, self.schedule.seen_classes(tid),
            batch_size=self.hp["batch_size"], drop_last=drop_last)

    def train_task(self, tid: int, loader) -> dict:
        # Fresh weights each task -- that is the whole point of the condition.
        self.model = build_model(self.cfg["model"], self.ctx.input_dim,
                                 self.schedule.n_classes, self.device)
        steps = training.train_epochs(
            self.model, loader, epochs=self.hp["task0_epochs"],
            lr=self.hp["task0_lr"],
            active_count=self.schedule.active_count(tid),
            device=self.device, label=f"joint task {tid}")
        return {"grad_steps": steps, "phase": "joint"}
