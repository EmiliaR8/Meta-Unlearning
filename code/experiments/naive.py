"""Naive sequential fine-tuning. The lower bound.

No replay, no distillation, no regularisation: each task is trained on its own
data only. Whatever accuracy the other methods retain above this line is what
their machinery bought.
"""

from __future__ import annotations

from core import training

from .base import Experiment, register


@register("naive")
class NaiveExperiment(Experiment):
    uses_buffer = False
    uses_si = False

    def train_task(self, tid: int, loader) -> dict:
        steps = training.train_epochs(
            self.model, loader, epochs=self.hp["task0_epochs"],
            lr=self.hp["task0_lr"],
            active_count=self.schedule.active_count(tid),
            device=self.device, label=f"task {tid}")
        return {"grad_steps": steps, "phase": "naive"}
