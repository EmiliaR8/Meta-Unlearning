"""MADAR replay ONLY -- no knowledge distillation, no synaptic intelligence.

The clean ablation of the MADAR condition: the buffer, its budgeting, the
anomaly/inlier split, the optimiser, the iteration count, BatchNorm freezing and
the evaluation are all unchanged. The ONLY difference is the loss, which becomes
plain cross-entropy over the current batch concatenated with a replay batch.

Two things this makes selectable, both material:

  * `--buffer-space raw|latent`. 'raw' fits the Isolation Forest on the scaled
    input features -- MADAR as published, where buffer selection is a property of
    the data. 'latent' fits it on the classifier's representation, so selection
    shifts as the model learns (MADAR-theta, what the main condition does).
    Which space the buffer is chosen in is a method difference, not a detail.

  * `--keep-rnt`. rnt = 1/(t+1) is the convex weight between the CE and KD terms.
    With KD removed there is nothing to weigh it against, so it degenerates into
    a bare scalar shrinking every gradient by 1/(t+1) -- a decaying learning rate,
    not a replay weight. Off by default; enable only to measure that confound.

SI is off by construction here (si_c is forced to 0), so this condition differs
from `madar` in exactly two named components rather than in an unstated mixture.
"""

from __future__ import annotations

from .base import ContinualExperiment, register


@register("er_only")
class ErOnlyExperiment(ContinualExperiment):
    uses_buffer = True
    uses_si = False          # no SI state is even constructed

    def __init__(self, ctx):
        super().__init__(ctx)
        # The shared continual step expects an SI object; give it an inert one so
        # the code path is identical and only the penalty is absent.
        from core import training
        self.si = training.SynapticIntelligence(self.model, si_c=0.0,
                                                eps=self.hp["si_eps"])

    def train_continual_phase(self, tid: int, loader) -> dict:
        from core import training
        info = training.train_continual(
            self.model, self.teacher, loader, self.buffer,
            iters=self.hp["cl_iters"],
            active_count=self.schedule.active_count(tid),
            prev_active=self.schedule.prev_active_count(tid),
            si=self.si, tid=tid, device=self.device, lr=self.hp["cl_lr"],
            momentum=self.hp["momentum"], weight_decay=self.hp["weight_decay"],
            kd_temp=self.hp["kd_temp"], grad_clip=self.hp["grad_clip"],
            batch_size=self.hp["batch_size"], rnt_mode=self.hp["rnt_mode"],
            rnt_floor=self.hp["rnt_floor"], rnt_value=self.hp["rnt_value"],
            use_replay=True, use_kd=False, use_rnt=bool(self.hp["keep_rnt"]),
            freeze_bn=bool(self.hp["freeze_bn"]))
        info["phase"] = "continual"
        return info

    def after_task(self, tid: int, loader, info: dict) -> None:
        # No SI boundary to close; the anchor advance is harmless and keeps the
        # recorded penalty (always 0 here) comparable in shape with `madar`.
        self.si.advance_anchor(self.model)
        self.refresh_teacher()
        buf = self.update_buffer(loader)
        if buf:
            info.setdefault("buffer_update", buf)
