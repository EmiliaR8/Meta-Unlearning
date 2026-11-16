"""Synaptic Intelligence ONLY -- no replay, no distillation.

Cross-entropy on the current task's batch plus the SI penalty. Pure
regularisation-based continual learning, which is the only setting in which
anything can be attributed to SI on its own.

THE MATCHED CONTROL IS `--si-c 0`. That gives the identical harness -- same
optimiser, same iteration count, same data order, BatchNorm handled the same way
-- with the penalty term removed. The SI-only number is not interpretable
without it, so run both. Note this control is NOT the same as the `naive`
condition, which trains by epochs rather than by a fixed iteration budget; only
the si_c=0 arm is budget-matched to si_only.

A PREDICTION WORTH TESTING. SI has been measured inert on EMBER 2018 -- penalty
magnitude 1e-4 to 1e-5 against a cross-entropy of order 1, at c = 1.0. If that
holds, si_c=1.0 should land on top of si_c=0 to within seed noise. If it does
not, the inertness finding is wrong somewhere. `si_penalty` is recorded per task
so this is measured rather than assumed.

ON THE PATH INTEGRAL. W accumulates from the gradient of the TOTAL loss, which
includes the SI penalty itself; strict Zenke et al. accumulates from the task
loss only. That behaviour is inherited unchanged for comparability with existing
runs, but it matters more here than in `madar`: SI is now the only regulariser,
so its own gradient feeds back into its own importance estimate. Negligible at
c = 1.0 on EMBER 2018; not obviously so at c = 100 (EMBER 2024, LAMDA).
"""

from __future__ import annotations

from core import training

from .base import ContinualExperiment, register


@register("si_only")
class SiOnlyExperiment(ContinualExperiment):
    uses_buffer = False      # no buffer is built at all
    uses_si = True

    def train_continual_phase(self, tid: int, loader) -> dict:
        info = training.train_continual(
            self.model, self.teacher, loader, None,
            iters=self.hp["cl_iters"],
            active_count=self.schedule.active_count(tid),
            prev_active=self.schedule.prev_active_count(tid),
            si=self.si, tid=tid, device=self.device, lr=self.hp["cl_lr"],
            momentum=self.hp["momentum"], weight_decay=self.hp["weight_decay"],
            kd_temp=self.hp["kd_temp"], grad_clip=self.hp["grad_clip"],
            batch_size=self.hp["batch_size"],
            # rnt weighted CE against KD; with neither replay nor KD there is
            # nothing to weigh, so it is dropped rather than left decaying the lr.
            use_replay=False, use_kd=False, use_rnt=bool(self.hp["keep_rnt"]),
            freeze_bn=bool(self.hp["freeze_bn"]))
        info["phase"] = "continual"
        return info

    def after_task(self, tid: int, loader, info: dict) -> None:
        if tid > 0:
            # Recorded so inertness is measured, not assumed.
            info["si_penalty"] = self.si.end_task(self.model)
        else:
            self.si.advance_anchor(self.model)
        self.refresh_teacher()
        # No buffer: nothing to update.
