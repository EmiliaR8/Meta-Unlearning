"""STAR + X-DER -- a continual-learning baseline, NOT combined with unlearning.

STAR (Eskandar et al., ICLR 2025) is a plug-in stability regulariser rather than
a method of its own: it adds one term to whatever rehearsal loss it wraps. Here
it wraps X-DER, so this row and the `xder` row share their objective exactly and
differ in the single term

    L_final = L_X-DER(theta) + lambda_STAR * max_delta L_FG(theta, theta + delta)

which is the comparison worth having -- STAR's own contribution -- rather than
two independently-tuned pipelines. The mechanics are in core/star.py.

STAR NEEDS SAMPLES THE MODEL GETS RIGHT. Its loss is defined over the buffer
entries currently classified correctly, so before the memory has been filled
there is nothing to stabilise. On task 0 the buffer is empty and STAR is
inactive by construction; `star_correct_frac` is logged per task so an
inactive regulariser shows up as a number rather than as an unexplained tie
with the `xder` row.

WHAT THE PAPER PAIRS IT WITH. The published combination is STAR + X-DER-RPC --
X-DER with a Regular Polytope Classifier replacing the contrastive future
preparation. RPC is not implemented here, so this is STAR + plain X-DER. Their
Table 8 grid for the RPC pairing (gamma 0.001-0.05, lambda 0.01-0.05) is what
the defaults are drawn from; it was tuned for a different future-head scheme.
"""

from __future__ import annotations

from core import star

from .base import register
from .xder import XDerExperiment


@register("star_xder")
class StarXDerExperiment(XDerExperiment):

    def extra_gradients(self, tid: int, xb, yb) -> dict:
        if self.memory.is_empty() or self.hp["star_lambda"] <= 0:
            return {"star_skipped": 1.0}
        # Not augmented, on purpose. STAR's M* is "the buffer samples the model
        # classifies CORRECTLY right now", and core/star.py commits to that
        # correctness being decided by the same masked-prefix argmax the reported
        # accuracy uses. A random crop would change which samples pass that gate,
        # so the set STAR stabilises would no longer be the set the metric counts.
        _, x, y, _, _ = self.memory.sample(self.hp["batch_size"])
        return star.star_gradient(
            self.model, x.to(self.device), y.to(self.device),
            active_count=self.schedule.active_count(tid),
            lam=self.hp["star_lambda"], gamma=self.hp["star_gamma"],
            eps=self.hp["star_eps"], steps=self.hp["star_steps"])
