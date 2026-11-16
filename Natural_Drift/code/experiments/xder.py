"""X-DER -- eXtended Dark Experience Replay (Boschini et al., TPAMI 2023).

A rehearsal baseline that is NOT combined with unlearning and does not use the
MADAR buffer. It replays stored model RESPONSES rather than distilling from a
frozen teacher, and it does three things DER++ does not (Eq. 12):

    L_X-DER = L_DER + L_S-CE + L_F

  L_DER    alpha * ||stored_logits - f(x)||^2 over a memory batch (Eq. 5).
  L_S-CE   cross-entropy with the softmax RESTRICTED to the current task's
           columns for stream examples, so past heads take no negative gradient
           from new data; ordinary softmax over all seen classes, weighted beta,
           for memory examples (Eq. 10, 13).
  L_F      lambda * future preparation (Eq. 9) + eta * the past/future
           constraint (Eq. 11).

plus the memory update of Eq. (7): the current task's slice of the live response
is implanted into the stored logits of replayed entries, rescaled to stay below
the ground-truth logit already in memory. In the paper's own ablation this is
the single most load-bearing component (-7.3 points at buffer 500 on CIFAR-100);
`--xder-implant false` turns it off and reproduces "X-DER w/o memory update".

WHAT DIFFERS FROM THE PUBLISHED METHOD, all of it forced by this setting:

1. THE SECOND VIEW. Eq. (8) needs strong data augmentation, which does not exist
   for a static malware feature vector. See core/xder.py for the substitute and
   its knobs. `--xder-lambda 0` removes the term rather than guessing.

2. TASK 0. Every other continual row here trains task 0 by the shared epoch
   schedule, and a method whose first task differs is not comparable at the
   continual phase. `--xder-task0 shared` (default) keeps that; `--xder-task0
   full` runs the X-DER objective from task 0, which is the published behaviour
   -- there the buffer terms are simply absent, since the buffer is empty.

3. BUFFER INSERTION AT TASK BOUNDARIES, class-balanced, at the same `mem_size`
   as the MADAR buffer. The published method inserts by reservoir sampling
   during the stream. Task-boundary insertion is what every other row in this
   table does, so holding it fixed keeps the comparison about the objective;
   `--logit-buffer-policy reservoir` is the other arm.

4. HYPERPARAMETERS. The X-DER paper's grid is in supplemental material that is
   not in the copy on hand. alpha/beta defaults are taken from the range the
   STAR paper reports for DER++ (alpha 0.15-0.4, beta 0.1-0.4, Table 8);
   gamma = 0.75 and margin = 0.3 are stated in the main text. lambda, eta and
   the contrastive temperature are OURS and have not been tuned on this corpus.
   Treat them as a starting point, not as the paper's numbers.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from core import training, xder as X
from core.buffer import LogitBuffer

from .base import ContinualExperiment, register


@register("xder")
class XDerExperiment(ContinualExperiment):
    uses_buffer = False      # the MADAR buffer is not constructed
    uses_si = False

    def __init__(self, ctx):
        super().__init__(ctx)
        self.memory = LogitBuffer(mem_size=self.hp["mem_size"],
                                  n_classes=self.schedule.n_classes,
                                  seed=self.cfg["seed"],
                                  policy=self.hp["logit_buffer_policy"])
        self.si = None

    # -- optimiser --------------------------------------------------------
    def _optimizer(self):
        return torch.optim.SGD(self.model.parameters(), lr=self.hp["cl_lr"],
                               momentum=self.hp["momentum"],
                               weight_decay=self.hp["weight_decay"])

    # -- the objective ----------------------------------------------------
    # -- resumable state --------------------------------------------------
    def extra_state(self) -> dict:
        return {"memory": self.memory.state_dict()}

    def load_extra_state(self, state: dict) -> None:
        if state.get("memory") is not None:
            self.memory.load_state_dict(state["memory"])

    def xder_loss(self, tid: int, xb, yb) -> tuple[torch.Tensor, dict]:
        """Eq. (12) for one step. Returns (loss, diagnostics).

        Also carries out the Eq. (7) implant on the entries it replayed, which
        is why the memory draw used for L_DER is the one whose row indices are
        kept: those are the rows whose stored responses are refreshed.
        """
        lo, hi, n_classes = X.head_bounds(self.schedule, tid)
        hp = self.hp
        parts = {}

        out_s = self.model(xb)
        # Eq. (10): softmax over the current task's columns only.
        loss = F.cross_entropy(out_s[:, lo:hi], yb - lo)
        parts["s_ce_stream"] = float(loss.item())

        f_logits, f_labels, f_inputs = out_s, yb, xb
        draw = self.memory.sample(hp["batch_size"]) if not self.memory.is_empty() else None
        if draw is not None:
            idx1, x1, y1, z1, _ = draw
            x1, y1, z1 = x1.to(self.device), y1.to(self.device), z1.to(self.device)
            # DELIBERATELY NOT AUGMENTED, and this is a departure from Mammoth
            # worth stating. z1 are the logits stored for THIS example; the DER
            # term (Eq. 5) is a squared distance between them and the live
            # response, and the implant (Eq. 7) writes the live response back
            # into that same entry. Feeding a fresh random crop makes both
            # operations compare responses to two different images, so the
            # distance picks up augmentation noise and the implanted value
            # describes a view the buffer does not hold. Mammoth re-augments
            # here and absorbs that noise; the buffer's augmented contribution
            # to feature learning is supplied instead by x2 below, which is
            # augmented and carries the cross-entropy term.
            out_1 = self.model(x1)

            der = hp["xder_alpha"] * F.mse_loss(out_1, z1)
            loss = loss + der
            parts["der"] = float(der.item())

            _, x2, y2, _, _ = self.memory.sample(hp["batch_size"])
            x2, y2 = x2.to(self.device), y2.to(self.device)
            x2 = training.augmented(self.model, x2)
            buf_ce = hp["xder_beta"] * F.cross_entropy(self.model(x2)[:, :hi], y2)
            loss = loss + buf_ce
            parts["s_ce_buffer"] = float(buf_ce.item())

            f_logits = torch.cat([out_s, out_1])
            f_labels = torch.cat([yb, y1])
            f_inputs = torch.cat([xb, x1])

            if hp["xder_implant"]:
                refreshed = X.implant_future_past(
                    z1, out_1.detach(), y1, lo, hi, hp["xder_gamma"])
                self.memory.update_logits(idx1, refreshed)
                parts["implanted"] = int(idx1.shape[0])

        if hp["xder_lambda"] > 0:
            # On an image corpus the second view is a real crop-and-flip, which
            # is what Eq. (8) was written for. The Gaussian/dropout substitute
            # documented in core/xder.py exists only because the malware corpora
            # have no such transform; using it here as well would throw away the
            # one corpus that can answer whether the substitution was adequate.
            img_aug = getattr(self.model, "augment", None)
            if img_aug is not None:
                aug = img_aug(f_inputs)
            else:
                aug = X.feature_augment(f_inputs, hp["xder_aug"],
                                        hp["xder_aug_sigma"], hp["xder_aug_drop"])
            fp = hp["xder_lambda"] * X.future_prep_loss(
                torch.cat([f_logits, self.model(aug)]),
                torch.cat([f_labels, f_labels]),
                self.schedule, tid, hp["xder_temp"])
            loss = loss + fp
            parts["future_prep"] = float(fp.item())

        if hp["xder_eta"] > 0:
            pfc = hp["xder_eta"] * X.past_future_penalty(
                f_logits, f_labels, lo, hi, hp["xder_margin"])
            loss = loss + pfc
            parts["pf_constraint"] = float(pfc.item())

        return loss, parts

    # -- extension points -------------------------------------------------
    # Everything layered on top of X-DER in this project enters through one of
    # these four, in the order the step runs them. Keeping them named means
    # STAR and DEDUCE do not each fork a copy of the training loop, and the
    # X-DER row and the STAR row therefore share one objective by construction.

    def pre_step(self, tid: int, xb, yb) -> dict:
        """Before anything is computed for this batch. May MODIFY parameters --
        DEDUCE's local unlearning step runs here."""
        return {}

    def extra_gradients(self, tid: int, xb, yb) -> dict:
        """Called with p.grad empty, before the CL backward, so an implementation
        may zero and rewrite gradients freely; whatever it leaves in p.grad is
        summed with the CL gradient. This is the `u <- lambda*grad L_STAR;
        u <- u + grad L_CL` of STAR's Algorithm 1."""
        return {}

    def extra_loss(self, tid: int, xb, yb):
        """An additive term inside the CL backward, or None."""
        return None

    def post_step(self, tid: int, xb, yb) -> dict:
        """After the optimiser step. DEDUCE's global unlearning runs here."""
        return {}

    # -- the task ---------------------------------------------------------
    def train_task(self, tid: int, loader) -> dict:
        if tid == 0 and self.hp["xder_task0"] == "shared":
            steps = training.train_epochs(
                self.model, loader, epochs=self.hp["task0_epochs"],
                lr=self.hp["task0_lr"], active_count=self.schedule.active_count(0),
                device=self.device)
            return {"grad_steps": steps, "phase": "task0"}
        if tid == 0:
            return self._loop(tid, loader, iters=self._task0_iters(loader),
                              phase="task0_xder")
        return self._loop(tid, loader, iters=self.hp["cl_iters"], phase=self.name)

    def _task0_iters(self, loader) -> int:
        """`--xder-task0 full` matches the SHARED task-0 budget in gradient
        steps, not in epochs, so the two arms differ in objective and not in how
        much training task 0 received."""
        return max(1, self.hp["task0_epochs"] * len(loader))

    def _loop(self, tid: int, loader, *, iters: int, phase: str) -> dict:
        opt = self._optimizer()
        self.model.train()
        if self.hp["freeze_bn"]:
            training.freeze_batchnorm(self.model)
        it = iter(training.cycle(loader))
        losses, tally = [], {}
        extra_tally = {}

        for step in range(iters):
            xb, yb = next(it)
            xb, yb = xb.to(self.device), yb.to(self.device)
            # One view of the stream batch, drawn here and reused by every term
            # and every extra gradient below. Identity on the tabular corpora.
            xb = training.augmented(self.model, xb)

            opt.zero_grad()
            _note(extra_tally, self.pre_step(tid, xb, yb))
            _note(extra_tally, self.extra_gradients(tid, xb, yb))

            loss, parts = self.xder_loss(tid, xb, yb)
            side = self.extra_loss(tid, xb, yb)
            if side is not None:
                loss = loss + side
                parts["extra_loss"] = float(side.item())
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at task {tid}, step {step}")
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.hp["grad_clip"])
            opt.step()
            _note(extra_tally, self.post_step(tid, xb, yb))

            losses.append(float(loss.item()))
            for k, v in parts.items():
                tally.setdefault(k, []).append(v)

        info = {"grad_steps": int(iters), "phase": phase,
                "terms": {k: float(np.mean(v)) for k, v in tally.items()},
                "loss_first": losses[0] if losses else None,
                "loss_last": losses[-1] if losses else None,
                "loss_mean": float(np.mean(losses)) if losses else None}
        for k, v in extra_tally.items():
            info[k] = float(np.mean(v)) if v else None
        return info

    # -- boundaries -------------------------------------------------------
    def after_task(self, tid: int, loader, info: dict) -> None:
        if self.hp["xder_implant"]:
            info["implant_sweep"] = self.sweep_implant(tid)
        X_task, y_task = loader.dataset.tensors
        z = training.logits_for(self.model, X_task, y_task, self.device,
                                self.hp["eval_batch_size"])
        info["memory"] = self.memory.add(X_task, y_task, z, tid)
        self.model.train()
        if self.hp["freeze_bn"]:
            training.freeze_batchnorm(self.model)

    @torch.no_grad()
    def sweep_implant(self, tid: int) -> int:
        """Eq. (7) applied once more over the WHOLE memory at the task boundary.

        The paper applies the update both during a task and at the end of it.
        During the task only replayed rows are reached, and with a buffer larger
        than the number of steps times the batch some rows are never drawn --
        those would keep responses that predate the current task entirely.
        """
        if self.memory.is_empty() or self.hp["xder_alpha"] < 0:
            return 0
        lo, hi, _ = X.head_bounds(self.schedule, tid)
        X_mem, y_mem = self.memory.tensors()
        z_mem, t_mem = self.memory.logits(), self.memory.task_ids()
        rows = (t_mem < tid).nonzero(as_tuple=True)[0]
        if not len(rows):
            return 0
        self.model.eval()
        live = []
        bs = self.hp["eval_batch_size"]
        for i in range(0, len(rows), bs):
            chunk = rows[i:i + bs]
            live.append(self.model(X_mem[chunk].to(self.device)).cpu())
        live = torch.cat(live)
        refreshed = X.implant_future_past(z_mem[rows], live, y_mem[rows], lo, hi,
                                          self.hp["xder_gamma"])
        return self.memory.update_logits(rows, refreshed)


def _note(tally: dict, values) -> None:
    """Fold a hook's diagnostics into the per-task running tally.

    Hooks report per-step scalars (a projection count, whether unlearning fired)
    and the task record wants their mean. A hook that returns nothing is the
    common case and costs nothing.
    """
    if not values:
        return
    for k, v in values.items():
        if v is None:
            continue
        tally.setdefault(k, []).append(float(v))
