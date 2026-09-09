"""DEDUCE wrapped around STAR + X-DER -- a continual-learning baseline.

This is the row DEDUCE's Table 1 calls `STAR(XDER) w/ OUR(G)`. It is NOT
combined with this project's unlearning: DEDUCE's unlearning is its own, aimed
at negative transfer between tasks, and it neither takes a forget set nor uses
the donut selector.

The stack, outermost first:

    DEDUCE      detect interference per batch, unlearn locally when found (LUM),
                reclaim dead units continuously (GUM)
    STAR        add the stability-perturbation gradient
    X-DER       the rehearsal objective and the (x, y, logits) memory

Because each layer enters through a named hook on the X-DER training step, the
three rows `xder`, `star_xder` and `deduce_star_xder` share one loop and one
objective. A difference between them is the layer that was added, which is the
only reading under which the ablation means anything.

ORDER WITHIN A STEP, which is Algorithm 3 read literally:

    1. detect      gradient conflict on this batch (or, once per task, the
                   transferability bound)
    2. LUM         if conflict: theta <- theta + delta F^-1 [alpha*prox - grad CE]
    3. STAR        p.grad <- lambda * grad L_STAR at the perturbed point
    4. learn       p.grad += grad [ L_X-DER + beta * (theta - theta_{t-1})^T F (...) ]
    5. GUM         reinitialise the lowest-contribution mature neurons

FIVE THINGS TO KNOW BEFORE QUOTING A NUMBER FROM THIS ROW.

* The paper gives an algorithm and no code. core/deduce.py marks four points
  where a faithful implementation still had to choose: the loss used by the
  detector, the Fisher variant, how the bound-based arm gets its target error,
  and what happens to BatchNorm statistics when a neuron is reinitialised.

* Its hyperparameters do not transfer. DEDUCE's published setting is ResNet-18,
  buffer 500, batch 32, 50 epochs per task on CIFAR. delta = 0.001,
  phi = 1e-5 and epsilon = 0.0 are carried over verbatim; alpha, beta, eta,
  the maturity threshold and k are NOT stated in the paper and are ours.

* GUM does not run during task 0 under `--xder-task0 shared`, because that phase
  is the shared epoch-based routine every other row uses. `--xder-task0 full`
  puts task 0 through this loop as well.

* COST. Detection is two extra backward passes per batch and STAR is two more,
  on top of X-DER's four forwards. `--deduce-detect-every N` thins the detector
  if that matters; N = 1 is the paper.

* `lum_fired` and `gum_resets` are logged per task. If `lum_fired` is ~0 the
  detector never triggered and this row is `star_xder` with extra compute; if
  it is ~1 the detector fires on every batch and "detection" is not selecting
  anything. Neither is visible from accuracy alone.
"""

from __future__ import annotations

import torch

from core import deduce as D
from core import training

from .base import register
from .star_xder import StarXDerExperiment


@register("deduce_star_xder")
class DeduceStarXDerExperiment(StarXDerExperiment):

    def __init__(self, ctx):
        super().__init__(ctx)
        self.fisher = D.DiagonalFisher(self.model)
        self.lum = D.LocalUnlearning(self.hp)
        self.gum = D.GlobalUnlearning(self.model, self.hp, seed=self.cfg["seed"])
        self.prev_task_params: dict = {}     # theta_{t-1}, Eq. (11)
        self.anchor: dict = {}               # theta_t^k, Eq. (9)
        self._step = 0
        self._conflict = 0.0                 # last decision, held between checks
        self._bound_note: dict = {}

    # -- 1. detect / 2. unlearn locally ------------------------------------
    def pre_step(self, tid: int, xb, yb) -> dict:
        out = {}
        every = max(1, int(self.hp["deduce_detect_every"]))
        active = self.schedule.active_count(tid)

        if self.hp["deduce_detect"] == "gradient" and not self.memory.is_empty():
            if self._step % every == 0:
                # Not augmented. The detector compares the gradient on the
                # current batch against the gradient on memory and calls a
                # negative inner product interference. Augmenting one side adds
                # variance to a quantity whose SIGN is the entire decision, and
                # the epsilon threshold has no way to tell that variance from
                # real conflict. The stream side (xb) arrives already augmented
                # from the training loop, so this is deliberately asymmetric:
                # the question is whether learning the augmented batch actually
                # being trained on conflicts with the memory as stored.
                _, mx, my, _, _ = self.memory.sample(self.hp["batch_size"])
                out = D.gradient_conflict(
                    self.model, (xb, yb), (mx.to(self.device), my.to(self.device)),
                    active, float(self.hp["deduce_epsilon"]))
                self._conflict = out["conflict"]
            else:
                out = {"conflict": self._conflict}

        if self._conflict and not self.fisher.is_empty():
            out.update(self.lum.step(self.model, xb, yb, fisher=self.fisher,
                                     anchor=self.anchor, active_count=active))
        else:
            out["lum_fired"] = 0.0

        self._step += 1
        if self._step == int(self.hp["deduce_k"]):
            # theta_t^k: the proximal term must not pull back to a point BEFORE
            # this task, or LUM would be undoing the current task as fast as it
            # is learned.
            self.anchor = {n: p.detach().clone()
                           for n, p in self.model.named_parameters()
                           if p.requires_grad}
        self.gum.arm()
        return out

    # -- 4. the learning-step regulariser ----------------------------------
    def extra_loss(self, tid: int, xb, yb):
        """beta * (theta - theta_{t-1})^T F (theta - theta_{t-1}), Eq. (11):
        steer new learning onto parameters the previous tasks did not rely on."""
        if not self.prev_task_params or self.fisher.is_empty():
            return None
        beta = float(self.hp["deduce_beta"])
        if beta <= 0:
            return None
        return beta * self.fisher.quadratic(self.model, self.prev_task_params)

    # -- 5. reclaim capacity ----------------------------------------------
    def post_step(self, tid: int, xb, yb) -> dict:
        return self.gum.step(self.model, self.fisher)

    # -- task boundaries ---------------------------------------------------
    def train_task(self, tid: int, loader) -> dict:
        self._step = 0
        self._conflict = 0.0
        self.anchor = {}
        if tid > 0 and self.hp["deduce_detect"] == "bound":
            self._conflict, self._bound_note = self._bound_decision(tid, loader)
        self.gum.attach(self.model)
        try:
            info = super().train_task(tid, loader)
        finally:
            self.gum.detach()
        if self._bound_note:
            info["deduce_bound"] = dict(self._bound_note)
        info["fisher"] = {"batches": self.fisher.n_batches}
        return info

    def after_task(self, tid: int, loader, info: dict) -> None:
        super().after_task(tid, loader, info)
        self.prev_task_params = {n: p.detach().clone()
                                 for n, p in self.model.named_parameters()
                                 if p.requires_grad}
        info["fisher_update"] = self._update_fisher(tid, loader)

    def _update_fisher(self, tid: int, loader) -> dict:
        """Algorithm 3, line 21. Estimated over the task's data AND the buffer,
        so F describes what the model relies on across everything seen rather
        than on the newest families alone."""
        batches = list(loader)
        if not self.memory.is_empty():
            X_mem, y_mem = self.memory.tensors()
            mem_loader = training.make_loader(
                X_mem, y_mem, sorted(set(int(v) for v in y_mem)),
                batch_size=self.hp["batch_size"])
            if mem_loader is not None:
                batches += list(mem_loader)
        return self.fisher.estimate(
            self.model, batches, self.schedule.active_count(tid), self.device,
            max_batches=int(self.hp["deduce_fim_batches"]))

    # -- OUR(B) ------------------------------------------------------------
    def _bound_decision(self, tid: int, loader) -> tuple[float, dict]:
        """Eq. (5): fires LUM for the whole task when the new task's error
        exceeds the transferability bound the previous tasks support."""
        if self.memory.is_empty():
            return 0.0, {}
        prev_active = self.schedule.prev_active_count(tid)
        X_mem, y_mem = self.memory.tensors()

        source_err = self._error_on(X_mem, y_mem, prev_active)
        X_task, y_task = loader.dataset.tensors
        eb = self.hp["eval_batch_size"]
        div = D.domain_divergence(
            training.latents_for(self.model, X_mem, y_mem, self.device, eb),
            training.latents_for(self.model, X_task, y_task, self.device, eb),
            seed=self.cfg["seed"])
        leep = D.leep_score(self.model, X_task, y_task, self.device, prev_active,
                            self.schedule.classes_for(tid))
        bound = (source_err + abs(1.0 - 2.0 * div)
                 + float(self.hp["deduce_bound_c"]) * abs(leep))
        target_err = D.probe_target_error(self, tid, loader)
        note = {"source_error": source_err, "domain_clf_error": div,
                "leep": leep, "bound": bound, "target_error": target_err}
        return float(target_err > bound), note

    @torch.no_grad()
    def _error_on(self, X, y, active_count: int) -> float:
        self.model.eval()
        wrong = total = 0
        bs = self.hp["eval_batch_size"]
        for i in range(0, len(y), bs):
            pred = self.model(X[i:i + bs].to(self.device))[:, :active_count]
            wrong += int((pred.argmax(dim=1).cpu() != y[i:i + bs]).sum())
            total += len(y[i:i + bs])
        self.model.train()
        if self.hp["freeze_bn"]:
            training.freeze_batchnorm(self.model)
        return wrong / max(1, total)
