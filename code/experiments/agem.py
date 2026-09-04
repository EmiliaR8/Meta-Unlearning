"""A-GEM -- Averaged Gradient Episodic Memory (Chaudhry et al., ICLR 2019).

NOT a rehearsal method, despite keeping a memory. Memory samples never enter the
training batch: the loss is cross-entropy on the current batch alone. The memory
is used only to compute a CONSTRAINT. Each step:

    g      = grad of CE on the current batch
    g_ref  = grad of CE on a reference batch drawn from episodic memory
    if <g, g_ref> < 0:                       # the update would hurt past tasks
        g <- g - (<g, g_ref> / <g_ref, g_ref>) * g_ref
    step with g

The projection is the minimum-norm change to g that removes the negative
component along g_ref, so the update no longer increases the average loss on
memory to first order. When the gradients already agree the step is untouched --
A-GEM constrains rather than regularises, and does nothing when there is no
conflict. `projection_rate` is logged per task so how often the constraint
actually binds is measured rather than assumed.

One constraint on the AVERAGE memory gradient, not GEM's per-task quadratic
program: that is the whole point of A-GEM, and why it costs one extra backward
pass rather than a QP over t constraints.

MEMORY POLICY. Reservoir sampling over the stream, at the same `mem_size` as the
MADAR buffer. A-GEM's own policy, so the two conditions differ in WHAT they
store rather than in how much -- pairing A-GEM's constraint with MADAR's
diversity-aware selection would report a hybrid under A-GEM's name.
`--buffer-policy madar` does exactly that hybrid if you want it as a separate
row, but it is not the default and is not A-GEM as published.

NO distillation, NO regulariser, NO unlearning. The comparison against MADAR is
therefore replay-and-distil versus gradient-projection at equal memory budget.

MOMENTUM WEAKENS THE GUARANTEE -- read this before quoting a result. A-GEM's
claim is about the GRADIENT: after projection, <g, g_ref> >= 0, so the step does
not increase the average memory loss to first order. The harness runs its
continual phase with SGD momentum 0.9, and with momentum the applied update is a
running average of past projected gradients rather than the projected gradient
itself. Each was non-conflicting at its own parameter point; their accumulation
need not be. Chaudhry et al. use plain SGD.

Momentum is kept at the harness default so this row is budget- and
optimiser-comparable with every other row in the table. Pass `--momentum 0.0`
for the faithful arm; it is recorded and aggregates as a separate condition, so
running both costs nothing but compute and settles whether it matters here.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from core import training
from core.buffer import ReservoirBuffer

from .base import ContinualExperiment, register


@register("agem")
class AGemExperiment(ContinualExperiment):
    uses_buffer = False      # the shared MADAR buffer is not constructed
    uses_si = False

    def __init__(self, ctx):
        super().__init__(ctx)
        policy = self.hp["buffer_policy"]
        if policy == "reservoir":
            self.memory = ReservoirBuffer(mem_size=self.hp["mem_size"],
                                          seed=self.cfg["seed"])
        elif policy == "madar":
            from core.buffer import ReplayBuffer
            self.memory = ReplayBuffer(mem_size=self.hp["mem_size"],
                                       contamination=self.hp["contamination"],
                                       seed=self.cfg["seed"],
                                       space=self.hp["buffer_space"])
        else:
            raise SystemExit(f"unknown --buffer-policy {policy!r}")
        self.policy = policy
        self._rng = np.random.default_rng(self.cfg["seed"] + 977)
        # An inert SI object: the shared continual step signature expects one,
        # and si_c = 0 makes the penalty identically zero.
        self.si = training.SynapticIntelligence(self.model, si_c=0.0,
                                                eps=self.hp["si_eps"])

    # -- gradient helpers -------------------------------------------------
    def _flat_grad(self) -> torch.Tensor:
        return torch.cat([(p.grad if p.grad is not None
                           else torch.zeros_like(p)).reshape(-1)
                          for p in self.model.parameters() if p.requires_grad])

    def _write_grad(self, flat: torch.Tensor) -> None:
        i = 0
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            n = p.numel()
            p.grad = flat[i:i + n].view_as(p).clone()
            i += n

    # -- the task ---------------------------------------------------------
    def train_continual_phase(self, tid: int, loader) -> dict:
        active = self.schedule.active_count(tid)
        opt = torch.optim.SGD(self.model.parameters(), lr=self.hp["cl_lr"],
                              momentum=self.hp["momentum"],
                              weight_decay=self.hp["weight_decay"])
        self.model.train()
        if self.hp["freeze_bn"]:
            training.freeze_batchnorm(self.model)
        mask = training.logit_mask(self.model.fc_last.out_features, active,
                                   self.device)
        it = iter(training.cycle(loader))
        ce = nn.CrossEntropyLoss()
        iters = self.hp["cl_iters"]
        losses, n_projected, dots = [], 0, []

        for _ in range(iters):
            xb, yb = next(it)
            xb, yb = xb.to(self.device), yb.to(self.device)

            g_ref = None
            if not self.memory.is_empty():
                mx, my = self._reference_batch()
                if mx is not None:
                    opt.zero_grad()
                    ce(self.model(mx.to(self.device)) + mask,
                       my.to(self.device)).backward()
                    g_ref = self._flat_grad().clone()

            opt.zero_grad()
            loss = ce(self.model(xb) + mask, yb)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at task {tid}")
            loss.backward()
            losses.append(float(loss.item()))

            if g_ref is not None:
                g = self._flat_grad()
                dot = float(torch.dot(g, g_ref))
                dots.append(dot)
                if dot < 0:
                    denom = float(torch.dot(g_ref, g_ref))
                    if denom > 0:
                        self._write_grad(g - (dot / denom) * g_ref)
                        n_projected += 1

            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.hp["grad_clip"])
            opt.step()

        return {"grad_steps": int(iters), "phase": "agem",
                "buffer_policy": self.policy,
                "projections": n_projected,
                # How often the constraint actually bound. If this is ~0, A-GEM
                # reduced to naive fine-tuning and any difference from `naive`
                # is noise, not method.
                "projection_rate": n_projected / max(1, iters),
                "grad_dot_mean": float(np.mean(dots)) if dots else None,
                "loss_first": losses[0] if losses else None,
                "loss_last": losses[-1] if losses else None,
                "loss_mean": float(np.mean(losses)) if losses else None}

    def _reference_batch(self):
        """A fresh draw every step, as A-GEM specifies.

        The reference gradient is meant to be a stochastic estimate of the
        average memory gradient. Re-seeding the generator per call would return
        the SAME indices at every step, turning that estimate into one fixed
        subset and quietly changing the method -- so the generator is created
        once and advanced.
        """
        n = self.hp["batch_size"]
        if self.policy == "reservoir":
            return self.memory.sample(n)
        X, y = self.memory.tensors()
        if X is None:
            return None, None
        idx = self._rng.choice(len(y), min(n, len(y)), replace=False)
        return X[idx], y[idx]

    def after_task(self, tid: int, loader, info: dict) -> None:
        X, y = loader.dataset.tensors
        if self.policy == "reservoir":
            upd = self.memory.add_stream(X, y)
        else:
            latents = training.latents_for(self.model, X, y, self.device,
                                           self.hp["eval_batch_size"])
            upd = self.memory.update(X, y, latents)
        info["memory"] = upd
        # No teacher and no SI boundary: A-GEM has neither.
