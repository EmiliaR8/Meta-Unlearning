"""GCR: Gradient Coreset based Replay Buffer Selection (Tiwari et al., 2022).

Four things, of which the third is the paper's actual contribution:

  1. A CANDIDATE POOL C_t, reservoir-sampled from the current task's stream,
     each entry stored with the logits the model produced at the moment it was
     seen (Alg. 1, lines 8-10).
  2. ADAPTIVE SAMPLING (Alg. 3): each replay draw comes from the candidate pool
     with probability n_t/n -- the share of everything seen so far that belongs
     to the current task -- and from the replay buffer otherwise. Early in a
     task the buffer dominates; as the task's own data accumulates the draw
     shifts toward it.
  3. GRADAPPROX (Alg. 2) at each task boundary: re-select the entire buffer
     from (buffer ++ candidate pool) so that the WEIGHTED subset gradient
     approximates the pool's. See core/coreset.py.
  4. A WEIGHTED loss (Eq. 5): distillation + label + supervised-contrastive
     over the replayed batch, each sample scaled by the weight GradApprox gave
     it. The weights are not optional -- the approximation the subset was
     chosen to satisfy holds only under them.

RELATION TO THE OTHER DER-FAMILY ROWS HERE. Terms (b) and (c) of Eq. 5 are
DER++'s objective, which `xder` already implements. GCR differs in that every
replay term is weighted, in the supervised-contrastive term, and above all in
how the buffer is chosen. Running it beside `xder` therefore isolates the
selection rule rather than the loss.

DEPARTURES, all deliberate and all visible in the record:

  * LAST-LAYER GRADIENTS for the coreset objective, which is what GRAD-MATCH
    (the paper's own reference for this machinery) uses. Named in coreset.py.
  * LOGITS AT ENCOUNTER, not at end of task. Alg. 1 stores z = h(x) inside the
    training loop, so this does too -- which differs from `xder` here, whose
    memory is filled after the task with end-of-task logits. The two
    conventions are not interchangeable and the faithful one is used.
  * SUPCON WITH A LOG. Eq. (6) as printed is a ratio of exponentials with no
    logarithm, which is not a contrastive loss and cannot be what was run; the
    standard supervised-contrastive form is used, with a temperature, which the
    paper does not state. `gcr_temp` is OURS and untuned.
  * ALPHA AND BETA default to this codebase's DER settings rather than the
    paper's CIFAR grid, so that GCR and X-DER differ in selection rather than
    in how hard each replay term is pulled. `gcr_gamma` and `gcr_lambda` are
    ours and untuned.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from core import coreset, training
from core.buffer import CoresetBuffer, LogitBuffer

from .base import ContinualExperiment, register


@register("gcr")
class GCRExperiment(ContinualExperiment):
    uses_buffer = False          # neither the MADAR buffer nor SI
    uses_si = False

    def __init__(self, ctx):
        super().__init__(ctx)
        n_classes = self.schedule.n_classes
        self.memory = CoresetBuffer(mem_size=self.hp["mem_size"],
                                    n_classes=n_classes,
                                    seed=self.cfg["seed"], policy="balanced")
        # The candidate pool is a RESERVOIR over the current task's stream --
        # Alg. 1 line 10 names reservoir sampling specifically, and it is what
        # keeps the pool an unbiased sample of a task whose length is not known
        # in advance.
        self.candidate = LogitBuffer(mem_size=self.hp["mem_size"],
                                     n_classes=n_classes,
                                     seed=self.cfg["seed"] + 1,
                                     policy="reservoir")
        self.si = None
        self._seen_total = 0
        self._seen_task = 0
        self._rng = np.random.default_rng(self.cfg["seed"] + 977)

    # -- optimiser / state -------------------------------------------------
    def _optimizer(self):
        return torch.optim.SGD(self.model.parameters(), lr=self.hp["cl_lr"],
                               momentum=self.hp["momentum"],
                               weight_decay=self.hp["weight_decay"])

    def extra_state(self) -> dict:
        return {"memory": self.memory.state_dict(),
                "candidate": self.candidate.state_dict(),
                "seen_total": self._seen_total, "seen_task": self._seen_task}

    def load_extra_state(self, state: dict) -> None:
        if state.get("memory") is not None:
            self.memory.load_state_dict(state["memory"])
        if state.get("candidate") is not None:
            self.candidate.load_state_dict(state["candidate"])
        self._seen_total = int(state.get("seen_total", 0))
        self._seen_task = int(state.get("seen_task", 0))

    # -- Alg. 3 ------------------------------------------------------------
    def _draw(self, size: int):
        """One replay batch, split between candidate pool and buffer.

        The paper draws per sample with probability p = n_t/n. Drawing the
        whole batch from one Binomial(B, p) is the same thing in expectation,
        at one call instead of B.
        """
        n_mem, n_cand = len(self.memory), len(self.candidate)
        if n_mem == 0 and n_cand == 0:
            return None
        p = (self._seen_task / self._seen_total) if self._seen_total else 1.0
        if n_cand == 0:
            k_cand = 0
        elif n_mem == 0:
            k_cand = size
        else:
            k_cand = int(self._rng.binomial(size, min(max(p, 0.0), 1.0)))

        xs, ys, zs, ws = [], [], [], []
        if k_cand:
            _, x, y, z, _ = self.candidate.sample(k_cand, replace=True)
            xs.append(x); ys.append(y); zs.append(z)
            # Alg. 1 line 11: candidates enter the pool at unit weight.
            ws.append(torch.ones(len(y)))
        k_mem = size - k_cand
        if k_mem and n_mem:
            idx, x, y, z, _ = self.memory.sample(k_mem, replace=True)
            xs.append(x); ys.append(y); zs.append(z)
            ws.append(self.memory.weights_at(idx))
        if not xs:
            return None
        return (torch.cat(xs).to(self.device), torch.cat(ys).to(self.device),
                torch.cat(zs).to(self.device), torch.cat(ws).to(self.device),
                k_cand, size - k_cand)

    # -- Eq. (6) -----------------------------------------------------------
    def _supcon(self, feats: torch.Tensor, labels: torch.Tensor,
                weights: torch.Tensor) -> torch.Tensor:
        """Weighted supervised contrastive loss over the replayed batch.

        Anchors and positives both come from the replayed batch, which is what
        the paper specifies ("data samples from the combined candidate pool and
        replay buffer as anchor points"). A sample whose class is alone in the
        batch has no positive and is skipped rather than contributing a zero --
        averaging in a term that could not be computed would quietly scale the
        loss by how many singletons the draw happened to contain.
        """
        z = F.normalize(feats, dim=1)
        sim = (z @ z.t()) / float(self.hp["gcr_temp"])
        n = len(labels)
        eye = torch.eye(n, dtype=torch.bool, device=sim.device)
        sim = sim.masked_fill(eye, float("-inf"))
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

        same = (labels[:, None] == labels[None, :]) & ~eye
        n_pos = same.sum(1)
        ok = n_pos > 0
        if not bool(ok.any()):
            return feats.sum() * 0.0
        pos_mean = (log_prob.masked_fill(~same, 0.0).sum(1)[ok]
                    / n_pos[ok].clamp(min=1))
        return -(weights[ok] * pos_mean).sum() / weights[ok].sum().clamp(min=1e-8)

    # -- Eq. (5) -----------------------------------------------------------
    def gcr_loss(self, tid: int, xb, yb) -> tuple[torch.Tensor, dict]:
        hp = self.hp
        active = self.schedule.active_count(tid)
        parts = {}

        out_s, feat_s = self.model(xb, return_latent=True)
        loss = F.cross_entropy(out_s[:, :active], yb)     # (a)
        parts["ce_stream"] = float(loss.item())

        draw = self._draw(hp["batch_size"])
        if draw is not None:
            x1, y1, z1, w1, k_cand, k_mem = draw
            out_1, feat_1 = self.model(x1, return_latent=True)
            wn = w1 / w1.sum().clamp(min=1e-8) * len(w1)   # mean-1 scaling

            # (b) weighted distillation against the stored responses. Full
            # width, as stored: truncating would compare a prefix of the live
            # response to all of a recorded one.
            der = (wn * ((out_1 - z1) ** 2).mean(dim=1)).mean()
            loss = loss + hp["gcr_alpha"] * der
            parts["distill"] = float(der.item())

            # (c) weighted label loss
            ce = (wn * F.cross_entropy(out_1[:, :active], y1,
                                       reduction="none")).mean()
            loss = loss + hp["gcr_beta"] * ce
            parts["ce_replay"] = float(ce.item())

            # (d) weighted supervised contrastive
            if hp["gcr_gamma"] > 0:
                sc = self._supcon(feat_1, y1, wn)
                loss = loss + hp["gcr_gamma"] * sc
                parts["supcon"] = float(sc.item())
            parts["draw_candidate"] = float(k_cand)
            parts["draw_memory"] = float(k_mem)

        # Alg. 1 line 10: the stream batch joins the candidate pool with the
        # logits it just produced.
        self.candidate.add(xb.detach().cpu(), yb.detach().cpu(),
                           out_s.detach().cpu(), tid)
        self._seen_total += len(yb)
        self._seen_task += len(yb)
        return loss, parts

    # -- training ----------------------------------------------------------
    def train_task(self, tid: int, loader) -> dict:
        self._seen_task = 0
        self.candidate = LogitBuffer(mem_size=self.hp["mem_size"],
                                     n_classes=self.schedule.n_classes,
                                     seed=self.cfg["seed"] + 1 + tid,
                                     policy="reservoir")
        if tid == 0 and self.hp["gcr_task0"] == "shared":
            steps = training.train_epochs(
                self.model, loader, epochs=self.hp["task0_epochs"],
                lr=self.hp["task0_lr"],
                active_count=self.schedule.active_count(0), device=self.device)
            # Task 0 under the shared protocol trains without the GCR loop, so
            # the candidate pool it would have built does not exist. Fill it
            # from the task data so the boundary solve has a pool to work with.
            X, y = loader.dataset.tensors
            z = training.logits_for(self.model, X, y, self.device,
                                    self.hp["eval_batch_size"])
            self.candidate.add(X, y, z, tid)
            self._seen_total += len(y)
            self._seen_task += len(y)
            return {"grad_steps": steps, "phase": "task0"}
        iters = (max(1, self.hp["task0_epochs"] * len(loader)) if tid == 0
                 else self.hp["cl_iters"])
        return self._loop(tid, loader, iters=iters,
                          phase="task0_gcr" if tid == 0 else "gcr")

    def _loop(self, tid: int, loader, *, iters: int, phase: str) -> dict:
        opt = self._optimizer()
        self.model.train()
        if self.hp["freeze_bn"]:
            training.freeze_batchnorm(self.model)
        it = iter(training.cycle(loader))
        losses, tally = [], {}

        for step in range(iters):
            xb, yb = next(it)
            xb, yb = xb.to(self.device), yb.to(self.device)
            xb = training.augmented(self.model, xb)

            opt.zero_grad()
            # No pre_step / extra_loss / post_step hooks: those are X-DER's own
            # extension points (xder.py:177-196), not the base class's, and
            # every GCR term lives in gcr_loss. Calling them here is what the
            # first real run tripped over.
            loss, parts = self.gcr_loss(tid, xb, yb)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at task {tid}, step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.hp["grad_clip"])
            opt.step()

            losses.append(float(loss.item()))
            for k, v in parts.items():
                tally.setdefault(k, []).append(v)

        info = {"grad_steps": int(iters), "phase": phase,
                "terms": {k: float(np.mean(v)) for k, v in tally.items()},
                "loss_first": losses[0] if losses else None,
                "loss_last": losses[-1] if losses else None,
                "loss_mean": float(np.mean(losses)) if losses else None}
        return info

    # -- Eq. (7): the boundary solve ---------------------------------------
    def after_task(self, tid: int, loader, info: dict) -> None:
        self.refresh_teacher()
        pool = self._pool()
        if pool is None:
            info["coreset"] = {"skipped": "empty pool"}
            return
        X, y, z, t, w = pool
        probs, feats = self._responses(X, y)
        out = coreset.grad_approx_lowrank(
            probs, feats, y.numpy(), w.numpy(),
            budget=self.hp["mem_size"], lam=float(self.hp["gcr_lambda"]))
        idx = torch.as_tensor(out["index"], dtype=torch.long)
        upd = self.memory.replace_with(X[idx], y[idx], z[idx], t[idx],
                                       torch.as_tensor(out["weights"]))
        info["coreset"] = {k: v for k, v in out.items()
                           if k not in ("index", "weights")}
        info["coreset"]["pool_size"] = int(len(y))
        info["memory"] = upd

    def _pool(self):
        """(buffer ++ candidate pool) with their weights. Eq. (7)."""
        parts = []
        if len(self.memory):
            X, y = self.memory.tensors()
            parts.append((X, y, self.memory.logits(), self.memory.task_ids(),
                          self.memory.weights()))
        if len(self.candidate):
            X, y = self.candidate.tensors()
            # Freshly pooled candidates carry unit weight (Alg. 1 line 11).
            parts.append((X, y, self.candidate.logits(),
                          self.candidate.task_ids(), torch.ones(len(y))))
        if not parts:
            return None
        return tuple(torch.cat([p[i] for p in parts]) for i in range(5))

    @torch.no_grad()
    def _responses(self, X, y):
        """Softmax probabilities and penultimate features over the pool."""
        self.model.eval()
        ps, fs = [], []
        bs = int(self.hp["eval_batch_size"])
        for i in range(0, len(y), bs):
            xb = X[i:i + bs].to(self.device)
            logits, feat = self.model(xb, return_latent=True)
            ps.append(torch.softmax(logits, dim=1).cpu().numpy())
            fs.append(feat.cpu().numpy())
        self.model.train()
        return np.concatenate(ps), np.concatenate(fs)
