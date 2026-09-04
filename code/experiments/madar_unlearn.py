"""MADAR + unlearning.

After the continual phase of each task, a fraction of the task's samples is
selected as a forget set. The model is then trained so that

    L = alpha * KL(uniform || p over the new-task logits)     [forget]
      + (1 - alpha) * (0.5 * CE + 0.5 * KD)  on retain + replay
      + si_c * SI

and the replay buffer is rebuilt from the RETAIN data only, so forgotten samples
never replay again.

alpha = 0 IS THE CONTROL, and is why the argument works. At alpha = 0 the forget
set is still selected, still excluded from the buffer rebuild, and the same
number of gradient steps still runs -- only the push toward uniform is gone. The
difference between alpha > 0 and alpha = 0 therefore separates the value of
CURATION (which samples are dropped) from the value of the FORGET OBJECTIVE
itself. Reporting the method without this arm would attribute to unlearning what
may simply be memory curation.

The SI anchor is deliberately NOT advanced before this phase: `p_old` still
points at the start of the task, so unlearning is penalised for dragging
important parameters away from where the task began, except through the forget
term. The teacher is the pre-unlearning model, so the retain anchor is "what you
believed a moment ago", not the previous task's model.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from core import training, selectors
from core.seeding import child_seed

from .base import ContinualExperiment, register


@register("madar_unlearn")
class MadarUnlearnExperiment(ContinualExperiment):

    def train_task(self, tid: int, loader) -> dict:
        info = super().train_task(tid, loader)
        if tid == 0:
            return info
        # SI boundary must close BEFORE unlearning so omega already contains this
        # task's contribution; the anchor stays at the start of the task.
        info["si_penalty"] = self.si.end_task(self.model, advance_anchor=False)
        info["_si_closed"] = True

        retain_loader, forget_loader, sel = self.select_forget_set(tid, loader)
        info["selection"] = sel
        if forget_loader is None:
            return info

        pre = training.clone_teacher(self.model)
        steps = self.unlearn(tid, forget_loader, retain_loader)
        info["grad_steps"] += steps
        info["unlearn_steps"] = steps
        info["unlearn_efficacy"] = self.measure(pre, forget_loader, retain_loader, tid)
        # The buffer is rebuilt from retain data only.
        info["_retain_loader"] = retain_loader
        return info

    def after_task(self, tid: int, loader, info: dict) -> None:
        if tid > 0 and not info.pop("_si_closed", False):
            info["si_penalty"] = self.si.end_task(self.model)
        elif tid == 0:
            self.si.advance_anchor(self.model)
        else:
            self.si.advance_anchor(self.model)   # anchor advances after unlearning
        self.refresh_teacher()
        buf = self.update_buffer(info.pop("_retain_loader", None) or loader)
        if buf:
            info.setdefault("buffer_update", buf)

    # -- selection --------------------------------------------------------
    def select_forget_set(self, tid: int, loader):
        X, y = loader.dataset.tensors
        n = len(y)
        name = self.cfg["selector"]
        ratio = self.hp["forget_ratio"]
        seed = child_seed(self.cfg["seed"], tid, 7)

        kw = {"n": n, "ratio": ratio, "seed": seed,
              "contamination": self.hp["contamination"]}
        if name == "donut":
            kw["latents"] = training.latents_for(self.model, X, y, self.device,
                                                 self.hp["eval_batch_size"])
        elif name == "leftover":
            budget = self.hp["mem_size"] // max(
                1, len(self.buffer.family_buffers) + len(np.unique(y.numpy())))
            kw.update({"features": X.numpy(), "labels": y.numpy(),
                       "budget_per_family": budget})

        forget_idx, retain_idx = selectors.split(name, **kw)
        summary = {"selector": name, "ratio": float(ratio), "n_candidates": int(n),
                   "n_forget": int(len(forget_idx)), "n_retain": int(len(retain_idx))}
        if len(forget_idx) == 0:
            summary["skipped"] = "empty forget set at this ratio"
            return loader, None, summary

        fams, counts = np.unique(y.numpy()[forget_idx], return_counts=True)
        summary["forget_per_family"] = dict(zip(fams.tolist(), counts.tolist()))

        bs = self.hp["batch_size"]
        forget_loader = data.DataLoader(
            data.TensorDataset(X[forget_idx], y[forget_idx]), batch_size=bs, shuffle=True)
        retain_loader = data.DataLoader(
            data.TensorDataset(X[retain_idx], y[retain_idx]), batch_size=bs, shuffle=True)
        return retain_loader, forget_loader, summary

    # -- the forget phase -------------------------------------------------
    def unlearn(self, tid: int, forget_loader, retain_loader) -> int:
        alpha = self.hp["alpha"]
        active = self.schedule.active_count(tid)
        prev_active = self.schedule.prev_active_count(tid)
        teacher = training.clone_teacher(self.model)

        self.model.train()
        training.freeze_batchnorm(self.model)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.hp["unlearn_lr"])
        mask = training.logit_mask(self.model.fc_last.out_features, active, self.device)

        retain_iter = iter(training.cycle(retain_loader))
        buf_loader = self.buffer.loader(self.hp["batch_size"]) if self.buffer else None
        buf_iter = iter(training.cycle(buf_loader)) if buf_loader else None
        steps = 0

        for _ in range(self.hp["unlearn_epochs"]):
            for fx, _fy in forget_loader:
                fx = fx.to(self.device)
                opt.zero_grad()

                # forget: flatten the NEW-task logit slice toward uniform
                cur = self.model(fx)[:, prev_active:active]
                log_p = F.log_softmax(cur, dim=1)
                uniform = torch.ones_like(log_p) / max(1, active - prev_active)
                forget_loss = F.kl_div(log_p, uniform, reduction="batchmean")

                # retain anchor: ground truth + distillation on retain + replay
                rx, ry = next(retain_iter)
                rx, ry = rx.to(self.device), ry.to(self.device)
                if buf_iter is not None:
                    mx, my = next(buf_iter)
                    rx = torch.cat([rx, mx.to(self.device)])
                    ry = torch.cat([ry, my.to(self.device)])
                with torch.no_grad():
                    t_logits = teacher(rx)[:, :active]
                s_logits = self.model(rx)
                retain_loss = (0.5 * nn.CrossEntropyLoss()(s_logits + mask, ry)
                               + 0.5 * training.kd_loss(s_logits[:, :active],
                                                        t_logits, self.hp["kd_temp"]))

                loss = (alpha * forget_loss + (1.0 - alpha) * retain_loss
                        + self.si.si_c * self.si.penalty(self.model))
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss unlearning task {tid}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.hp["unlearn_grad_clip"])
                opt.step()
                steps += 1
        return steps

    @torch.no_grad()
    def measure(self, before, forget_loader, retain_loader, tid: int) -> dict:
        """Did the forget set actually move, and did the retain set survive?"""
        active = self.schedule.active_count(tid)

        def scope(loader):
            self.model.eval(); before.eval()
            acc_b = acc_a = total = 0
            ents, shifts = [], []
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                lb, latb = before(xb, return_latent=True)
                la, lata = self.model(xb, return_latent=True)
                lb, la = lb[:, :active], la[:, :active]
                probs = F.softmax(la, dim=1)
                acc_b += int((torch.argmax(lb, 1) == yb).sum())
                acc_a += int((torch.argmax(la, 1) == yb).sum())
                ents.append(-(probs * torch.log(probs + 1e-9)).sum(1).cpu())
                shifts.append(torch.norm(lata - latb, p=2, dim=1).cpu())
                total += int(yb.numel())
            if not total:
                return {}
            ent = torch.cat(ents); sh = torch.cat(shifts)
            return {"acc_before": acc_b / total * 100, "acc_after": acc_a / total * 100,
                    "entropy": float(ent.mean()),
                    "entropy_norm": float(ent.mean() / np.log(max(2, active))),
                    "latent_shift_mean": float(sh.mean()),
                    "latent_shift_max": float(sh.max())}

        return {"forget": scope(forget_loader), "retain": scope(retain_loader)}
