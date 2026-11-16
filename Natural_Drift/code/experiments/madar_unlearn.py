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

from core import nn1, training, selectors
from core.nn1_features import task_features
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
        # The buffer is rebuilt from retain data only -- or, under the
        # complement pool, from MADAR's own keep-set.
        info["_retain_loader"] = retain_loader
        prot = getattr(self, "_protected_idx", None)
        if prot is not None:
            X, y = loader.dataset.tensors
            info["_buffer_loader"] = data.DataLoader(
                data.TensorDataset(X[prot], y[prot]),
                batch_size=self.hp["batch_size"])
            self._protected_idx = None
        return info

    def after_task(self, tid: int, loader, info: dict) -> None:
        if tid > 0 and not info.pop("_si_closed", False):
            info["si_penalty"] = self.si.end_task(self.model)
        elif tid == 0:
            self.si.advance_anchor(self.model)
        else:
            self.si.advance_anchor(self.model)   # anchor advances after unlearning
        self.refresh_teacher()
        buf = self.update_buffer(info.pop("_buffer_loader", None)
                                 or info.pop("_retain_loader", None) or loader)
        info.pop("_retain_loader", None)
        if buf:
            info.setdefault("buffer_update", buf)

    # -- selection --------------------------------------------------------
    def select_forget_set(self, tid: int, loader):
        X, y = loader.dataset.tensors
        n = len(y)
        name = self.cfg["selector"]
        ratio = self.hp["forget_ratio"]
        seed = child_seed(self.cfg["seed"], tid, 7)

        # Which samples may be forgotten at all. See config.BASE_HYPERPARAMS.
        cand, protected, pool = self._candidate_pool(X, y, n)
        Xc, yc = (X, y) if len(cand) == n else (X[cand], y[cand])
        nc = len(cand)

        kw = {"n": nc, "ratio": ratio, "seed": seed,
              "contamination": self.hp["contamination"]}
        if name in ("donut", "anomalous", "representative",
                    "anomalous-joint", "representative-joint"):
            kw["latents"] = training.latents_for(self.model, Xc, yc, self.device,
                                                 self.hp["eval_batch_size"])
            if name in ("anomalous", "representative"):
                kw["labels"] = yc.numpy()
        elif name == "nn1":
            kw["scores"] = self._nn1_scores(tid, Xc, yc)
        elif name == "leftover":
            budget = self.hp["mem_size"] // max(
                1, len(self.buffer.family_buffers) + len(np.unique(yc.numpy())))
            # `leftover` fits an Isolation Forest in RAW feature space, which is
            # MADAR-as-published and correct on a tabular corpus. On an image
            # corpus "raw" is pixels: a (N, 3, 64, 64) tensor the forest cannot
            # even accept, and which core/buffer.py already refuses on the
            # grounds that pixel-space anomalies track brightness and
            # background rather than class. Latents are what the buffer itself
            # selects in there, so they are what this emulates.
            if Xc.dim() > 2:
                feats = training.latents_for(self.model, Xc, yc, self.device,
                                             self.hp["eval_batch_size"])
                summary_space = "latent"
            else:
                feats = Xc.numpy()
                summary_space = "raw"
            self._leftover_space = summary_space
            kw.update({"features": feats, "labels": yc.numpy(),
                       "budget_per_family": budget})

        if nc == 0:
            return loader, None, {"selector": name, "forget_pool": pool,
                                  "skipped": "empty candidate pool"}

        f_local, r_local = selectors.split(name, **kw)
        # Back to task indexing. The protected points are retained by
        # construction, not by the selector's choice.
        forget_idx = cand[np.asarray(f_local, dtype=int)]
        retain_idx = np.sort(np.concatenate(
            [cand[np.asarray(r_local, dtype=int)], protected]).astype(np.int64))
        summary = {"selector": name, "ratio": float(ratio), "n_candidates": int(nc),
                   "forget_pool": pool, "n_protected": int(len(protected)),
                   "n_task_samples": int(n),
                   "n_forget": int(len(forget_idx)), "n_retain": int(len(retain_idx))}
        summary["forget_idx"] = np.asarray(forget_idx, dtype=int).tolist()
        if name == "leftover":
            summary["leftover_space"] = getattr(self, "_leftover_space", "raw")
        if len(forget_idx) == 0:
            summary["skipped"] = "empty forget set at this ratio"
            return loader, None, summary

        fams, counts = np.unique(y.numpy()[forget_idx], return_counts=True)
        summary["forget_per_family"] = dict(zip(fams.tolist(), counts.tolist()))

        # Under `complement`, the buffer is built from THESE indices, not from
        # the retain set. Excluding protected points from the forget set is not
        # enough on its own: update() re-fits an Isolation Forest on whatever it
        # is handed, so a different forget set changes the fit and therefore the
        # keeps. Handing it MADAR's own choice is what actually makes the buffer
        # identical across arms -- and it is an identity, since each family's
        # protected set is already exactly its budget.
        summary["buffer_from"] = "protected" if len(protected) else "retain"
        self._protected_idx = protected if len(protected) else None

        bs = self.hp["batch_size"]
        forget_loader = data.DataLoader(
            data.TensorDataset(X[forget_idx], y[forget_idx]), batch_size=bs, shuffle=True)
        retain_loader = data.DataLoader(
            data.TensorDataset(X[retain_idx], y[retain_idx]), batch_size=bs, shuffle=True)
        return retain_loader, forget_loader, summary

    def _candidate_pool(self, X, y, n: int):
        """(candidate indices, protected indices, pool name).

        Under `complement` the forget set may only be drawn from what MADAR was
        going to discard, so the buffer this task produces is the same whichever
        selector ran -- the comparison between arms then isolates the forget set
        from the curation it would otherwise also be changing.

        With no buffer there is nothing to take a complement OF, so it falls
        back to the whole task and SAYS so in the summary rather than quietly
        behaving like a different setting than the one requested.
        """
        pool = str(self.hp.get("forget_pool", "all"))
        if pool not in ("all", "complement"):
            raise SystemExit(f"--forget-pool must be 'all' or 'complement', "
                             f"got {pool!r}")
        empty = np.zeros(0, dtype=np.int64)
        if pool == "all":
            return np.arange(n, dtype=np.int64), empty, "all"
        if self.buffer is None:
            return np.arange(n, dtype=np.int64), empty, "all (no buffer)"
        lat = training.latents_for(self.model, X, y, self.device,
                                   self.hp["eval_batch_size"])
        protected = self.buffer.would_keep(y.numpy(), lat, raw=X.numpy())
        cand = np.setdiff1d(np.arange(n, dtype=np.int64), protected)
        return cand, protected, "complement"

    def _zero_features(self) -> list[str]:
        raw = self.hp.get("zero_features") or ""
        return [f for f in str(raw).replace(",", " ").split() if f]

    def _nn1_scores(self, tid: int, X, y):
        """Mode-A scores for the arriving task's samples, highest = forget first.

        The feature-count check is not a formality. A mode-B scorer has nine
        weights whose last two are for grad_conflict and density_ratio, features
        that do not exist here; loading one would silently reinterpret those
        weights as belonging to whichever columns happened to line up, and the
        run would look fine and mean nothing.
        """
        path = self.hp.get("scorer_weights")
        if not path:
            raise SystemExit(
                "selector 'nn1' needs --scorer-weights pointing at a saved "
                "scorer (.npz). Train one with `./iclr meta --mode a`, or use "
                "--selector random / donut / leftover for a control.")
        params, hidden, nf = nn1.load(path)
        if nf != nn1.N_FEATURES:
            raise SystemExit(
                f"madar_unlearn is mode A and needs a {nn1.N_FEATURES}-feature "
                f"scorer; {path} has {nf}. A mode-B scorer is not "
                f"interchangeable -- its last two weights are for features mode "
                f"A does not compute.")
        feats = task_features(
            self.model, X, y,
            active_count=self.schedule.active_count(tid), device=self.device,
            batch_size=self.hp["eval_batch_size"],
            seed=child_seed(self.cfg["seed"], tid, 11),
            contamination=self.hp["contamination"],
            zero_features=self._zero_features())
        return nn1.score(params, feats["features"], hidden)

    # -- the forget phase -------------------------------------------------
    def flatten_range(self, tid: int) -> tuple[int, int]:
        """Which logit slice the forget set is pushed toward uniform.

        The NEW-task classes by default: these are current-task samples, and
        forgetting them means the model should not have learned to name them.
        Mode B overrides this -- it forgets BUFFER points, whose knowledge lives
        in the OLD slice, and flattening the new-task columns for them would
        target classes the sample was never about.
        """
        return self.schedule.prev_active_count(tid), self.schedule.active_count(tid)

    def unlearn(self, tid: int, forget_loader, retain_loader) -> int:
        alpha = self.hp["alpha"]
        active = self.schedule.active_count(tid)
        teacher = training.clone_teacher(self.model)

        self.model.train()
        training.freeze_batchnorm(self.model)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.hp["unlearn_lr"])
        mask = training.logit_mask(self.model.fc_last.out_features, active, self.device)
        flat_lo, flat_hi = self.flatten_range(tid)

        retain_iter = iter(training.cycle(retain_loader))
        buf_loader = self.buffer.loader(self.hp["batch_size"]) if self.buffer else None
        buf_iter = iter(training.cycle(buf_loader)) if buf_loader else None
        steps = 0

        for _ in range(self.hp["unlearn_epochs"]):
            for fx, _fy in forget_loader:
                fx = fx.to(self.device)
                opt.zero_grad()

                # forget: flatten the chosen logit slice toward uniform
                cur = self.model(fx)[:, flat_lo:flat_hi]
                log_p = F.log_softmax(cur, dim=1)
                uniform = torch.ones_like(log_p) / max(1, flat_hi - flat_lo)
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
                # One view, shared by the frozen reference and the live model:
                # the unlearning loss is a comparison between the two ON THE
                # SAME INPUT, and two independent crops would make the KL a
                # measure of augmentation rather than of what was forgotten.
                xb = training.augmented(self.model, xb)
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
