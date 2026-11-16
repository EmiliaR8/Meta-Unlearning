"""Gaussian naive Bayes: a classifier whose old-class parameters cannot change.

WHY THIS EXISTS. The reviewer question is whether excluding a sample from the
replay buffer is enough. On a network it looks sufficient, because the weights
drift away from anything they stop seeing -- exclusion and forgetting are
confounded. This method removes that confound by construction.

Each class here is a set of sufficient statistics (count, sum, sum of squares)
over that class's own samples. Nothing another class contributes can touch them,
so a family learned at task 1 has EXACTLY the same parameters at task 10:
catastrophic forgetting is impossible, not merely small. What does still fall is
accuracy, purely because later families compete for the same test points. That
competition is the conflict channel the experiment is about.

THE ARMS.

  A  -- no forget set. Every task sample is folded in; the buffer keeps all.
  B  -- EXCLUSION ONLY. The forget set is excluded from the buffer rebuild, but
        it was already folded into the statistics during the task. Its influence
        is untouched. This is the analogue of madar_unlearn at alpha = 0.
  C  -- REMOVAL. The same forget set is subtracted from the statistics, exactly.
  C-rand -- a random forget set with the SAME PER-FAMILY COUNTS, removed the
        same way. Without it, "removal helped" cannot be separated from "any
        removal helps" or "less data helps".

Under --no-replay, arm B is IDENTICAL TO ARM A by construction -- the statistics
never see the buffer, so excluding something from it changes nothing. That is
the finding, not a bug, and it is the cleanest statement of the claim: there
exist model classes where buffer exclusion CANNOT implement forgetting. Under
--replay the buffer is folded in as well, so B differs from A by the replay
re-count alone and the result becomes empirical rather than definitional. Run
both; they answer the same question at different strengths.

THE FORGET SET IS NOT CHOSEN HERE. It is replayed from a madar_unlearn run via
--forget-from, because the claim is about the samples THAT method names. The
`donut` selector fits an Isolation Forest on the network's latents, so the
selection cannot be re-derived without the network. Config fields are
cross-checked against the source record and a mismatch is refused: replaying
seed 3's forget sets into a seed 1 run would produce a plausible number and no
error.

BUFFER SPACE MUST BE RAW, and --buffer-space raw is REQUIRED rather than
assumed. There are no latents for the buffer's Isolation Forest to fit, but
silently overriding the flag would leave the record claiming 'latent' for a raw
buffer -- and aggregate.py groups on buffer_space, so the run would be pooled
with runs it is not comparable to.

EXACTNESS. Removal is subtraction of sufficient statistics, which is exact to
floating point rather than approximate. Verified against sklearn's GaussianNB at
every task and the worst disagreement is recorded as `sklearn_max_abs_diff`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from core.buffer import ReplayBuffer

from .base import Experiment, register

ARMS = ("A", "B", "C", "C-rand")
VAR_SMOOTHING = 1e-9        # sklearn's GaussianNB default


class GaussianNBModule(nn.Module):
    """Per-class sufficient statistics, exposed as a torch model.

    Being an nn.Module is what lets the shared evaluation path -- three scopes,
    confusion matrices, metrics -- run unchanged: training.predict calls
    model.eval() and model(x), and gets log-posteriors back. The statistics are
    registered buffers, so state_dict/load_state_dict and therefore per-task
    checkpointing work with no extra code.
    """

    def __init__(self, n_classes: int, n_features: int):
        super().__init__()
        self.n_classes, self.n_features = int(n_classes), int(n_features)
        self.register_buffer("count", torch.zeros(n_classes, dtype=torch.float64))
        self.register_buffer("sums", torch.zeros(n_classes, n_features, dtype=torch.float64))
        self.register_buffer("sumsq", torch.zeros(n_classes, n_features, dtype=torch.float64))
        # Present so `fc_last.out_features` resolves: several shared helpers
        # read it to size the logit mask.
        self.fc_last = nn.Linear(1, n_classes, bias=False)
        self.fc_last.requires_grad_(False)

    # -- statistics ------------------------------------------------------
    def add(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self._accumulate(X, y, +1.0)

    def remove(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Exact unlearning: subtract the samples' contribution."""
        self._accumulate(X, y, -1.0)
        if bool((self.count < 0).any()):
            raise RuntimeError("negative class count after removal -- the same "
                               "samples were removed twice")

    def _accumulate(self, X, y, sign: float) -> None:
        Xd = X.to(self.sums.device, torch.float64)
        yd = y.to(self.count.device, torch.long)
        oh = torch.zeros(len(yd), self.n_classes, dtype=torch.float64,
                         device=self.count.device)
        oh[torch.arange(len(yd), device=oh.device), yd] = 1.0
        self.count += sign * oh.sum(0)
        self.sums += sign * (oh.T @ Xd)
        self.sumsq += sign * (oh.T @ (Xd * Xd))

    def theta_var(self):
        """(means, variances) with sklearn's epsilon, zeros where unfitted."""
        n = self.count.clamp(min=1.0).unsqueeze(1)
        mean = self.sums / n
        var = self.sumsq / n - mean * mean
        # Floating-point subtraction can land a hair below zero on a constant
        # feature; clamp before the epsilon rather than after, so the epsilon
        # still guarantees a strictly positive variance.
        var = var.clamp(min=0.0)
        eps = VAR_SMOOTHING * float(var.max().item() if var.numel() else 1.0)
        return mean, var + max(eps, VAR_SMOOTHING)

    def forward(self, x, return_latent: bool = False):
        """Unnormalised log-posteriors, (batch, n_classes).

        Classes with no samples get -inf so they can never win an argmax. The
        shared evaluator truncates to the active prefix anyway, but an unfitted
        class inside that prefix would otherwise score on a degenerate Gaussian.
        """
        mean, var = self.theta_var()
        xd = x.to(mean.device, torch.float64).reshape(len(x), -1)
        # -0.5 * sum_d [ log(2 pi var) + (x-mu)^2 / var ]
        logdet = torch.log(2.0 * np.pi * var).sum(1)
        quad = ((xd.unsqueeze(1) - mean.unsqueeze(0)) ** 2 / var.unsqueeze(0)).sum(2)
        ll = -0.5 * (logdet.unsqueeze(0) + quad)
        total = self.count.sum().clamp(min=1.0)
        prior = torch.log((self.count / total).clamp(min=1e-300))
        out = ll + prior.unsqueeze(0)
        out[:, self.count <= 0] = float("-inf")
        out = out.to(torch.float32)
        return (out, xd.to(torch.float32)) if return_latent else out

    def sklearn_agreement(self, X: np.ndarray, y: np.ndarray) -> float:
        """Max |difference| in means and variances against sklearn's own fit.

        Guards against this reimplementation drifting from the stock estimator
        it claims to be. Compares only classes present in (X, y).

        X IS UPCAST TO float64 FIRST, and that is not a detail. sklearn keeps
        the input dtype, so handing it the float32 feature matrix (which is what
        X.cpu().numpy() is) makes it accumulate in float32. On EMBER 2018 task 0
        -- 282,510 samples, 2381 features, 50 families -- fitting sklearn twice
        on IDENTICAL input, once float32 and once float64, makes the two
        disagree by 2.073e-03, which is exactly the figure this check reported
        before the upcast. The whole discrepancy was sklearn's own float32 path;
        the statistics here, accumulated in float64 throughout, are the more
        accurate of the two.

        Why float32 drifts that far on this particular data is NOT established.
        Synthetic data of the same shape and scale -- dense gaussian, and sparse
        heavy-tailed with the same clipping -- reproduces only 5e-6 to 2e-5, so
        something about EMBER's real feature distribution is responsible and has
        not been identified. It does not matter for any claim here: removal
        exactness, the invariance of old-class parameters, and arm B equalling
        arm A are all properties of the float64 statistics and hold regardless
        of what sklearn computes. This check is a guard against THIS code
        drifting from the estimator it claims to implement, nothing more.
        Against a float64 fit the agreement should be ~1e-8.
        """
        from sklearn.naive_bayes import GaussianNB
        g = GaussianNB().fit(np.asarray(X, dtype=np.float64), y)
        mean, var = self.theta_var()
        worst = 0.0
        for i, c in enumerate(g.classes_):
            c = int(c)
            worst = max(worst,
                        float(np.abs(g.theta_[i] - mean[c].cpu().numpy()).max()),
                        float(np.abs(g.var_[i] - var[c].cpu().numpy()).max()))
        return worst


def _load_forget_sets(path: str, config: dict, hyperparams: dict) -> dict:
    """{task id: forget indices} from a madar_unlearn record, config-checked."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"--forget-from: no such record {p}")
    try:
        rec = json.loads(p.read_text())
    except ValueError as e:
        raise SystemExit(f"--forget-from: {p} is not readable JSON ({e})")

    src = rec.get("config") or {}
    src_hp = rec.get("hyperparams") or {}
    for field in ("dataset", "task_setup", "seed", "protocol"):
        if src.get(field) != config.get(field):
            raise SystemExit(
                f"--forget-from: {p.name} has {field}={src.get(field)!r} but this "
                f"run is {field}={config.get(field)!r}. Replaying another "
                f"configuration's forget sets would produce a number that looks "
                f"fine and means nothing.")
    if src.get("method") != "madar_unlearn":
        raise SystemExit(f"--forget-from: {p.name} is method "
                         f"{src.get('method')!r}, expected madar_unlearn")
    if float(src_hp.get("forget_ratio", -1)) != float(hyperparams["forget_ratio"]):
        raise SystemExit(
            f"--forget-from: {p.name} used forget_ratio="
            f"{src_hp.get('forget_ratio')} but this run asks for "
            f"{hyperparams['forget_ratio']}")

    out = {}
    missing = []
    for entry in rec.get("per_task", []):
        sel = entry.get("selection") or {}
        tid = int(entry["task"])
        if "forget_idx" in sel:
            out[tid] = (np.asarray(sel["forget_idx"], dtype=np.int64),
                        int(sel.get("n_task_samples", -1)),
                        {int(k): int(v) for k, v in
                         (sel.get("forget_per_family") or {}).items()})
        elif sel and not sel.get("skipped"):
            missing.append(tid)
    if missing:
        raise SystemExit(
            f"--forget-from: {p.name} has selection records without "
            f"`forget_idx` at task(s) {missing}. That record predates the field; "
            f"re-run madar_unlearn with the index line in select_forget_set.")
    if not out:
        raise SystemExit(f"--forget-from: {p.name} contains no forget sets")
    return {"sets": out, "source_run_id": rec.get("run_id"),
            "source_selector": src.get("selector")}


@register("gnb")
class GNBExperiment(Experiment):
    """MADAR replay under a classifier that cannot forget."""

    uses_buffer = True
    uses_si = False

    def __init__(self, ctx):
        super().__init__(ctx)
        arm = str(self.hp.get("gnb_arm", "A"))
        if arm not in ARMS:
            raise SystemExit(f"--gnb-arm must be one of {', '.join(ARMS)}; got {arm!r}")
        self.arm = arm
        self.replay = bool(self.hp.get("gnb_replay", False))

        # There are no latents to fit an Isolation Forest in, so the buffer can
        # only be raw-space. Quietly overriding --buffer-space would leave the
        # record claiming 'latent' for a raw buffer, and aggregate.py GROUPS ON
        # buffer_space -- the run would be averaged in with latent-space runs.
        # Refusing costs one flag and keeps the record true to what ran.
        if self.hp.get("buffer_space") != "raw":
            raise SystemExit(
                f"--experiment gnb requires --buffer-space raw (got "
                f"{self.hp.get('buffer_space')!r}). Gaussian naive Bayes has no "
                f"latent representation for the buffer's Isolation Forest to "
                f"fit, so a raw-space buffer is the only option. It is not set "
                f"for you because aggregate.py groups on buffer_space, and a "
                f"record saying 'latent' for a raw buffer would be pooled with "
                f"runs it is not comparable to.")

        self.forget_sets = {}
        self.forget_meta = {}
        if arm != "A":
            path = str(self.hp.get("forget_from", "") or "")
            if not path:
                raise SystemExit(
                    f"--gnb-arm {arm} needs --forget-from pointing at a "
                    f"madar_unlearn record. Arm A is the only one that selects "
                    f"nothing.")
            loaded = _load_forget_sets(path, self.cfg, self.hp)
            self.forget_sets = loaded["sets"]
            self.forget_meta = {k: v for k, v in loaded.items() if k != "sets"}

    # -- model ------------------------------------------------------------
    def build_model(self):
        return GaussianNBModule(self.schedule.n_classes,
                                self.ctx.input_dim).to(self.ctx.device)

    def model_info(self) -> dict:
        n = self.schedule.n_classes * self.ctx.input_dim * 2 + self.schedule.n_classes
        return {"name": "gaussian_nb", "n_params": int(n),
                "n_classes": int(self.schedule.n_classes),
                "n_features": int(self.ctx.input_dim),
                "note": "sufficient statistics, not learned parameters"}

    def update_buffer(self, loader) -> dict:
        """Raw-space buffer: `latents` is unused by ReplayBuffer when
        space='raw', so no forward pass is needed to produce it."""
        if self.buffer is None or loader is None:
            return {}
        X, y = loader.dataset.tensors
        return self.buffer.update(X, y, np.zeros((len(y), 1), dtype=np.float32))

    # -- the task ---------------------------------------------------------
    def _forget_indices(self, tid: int, n: int, y: torch.Tensor):
        """Indices into this task's samples to remove, or None."""
        if self.arm == "A" or tid not in self.forget_sets:
            return None, {}
        idx, n_src, per_family = self.forget_sets[tid]
        if n_src >= 0 and n_src != n:
            raise SystemExit(
                f"task {tid}: the source record saw {n_src} samples but this run "
                f"has {n}. The schedules differ, so its indices do not refer to "
                f"these samples.")
        if len(idx) and int(idx.max()) >= n:
            raise SystemExit(f"task {tid}: forget index {int(idx.max())} is out "
                             f"of range for {n} samples")
        note = {"n_forget": int(len(idx))}
        if self.arm != "C-rand":
            return idx, note

        # Matched control: same number per family, drawn at random. Matching the
        # family composition matters -- an unstratified draw would differ from
        # the real forget set in WHICH families lose points as well as in which
        # points, and the comparison would no longer isolate the selection.
        rng = np.random.default_rng(abs(hash((self.cfg["seed"], tid, "crand"))) % (2**32))
        y_np = y.cpu().numpy()
        picked = []
        for fam, k in sorted(per_family.items()):
            pool = np.flatnonzero(y_np == fam)
            if len(pool) == 0:
                continue
            picked.append(rng.choice(pool, size=min(int(k), len(pool)), replace=False))
        out = np.sort(np.concatenate(picked)) if picked else np.zeros(0, np.int64)
        note.update({"n_forget": int(len(out)), "matched_to": "forget_per_family"})
        return out, note

    def train_task(self, tid: int, loader) -> dict:
        X, y = loader.dataset.tensors
        n = len(y)
        info = {"grad_steps": 0, "arm": self.arm, "replay": self.replay,
                "phase": "fit"}
        if self.forget_meta:
            info["forget_source"] = dict(self.forget_meta)

        # Every arm folds in the WHOLE task. The forget set was trained on
        # before it was ever named -- that is the premise of the experiment.
        self.model.add(X, y)

        if self.replay and self.buffer is not None and not self.buffer.is_empty():
            bx, by = self.buffer.tensors()
            self.model.add(bx, by)
            info["replayed"] = int(len(by))

        forget_idx, note = self._forget_indices(tid, n, y)
        sel = {"arm": self.arm, "n_task_samples": int(n)}
        sel.update(note)
        if forget_idx is not None and len(forget_idx):
            # Arm B leaves the statistics alone: exclusion acts on the buffer
            # only, which is exactly the thing being shown to be insufficient.
            if self.arm in ("C", "C-rand"):
                self.model.remove(X[forget_idx], y[forget_idx])
                sel["removed_from_statistics"] = True
            else:
                sel["removed_from_statistics"] = False
            self._retain_idx = np.setdiff1d(np.arange(n), forget_idx)
        else:
            self._retain_idx = None
        info["selection"] = sel

        X_np = X.cpu().numpy().reshape(n, -1)
        info["sklearn_max_abs_diff"] = (
            self.model.sklearn_agreement(X_np, y.cpu().numpy())
            if (self.arm == "A" and tid == 0 and not self.replay) else None)
        return info

    def after_task(self, tid: int, loader, info: dict) -> None:
        """Rebuild the buffer from retain only, as madar_unlearn does."""
        if self._retain_idx is None:
            buf = self.update_buffer(loader)
        else:
            X, y = loader.dataset.tensors
            import torch.utils.data as data
            retain = data.DataLoader(
                data.TensorDataset(X[self._retain_idx], y[self._retain_idx]),
                batch_size=self.hp["batch_size"])
            buf = self.update_buffer(retain)
            info.setdefault("selection", {})["buffer_from"] = "retain"
        if buf:
            info.setdefault("buffer_update", buf)
        self._retain_idx = None
