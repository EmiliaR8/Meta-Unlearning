"""Training primitives: task-0 fitting, the continual step, SI, and evaluation.

The continual objective is

    rnt * CE(current + replay)  +  (1 - rnt) * KD(replay || previous model)
                                +  si_c * SI

with rnt = 1/(t+1) by default, i.e. every task seen so far weighted equally.

BatchNorm is frozen to eval() during the continual phase: its running statistics
were estimated on task 0 and updating them from a stream dominated by the newest
families is a channel of forgetting that has nothing to do with the weights.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from .metrics import confusion_matrix, metrics_from_confusion


# ---------------------------------------------------------------- helpers
def cycle(iterable):
    while True:
        for item in iterable:
            yield item


def make_loader(X, y, classes, batch_size: int, shuffle: bool = True,
                drop_last: bool = False):
    """Loader over the rows whose label is in `classes`, or None if empty."""
    mask = torch.isin(y, torch.as_tensor(list(classes)))
    if not bool(mask.any()):
        return None
    return data.DataLoader(data.TensorDataset(X[mask], y[mask]),
                           batch_size=batch_size, shuffle=shuffle,
                           drop_last=drop_last)


def augmented(model, x):
    """One random view of a batch, or the batch unchanged.

    Identity unless the model defines `augment` -- only the convolutional
    backbone does -- and identity in eval mode, so evaluation never sees a
    randomly cropped image. Call it ONCE per batch and reuse the result: a
    replay batch is consumed by the cross-entropy term AND, under distillation,
    by both the teacher and the student, and drawing a fresh crop for each would
    make the distillation target describe a different image from the one the
    student is being scored on.
    """
    aug = getattr(model, "augment", None)
    if aug is None or not model.training:
        return x
    return aug(x)


def logit_mask(num_classes: int, active_count: int, device) -> torch.Tensor:
    """Additive mask restricting the loss to the active class prefix."""
    m = torch.full((num_classes,), -1e9, device=device)
    m[:active_count] = 0.0
    return m


def kd_loss(student_logits, teacher_logits, temperature: float = 2.0):
    log_p = F.log_softmax(student_logits / temperature, dim=1)
    q = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(log_p, q, reduction="batchmean") * (temperature ** 2)


def rnt_for(tid: int, mode: str = "inverse", floor: float = 0.25,
            value: float = 0.5) -> float:
    """Weight on the current-task loss.

    'inverse' (1/(t+1)) weights every task seen so far equally. That is
    principled when tasks are interchangeable; when task 0 is far larger than the
    increments, the newest families end up with a few percent of the gradient.
    The alternatives exist to test that, and default to off.
    """
    if mode == "inverse":
        return 1.0 / (tid + 1)
    if mode == "floor":
        return max(1.0 / (tid + 1), floor)
    if mode == "sqrt":
        return 1.0 / math.sqrt(tid + 1)
    if mode == "fixed":
        return value
    raise ValueError(f"unknown rnt mode {mode!r}")


def freeze_batchnorm(model) -> None:
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.eval()


# ---------------------------------------------------------------- SI
class SynapticIntelligence:
    """Path-integral parameter importance (Zenke et al.).

    `W` accumulates -grad * delta over each step; at a task boundary it becomes
    omega, normalised by the squared distance the parameter actually moved.
    `p_old` anchors to the START of the current task and is advanced only by
    `end_task`, so an unlearning phase running after the boundary is penalised
    for drifting important parameters away from where the task began.
    """

    def __init__(self, model, si_c: float = 1.0, eps: float = 0.1):
        self.si_c = float(si_c)
        self.eps = float(eps)
        self.W = {self._key(n): torch.zeros_like(p)
                  for n, p in model.named_parameters() if p.requires_grad}
        self.omega = {k: torch.zeros_like(v) for k, v in self.W.items()}
        self.p_old = {self._key(n): p.detach().clone()
                      for n, p in model.named_parameters() if p.requires_grad}

    @staticmethod
    def _key(name: str) -> str:
        return name.replace(".", "__")

    def penalty(self, model):
        if self.si_c == 0:
            return torch.zeros((), device=next(model.parameters()).device)
        total = 0.0
        for n, p in model.named_parameters():
            if p.requires_grad:
                k = self._key(n)
                total = total + (self.omega[k] * (p - self.p_old[k]) ** 2).sum()
        return total

    def snapshot(self, model) -> dict:
        return {self._key(n): (p.grad.detach().clone(), p.detach().clone())
                for n, p in model.named_parameters()
                if p.requires_grad and p.grad is not None}

    def accumulate(self, model, snap: dict) -> None:
        """Call AFTER optimizer.step() with the pre-step snapshot."""
        for n, p in model.named_parameters():
            k = self._key(n)
            if p.requires_grad and k in snap:
                grad, before = snap[k]
                self.W[k].add_(-grad * (p.detach() - before))

    def end_task(self, model, advance_anchor: bool = True) -> float:
        """Fold W into omega. Returns the penalty magnitude for diagnostics."""
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            k = self._key(n)
            current = p.detach().clone()
            self.omega[k] += self.W[k] / ((current - self.p_old[k]) ** 2 + self.eps)
            self.W[k].zero_()
            if advance_anchor:
                self.p_old[k] = current
        with torch.no_grad():
            return float(self.penalty(model).item())

    def advance_anchor(self, model) -> None:
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.p_old[self._key(n)] = p.detach().clone()


# ---------------------------------------------------------------- training
def train_epochs(model, loader, *, epochs: int, lr: float, active_count: int,
                 device, label: str = "task 0") -> int:
    """Plain supervised training (task 0, and the joint baseline). Adam."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    mask = logit_mask(model.fc_last.out_features, active_count, device)
    steps = 0
    for epoch in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            xb = augmented(model, xb)
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(model(xb) + mask, yb)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss during {label}, epoch {epoch} -- diverged")
            loss.backward()
            optimizer.step()
            steps += 1
    return steps


def train_continual(model, teacher, loader, buffer, *, iters: int,
                    active_count: int, prev_active: int, si, tid: int,
                    device, lr: float = 1e-4, momentum: float = 0.9,
                    weight_decay: float = 1e-6, kd_temp: float = 2.0,
                    grad_clip: float = 1.0, batch_size: int = 256,
                    rnt_mode: str = "inverse", rnt_floor: float = 0.25,
                    rnt_value: float = 0.5, use_replay: bool = True,
                    use_kd: bool = True, use_rnt: bool = True,
                    freeze_bn: bool = True) -> dict:
    """One continual task. Replay, distillation and SI are each switchable.

    The three components are independent so that an ablation removes exactly one
    thing. Two consequences are worth stating rather than discovering:

    * WITHOUT KD, rnt HAS NOTHING TO WEIGHT. rnt is the convex weight between the
      CE term and the KD term; with KD gone it degenerates into a bare scalar on
      the whole loss, shrinking every gradient by 1/(t+1) -- a decaying learning
      rate wearing a replay-weight costume. It is therefore off by default when
      KD is off, and `use_rnt=True` retains it only if you mean to measure that
      confound rather than remove it.

    * SI is controlled by si.si_c. At si_c = 0 the penalty is identically zero,
      which is the matched control: same optimiser, same iteration count, same
      data order, no regulariser.
    """
    if use_replay and (buffer is None or buffer.is_empty()):
        raise RuntimeError("replay is enabled but the buffer is empty")
    if use_kd and not use_replay:
        raise RuntimeError("knowledge distillation here distils over replayed "
                           "samples, so it cannot be used without replay")

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                weight_decay=weight_decay)
    model.train()
    if freeze_bn:
        freeze_batchnorm(model)
    mask = logit_mask(model.fc_last.out_features, active_count, device)
    rnt = rnt_for(tid, rnt_mode, rnt_floor, rnt_value) if use_rnt else 1.0

    cur_iter = iter(cycle(loader))
    buf_iter = iter(cycle(buffer.loader(batch_size))) if use_replay else None
    losses = []

    for step in range(iters):
        xb, yb = next(cur_iter)
        xb, yb = xb.to(device), yb.to(device)
        xb = augmented(model, xb)

        optimizer.zero_grad()
        if use_replay:
            mx, my = next(buf_iter)
            mx, my = mx.to(device), my.to(device)
            mx = augmented(model, mx)
            loss_cur = nn.CrossEntropyLoss()(
                model(torch.cat([xb, mx])) + mask, torch.cat([yb, my]))
        else:
            loss_cur = nn.CrossEntropyLoss()(model(xb) + mask, yb)

        if use_kd:
            with torch.no_grad():
                teacher_logits = teacher(mx)[:, :prev_active]
            loss_kd = kd_loss(model(mx)[:, :prev_active], teacher_logits, kd_temp)
            loss = rnt * loss_cur + (1.0 - rnt) * loss_kd
        else:
            loss = rnt * loss_cur
        loss = loss + si.si_c * si.penalty(model)

        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at task {tid}, step {step} -- diverged")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        snap = si.snapshot(model)
        optimizer.step()
        si.accumulate(model, snap)
        losses.append(float(loss.item()))

    return {"grad_steps": int(iters), "rnt": float(rnt),
            "components": {"replay": use_replay, "kd": use_kd,
                           "si_c": float(si.si_c), "rnt_applied": use_rnt},
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "loss_mean": float(np.mean(losses)) if losses else None}


def clone_teacher(model):
    teacher = copy.deepcopy(model)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


# ---------------------------------------------------------------- inference
@torch.no_grad()
def predict(model, X, y, classes, active_count: int, device,
            batch_size: int = 512):
    """Masked-prefix predictions over the rows whose label is in `classes`."""
    model.eval()
    mask = torch.isin(y, torch.as_tensor(list(classes)))
    if not bool(mask.any()):
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    loader = data.DataLoader(data.TensorDataset(X[mask], y[mask]),
                             batch_size=batch_size)
    trues, preds = [], []
    for xb, yb in loader:
        logits = model(xb.to(device))[:, :active_count]
        preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        trues.append(yb.numpy())
    return np.concatenate(trues), np.concatenate(preds)


def evaluate(model, X, y, classes, active_count: int, device,
             n_classes: int, batch_size: int = 512):
    """(metrics dict, confusion matrix) over one evaluation scope."""
    y_true, y_pred = predict(model, X, y, classes, active_count, device, batch_size)
    cm = confusion_matrix(y_true, y_pred, n_classes)
    return metrics_from_confusion(cm), cm


@torch.no_grad()
def latents_for(model, X, y, device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    loader = data.DataLoader(data.TensorDataset(X, y), batch_size=batch_size)
    out = []
    for xb, _ in loader:
        _, lat = model(xb.to(device), return_latent=True)
        out.append(lat.cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.latent_dim), np.float32)


@torch.no_grad()
def logits_for(model, X, y, device, batch_size: int = 512) -> torch.Tensor:
    """FULL-WIDTH responses for a dataset, on CPU.

    Not truncated to the active prefix: the DER family stores these in memory,
    and X-DER's future heads are exactly the columns beyond that prefix.
    """
    model.eval()
    loader = data.DataLoader(data.TensorDataset(X, y), batch_size=batch_size)
    out = [model(xb.to(device)).cpu() for xb, _ in loader]
    return (torch.cat(out) if out
            else torch.zeros((0, model.fc_last.out_features), dtype=torch.float32))
