"""DEDUCE -- Detect, Decide, Unlearn (Wang, Benavides-Prado & Koh, ICLR 2026).

A wrapper around an existing continual learner, not a learner itself. Before
each batch it asks whether learning that batch will interfere with what the
model already knows; if so it takes one deliberate step of UNLEARNING first
(LUM), and throughout it reclaims dead capacity by reinitialising neurons that
are both inactive and unimportant (GUM).

    detect   Eq. (7)  gradient conflict, or Eq. (5) the transferability bound
    decide   run LUM on this batch, or do not
    unlearn  LUM  Eq. (10): FIM-scaled ascent on the current batch's CE
             GUM  Eq. (12): contribution-ranked neuron reinitialisation

The paper publishes the algorithm and its hyperparameters but no code, so the
places where a faithful implementation still had to choose are marked CHOICE
below and repeated in the experiment module's docstring. Every one of them is a
place where a different reading is defensible.

CHOICE (detector loss). Eq. (6)-(7) compare grad L(f, M) with grad L(f, D_t)
without naming L. Both are taken here as cross-entropy over the ACTIVE class
prefix -- the same masked rule the evaluation uses -- rather than the wrapped
method's own loss. Using X-DER's stream loss for one side and its memory loss
for the other would compare two different objectives' gradients and call the
angle between them interference.

CHOICE (Fisher). F is the diagonal empirical Fisher, E[(grad log p(y|x))^2] at
the TRUE labels, estimated at each task boundary from the task's data and the
buffer together. Sampling y from the model instead gives the true Fisher; the
empirical form is what EWC-style work overwhelmingly uses and is cheaper.

CHOICE (Fisher scale), and this one is not optional. Eq. (10) steps by
delta * F^-1 * (...), but F is an unnormalised expectation of squared gradients:
its magnitude depends on the loss scale, the batch size, and how converged the
model is. On a fitted model its entries here are ~1e-8, so F^-1 under any small
damping is ~1e3 and the paper's delta = 0.001 becomes an effective learning rate
of order 1 -- in testing, ONE unlearning step took task-0 accuracy from 100 to
0. BOTH F and the preconditioner F^-1 are therefore normalised to unit mean over
the whole parameter vector. That preserves exactly what Eq. (10) is for, the
RELATIVE protection of high-Fisher parameters, while making delta a learning
rate whose published value transfers. The same normalisation is what makes beta
in Eq. (11) mean anything. The unlearning step is additionally norm-capped at
grad_clip * delta, so the one update in the loop that ASCENDS a loss cannot be
the one update with no bound on its size.

CHOICE (bound-based detection). Eq. (5) needs the target error, which the paper
obtains from "one full pass (epoch) of the task data". That pass is run here on
a DEEP COPY of the model and thrown away, so the OUR(B) and OUR(G) arms follow
the same training trajectory and differ only in the decision. Training the real
model for that epoch would give OUR(B) an extra epoch per task that OUR(G) does
not get, and the two arms would no longer be comparable.

CHOICE (BatchNorm on reinitialisation). The paper resets a neuron's input
weights and zeroes its outgoing weights. Its backbone comment does not cover
what happens to normalisation statistics, which here would be left describing a
neuron that no longer exists. The unit's BatchNorm affine is reset to (1, 0) and
its running statistics to (0, 1) along with the weights.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ Fisher
class DiagonalFisher:
    """Diagonal empirical FIM over the trainable parameters."""

    def __init__(self, model):
        self.F = {n: torch.zeros_like(p) for n, p in model.named_parameters()
                  if p.requires_grad}
        self.P: dict = {}          # unit-mean preconditioner, see module docstring
        self.n_batches = 0
        self.raw_mean = 0.0        # the pre-normalisation scale, logged

    def is_empty(self) -> bool:
        return self.n_batches == 0

    def estimate(self, model, batches, active_count: int, device,
                 max_batches: int = 32) -> dict:
        """Accumulate squared gradients of the log-likelihood at the true label."""
        new = {n: torch.zeros_like(p) for n, p in model.named_parameters()
               if p.requires_grad}
        used = 0
        was = model.training
        model.eval()
        for xb, yb in batches:
            if used >= max_batches:
                break
            model.zero_grad(set_to_none=True)
            logits = model(xb.to(device))[:, :active_count]
            F.nll_loss(F.log_softmax(logits, dim=1), yb.to(device)).backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    new[n] += p.grad.detach() ** 2
            used += 1
        model.zero_grad(set_to_none=True)
        model.train(was)
        if not used:
            return {"fim_batches": 0, "fim_raw_mean": 0.0}
        for n in new:
            new[n] /= used
        count = sum(v.numel() for v in new.values())
        self.raw_mean = sum(float(v.sum()) for v in new.values()) / max(1, count)
        scale = self.raw_mean if self.raw_mean > 0 else 1.0
        self.F = {n: v / scale for n, v in new.items()}
        self.P = {}
        self.n_batches = used
        return {"fim_batches": used, "fim_raw_mean": self.raw_mean}

    def preconditioner(self, damping: float) -> dict:
        """Unit-mean F^-1, cached until the Fisher is re-estimated."""
        if self.P.get("_key") == damping:
            return self.P
        raw = {n: 1.0 / (v + damping) for n, v in self.F.items()}
        count = sum(v.numel() for v in raw.values())
        mean = sum(float(v.sum()) for v in raw.values()) / max(1, count)
        self.P = {n: v / mean for n, v in raw.items()}
        self.P["_key"] = damping
        return self.P

    def quadratic(self, model, anchor: dict) -> torch.Tensor:
        """(theta - anchor)^T F (theta - anchor), differentiable in theta."""
        total = None
        for n, p in model.named_parameters():
            if not p.requires_grad or n not in anchor:
                continue
            term = (self.F[n] * (p - anchor[n]) ** 2).sum()
            total = term if total is None else total + term
        return total if total is not None else torch.zeros(
            (), device=next(model.parameters()).device)

    def neuron_importance(self, weight_name: str, model) -> torch.Tensor:
        """F_{l,i} = sum over the OUTGOING connections of neuron i (Eq. 12's
        aggregation), where `weight_name` names the next layer's weight."""
        f = self.F[weight_name]
        # Linear: (out, in) -> sum over out. Conv2d: (out, in, kh, kw) -> sum
        # over out AND the kernel, so each input channel is scored by all the
        # connections it actually has rather than by one arbitrary tap.
        dims = (0,) + tuple(range(2, f.dim()))
        return f.sum(dim=dims)


# ------------------------------------------------------------------ detect
def _flat_grad(model) -> torch.Tensor:
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                      for p in model.parameters() if p.requires_grad])


def gradient_conflict(model, cur, mem, active_count: int, epsilon: float = 0.0) -> dict:
    """Definition 2 / Eq. (7). Leaves p.grad cleared.

    Returns {'conflict': 1.0|0.0, 'cosine': ...}. `cur` and `mem` are
    (x, y) pairs already on the device.
    """
    ce = nn.CrossEntropyLoss()

    def grad_of(x, y):
        model.zero_grad(set_to_none=True)
        ce(model(x)[:, :active_count], y).backward()
        return _flat_grad(model).clone()

    g_mem = grad_of(*mem)
    g_cur = grad_of(*cur)
    model.zero_grad(set_to_none=True)

    denom = float(g_mem.norm() * g_cur.norm())
    if denom <= 0:
        return {"conflict": 0.0, "cosine": 0.0}
    cos = float(torch.dot(g_mem, g_cur)) / denom
    return {"conflict": float(cos <= epsilon), "cosine": cos}


def leep_score(model, X, y, device, source_classes: int, target_classes,
               batch_size: int = 512) -> float:
    """LEEP (Nguyen et al., 2020), Definition 1. Always negative; nearer zero is
    better transferability."""
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(y), batch_size):
            out = model(X[i:i + batch_size].to(device))[:, :source_classes]
            probs.append(F.softmax(out, dim=1).cpu())
    theta = torch.cat(probs) if probs else torch.zeros(0, source_classes)
    if not len(theta):
        return 0.0
    remap = {int(c): j for j, c in enumerate(sorted(target_classes))}
    yt = torch.tensor([remap[int(v)] for v in y], dtype=torch.long)

    n, k = len(yt), len(remap)
    joint = torch.zeros(k, source_classes)
    joint.index_add_(0, yt, theta)
    joint /= n
    marginal = joint.sum(dim=0).clamp_min(1e-12)          # P(y_s)
    conditional = joint / marginal                        # P(y_t | y_s)
    per_sample = (conditional[yt] * theta).sum(dim=1).clamp_min(1e-12)
    return float(torch.log(per_sample).mean())


def domain_divergence(latents_source: np.ndarray, latents_target: np.ndarray,
                      seed: int = 0) -> float:
    """The 2|1 - 2*eps(h_d)| of Eq. (2), via a logistic-regression domain
    classifier on the model's own representation. Returns eps(h_d) and the
    divergence together is not needed; the caller wants |1 - 2*eps|."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X = np.concatenate([latents_source, latents_target]).astype(np.float64)
    y = np.concatenate([np.zeros(len(latents_source)),
                        np.ones(len(latents_target))])
    if len(set(y.tolist())) < 2 or len(y) < 20:
        return 0.5
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed,
                                          stratify=y)
    clf = LogisticRegression(max_iter=200, n_jobs=1).fit(Xtr, ytr)
    return float(1.0 - clf.score(Xte, yte))


# ------------------------------------------------------------------ LUM
class LocalUnlearning:
    """Eq. (9)-(10): one FIM-scaled ascent step on the current batch's CE.

    theta' = theta + delta * F^-1 [ alpha * grad D(theta, theta_k) - grad CE ]

    The minus sign on the CE gradient is the whole module: this is deliberate
    ascent on the batch about to be learned, which removes the prior knowledge
    that would fight it. The proximal term pulls back towards theta_k, the
    parameters after the first k batches of THIS task, so the step cannot
    unlearn what the current task has just established.

    F^-1 is what makes it selective: parameters with high Fisher information
    barely move, so the ascent lands on low-importance directions.
    """

    def __init__(self, hp):
        self.delta = float(hp["deduce_delta"])
        self.alpha = float(hp["deduce_alpha"])
        self.damping = float(hp["deduce_fim_damping"])
        self.clip = float(hp["grad_clip"])

    @torch.no_grad()
    def _apply(self, model, fisher, anchor) -> float:
        precond = fisher.preconditioner(self.damping)
        steps, moved = {}, 0.0
        for n, p in model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            prox = 2.0 * (p.detach() - anchor[n]) if anchor else torch.zeros_like(p)
            step = self.delta * precond[n] * (self.alpha * prox - p.grad.detach())
            steps[n] = step
            moved += float(step.norm() ** 2)
        moved = math.sqrt(moved)
        cap = self.clip * self.delta
        scale = min(1.0, cap / moved) if moved > 0 else 1.0
        for n, p in model.named_parameters():
            if n in steps:
                p.add_(steps[n] * scale)
        return moved * scale

    def step(self, model, xb, yb, *, fisher, anchor, active_count: int) -> dict:
        model.zero_grad(set_to_none=True)
        nn.CrossEntropyLoss()(model(xb)[:, :active_count], yb).backward()
        moved = self._apply(model, fisher, anchor)
        model.zero_grad(set_to_none=True)
        return {"lum_fired": 1.0, "lum_step_norm": moved}


# ------------------------------------------------------------------ GUM
def _resolve(model, name: str):
    """Fetch a submodule by dotted name: 'layer1.0.conv1' as well as 'fc1'."""
    obj = model
    for part in str(name).split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def _fan_in(layer) -> int:
    """Number of inputs feeding one unit -- in_features, or in_ch*kh*kw."""
    if hasattr(layer, "in_features"):
        return int(layer.in_features)
    w = layer.weight
    return int(w.shape[1] * w.shape[2] * w.shape[3])


def _n_units(layer) -> int:
    return int(getattr(layer, "out_features", None) or layer.out_channels)


class GlobalUnlearning:
    """Eq. (12) plus Algorithm 2: reinitialise low-contribution mature neurons.

    Contribution combines how much a unit actually drives the next layer
    (|h| times the sum of its outgoing weight magnitudes, smoothed with decay
    eta) with a gate on its historical importance, so a unit that is quiet now
    but was important before is protected. Reinitialised units are given fresh
    input weights and ZEROED outgoing weights -- they start inert and have to
    earn their way back -- and are shielded by a maturity threshold so they
    cannot be reset again immediately, when their contribution is zero by
    construction rather than by disuse.

    Wired to EmberNN's four hidden blocks. A unit in block l has input weights
    in fc{l} (plus its BatchNorm affine) and outgoing weights in column i of the
    next block's weight matrix; the last block's successor is the classifier.
    """

    # EmberNN's four hidden blocks, used when the model does not describe its
    # own. A model that is not the MLP -- ResNet-18 on Tiny ImageNet -- supplies
    # `gum_blocks()` instead; hardcoding this list here was the one place the
    # DEDUCE implementation assumed the backbone.
    DEFAULT_BLOCKS = [("fc1", "fc1_bn", "fc2"), ("fc2", "fc2_bn", "fc3"),
                      ("fc3", "fc3_bn", "fc4"), ("fc4", "fc4_bn", "fc_last")]

    def __init__(self, model, hp, seed: int = 0):
        self.eta = float(hp["deduce_eta"])
        self.phi = float(hp["deduce_phi"])
        self.maturity = int(hp["deduce_maturity"])
        self.gen = torch.Generator(device="cpu").manual_seed(int(seed) + 5501)
        describe = getattr(model, "gum_blocks", None)
        self.BLOCKS = list(describe()) if describe else list(self.DEFAULT_BLOCKS)
        self.C, self.age, self.credit = {}, {}, {}
        for lin, _, _ in self.BLOCKS:
            n = _n_units(_resolve(model, lin))
            self.C[lin] = torch.zeros(n)
            self.age[lin] = torch.zeros(n)
            self.credit[lin] = 0.0
        self._h: dict = {}
        self._handles = []

    # -- activations ------------------------------------------------------
    def attach(self, model) -> None:
        """Record post-ReLU activations. Hooked on the BatchNorm modules because
        EmberNN reuses ONE nn.ReLU instance for all four blocks, so a hook there
        could not tell the blocks apart."""
        self.detach()
        for lin, bn, _ in self.BLOCKS:
            def hook(_m, _inp, out, key=lin):
                # Record the FIRST forward after arm() and no others. A training
                # step here runs several forwards -- stream, two memory draws, an
                # augmented view -- and contribution is meant to describe the
                # unit's response to the data being learned, not to whichever
                # pass happened to run last.
                if key not in self._h:
                    a = out.detach().relu().abs()
                    # (N, C) for the MLP; (N, C, H, W) for a convolution, where
                    # a unit is a CHANNEL and its response is averaged over the
                    # spatial positions before the batch mean.
                    if a.dim() > 2:
                        a = a.mean(dim=tuple(range(2, a.dim())))
                    self._h[key] = a.mean(dim=0).cpu()
            self._handles.append(
                _resolve(model, bn).register_forward_hook(hook))

    def arm(self) -> None:
        """Open the window for one forward pass to be recorded."""
        self._h = {}

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    # -- the step ---------------------------------------------------------
    @torch.no_grad()
    def step(self, model, fisher) -> dict:
        if not self._h:
            return {}
        resets = 0
        for lin, bn, nxt in self.BLOCKS:
            h = self._h.get(lin)
            if h is None:
                continue
            nxt_w = _resolve(model, nxt).weight
            # Total outgoing weight magnitude PER UNIT of this block. Linear:
            # (out, in) -> sum over out. Conv2d: (out, in, kh, kw) -> sum over
            # out and the kernel too, or the result keeps its spatial dimensions
            # and cannot multiply the per-channel activation h.
            dims = (0,) + tuple(range(2, nxt_w.dim()))
            outgoing = nxt_w.abs().sum(dim=dims).detach().cpu()

            imp = fisher.neuron_importance(f"{nxt}.weight", model).detach().cpu()
            span = float(imp.max() - imp.min())
            norm = (imp - imp.min()) / span if span > 0 else torch.zeros_like(imp)
            self.C[lin] = ((1 - self.eta) * h * outgoing
                           + self.eta * self.C[lin]) * torch.sigmoid(norm)
            self.age[lin] += 1

            eligible = self.age[lin] > self.maturity
            n_eligible = int(eligible.sum())
            if not n_eligible:
                continue
            self.credit[lin] += self.phi * n_eligible
            while self.credit[lin] >= 1.0:
                scores = self.C[lin].clone()
                scores[~eligible] = float("inf")
                r = int(torch.argmin(scores))
                if not bool(eligible[r]):
                    break
                self._reinit(model, lin, bn, nxt, r)
                self.age[lin][r] = 0
                self.C[lin][r] = 0.0
                eligible[r] = False
                self.credit[lin] -= 1.0
                resets += 1
        return {"gum_resets": float(resets)}

    @torch.no_grad()
    def _reinit(self, model, lin: str, bn: str, nxt: str, r: int) -> None:
        layer, norm, following = (_resolve(model, lin), _resolve(model, bn),
                                  _resolve(model, nxt))
        bound = 1.0 / math.sqrt(_fan_in(layer))
        shape = tuple(layer.weight.shape[1:])          # () for Linear rows too
        fresh = (torch.rand(shape, generator=self.gen) * 2 - 1) * bound
        layer.weight[r].copy_(fresh.to(layer.weight.dtype))
        if layer.bias is not None:
            layer.bias[r] = float((torch.rand(1, generator=self.gen) * 2 - 1) * bound)
        norm.weight[r] = 1.0
        norm.bias[r] = 0.0
        if norm.running_mean is not None:
            norm.running_mean[r] = 0.0
            norm.running_var[r] = 1.0
        # Zero every outgoing connection this unit has in the following layer.
        # For a Linear that is one column; for a Conv2d it is one INPUT-CHANNEL
        # slice across the whole kernel, so the trailing spatial dimensions have
        # to be taken too -- `following.weight[:, r] = 0` happens to do both,
        # since indexing a (out, in, kh, kw) tensor on dim 1 selects the plane.
        following.weight[:, r] = 0.0


# ------------------------------------------------------------------ bound
def probe_target_error(experiment, tid: int, loader) -> float:
    """Error on the new task after one epoch, measured on a THROWAWAY copy.

    See the module docstring: doing this on the live model would hand the
    bound-based arm an extra epoch per task and stop the two detection
    strategies from being comparable.
    """
    clone = copy.deepcopy(experiment.model)
    opt = torch.optim.SGD(clone.parameters(), lr=experiment.hp["cl_lr"],
                          momentum=experiment.hp["momentum"])
    active = experiment.schedule.active_count(tid)
    ce = nn.CrossEntropyLoss()
    clone.train()
    from core.training import freeze_batchnorm
    if experiment.hp["freeze_bn"]:
        freeze_batchnorm(clone)
    for xb, yb in loader:
        xb, yb = xb.to(experiment.device), yb.to(experiment.device)
        opt.zero_grad()
        ce(clone(xb)[:, :active], yb).backward()
        opt.step()
    clone.eval()
    wrong = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = clone(xb.to(experiment.device))[:, :active].argmax(dim=1).cpu()
            wrong += int((pred != yb).sum())
            total += len(yb)
    del clone
    return wrong / max(1, total)
