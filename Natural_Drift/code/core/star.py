"""STAR -- Stability-inducing weight perturbation (Eskandar et al., ICLR 2025).

A plug-in regulariser for any rehearsal-based continual learner. The idea is to
stop measuring forgetting after the fact and instead penalise it in advance: if
the model's output distribution over samples it CURRENTLY gets right is flat in
a neighbourhood of the present weights, then whatever direction future updates
take, those samples stay right.

    L_STAR(theta) = max_{||delta|| <= d} sum_{(x,y) in M*} KL( q_theta(x) || q_{theta+delta}(x) )
    L_final       = L_CL(theta) + lambda * L_STAR(theta)                       Eq. (7), (8)

    M* = the buffer samples the model classifies CORRECTLY right now.

Three implementation points, all from the paper's Section 4.3 and Algorithm 1:

  * delta = 0 is the global minimiser of the KL term and so has zero gradient.
    The ascent therefore starts from noise, delta_0 ~ N(0, eps*||theta_l||),
    scaled per layer (Eq. 9). With eps = 0 the perturbation is identically zero
    and STAR silently becomes a no-op -- which is why it is a parameter with a
    positive default rather than a hardcoded constant.

  * The perturbation is normalised PER PARAMETER TENSOR, not globally:
    delta_l = delta_0 + gamma * (||theta_l|| / ||g_l||) * g_l (Eq. 11). Weights
    are scale-invariant across layers -- multiplying one layer by 10 and
    dividing the next by 10 gives the same network -- so a single global norm
    constraint would perturb the layers unequally for no principled reason.

  * The descent gradient is approximated by the gradient AT THE PERTURBED POINT,
    grad_theta L_FG ~= grad_{theta+delta} L_FG, avoiding a Hessian-vector
    product. This is an approximation and the paper says so; it is also why more
    ascent steps stop helping past 1-3 (their Appendix D).

COST: two extra forward/backward passes per step, plus one no-grad forward for
the reference distribution. On the 3M-parameter MLP here that is cheap; on a
ResNet it is not, which is why the paper argues about it and this does not.

WHAT M* IS EVALUATED UNDER. "Correctly classified" uses the same masked-prefix
argmax the evaluation protocol uses, so a sample counts as correct here exactly
when the reported accuracy counts it as correct. If none of the drawn buffer
samples are correct, the step contributes no STAR gradient and says so
(`star_skipped`) rather than dividing by zero.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F


@contextlib.contextmanager
def batchnorm_eval(model):
    """Hold every BatchNorm in eval for the duration, then restore each module's
    own prior flag.

    STAR runs three extra forward passes per step, two of them at PERTURBED
    weights. In train mode those would fold perturbed-network activations into
    the running statistics -- silently, and only for the STAR rows. Toggling
    `model.eval()` and then `model.train()` would be worse still: it would undo
    the harness's `--freeze-bn`, so the fix has to restore flags per module
    rather than en masse.
    """
    saved = [(m, m.training) for m in model.modules()
             if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
    for m, _ in saved:
        m.eval()
    try:
        yield
    finally:
        for m, was in saved:
            m.train(was)


def _trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


@torch.no_grad()
def _zero_grads(params) -> None:
    for p in params:
        if p.grad is not None:
            p.grad = None


@torch.no_grad()
def correct_mask(model, x, y, active_count: int) -> torch.Tensor:
    """Which of these buffer samples the model currently gets right, under the
    same masked-prefix rule the evaluation uses."""
    with batchnorm_eval(model):
        pred = model(x)[:, :active_count].argmax(dim=1)
    return pred == y


def _kl(model, x, reference: torch.Tensor, active_count: int) -> torch.Tensor:
    """KL( q_theta(x) || q_perturbed(x) ), with `reference` = q_theta(x) fixed.

    F.kl_div(input, target) computes sum target * log(target/input), so the
    reference distribution goes in as the TARGET. Getting this round the wrong
    way computes the reverse KL, which is a different (mode-seeking) objective
    and would still train.
    """
    log_q = F.log_softmax(model(x)[:, :active_count], dim=1)
    return F.kl_div(log_q, reference, reduction="batchmean")


def star_gradient(model, x, y, *, active_count: int, lam: float, gamma: float,
                  eps: float, steps: int = 1) -> dict:
    """Add lambda * grad L_STAR to p.grad, leaving parameters unchanged.

    Call with p.grad empty and before the CL backward: this is the
    `u <- lambda * grad L_STAR; u <- u + grad L_CL` of Algorithm 1. Returns
    diagnostics; `star_kl` is the value of the stability term at the perturbed
    point, which is what the regulariser is actually pushing down.
    """
    with batchnorm_eval(model):
        return _star_gradient(model, x, y, active_count=active_count, lam=lam,
                              gamma=gamma, eps=eps, steps=steps)


def _star_gradient(model, x, y, *, active_count, lam, gamma, eps, steps) -> dict:
    params = _trainable(model)
    keep = correct_mask(model, x, y, active_count)
    if not bool(keep.any()):
        return {"star_skipped": 1.0}
    x = x[keep]

    with torch.no_grad():
        reference = F.softmax(model(x)[:, :active_count], dim=1)

    theta0 = [p.detach().clone() for p in params]
    # Eq. (9): a per-tensor noise floor, so the ascent has a non-zero gradient
    # to follow out of the exact minimum at delta = 0.
    with torch.no_grad():
        delta = [torch.randn_like(p) * (eps * p.norm()) for p in params]
        for p, d in zip(params, delta):
            p.add_(d)

    for _ in range(max(1, int(steps))):
        _zero_grads(params)
        _kl(model, x, reference, active_count).backward()
        with torch.no_grad():
            for p, t0, d in zip(params, theta0, delta):
                g = p.grad
                if g is None:
                    continue
                gn = g.norm()
                if gn <= 0:
                    continue
                # Eq. (11), applied per tensor: ||delta_l|| / ||theta_l|| ~= gamma.
                # With --star-steps > 1 the increments ACCUMULATE, so the total
                # ratio is roughly steps * gamma. The paper reports multi-step
                # results without restating the constraint, and rescaling to hold
                # the ratio at gamma would make the sweep measure something else.
                d.add_(gamma * (t0.norm() / gn) * g)
                p.copy_(t0 + d)

    _zero_grads(params)
    kl = _kl(model, x, reference, active_count)
    (lam * kl).backward()

    with torch.no_grad():
        for p, t0 in zip(params, theta0):
            p.copy_(t0)

    return {"star_kl": float(kl.item()),
            "star_correct_frac": float(keep.float().mean().item()),
            "star_skipped": 0.0}
