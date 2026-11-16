"""Objective pieces for X-DER (Boschini et al., "Class-Incremental Continual
Learning into the eXtended DER-verse", TPAMI 45(5), 2023).

The method is DER++ plus three things, each isolated in one function here so the
experiment module reads as the paper's Eq. (12) and an ablation switches one
term off rather than editing a loop:

  implant_future_past   Eq. (7)  -- keep stored logits current, rescaled
  future_prep_loss      Eq. (9)  -- supervised contrastive warm-up of unseen heads
  past_future_penalty   Eq. (11) -- hold past/future responses below ground truth

and one change to the cross-entropy itself (Eq. 10), which is a two-line slice
and lives in the experiment.

THE ONE SUBSTITUTION. Eq. (8) needs a strongly augmented second view of every
example. X-DER is an image method; crop, flip and colour jitter have no analogue
for a 2381-dimensional static malware feature vector, so a substitute is
unavoidable and is made explicit rather than smuggled in:

  * `noise`   -- additive Gaussian, scaled by each feature's own standard
                 deviation IN THE BATCH. Scaling by the batch std rather than a
                 fixed sigma is what makes one setting valid across corpora:
                 EMBER is standardised to unit variance, LAMDA is raw binary
                 indicators, and a constant sigma would be imperceptible on one
                 and destructive on the other.
  * `dropout` -- zero a random subset of features. On a standardised corpus zero
                 is the task-0 mean, and on a binary corpus it is "this feature
                 was not observed"; both read as masking, which is the closest
                 honest analogue of the cutout-style augmentation X-DER uses.
  * `both`    -- dropout then noise (default).
  * `none`    -- the two views are IDENTICAL. Eq. (8) still runs but every
                 anchor's positive set contains an exact copy of itself, so the
                 objective is nearly trivially satisfiable. Kept only as a
                 control that measures how much of L_FP's effect is the
                 augmentation.

`--xder-lambda 0` disables L_FP outright, which reproduces the paper's own
"X-DER w/ CE future heads" territory without pretending the augmentation
question has been answered.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

AUG_MODES = ("both", "noise", "dropout", "none")


# ------------------------------------------------------------------ views
def feature_augment(x: torch.Tensor, mode: str = "both", sigma: float = 0.1,
                    drop: float = 0.1, generator=None) -> torch.Tensor:
    """A second view of `x`. See the module docstring for why this is a choice."""
    if mode not in AUG_MODES:
        raise ValueError(f"--xder-aug must be one of {AUG_MODES}, got {mode!r}")
    if mode == "none":
        return x
    out = x
    if mode in ("both", "dropout") and drop > 0:
        keep = torch.rand(out.shape, device=out.device, generator=generator) >= drop
        out = out * keep
    if mode in ("both", "noise") and sigma > 0:
        # Per-feature scale from the batch itself, so one sigma is meaningful on
        # a standardised corpus and on a binary one alike.
        scale = out.std(dim=0, keepdim=True).clamp_min(1e-6)
        noise = torch.randn(out.shape, device=out.device, generator=generator)
        out = out + sigma * scale * noise
    return out


# ------------------------------------------------------------------ heads
def head_bounds(schedule, tid: int) -> tuple[int, int, int]:
    """(present_lo, present_hi, n_classes) for task `tid`.

    Past heads are [0, present_lo), present [present_lo, present_hi), future
    [present_hi, n_classes). Task 0 introduces the base set, so its present
    block starts at 0 rather than at prev_active_count.
    """
    hi = schedule.active_count(tid)
    lo = 0 if tid == 0 else schedule.prev_active_count(tid)
    return lo, hi, schedule.n_classes


def future_task_blocks(schedule, tid: int) -> list[tuple[int, int]]:
    """The [lo, hi) column block of each task after `tid`."""
    return [(schedule.prev_active_count(j), schedule.active_count(j))
            for j in range(tid + 1, schedule.n_tasks)]


# ------------------------------------------------------------------ Eq. (9)
def _supcon(view: torch.Tensor, labels: torch.Tensor, temp: float) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al.) over one L2-normalised block.

    `view` is [2N, k] already normalised; `labels` is [2N]. Averaged over
    anchors, and within an anchor over its positives -- the 1/|P(i)| of Eq. (9).
    """
    n = view.shape[0]
    sim = (view @ view.T) / temp
    eye = torch.eye(n, dtype=torch.bool, device=view.device)
    sim = sim.masked_fill(eye, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

    positive = (labels[:, None] == labels[None, :]) & ~eye
    n_pos = positive.sum(1)
    have = n_pos > 0
    if not bool(have.any()):
        return view.new_zeros(())
    per_anchor = (log_prob.masked_fill(~positive, 0.0) * positive).sum(1)
    return -(per_anchor[have] / n_pos[have]).mean()


def future_prep_loss(logits: torch.Tensor, labels: torch.Tensor, schedule,
                     tid: int, temp: float = 0.1) -> torch.Tensor:
    """Eq. (9): supervised contrastive loss on each future task's head block.

    `logits` is [2N, C] -- the original examples stacked on their augmented
    views -- and `labels` is the matching [2N]. Returns 0 when no future heads
    remain, which is the last task rather than an error.
    """
    blocks = future_task_blocks(schedule, tid)
    if not blocks:
        return logits.new_zeros(())
    total = logits.new_zeros(())
    for lo, hi in blocks:
        block = F.normalize(logits[:, lo:hi], dim=1)
        total = total + _supcon(block, labels, temp)
    return total / len(blocks)


# ------------------------------------------------------------------ Eq. (11)
def past_future_penalty(logits: torch.Tensor, labels: torch.Tensor,
                        present_lo: int, present_hi: int,
                        margin: float = 0.3) -> torch.Tensor:
    """Eq. (11): the largest past and largest future response must stay at least
    `margin` below the ground-truth logit.

    The ground-truth column is excluded from both maxima. Without that exclusion
    a replayed example whose own class is a past class would be penalised for
    its own correct answer -- the constraint would fight the objective.
    """
    n, c = logits.shape
    gt = logits.gather(1, labels[:, None]).squeeze(1)
    idx = torch.arange(c, device=logits.device)[None, :].expand(n, c)
    not_gt = idx != labels[:, None]

    loss = logits.new_zeros(())
    for lo, hi in ((0, present_lo), (present_hi, c)):
        if hi <= lo:
            continue
        block = logits[:, lo:hi].masked_fill(~not_gt[:, lo:hi], float("-inf"))
        block_max = block.max(dim=1).values
        finite = torch.isfinite(block_max)
        if not bool(finite.any()):
            continue
        loss = loss + F.relu(block_max[finite] - gt[finite] + margin).mean()
    return loss


# ------------------------------------------------------------------ Eq. (7)
@torch.no_grad()
def implant_future_past(stored: torch.Tensor, live: torch.Tensor,
                        stored_y: torch.Tensor, lo: int, hi: int,
                        gamma: float = 0.75) -> torch.Tensor:
    """Eq. (7): write the current task's responses into stored memory logits.

    The implanted block is rescaled so its maximum stays below the ground-truth
    logit ALREADY IN MEMORY, attenuated by gamma. Overwriting without that
    rescaling would feed the model's own bias towards present classes straight
    back through replay, which is the failure the equation exists to prevent.

    Two degenerate cases the paper does not spell out and that do occur here:
    when the block maximum is already <= 0 no rescaling is needed (any
    non-negative scale keeps it below a positive ground truth), and when the
    stored ground-truth logit is itself <= 0 no positive scale satisfies the
    constraint, so the implant is suppressed rather than allowed through.
    """
    out = stored.clone()
    block = live[:, lo:hi]
    fp_max = block.max(dim=1).values
    gt = stored.gather(1, stored_y[:, None]).squeeze(1)

    scale = torch.ones_like(fp_max)
    needs = fp_max > 0
    ok = needs & (gt > 0)
    scale[ok] = torch.clamp(gamma * gt[ok] / fp_max[ok], max=1.0)
    scale[needs & ~ok] = 0.0

    out[:, lo:hi] = block * scale[:, None]
    return out
