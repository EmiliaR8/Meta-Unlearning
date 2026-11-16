"""Mode B: unlearn BUFFER points that have stopped earning their place.

Mode A asks which of the ARRIVING data to forget. This asks the opposite
question about the other side of the replay boundary: now that a new task has
appeared, which RETAINED memories are actively getting in the way?

Each buffer point is scored (see core/nn1_features.py), the top `forget_ratio`
fraction is chosen, and those points are then:

  1. REMOVED FROM THE BUFFER, permanently. They never replay again. This is the
     part that makes it curation rather than regularisation, and it happens
     BEFORE the unlearning step so the retain stream no longer contains the very
     points being forgotten -- otherwise the two halves of the loss would be
     pulling on the same samples.
  2. pushed toward uniform over the OLD-knowledge logit slice [0, prev_active).
     Not the new-task slice madar_unlearn flattens: a buffer point's knowledge
     lives in the classes it belongs to, and flattening columns it was never
     about would be a no-op dressed as an intervention.

The current task's data is fully retained here -- mode B never deletes arriving
data -- so the retain anchor is the whole task loader plus the cleaned buffer.

SELECTORS. `nn1` is the learned scorer. The others are the controls that decide
whether learning bought anything: `none` (no forgetting at all), `random` (same
budget, no signal), and -- the two that matter -- `gradconflict` and
`densityratio`, which rank on a SINGLE one of NN-1's own features. Beating
random is not evidence; beating those two is.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.utils.data as data

from core import nn1, training
from core.nn1_features import buffer_features

from .base import register
from .madar_unlearn import MadarUnlearnExperiment

BUFFER_SELECTORS = ("nn1", "none", "random", "gradconflict", "densityratio")


@register("meta_unlearn")
class MetaUnlearnExperiment(MadarUnlearnExperiment):
    """madar_unlearn's machinery, pointed at the buffer instead of the task."""

    def flatten_range(self, tid: int) -> tuple[int, int]:
        # The OLD-knowledge slice; see the module docstring.
        return 0, self.schedule.prev_active_count(tid)

    def _apply_family_cap(self, forget_idx, buf_y, n_forget: int, summary: dict):
        """Take `n_forget` from a full ranking, bounding the depth per family.

        With the cap off this is just the top `n_forget` of the ranking.

        WHY THIS EXISTS. Matched memory (refill_policy) equalises how many
        replay SLOTS survive, and on EMBER 2018 that changed the ranking by
        0.01 points -- capacity was not what separated the selectors. What still
        differed was DEPTH: at equal total volume, `gradconflict` took about a
        third of every family it touched while `random` took about a tenth, and
        duplication restores a cut family's slots but not its distinct points.
        Depth and selection quality are therefore still tangled, and capping
        depth is what separates them: under a cap every selector cuts equally
        deep, so a surviving gap can only be about WHICH points were chosen.

        VOLUME IS PRESERVED, deliberately. The ranked order is walked and a
        point is admitted only while its family is under cap, continuing past
        skipped points until `n_forget` are admitted. A cap that simply dropped
        the overflow would shrink the forget set for concentrated selectors and
        trade a depth confound for a volume one.

        If the cap is so tight that the whole buffer cannot supply `n_forget`
        admissible points, fewer are removed and the shortfall is recorded
        rather than passed over in silence.
        """
        forget_idx = np.asarray(forget_idx, dtype=int)
        frac = float(self.hp.get("forget_per_family_cap", 0.0) or 0.0)
        if frac <= 0:
            return forget_idx[:n_forget]
        if frac > 1:
            raise SystemExit(
                f"--forget-per-family-cap is a fraction in (0, 1], got {frac}")

        y = buf_y.numpy()
        sizes = np.bincount(y, minlength=int(y.max()) + 1)
        # At least one, or a tight cap on a small family would forbid every
        # removal there and quietly concentrate the quota on the large families.
        caps = {int(f): max(1, int(np.floor(sizes[f] * frac)))
                for f in np.unique(y)}
        taken: dict[int, int] = {}
        admitted, skipped = [], 0
        for i in np.asarray(forget_idx, dtype=int):
            fam = int(y[i])
            if taken.get(fam, 0) >= caps[fam]:
                skipped += 1
                continue
            taken[fam] = taken.get(fam, 0) + 1
            admitted.append(int(i))
            if len(admitted) >= n_forget:
                break
        summary["family_cap"] = {
            "fraction": frac, "skipped_over_cap": int(skipped),
            "admitted": len(admitted), "requested": int(n_forget),
            "shortfall": int(max(0, n_forget - len(admitted))),
            "max_depth_allowed": max(caps.values()) if caps else 0}
        return np.asarray(admitted, dtype=int)

    def select_forget_set(self, tid: int, loader):
        name = self.cfg.get("selector") or "nn1"
        if name not in BUFFER_SELECTORS:
            raise SystemExit(
                f"meta_unlearn takes a buffer selector "
                f"({', '.join(BUFFER_SELECTORS)}), got {name!r}")

        buf_X, buf_y = (self.buffer.tensors() if self.buffer else (None, None))
        summary = {"selector": name, "target": "buffer"}
        if buf_X is None or len(buf_y) == 0:
            summary["skipped"] = "empty buffer"
            return loader, None, summary

        ratio = float(self.hp["forget_ratio"])
        n = int(len(buf_y))
        n_forget = int(n * ratio)
        summary.update({"ratio": ratio, "n_candidates": n})
        if name == "none" or n_forget == 0:
            summary["skipped"] = ("selector 'none'" if name == "none"
                                  else "empty forget set at this ratio")
            return loader, None, summary

        new_X, new_y = loader.dataset.tensors
        active = self.schedule.active_count(tid)
        seed = int(self.cfg["seed"])

        # Every selector produces a FULL ranking, best-to-remove first, and the
        # truncation to `n_forget` happens afterwards. That ordering matters:
        # under a per-family depth cap the walk skips points whose family is at
        # cap and has to keep going down the ranking to make up the volume. If
        # a selector handed back only its top `n_forget`, a cap would leave it
        # with nothing to fall back on and would shrink the forget set --
        # replacing the depth confound with a volume one.
        if name == "random":
            rng = np.random.default_rng(seed * 1000 + tid)
            ranked = rng.permutation(n)
        else:
            feats = buffer_features(
                self.model, buf_X, buf_y, new_X, new_y,
                active_count=active, device=self.device,
                batch_size=self.hp["eval_batch_size"], seed=seed,
                contamination=self.hp["contamination"],
                zero_features=self._zero_features())
            if name == "gradconflict":
                # Most negative cosine first: these fight the new task hardest.
                ranked = np.argsort(feats["grad_conflict"])
            elif name == "densityratio":
                # Least likely under the new distribution first.
                ranked = np.argsort(feats["density_ratio"])
            else:
                ranked = self._nn1_ranking(feats["features"], summary)

        forget_idx = self._apply_family_cap(ranked, buf_y, n_forget, summary)

        mask = np.zeros(n, dtype=bool)
        mask[np.asarray(forget_idx, dtype=int)] = True
        fam, counts = np.unique(buf_y.numpy()[mask], return_counts=True)
        summary.update({"n_forget": int(mask.sum()),
                        "n_old_families_touched": int(len(fam)),
                        "max_removed_from_one_family": int(counts.max()) if len(counts) else 0})

        # Remove BEFORE unlearning: see the module docstring.
        before = self.buffer.family_sizes()
        summary["removed_from_buffer"] = int(self.buffer.remove(mask))
        summary["buffer_size_after"] = int(len(self.buffer))

        policy = self.hp.get("refill_policy", "none")
        if policy not in ("none", "family"):
            raise SystemExit(
                f"--refill-policy must be 'none' or 'family', got {policy!r}")
        if policy == "family":
            # Matched memory: every arm hands the same number of replay slots
            # per family to everything downstream, so the reward reflects the
            # selection and not the capacity the selection happened to cost.
            # Seeded off the task so refills differ per task but reproduce.
            summary["refill"] = self.buffer.refill_to(
                before, seed=int(self.cfg["seed"]) * 7919 + tid)
        summary["refill_policy"] = policy

        forget_loader = data.DataLoader(
            data.TensorDataset(buf_X[mask], buf_y[mask]),
            batch_size=self.hp["batch_size"], shuffle=True)
        # Current-task data is untouched in mode B, so it IS the retain anchor.
        return loader, forget_loader, summary

    # -- the learned scorer ------------------------------------------------
    # `_zero_features` is inherited from MadarUnlearnExperiment: both modes read
    # the same hyperparameter the same way, and two identical copies is two
    # things to keep in step.
    def _nn1_ranking(self, features, summary: dict):
        path = self.hp.get("scorer_weights")
        if not path:
            raise SystemExit(
                "selector 'nn1' needs --scorer-weights pointing at a saved "
                "scorer (.npz). Train one with `./iclr meta`, or use "
                "--selector random / gradconflict / densityratio for a control.")
        params, hidden, nf = nn1.load(path)
        if nf != nn1.N_FEATURES_B:
            raise SystemExit(
                f"meta_unlearn is mode B and needs a {nn1.N_FEATURES_B}-feature "
                f"scorer; {path} has {nf}. Grow a mode-A scorer with nn1.grow "
                f"if that is what you meant.")
        extra = nn1.check_ablation(path, self._zero_features())
        if extra:
            summary["ablation_note"] = (
                f"zeroing {extra} on a scorer trained with them active")
        summary.update({"scorer_weights": str(path), "scorer_hidden": int(hidden)})
        # The FULL order, highest score (most forgettable) first -- the same
        # ranking nn1.select_forget thresholds, left untruncated so the caller
        # can apply the depth cap and the volume target itself.
        return np.argsort(nn1.score(params, features, hidden))[::-1]
