"""MADAR-style diversity-aware replay buffer.

Per family, an Isolation Forest is fitted in the classifier's LATENT space and
samples are ranked by decision_function. The buffer takes the most anomalous
half and the most prototypical half, INTERLEAVED (anomaly, inlier, anomaly, ...)
so that later truncation to a smaller budget stays balanced between the two ends
rather than keeping whichever end happens to sit at the front of the list.

Note on `contamination`: it only moves Isolation Forest's decision threshold.
Selection here is purely rank-based on decision_function scores, so the value
does not change which samples are chosen. It is kept because the estimator
requires it.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.utils.data as data
from sklearn.ensemble import IsolationForest


class ReplayBuffer:
    """A bounded, per-family replay buffer.

    Entries are kept as CPU tensors. Batches move to the device per step; a
    buffer holding CUDA tensors breaks the moment a run resumes on a different
    device, and mixing the two breaks torch.stack outright.
    """

    def __init__(self, mem_size: int, contamination: float = 0.1, seed: int = 0,
                 space: str = "latent", fill: str = "both"):
        self.mem_size = int(mem_size)
        self.contamination = float(contamination)
        self.seed = int(seed)
        if space not in ("latent", "raw"):
            raise ValueError(f"buffer space must be 'latent' or 'raw', got {space!r}")
        # 'raw' on an image corpus would fit an Isolation Forest on 12,288 raw
        # pixel values per example. That is not MADAR-as-published transplanted
        # to images -- it is a different method that happens to share its name.
        # Pixel space has no per-feature meaning for a forest to split on, the
        # fit dominates the run's wall clock, and the resulting "anomalies" track
        # brightness and background rather than anything about the class. Refused
        # rather than silently run; latent space is the defensible choice here.
        self._refuse_raw_images = space == "raw"
        # 'raw'    -- Isolation Forest fitted on the scaled INPUT features. This is
        #             MADAR as published: selection is a property of the data.
        # 'latent' -- fitted on the classifier's penultimate representation, so
        #             selection moves as the model learns (MADAR-theta).
        # The distinction is not cosmetic: it decides whether the buffer is chosen
        # in a fixed space or one the model is simultaneously reshaping.
        self.space = space
        if fill not in ("both", "anomalous", "representative", "middle"):
            raise ValueError(f"buffer fill must be both|anomalous|"
                             f"representative|middle, got {fill!r}")
        # Which part of the ranking the buffer keeps. `both` is MADAR as
        # published and always refills from whatever it is given, so a band
        # removed from the TASK cannot be kept out of the BUFFER -- the next
        # rung is promoted instead. The one-sided fills can exclude a band
        # outright, which is the only way to test what replaying it costs.
        self.fill = fill
        self.family_buffers: dict[int, list] = {}

    # -- state ------------------------------------------------------------
    def __len__(self) -> int:
        return sum(len(v) for v in self.family_buffers.values())

    def is_empty(self) -> bool:
        return len(self) == 0

    def composition(self) -> dict:
        return {int(k): len(v) for k, v in sorted(self.family_buffers.items())}

    def tensors(self):
        entries = [e for fam in self.family_buffers.values() for e in fam]
        if not entries:
            return None, None
        X = torch.stack([e[0] for e in entries])
        y = torch.tensor([int(e[1]) for e in entries], dtype=torch.long)
        return X, y

    def loader(self, batch_size: int):
        X, y = self.tensors()
        if X is None:
            return None
        return data.DataLoader(data.TensorDataset(X, y),
                               batch_size=batch_size, shuffle=True)

    # -- update -----------------------------------------------------------
    def update(self, X: torch.Tensor, y: torch.Tensor, latents: np.ndarray) -> dict:
        if self._refuse_raw_images and X.dim() > 2:
            raise SystemExit(
                "--buffer-space raw is not available on an image corpus: it "
                "would fit the Isolation Forest on raw pixels. Use "
                "--buffer-space latent (the default).")
        """Fold one task's samples in, then rebalance every family to budget."""
        X_np = X.cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
        y_np = y.cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
        new_families = np.unique(y_np)

        total_families = len(set(self.family_buffers) | set(int(f) for f in new_families))
        budget = max(1, self.mem_size // max(1, total_families))

        # Shrink existing families first. The interleave means a prefix of the
        # list is already an even mix of anomalies and inliers.
        for fam, buf in self.family_buffers.items():
            if len(buf) > budget:
                self.family_buffers[fam] = buf[:budget]

        for fam in new_families:
            mask = y_np == fam
            X_fam, L_fam = X_np[mask], latents[mask]
            n_select = min(budget, len(X_fam))
            if n_select == 0:
                continue
            order = self._rank(L_fam if self.space == "latent" else X_fam)
            idx = self._pick(order, n_select)
            self.family_buffers[int(fam)] = [
                (torch.as_tensor(X_fam[i]).clone(), int(fam)) for i in idx]

        return {"budget_per_family": int(budget), "size": len(self),
                "n_families": len(self.family_buffers), "space": self.space}

    def would_keep(self, y: np.ndarray, latents: np.ndarray,
                   raw: np.ndarray | None = None) -> np.ndarray:
        """Indices this policy WOULD retain from one task, without storing them.

        Mode A's `--forget-pool complement` needs to know MADAR's choice before
        the forget set is picked, so the forget set can be drawn from what MADAR
        was going to discard anyway. Answering that by rerunning the selection
        here -- same budget arithmetic, same `_rank`, same interleave -- is what
        stops the two from drifting: a change to `update` that this method did
        not follow would show up immediately as a keep-set that does not match
        the buffer the same task produces.

        Returns indices into `y`, sorted.
        """
        y = np.asarray(y)
        new_families = np.unique(y)
        total = len(set(self.family_buffers) | {int(f) for f in new_families})
        budget = max(1, self.mem_size // max(1, total))
        keep = []
        for fam in new_families:
            idx = np.flatnonzero(y == fam)
            n_select = min(budget, len(idx))
            if n_select == 0:
                continue
            feats = latents[idx] if self.space == "latent" else raw[idx]
            order = self._rank(feats)
            picked = self._pick(order, n_select)
            keep.extend(int(idx[i]) for i in picked[:n_select])
        return np.array(sorted(set(keep)), dtype=np.int64)

    def _pick(self, order: np.ndarray, n_select: int) -> list:
        """Which `n_select` of a family's ranking this buffer keeps.

        `order` is ascending by decision_function, so index 0 is the most
        anomalous and -1 the most prototypical.

        Shared by `update` and `would_keep` deliberately. The two must agree:
        `--forget-pool complement` asks `would_keep` what the buffer will do and
        builds the forget set from the complement, so a rule applied in one and
        not the other yields a forget set that silently overlaps the buffer.

        `both` interleaves so a prefix of the stored list stays an even mix of
        the two ends, which is what lets the shrink step (`buf[:budget]`) keep
        that balance when a family's budget falls. The one-sided fills need no
        interleave -- a prefix of a one-sided list is already the right thing --
        but `representative` is reversed so that its prefix is the MOST
        prototypical, keeping the same "prefix is what you want most" contract.
        """
        if self.fill == "anomalous":
            return list(order[:n_select])
        if self.fill == "representative":
            return list(order[-n_select:][::-1])
        if self.fill == "middle":
            start = max(0, (len(order) // 2) - (n_select // 2))
            return list(order[start:start + n_select])
        half = n_select // 2
        anomalies = order[:half]
        inliers = order[-(n_select - half):]
        idx = [i for pair in zip(anomalies, inliers) for i in pair]
        if n_select % 2:
            idx.append(inliers[-1])
        return idx[:n_select]

    def _rank(self, latents: np.ndarray) -> np.ndarray:
        """Ascending decision_function order: most anomalous first."""
        if len(latents) < 2:
            return np.arange(len(latents))
        iso = IsolationForest(contamination=self.contamination, n_jobs=-1,
                              random_state=self.seed).fit(latents)
        return np.argsort(iso.decision_function(latents))

    def remove(self, mask: np.ndarray) -> int:
        """Drop flagged entries, indexed in `tensors()` order.

        Removal breaks the anomaly/inlier interleave, so subsequent truncation
        degrades from balanced to approximately-stratified. Documented rather
        than silently accepted.
        """
        mask = np.asarray(mask, dtype=bool)
        pos, removed = 0, 0
        for fam in list(self.family_buffers):
            entries = self.family_buffers[fam]
            keep = [e for j, e in enumerate(entries) if not mask[pos + j]]
            removed += len(entries) - len(keep)
            self.family_buffers[fam] = keep
            pos += len(entries)
        return removed

    def family_sizes(self) -> dict:
        """Per-family occupancy, for taking a target before a removal."""
        return {int(k): len(v) for k, v in self.family_buffers.items()}

    def refill_to(self, targets: dict, seed: int = 0) -> dict:
        """Resample each family back up to `targets[fam]` slots, WITH replacement.

        THE MATCHED-MEMORY CONTROL. Without it, arms are separated partly by how
        much replay capacity survived their removals rather than by which points
        they chose, and a concentrated selector is penalised for being
        concentrated: a removal costs nothing unless it drops a family below the
        shrinking per-family budget, so removals spread thin are nearly free and
        removals piled on one family are not. A ranking scorer is concentrated by
        construction, so that penalty tracks the method rather than its quality.

        WHY DUPLICATION AND NOT FRESH DATA. Removals land on OLD families, whose
        stream is gone -- the buffer is the only surviving record of them, so
        there is nothing else of that family left to draw. Topping up from the
        current task instead would hold total occupancy fixed while quietly
        converting old-family slots into new-family slots, which trades a size
        confound for a composition one and would flatter exactly the arms that
        removed the most. Duplicating survivors keeps each family's slot count
        -- and therefore its weight in the replay loss -- identical to the
        no-removal arm, so what differs between arms is only which DISTINCT
        points carry that weight. That is the comparison worth reporting.

        What it does NOT restore is diversity: a refilled family replays fewer
        distinct points more often. That is the honest residual of this control
        and belongs in the write-up, not in a footnote.

        A family removed down to zero cannot be refilled -- there is nothing to
        resample -- and is reported in `emptied` rather than silently skipped.
        """
        rng = np.random.default_rng(int(seed))
        added, emptied = 0, []
        for fam, target in sorted(targets.items()):
            fam = int(fam)
            have = self.family_buffers.get(fam, [])
            short = int(target) - len(have)
            if short <= 0:
                continue
            if not have:
                emptied.append(fam)
                continue
            # Appended at the end so that update()'s truncation to a smaller
            # budget drops duplicates before it drops distinct points.
            pick = rng.integers(0, len(have), size=short)
            self.family_buffers[fam] = have + [have[int(i)] for i in pick]
            added += short
        return {"refilled": int(added), "emptied_families": emptied,
                "size_after_refill": len(self)}

    # -- persistence ------------------------------------------------------
    def state_dict(self) -> dict:
        return {"mem_size": self.mem_size, "contamination": self.contamination,
                "seed": self.seed, "space": self.space,
                "family_buffers": self.family_buffers}

    def load_state_dict(self, state: dict) -> None:
        self.mem_size = int(state["mem_size"])
        self.contamination = float(state["contamination"])
        self.seed = int(state["seed"])
        self.space = state.get("space", "latent")
        # BACK TO CPU, ALWAYS. Both callers -- core/checkpoint.py's resume and
        # core/task0.py's shared task 0 -- load with map_location=device, which
        # is right for the model and wrong for this buffer: every stored entry
        # comes back on CUDA, while update() keeps adding new families as CPU
        # tensors. The two coexist happily until tensors() calls torch.stack on
        # the mix, and then the run dies on the FIRST task after the restore
        # with "expected all tensors to be on the same device".
        #
        # The class contract says these are CPU tensors, so the restore is where
        # it gets enforced. The bug is invisible under --device cpu, where
        # map_location is "cpu" and there is nothing to mix.
        self.family_buffers = {
            int(k): [(self._to_cpu(x), int(lab)) for x, lab in v]
            for k, v in state["family_buffers"].items()}

    @staticmethod
    def _to_cpu(t):
        return t.detach().cpu() if torch.is_tensor(t) else torch.as_tensor(t)


class ReservoirBuffer:
    """Uniform-random episodic memory over the whole stream (Vitter 1985).

    A-GEM's memory, as published: samples are admitted with probability that
    keeps the buffer a uniform sample of everything seen so far, with no notion
    of which samples are informative. That is the point of using it here rather
    than the MADAR buffer -- A-GEM's contribution is the gradient constraint,
    and pairing it with a diversity-aware selection rule would report a hybrid
    while calling it A-GEM.

    Same `mem_size` as the MADAR buffer, so the two differ in WHAT they store,
    not HOW MUCH.
    """

    def __init__(self, mem_size: int, seed: int = 0):
        self.mem_size = int(mem_size)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.X: list = []
        self.y: list = []
        self.n_seen = 0

    def __len__(self) -> int:
        return len(self.X)

    def is_empty(self) -> bool:
        return not self.X

    def add_stream(self, X, y) -> dict:
        """Reservoir-insert a task's samples, in order."""
        X = X.cpu() if torch.is_tensor(X) else torch.as_tensor(X)
        y = y.cpu() if torch.is_tensor(y) else torch.as_tensor(y)
        admitted = replaced = 0
        for i in range(len(y)):
            self.n_seen += 1
            if len(self.X) < self.mem_size:
                self.X.append(X[i].clone()); self.y.append(int(y[i]))
                admitted += 1
            else:
                j = int(self.rng.integers(0, self.n_seen))
                if j < self.mem_size:
                    self.X[j] = X[i].clone(); self.y[j] = int(y[i])
                    replaced += 1
        return {"size": len(self.X), "n_seen": self.n_seen,
                "admitted": admitted, "replaced": replaced,
                "n_families": len(set(self.y))}

    def sample(self, n: int):
        """A reference batch, drawn fresh each step as A-GEM specifies."""
        if not self.X:
            return None, None
        k = min(int(n), len(self.X))
        idx = self.rng.choice(len(self.X), k, replace=False)
        return (torch.stack([self.X[i] for i in idx]),
                torch.tensor([self.y[i] for i in idx], dtype=torch.long))

    # -- persistence ------------------------------------------------------
    def state_dict(self) -> dict:
        """Contents, the stream counter, AND the generator state.

        `n_seen` and the RNG are as load-bearing as the samples: Vitter's
        admission probability is 1/n_seen, so a buffer restored without the
        counter would start admitting new samples far too eagerly and stop being
        a uniform sample of the stream. And `rng` is the same generator that
        draws every reference batch, so restarting it would make a resumed run
        reuse the draws the earlier tasks already made.
        """
        return {"X": [x.clone() for x in self.X], "y": list(self.y),
                "n_seen": int(self.n_seen), "mem_size": int(self.mem_size),
                "rng": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        if int(state.get("mem_size", self.mem_size)) != self.mem_size:
            raise SystemExit(
                f"reservoir buffer was saved at mem_size "
                f"{state['mem_size']} but this run uses {self.mem_size}; "
                f"resuming would change the memory budget mid-run")
        # .cpu() for the reason spelled out in ReplayBuffer.load_state_dict:
        # torch.as_tensor on a CUDA tensor leaves it on CUDA, and sample()
        # stacks these against entries admitted later on the CPU side.
        self.X = [torch.as_tensor(x).detach().cpu() for x in state["X"]]
        self.y = [int(v) for v in state["y"]]
        self.n_seen = int(state["n_seen"])
        if state.get("rng") is not None:
            self.rng.bit_generator.state = state["rng"]

    def composition(self) -> dict:
        from collections import Counter
        return dict(sorted(Counter(self.y).items()))


class LogitBuffer:
    """Episodic memory storing model RESPONSES alongside (x, y).

    Required by DER/X-DER, which distil from logits recorded at insertion time
    rather than from a frozen teacher network. Two things make this more than
    `ReservoirBuffer` with an extra column:

    1. `sample` returns the buffer POSITIONS it drew. X-DER's "logits of future
       past" (Eq. 7) rewrites the stored responses of the very entries it just
       replayed, so the caller has to be able to address them. A sampler that
       returns only tensors makes that update impossible to express.
    2. `task_id` is kept per entry -- the task during which the entry was
       inserted -- so a caller can tell which heads were future at that moment.

    Logits are stored at FULL WIDTH (`n_classes`), not at the active prefix.
    X-DER's future heads are exactly the columns beyond the active prefix, so
    truncating at insertion would delete the thing the method operates on.

    POLICY. `balanced` (default) holds `mem_size // n_classes_seen` entries per
    class, trimming older classes as new ones arrive -- the same Uniform
    budgeting `ReplayBuffer` uses, so a DER-family row and a MADAR row differ in
    WHAT they keep, not how much. `reservoir` is Vitter uniform sampling over
    the stream, which under a long task-0 leaves later families barely present.
    The published X-DER uses a class-balanced buffer; the exact insertion
    routine in the authors' code is not reproduced here and has not been
    verified against it, so `reservoir` is kept selectable rather than
    silently ruled out.
    """

    def __init__(self, mem_size: int, n_classes: int, seed: int = 0,
                 policy: str = "balanced"):
        if policy not in ("balanced", "reservoir"):
            raise ValueError(
                f"logit buffer policy must be 'balanced' or 'reservoir', got {policy!r}")
        self.mem_size = int(mem_size)
        self.n_classes = int(n_classes)
        self.seed = int(seed)
        self.policy = policy
        self.rng = np.random.default_rng(seed)
        self.n_seen = 0
        self._X: torch.Tensor | None = None      # [m, d]  float32, CPU
        self._y = torch.zeros(self.mem_size, dtype=torch.long)
        self._z: torch.Tensor | None = None      # [m, C]  float32, CPU
        self._t = torch.zeros(self.mem_size, dtype=torch.long)
        self._n = 0                              # rows currently filled

    # -- state ------------------------------------------------------------
    def __len__(self) -> int:
        return self._n

    def is_empty(self) -> bool:
        return self._n == 0

    def composition(self) -> dict:
        from collections import Counter
        return dict(sorted(Counter(int(v) for v in self._y[:self._n]).items()))

    def tensors(self):
        if self._n == 0:
            return None, None
        return self._X[:self._n].clone(), self._y[:self._n].clone()

    def logits(self):
        if self._n == 0:
            return None
        return self._z[:self._n].clone()

    def task_ids(self):
        if self._n == 0:
            return None
        return self._t[:self._n].clone()

    def _allocate(self, shape, dtype=torch.float32) -> None:
        """Reserve the store from the PER-EXAMPLE SHAPE, not a feature count.

        `shape` is everything after the batch dimension: (d,) for a feature
        vector, (C, H, W) for an image. Taking only `X.shape[1]` here allocated a
        (mem_size, 3) store for images -- the channel count read as a feature
        count -- and the mismatch only surfaced on the first write.

        The dtype follows the incoming data so an image buffer stays uint8: 2000
        Tiny ImageNet images are 24 MB as uint8 and 98 MB as float32, and the
        float copy would buy nothing, since the model normalises on the way in.
        """
        if self._X is None:
            self._X = torch.zeros(self.mem_size, *tuple(shape), dtype=dtype)
            self._z = torch.zeros(self.mem_size, self.n_classes, dtype=torch.float32)

    @staticmethod
    def _cpu(t, dtype=None):
        t = t.detach().cpu() if torch.is_tensor(t) else torch.as_tensor(t)
        return t.to(dtype) if dtype is not None else t

    # -- insertion --------------------------------------------------------
    def add(self, X, y, logits, task_id: int) -> dict:
        """Fold one task's samples in under the configured policy."""
        if self.policy == "reservoir":
            return self.add_stream(X, y, logits, task_id)
        return self.add_balanced(X, y, logits, task_id)

    def add_stream(self, X, y, logits, task_id: int) -> dict:
        """Vitter reservoir insertion, one pass in stream order."""
        # Dtype preserved, not forced to float32: a tabular corpus already
        # arrives as float32, and an image corpus stays uint8 all the way to the
        # model, which normalises on the way in.
        X = self._cpu(X)
        y = self._cpu(y, torch.long)
        z = self._cpu(logits, torch.float32)
        self._check_width(z)
        self._allocate(X.shape[1:], X.dtype)
        admitted = replaced = 0
        for i in range(len(y)):
            self.n_seen += 1
            if self._n < self.mem_size:
                j, self._n = self._n, self._n + 1
                admitted += 1
            else:
                j = int(self.rng.integers(0, self.n_seen))
                if j >= self.mem_size:
                    continue
                replaced += 1
            self._write(j, X[i], y[i], z[i], task_id)
        return {"size": self._n, "n_seen": self.n_seen, "admitted": admitted,
                "replaced": replaced, "n_families": len(set(
                    int(v) for v in self._y[:self._n])), "policy": self.policy}

    def add_balanced(self, X, y, logits, task_id: int) -> dict:
        """Trim every held class to the new per-class budget, then fill the
        arriving classes to the same budget with a uniform random subset."""
        # Dtype preserved, not forced to float32: a tabular corpus already
        # arrives as float32, and an image corpus stays uint8 all the way to the
        # model, which normalises on the way in.
        X = self._cpu(X)
        y = self._cpu(y, torch.long)
        z = self._cpu(logits, torch.float32)
        self._check_width(z)
        self._allocate(X.shape[1:], X.dtype)
        self.n_seen += len(y)

        new_classes = sorted(set(int(v) for v in y))
        held = self._y[:self._n]
        old_classes = sorted(set(int(v) for v in held) - set(new_classes))
        budget = max(1, self.mem_size // max(1, len(old_classes) + len(new_classes)))

        keep_rows: list[int] = []
        for c in old_classes:
            rows = (held == c).nonzero(as_tuple=True)[0].tolist()
            keep_rows.extend(rows[:budget])
        # Rows arriving now overwrite everything else, so gather the survivors
        # before any position is reused.
        kept = (self._X[keep_rows].clone(), self._y[keep_rows].clone(),
                self._z[keep_rows].clone(), self._t[keep_rows].clone())

        self._n = 0
        for k in range(len(keep_rows)):
            self._write(self._n, kept[0][k], kept[1][k], kept[2][k], int(kept[3][k]))
            self._n += 1

        admitted = 0
        for c in new_classes:
            rows = (y == c).nonzero(as_tuple=True)[0].numpy()
            take = min(budget, len(rows), self.mem_size - self._n)
            if take <= 0:
                continue
            pick = self.rng.choice(rows, take, replace=False)
            for i in pick:
                self._write(self._n, X[i], y[i], z[i], task_id)
                self._n += 1
                admitted += 1

        return {"size": self._n, "n_seen": self.n_seen,
                "budget_per_class": int(budget), "admitted": admitted,
                "n_families": len(set(int(v) for v in self._y[:self._n])),
                "policy": self.policy}

    def _write(self, j: int, x, yi, zi, task_id: int) -> None:
        self._X[j] = x
        self._y[j] = int(yi)
        self._z[j] = zi
        self._t[j] = int(task_id)

    def _check_width(self, z: torch.Tensor) -> None:
        if z.ndim != 2 or z.shape[1] != self.n_classes:
            raise ValueError(
                f"logit buffer expects full-width responses [n, {self.n_classes}], "
                f"got {tuple(z.shape)}. Storing the active prefix only would "
                f"discard the future heads X-DER regularises.")

    # -- sampling ---------------------------------------------------------
    def sample(self, n: int, replace: bool = False):
        """Draw a replay batch. Returns (idx, X, y, logits, task_ids).

        `idx` addresses buffer rows and is what `update_logits` expects: the
        caller replays these entries and then writes the refreshed responses
        back to the same positions.
        """
        if self._n == 0:
            return None
        k = int(n) if replace else min(int(n), self._n)
        idx = torch.as_tensor(
            self.rng.choice(self._n, k, replace=replace), dtype=torch.long)
        return (idx, self._X[idx].clone(), self._y[idx].clone(),
                self._z[idx].clone(), self._t[idx].clone())

    def update_logits(self, idx: torch.Tensor, z: torch.Tensor) -> int:
        """Overwrite stored responses at `idx`. Rows, not columns: the caller
        decides which columns changed and passes the full row back."""
        idx = self._cpu(idx, torch.long)
        z = self._cpu(z, torch.float32)
        if z.shape[0] != idx.shape[0]:
            raise ValueError(f"update_logits: {idx.shape[0]} rows requested, "
                             f"{z.shape[0]} supplied")
        self._check_width(z)
        if self._n and int(idx.max()) >= self._n:
            raise IndexError(f"update_logits: row {int(idx.max())} outside "
                             f"buffer of size {self._n}")
        self._z[idx] = z
        return int(idx.shape[0])

    def remove(self, mask: np.ndarray) -> int:
        """Drop flagged entries, indexed in `tensors()` order."""
        mask = np.asarray(mask, dtype=bool)
        keep = np.flatnonzero(~mask[:self._n])
        if len(keep) == self._n:
            return 0
        removed = self._n - len(keep)
        kept = (self._X[keep].clone(), self._y[keep].clone(),
                self._z[keep].clone(), self._t[keep].clone())
        self._n = 0
        for k in range(len(keep)):
            self._write(self._n, kept[0][k], kept[1][k], kept[2][k], int(kept[3][k]))
            self._n += 1
        return removed

    # -- persistence ------------------------------------------------------
    def state_dict(self) -> dict:
        """Contents AND the generator that draws from them.

        `rng` is not optional. It is the stream every replay batch, every
        reservoir admission decision and -- through them -- X-DER's implant
        target and STAR's correctness gate are drawn from. Restoring the samples
        while restarting the generator gives a resumed run a different sequence
        of replay batches from the one a straight-through run would have seen,
        which is enough to move the final accuracy. Leaving it out was silent:
        the run resumed, finished, and simply produced a different number.
        """
        return {"mem_size": self.mem_size, "n_classes": self.n_classes,
                "seed": self.seed, "policy": self.policy, "n_seen": self.n_seen,
                "n": self._n,
                "rng": self.rng.bit_generator.state,
                "X": None if self._X is None else self._X[:self._n].clone(),
                "y": self._y[:self._n].clone(),
                "z": None if self._z is None else self._z[:self._n].clone(),
                "t": self._t[:self._n].clone()}

    def load_state_dict(self, state: dict) -> None:
        self.mem_size = int(state["mem_size"])
        self.n_classes = int(state["n_classes"])
        self.seed = int(state["seed"])
        self.policy = state.get("policy", "balanced")
        self.n_seen = int(state.get("n_seen", 0))
        n = int(state["n"])
        self._y = torch.zeros(self.mem_size, dtype=torch.long)
        self._t = torch.zeros(self.mem_size, dtype=torch.long)
        self._X = self._z = None
        self._n = 0
        if n:
            self._allocate(state["X"].shape[1:], state["X"].dtype)
            self._X[:n] = state["X"]; self._y[:n] = state["y"]
            self._z[:n] = state["z"]; self._t[:n] = state["t"]
            self._n = n
        if state.get("rng") is not None:
            self.rng.bit_generator.state = state["rng"]


class CoresetBuffer(LogitBuffer):
    """A LogitBuffer whose entries carry a coreset WEIGHT.

    GCR's replay terms are weighted sums, with the weights produced by the same
    optimisation that chose the entries (see core/coreset.py). A weight is
    therefore not decoration: dropping it changes the objective, because the
    gradient approximation the subset was selected to satisfy only holds for
    that weighting.

    SUBCLASSED RATHER THAN FOLDED IN. X-DER and its descendants share
    LogitBuffer and have no notion of a weight; widening their memory to carry
    one would change a type three running methods depend on, to no purpose.

    REPLACED WHOLESALE, not appended to. GCR re-solves over (buffer ++ candidate
    pool) at every task boundary, so the buffer after task t is a fresh
    selection rather than the previous one with entries added -- `replace_with`
    is the operation, and `add` is inherited only for building candidate pools.
    """

    def __init__(self, mem_size: int, n_classes: int, seed: int = 0,
                 policy: str = "balanced"):
        super().__init__(mem_size, n_classes, seed=seed, policy=policy)
        self._w = torch.ones(self.mem_size, dtype=torch.float32)

    def weights(self) -> torch.Tensor:
        return self._w[:self._n].clone()

    def weights_at(self, idx) -> torch.Tensor:
        return self._w[torch.as_tensor(idx, dtype=torch.long)].clone()

    def replace_with(self, X, y, z, task_ids, weights) -> dict:
        """Install a selected subset as the entire buffer."""
        n = len(y)
        if n > self.mem_size:
            raise ValueError(f"coreset of {n} exceeds mem_size {self.mem_size}")
        self._allocate(tuple(X.shape[1:]))
        self._X[:n] = self._cpu(X, torch.float32)
        self._y[:n] = self._cpu(y, torch.long)
        self._z[:n] = self._cpu(z, torch.float32)
        self._t[:n] = self._cpu(torch.as_tensor(task_ids), torch.long)
        self._w[:n] = self._cpu(torch.as_tensor(weights), torch.float32)
        self._n = int(n)
        return {"size": self._n, "n_families": len(self.composition()),
                "weight_mean": float(self._w[:n].mean()) if n else 0.0,
                "weight_max": float(self._w[:n].max()) if n else 0.0,
                "weight_min": float(self._w[:n].min()) if n else 0.0}

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["w"] = self._w[:self._n].clone()
        return state

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        self._w = torch.ones(self.mem_size, dtype=torch.float32)
        w = state.get("w")
        if w is not None:
            self._w[:len(w)] = torch.as_tensor(w, dtype=torch.float32)
