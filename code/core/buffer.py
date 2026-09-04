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
                 space: str = "latent"):
        self.mem_size = int(mem_size)
        self.contamination = float(contamination)
        self.seed = int(seed)
        if space not in ("latent", "raw"):
            raise ValueError(f"buffer space must be 'latent' or 'raw', got {space!r}")
        # 'raw'    -- Isolation Forest fitted on the scaled INPUT features. This is
        #             MADAR as published: selection is a property of the data.
        # 'latent' -- fitted on the classifier's penultimate representation, so
        #             selection moves as the model learns (MADAR-theta).
        # The distinction is not cosmetic: it decides whether the buffer is chosen
        # in a fixed space or one the model is simultaneously reshaping.
        self.space = space
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
            half = n_select // 2
            anomalies = order[:half]
            inliers = order[-(n_select - half):]
            idx = [i for pair in zip(anomalies, inliers) for i in pair]
            if n_select % 2:
                idx.append(inliers[-1])
            idx = idx[:n_select]
            self.family_buffers[int(fam)] = [
                (torch.as_tensor(X_fam[i]).clone(), int(fam)) for i in idx]

        return {"budget_per_family": int(budget), "size": len(self),
                "n_families": len(self.family_buffers), "space": self.space}

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
        self.family_buffers = {int(k): v for k, v in state["family_buffers"].items()}


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

    def composition(self) -> dict:
        from collections import Counter
        return dict(sorted(Counter(self.y).items()))
