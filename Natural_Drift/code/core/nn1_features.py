"""Features NN-1 scores samples on.

TWO MODES, TWO FUNCTIONS. `buffer_features` scores RETAINED points against an
arriving task (mode B, nine features). `task_features` scores the ARRIVING
task's own samples (mode A, seven). The split is not cosmetic: mode A runs
before the buffer exists for this task, so the two relational features have no
second population to be relative to, and computing them would mean inventing
one.

--- mode B ---

Nine numbers per retained sample, describing it from three angles:

  how unusual it is          iso_latent, iso_raw, centroid_dist, log_family_size
  how the model handles it   ce_loss, entropy, margin
  how it relates to the NEW task   grad_conflict, density_ratio

The last two are what make mode B a different question from mode A. A buffer
point is not scored on its own merits but on whether it still earns its place
now that a new task has arrived:

  grad_conflict  cosine between this point's last-layer gradient and the MEAN
                 gradient of the incoming task. Negative means learning the new
                 task actively fights this memory.
  density_ratio  log-odds, from a logistic probe in latent space, that the point
                 came from the new task's distribution rather than the buffer's.
                 Low means the new distribution has moved away from it.

--- mode A ---

The same seven intrinsic features, over the arriving task's samples:

  how unusual it is          iso_latent, iso_raw, centroid_dist, log_family_size
  how the model handles it   ce_loss, entropy, margin

`grad_conflict` and `density_ratio` are deliberately absent, which is what
nn1.N_FEATURES = 7 already encodes. In mode A the candidate pool IS the new
task, so "does this point fight the new task?" has no meaning -- the reference
gradient would be the mean of the very population being scored, and every
cosine would be measured against its own average. A relational feature needs
two populations, and at this point in the task there is only one.

Everything is z-scored over the population being scored -- see nn1.zscore for
why that is per-task rather than global.

COST. The gradient feature materialises an (n, active*latent) matrix. At 2,500
buffer points, 80 active classes and a 128-dimensional latent that is 2.5 GB in
float32, so the new-task side is subsampled (`grad_max_n`) and the buffer side
is computed in batches. This is the expensive feature; if a budget forces one
to be dropped, it is this one, and `--zero-features grad_conflict` does so
honestly by recording the ablation in the scorer file.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression

from core import nn1


@torch.no_grad()
def _responses(model, X, y, active_count: int, device, batch_size: int):
    """Latents, per-sample loss, entropy and margin in one pass."""
    model.eval()
    lats, losses, probs = [], [], []
    loader = data.DataLoader(data.TensorDataset(X, y), batch_size=batch_size)
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits, lat = model(xb, return_latent=True)
        logits = logits[:, :active_count]
        lats.append(lat.cpu())
        losses.append(F.cross_entropy(logits, yb, reduction="none").cpu())
        probs.append(F.softmax(logits, dim=1).cpu())
    L = torch.cat(lats).numpy()
    loss = torch.cat(losses).numpy()
    p = torch.cat(probs).numpy()
    ent = -(p * np.log(p + 1e-9)).sum(1) / np.log(max(2, active_count))
    top2 = np.sort(p, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    return L, loss, ent, margin


@torch.no_grad()
def _last_layer_grads(model, X, y, active_count: int, device, batch_size: int,
                      max_n: int | None = None, seed: int = 0):
    """Per-sample gradient of the loss w.r.t. the classifier weights.

    For a softmax head the gradient is the outer product of the prediction error
    with the latent, so it needs no autograd pass -- which matters, because an
    autograd call per sample would dominate the run.
    """
    model.eval()
    if max_n is not None and len(X) > max_n:
        idx = np.random.default_rng(seed).choice(len(X), max_n, replace=False)
        X, y = X[idx], y[idx]
    out = []
    for xb, yb in data.DataLoader(data.TensorDataset(X, y), batch_size=batch_size):
        xb = xb.to(device)
        logits, lat = model(xb, return_latent=True)
        err = F.softmax(logits[:, :active_count], dim=1)
        err[torch.arange(len(yb)), yb.to(device)] -= 1.0
        g = torch.einsum("bc,bd->bcd", err, lat).reshape(len(yb), -1)
        out.append(g.cpu())
    return torch.cat(out).numpy().astype(np.float32)


def _intrinsic(model, X, y, active_count, device, batch_size, seed,
               contamination):
    """The seven features a sample has on its own, independent of any other set.

    Shared by both modes so a weight means the same thing in each: mode B's
    first seven columns and mode A's only seven are computed by this one
    function, not by two that drifted apart.
    """
    L, loss, ent, margin = _responses(model, X, y, active_count, device,
                                      batch_size)
    Xn = X.numpy().reshape(len(y), -1)
    iso_lat = IsolationForest(contamination=contamination, n_jobs=-1,
                              random_state=seed).fit(L).decision_function(L)
    iso_raw = IsolationForest(contamination=contamination, n_jobs=-1,
                              random_state=seed).fit(Xn).decision_function(Xn)
    yn = y.numpy()
    centroid = np.zeros(len(yn))
    fam_size = np.zeros(len(yn))
    for fam in np.unique(yn):
        m = yn == fam
        centroid[m] = np.linalg.norm(L[m] - L[m].mean(0), axis=1)
        fam_size[m] = np.log10(max(int(m.sum()), 1))
    return L, yn, [iso_lat, iso_raw, loss, ent, margin, centroid, fam_size]


def task_features(model, X, y, *, active_count: int, device,
                  batch_size: int = 512, seed: int = 0,
                  contamination: float = 0.1, zero_features=()) -> dict:
    """Mode A: the seven z-scored features over the ARRIVING task's samples.

    No new-task-relative columns, for the reason in the module docstring. The
    return shape matches buffer_features so a caller ranking by score does not
    need to know which mode produced it.
    """
    _, yn, cols = _intrinsic(model, X, y, active_count, device, batch_size,
                             seed, contamination)
    feats = nn1.apply_ablation(nn1.zscore(np.stack(cols, axis=1)),
                               zero_features, nn1.FEATURE_NAMES)
    return {"features": feats, "n": len(yn)}


def buffer_features(model, buf_X, buf_y, new_X, new_y, *, active_count: int,
                    device, batch_size: int = 512, seed: int = 0,
                    contamination: float = 0.1, grad_max_n: int = 2048,
                    density_max_n: int = 5000, zero_features=()) -> dict:
    """The nine z-scored features, plus the two raw columns baselines rank on.

    `grad_conflict` and `density_ratio` are returned raw as well, because two of
    mode B's baselines ARE those single features -- ranking on one column is the
    control that says whether the learned combination beat its own ingredients.
    """
    L, yn, cols = _intrinsic(model, buf_X, buf_y, active_count, device,
                             batch_size, seed, contamination)

    # grad_conflict
    g_buf = _last_layer_grads(model, buf_X, buf_y, active_count, device,
                              batch_size, seed=seed)
    g_new = _last_layer_grads(model, new_X, new_y, active_count, device,
                              batch_size, max_n=grad_max_n, seed=seed)
    ref = g_new.mean(0)
    ref = ref / (np.linalg.norm(ref) + 1e-12)
    conflict = (g_buf @ ref) / (np.linalg.norm(g_buf, axis=1) + 1e-12)

    # density_ratio
    take = np.random.default_rng(seed + 1).choice(
        len(new_y), min(len(new_y), density_max_n), replace=False)
    Ln, *_ = _responses(model, new_X[take], new_y[take], active_count,
                        device, batch_size)
    probe = LogisticRegression(max_iter=300)
    probe.fit(np.vstack([L, Ln]),
              np.concatenate([np.zeros(len(L)), np.ones(len(Ln))]))
    density = probe.decision_function(L)

    raw = np.stack(cols + [conflict, density], axis=1)
    feats = nn1.apply_ablation(nn1.zscore(raw), zero_features,
                               nn1.FEATURE_NAMES_B)
    return {"features": feats, "grad_conflict": conflict, "density_ratio": density,
            "n": len(yn)}
