"""NN-1: the tiny per-sample scorer whose weights are meta-learned.

The whole point of the architecture is that it is SMALL. Evolution Strategies
estimates a gradient from scalar rewards, and each reward costs a full
continual-learning run; a scorer with thousands of parameters would need more
evaluations than the compute budget allows. The default is linear -- one weight
per feature plus a bias -- which is both within reach of a small population and
directly readable: each learned weight answers "does this property make a sample
worth forgetting?". A one-hidden-layer variant exists for once the linear form
shows signal, and will need a substantially larger budget.

MODE A scores CURRENT-TASK samples (7 features): which of the arriving data to
unlearn. MODE B scores BUFFER points against the new task's distribution (the
same 7 plus `grad_conflict` and `density_ratio`): which retained memories have
stopped paying their way. B is a strict superset of A's feature set, so a
scorer trained on A can be grown into B by zero-initialising the two new
weights -- see `grow`.

SIGN IS LEARNED. Selection forgets the TOP `ratio` fraction by score, and ES is
free to give any feature a negative weight, so no feature needs a hand-chosen
orientation.

ABLATIONS ARE RECORDED IN THE FILE. `zero_features` is stored alongside the
weights because ES keeps perturbing a zeroed feature's weight -- it simply has
no effect on the reward, so it drifts. Evaluating such a scorer WITHOUT the same
ablation would activate a weight that was never trained against anything, which
looks like a trained policy and is not one. Callers are expected to check.
"""

from __future__ import annotations

import numpy as np

N_FEATURES = 7            # mode A: current-task selection
N_FEATURES_B = 9          # mode B: buffer selection against the new task

FEATURE_NAMES = ["iso_latent", "iso_raw", "ce_loss", "entropy",
                 "margin", "centroid_dist", "log_family_size"]
FEATURE_NAMES_B = FEATURE_NAMES + ["grad_conflict", "density_ratio"]


def feature_names(n_features: int) -> list[str]:
    if n_features == N_FEATURES:
        return list(FEATURE_NAMES)
    if n_features == N_FEATURES_B:
        return list(FEATURE_NAMES_B)
    raise ValueError(f"no feature names for a {n_features}-feature scorer")


def n_params(hidden: int = 0, n_features: int = N_FEATURES) -> int:
    if hidden == 0:
        return n_features + 1                       # linear + bias
    return (n_features * hidden + hidden) + (hidden + 1)


def init_params(hidden: int = 0, seed: int = 0, scale: float = 0.3,
                n_features: int = N_FEATURES) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n_params(hidden, n_features)) * scale).astype(np.float64)


def score(params, features, hidden: int = 0) -> np.ndarray:
    """features: (n_samples, n_features), already z-scored. -> (n_samples,)"""
    p = np.asarray(params, dtype=np.float64)
    nf = features.shape[1]
    want = n_params(hidden, nf)
    if p.shape != (want,):
        raise ValueError(
            f"expected {want} parameters for hidden={hidden}, n_features={nf}; "
            f"got {p.shape}")
    if hidden == 0:
        return features @ p[:nf] + p[nf]
    i = 0
    W1 = p[i:i + nf * hidden].reshape(nf, hidden); i += nf * hidden
    b1 = p[i:i + hidden]; i += hidden
    W2 = p[i:i + hidden]; i += hidden
    b2 = p[i]
    return np.tanh(features @ W1 + b1) @ W2 + b2


def select_forget(params, features, ratio: float, hidden: int = 0):
    """-> (forget_idx, retain_idx). Forgets the top `ratio` fraction by score."""
    s = score(params, features, hidden)
    n_forget = int(len(s) * ratio)
    order = np.argsort(s)
    if n_forget <= 0:
        return np.array([], dtype=int), order
    return order[-n_forget:], order[:-n_forget]


def zscore(X, eps: float = 1e-8) -> np.ndarray:
    """Standardise each feature over the population being scored.

    Done per task rather than globally: the features have wildly different
    natural scales (an Isolation Forest score, a cross-entropy, a log count),
    and their distributions move as the model trains. Without this the learned
    weights would be fitting scale rather than signal, and a weight learned at
    task 1 would mean something different at task 10.
    """
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    return (X - mu) / (sd + eps)


def apply_ablation(features: np.ndarray, zero_features, names) -> np.ndarray:
    """Zero the named feature columns so they cannot contribute to the score."""
    if not zero_features:
        return features
    out = np.array(features, copy=True)
    unknown = [f for f in zero_features if f not in names]
    if unknown:
        raise SystemExit(
            f"unknown feature(s) to zero: {unknown}. Known: {list(names)}")
    for f in zero_features:
        out[:, names.index(f)] = 0.0
    return out


# ------------------------------------------------------------------ files
def save(path, params, hidden: int = 0, n_features: int = N_FEATURES,
         zero_features=()) -> None:
    np.savez(str(path), params=np.asarray(params), hidden=np.array(hidden),
             n_features=np.array(n_features),
             zero_features=np.array(list(zero_features), dtype=object))


def load(path):
    """-> (params, hidden, n_features)."""
    z = np.load(str(path), allow_pickle=True)
    hidden = int(z["hidden"]) if "hidden" in z else 0
    nf = int(z["n_features"]) if "n_features" in z else N_FEATURES
    return z["params"], hidden, nf


def zero_features_of(path) -> list[str]:
    """The ablation a scorer was trained under; [] for files predating the field."""
    z = np.load(str(path), allow_pickle=True)
    if "zero_features" not in z:
        return []
    return [str(x) for x in z["zero_features"]]


def check_ablation(path, zero_features) -> list[str]:
    """Refuse to evaluate a scorer outside the ablation it was trained in.

    Returns the features being zeroed here but NOT during training, which is
    merely informative -- that measures how much a trained policy leans on them.
    The reverse is an error: a weight that drifted freely during ES because its
    feature was zeroed is not a trained weight, and activating it now evaluates
    a policy that never existed.
    """
    trained = zero_features_of(path)
    missing = [f for f in trained if f not in zero_features]
    if missing:
        raise SystemExit(
            f"ablation mismatch: {path} was trained with zero_features={trained} "
            f"but this run does not zero {missing}. Those weights drifted during "
            f"ES without affecting the reward, so activating them now would "
            f"evaluate a policy that was never trained.\n"
            f"Pass --zero-features {' '.join(trained)}.")
    return [f for f in zero_features if f not in trained]


def grow(params, from_features: int, to_features: int, hidden: int = 0):
    """Warm-start a larger feature set: shared weights kept, new ones zeroed."""
    if hidden != 0:
        raise SystemExit("only a linear scorer can be grown across feature sets")
    if to_features <= from_features:
        raise SystemExit(f"cannot grow {from_features} -> {to_features}")
    out = np.zeros(to_features + 1)
    out[:from_features] = params[:from_features]
    out[-1] = params[-1]                       # bias
    return out
