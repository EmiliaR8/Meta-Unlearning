"""Model definitions and the --model axis.

EmberNN is the 4-block BatchNorm MLP used throughout. Widths are a parameter so
that model size is selectable, but the layer ATTRIBUTE NAMES and the block COUNT
are fixed, which keeps state_dicts compatible with existing checkpoints at the
default widths.

The latent (dims[3], the fc4 output) is deliberately separable from the trunk.
It is not just another hidden layer: MADAR buffer selection, the donut selector
and several NN-1 features all operate in that space. Shrinking it changes the
geometry every selection rule sees, which confounds a capacity measurement --
hence the 'trunk' preset, which cuts capacity while holding the latent at 128.
"""

from __future__ import annotations

import torch.nn as nn

DEFAULT_DIMS = (1024, 512, 256, 128)

# name -> (fc1, fc2, fc3, fc4/latent)
PRESETS = {
    "full":    (1024, 512, 256, 128),   # 3.14M params at d=2381
    "trunk":   (512, 256, 128, 128),    # capacity down, latent geometry HELD FIXED
    "half":    (512, 256, 128, 64),     # uniform half width
    "quarter": (256, 128, 64, 32),
    "tiny":    (128, 64, 32, 32),
}


def get_dims(model: str) -> tuple[int, int, int, int]:
    """Resolve a --model value: a preset name or four explicit widths."""
    key = str(model).lower()
    if key in PRESETS:
        return PRESETS[key]
    parts = [p for p in str(model).replace(",", " ").split() if p]
    try:
        widths = [int(p) for p in parts]
    except ValueError:
        raise ValueError(
            f"--model {model!r} is neither a preset ({', '.join(sorted(PRESETS))}) "
            f"nor four integer widths like '512 256 128 64'")
    if len(widths) != 4:
        raise ValueError(
            f"--model {model!r}: expected 4 widths, got {len(widths)}")
    if any(w <= 0 for w in widths):
        raise ValueError(f"--model {model!r}: widths must be positive")
    return tuple(widths)


def count_params(input_dim: int, num_classes: int, dims=DEFAULT_DIMS) -> int:
    """Trainable parameter count, without instantiating the model."""
    d1, d2, d3, d4 = dims
    linear = ((input_dim * d1 + d1) + (d1 * d2 + d2) + (d2 * d3 + d3) +
              (d3 * d4 + d4) + (d4 * num_classes + num_classes))
    bn = 2 * (d1 + d2 + d3 + d4)
    return linear + bn


class EmberNN(nn.Module):
    """4-block MLP with BatchNorm.

    forward(x, return_latent=True) -> (logits, latent), latent = post-ReLU fc4.
    """

    def __init__(self, input_dim: int, num_classes: int, dims=DEFAULT_DIMS):
        super().__init__()
        d1, d2, d3, d4 = dims
        self.dims = tuple(dims)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.latent_dim = d4
        self.fc1 = nn.Linear(input_dim, d1); self.fc1_bn = nn.BatchNorm1d(d1)
        self.fc2 = nn.Linear(d1, d2);        self.fc2_bn = nn.BatchNorm1d(d2)
        self.fc3 = nn.Linear(d2, d3);        self.fc3_bn = nn.BatchNorm1d(d3)
        self.fc4 = nn.Linear(d3, d4);        self.fc4_bn = nn.BatchNorm1d(d4)
        self.relu = nn.ReLU()
        self.fc_last = nn.Linear(d4, num_classes)

    def forward(self, x, return_latent: bool = False):
        x = self.relu(self.fc1_bn(self.fc1(x)))
        x = self.relu(self.fc2_bn(self.fc2(x)))
        x = self.relu(self.fc3_bn(self.fc3(x)))
        latent = self.relu(self.fc4_bn(self.fc4(x)))
        logits = self.fc_last(latent)
        return (logits, latent) if return_latent else logits


def build_model(model: str, input_dim: int, num_classes: int, device=None):
    dims = get_dims(model)
    net = EmberNN(input_dim, num_classes, dims)
    if device is not None:
        net = net.to(device)
    return net


def model_info(model: str, input_dim: int, num_classes: int) -> dict:
    """The 'model size' block recorded in every run's log."""
    dims = get_dims(model)
    return {"model": str(model), "dims": list(dims), "latent_dim": dims[3],
            "input_dim": int(input_dim), "num_classes": int(num_classes),
            "n_params": count_params(input_dim, num_classes, dims)}
