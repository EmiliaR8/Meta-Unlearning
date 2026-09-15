"""
Shared model definition for the EMBER / LAMDA continual-learning pipeline.

Previously EmberNN was duplicated verbatim in five scripts. It is defined
once here so that architecture becomes a controllable experimental variable
rather than five places to edit and drift.

IMPORTANT - state_dict compatibility:
The layer ATTRIBUTE NAMES (fc1, fc1_bn, ..., fc4, fc4_bn, fc_last) and the
layer COUNT (4 hidden blocks) are unchanged. Only the widths are
parameterised. A model built with dims=DEFAULT_DIMS therefore produces a
state_dict identical in keys and shapes to the original class, so existing
checkpoints (task0_ckpt.pt, task0_ember2024.pt) still load unmodified and
prior runs stay valid.

The LATENT dim (dims[3], the output of fc4) is deliberately separable from
the trunk widths. It is not just a hidden layer: MADAR buffer selection, the
donut selector, and NN-1's iso_latent / centroid_dist / density_ratio
features all operate in that space, and _last_layer_grads produces vectors
of dim active_count * dims[3]. Shrinking it changes the geometry every
selection rule sees, which confounds a capacity measurement. Use the
'trunk' preset to shrink capacity while holding the latent geometry fixed.
"""

import torch.nn as nn

DEFAULT_DIMS = (1024, 512, 256, 128)

# name -> (fc1_out, fc2_out, fc3_out, fc4_out/latent)
ARCH_PRESETS = {
    'full':    (1024, 512, 256, 128),   # original; 3.14M params at d=2381
    'trunk':   (512, 256, 128, 128),    # capacity down, latent geometry HELD FIXED
    'half':    (512, 256, 128, 64),     # uniform half width
    'quarter': (256, 128, 64, 32),      # uniform quarter width
    'tiny':    (128, 64, 32, 32),       # aggressive floor
}


def get_dims(arch):
    """Resolve an --arch value to a 4-tuple of layer widths."""
    if arch in ARCH_PRESETS:
        return ARCH_PRESETS[arch]
    parts = [int(p) for p in str(arch).replace(',', ' ').split()]
    if len(parts) != 4:
        raise ValueError(
            f"--arch must be one of {sorted(ARCH_PRESETS)} or four widths "
            f"like '512 256 128 64'; got {arch!r}")
    return tuple(parts)


def count_params(input_dim, num_classes, dims=DEFAULT_DIMS):
    """Trainable parameter count, without instantiating torch."""
    d1, d2, d3, d4 = dims
    linear = ((input_dim * d1 + d1) + (d1 * d2 + d2) + (d2 * d3 + d3) +
              (d3 * d4 + d4) + (d4 * num_classes + num_classes))
    bn = 2 * (d1 + d2 + d3 + d4)          # weight + bias per BatchNorm1d
    return linear + bn


class EmberNN(nn.Module):
    """4-block MLP with BatchNorm. Identical to the original at DEFAULT_DIMS.

    forward(x, return_latent=True) returns (logits, latent) where latent is
    the post-ReLU output of the fc4 block.
    """

    def __init__(self, input_dim, num_classes, dims=DEFAULT_DIMS):
        super().__init__()
        d1, d2, d3, d4 = dims
        self.dims = tuple(dims)
        self.latent_dim = d4
        self.fc1 = nn.Linear(input_dim, d1); self.fc1_bn = nn.BatchNorm1d(d1)
        self.fc2 = nn.Linear(d1, d2);        self.fc2_bn = nn.BatchNorm1d(d2)
        self.fc3 = nn.Linear(d2, d3);        self.fc3_bn = nn.BatchNorm1d(d3)
        self.fc4 = nn.Linear(d3, d4);        self.fc4_bn = nn.BatchNorm1d(d4)
        self.relu = nn.ReLU()
        self.fc_last = nn.Linear(d4, num_classes)

    def forward(self, x, return_latent=False):
        x = self.relu(self.fc1_bn(self.fc1(x)))
        x = self.relu(self.fc2_bn(self.fc2(x)))
        x = self.relu(self.fc3_bn(self.fc3(x)))
        latent = self.relu(self.fc4_bn(self.fc4(x)))
        logits = self.fc_last(latent)
        return (logits, latent) if return_latent else logits


if __name__ == '__main__':
    for d in (2381, 2568, 4561):
        print(f"\ninput_dim={d}, num_classes=100")
        base = count_params(d, 100, DEFAULT_DIMS)
        for name, dims in ARCH_PRESETS.items():
            n = count_params(d, 100, dims)
            print(f"  {name:8s} {str(dims):22s} {n:>10,}  ({100*n/base:5.1f}% of full)")