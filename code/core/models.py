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


# Convolutional presets, kept in their own table because they are named by
# DEPTH AND WIDTH rather than by four layer sizes; a `--model` value is looked up
# here first and only then parsed as MLP widths.
CONV_PRESETS = {
    "resnet18":         (64, (2, 2, 2, 2)),   # 11.27M params at 200 classes
    "resnet18_half":    (32, (2, 2, 2, 2)),   # width down, depth held
    "resnet18_quarter": (16, (2, 2, 2, 2)),
    "resnet10":         (64, (1, 1, 1, 1)),   # depth down, width held
}


def is_conv(model: str) -> bool:
    return str(model).lower() in CONV_PRESETS


def build_model(model: str, input_dim: int, num_classes: int, device=None,
                input_shape=None):
    """The network for this run.

    `input_shape` is the per-example shape without the batch dimension: (d,) for
    a tabular corpus, (C, H, W) for images. It is what decides whether a
    convolutional preset is even admissible -- asking for `--model resnet18` on
    EMBER is a mistake worth catching at build time rather than at the first
    forward pass, where it would surface as a shape error four frames deep.
    """
    if is_conv(model):
        if input_shape is None or len(input_shape) != 3:
            raise SystemExit(
                f"--model {model} is convolutional and needs an image corpus; "
                f"this dataset gives examples of shape {tuple(input_shape or (input_dim,))}. "
                f"Use an MLP preset ({', '.join(sorted(PRESETS))}) instead.")
        width, blocks = CONV_PRESETS[str(model).lower()]
        net = ResNet18(num_classes, in_ch=int(input_shape[0]), width=width,
                       blocks=blocks)
        net.input_dim = int(input_dim)
    else:
        if input_shape is not None and len(input_shape) == 3:
            raise SystemExit(
                f"--model {model} is an MLP but this corpus gives "
                f"{tuple(input_shape)} images. Use a convolutional preset "
                f"({', '.join(sorted(CONV_PRESETS))}).")
        net = EmberNN(input_dim, num_classes, get_dims(model))
    if device is not None:
        net = net.to(device)
    return net


def model_info(model: str, input_dim: int, num_classes: int,
               input_shape=None) -> dict:
    """The 'model size' block recorded in every run's log."""
    if is_conv(model):
        width, blocks = CONV_PRESETS[str(model).lower()]
        in_ch = int(input_shape[0]) if input_shape is not None else 3
        net = ResNet18(num_classes, in_ch=in_ch, width=width, blocks=blocks)
        n = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {"model": str(model), "family": "resnet", "width": width,
                "blocks": list(blocks), "latent_dim": width * 8,
                "input_dim": int(input_dim),
                "input_shape": list(input_shape) if input_shape else None,
                "num_classes": int(num_classes), "n_params": int(n)}
    dims = get_dims(model)
    return {"model": str(model), "family": "mlp", "dims": list(dims),
            "latent_dim": dims[3],
            "input_dim": int(input_dim),
            "input_shape": list(input_shape) if input_shape else [int(input_dim)],
            "num_classes": int(num_classes),
            "n_params": count_params(input_dim, num_classes, dims)}


# ===================================================================== images
# Everything above assumes a tabular corpus: one feature vector per example, a
# 4-block MLP, and a `--model` axis that is four widths. Tiny ImageNet is the
# first corpus that is none of those, so the pieces it needs live below rather
# than being folded into EmberNN's argument list.

import torch
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """The standard two-conv residual block. No bottleneck, expansion 1."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.short = None
        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + (x if self.short is None else self.short(x))
        return F.relu(out)


class ResNet18(nn.Module):
    """ResNet-18 with a 3x3 stride-1 stem and NO max-pool.

    This is the CIFAR/Tiny-ImageNet variant the continual-learning literature
    uses, not torchvision's ImageNet model. The difference is not cosmetic: the
    ImageNet stem is a 7x7 stride-2 convolution followed by a stride-2 max-pool,
    which takes a 64x64 input down to 16x16 before the first residual block and
    throws away most of the spatial detail the later stages are meant to use.
    Mammoth -- the code base X-DER, DER++ and STAR were all evaluated in -- uses
    the variant implemented here, so numbers from those papers are comparable to
    these only with this stem.

    NORMALISATION LIVES IN THE MODEL, and the statistics are buffers rather than
    constants. Two reasons. First, inputs stay uint8 all the way to the forward
    pass: 100k Tiny ImageNet images are 1.2 GB as uint8 and 4.9 GB as float32,
    and the float32 copy would have to sit in memory alongside the buffer's own
    copies. Second, the statistics are fitted on TASK 0 ONLY, exactly as
    `scale_features` does for the tabular corpora -- normalising with statistics
    computed over all 200 classes would leak the distribution of classes the
    learner has not met. Being buffers, they travel with the state_dict, so a
    teacher clone and a STAR perturbation see identical normalisation.

    AUGMENTATION IS NOT IN forward(). It is a separate method the training loop
    calls once per batch. Putting it in forward would silently corrupt every
    method here that runs more than one forward over the same batch: STAR's
    KL is between the clean and perturbed responses TO THE SAME INPUT, and
    X-DER's implant compares a live logit against a stored one. Two independent
    random crops in those places would be measuring augmentation noise.
    """

    def __init__(self, num_classes: int, in_ch: int = 3, width: int = 64,
                 blocks=(2, 2, 2, 2)):
        super().__init__()
        w = int(width)
        self.widths = (w, w * 2, w * 4, w * 8)
        self.in_channels = int(in_ch)
        self.num_classes = int(num_classes)
        self.latent_dim = self.widths[3]
        self.input_dim = None            # set by build_model; images have a shape

        self.conv1 = nn.Conv2d(in_ch, w, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(w)
        self._cur = w
        self.layer1 = self._stage(self.widths[0], blocks[0], 1)
        self.layer2 = self._stage(self.widths[1], blocks[1], 2)
        self.layer3 = self._stage(self.widths[2], blocks[2], 2)
        self.layer4 = self._stage(self.widths[3], blocks[3], 2)
        self.fc_last = nn.Linear(self.widths[3], num_classes)

        # Overwritten by set_norm() with task-0 statistics. The ImageNet values
        # are a placeholder so an un-fitted model still produces sane scales.
        self.register_buffer("norm_mean",
                             torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer("norm_std",
                             torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))
        self.aug_pad = 4
        self.aug_flip = True

    def _stage(self, out_ch: int, n: int, stride: int) -> nn.Sequential:
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(BasicBlock(self._cur, out_ch, s))
            self._cur = out_ch
        return nn.Sequential(*layers)

    # -- input handling ---------------------------------------------------
    def set_norm(self, mean, std) -> None:
        """Install per-channel statistics, in 0-255 units to match uint8 input."""
        m = torch.as_tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
        s = torch.as_tensor(std, dtype=torch.float32).view(1, -1, 1, 1)
        if int(s.min()) == 0 or bool((s <= 0).any()):
            raise ValueError("per-channel std must be positive; got a zero or "
                             "negative channel, which means a constant channel")
        self.norm_mean = m.to(self.norm_mean.device)
        self.norm_std = s.to(self.norm_std.device)

    def normalise(self, x):
        return (x.float() - self.norm_mean) / self.norm_std

    def augment(self, x):
        """Random crop with reflection padding, plus a horizontal flip.

        Operates on the uint8 batch ON THE DEVICE it already lives on, so no
        host round-trip and no per-sample Python loop. One crop offset and one
        flip decision PER SAMPLE, not per batch -- a single shared offset would
        make the whole batch's augmentation perfectly correlated, which is
        materially weaker regularisation and a known way to make replay look
        better than it is.
        """
        if self.aug_pad <= 0 and not self.aug_flip:
            return x
        n, c, h, w = x.shape
        out = x
        if self.aug_pad > 0:
            p = self.aug_pad
            padded = F.pad(out.float(), (p, p, p, p), mode="reflect")
            ox = torch.randint(0, 2 * p + 1, (n,), device=x.device)
            oy = torch.randint(0, 2 * p + 1, (n,), device=x.device)
            rows = (torch.arange(h, device=x.device).view(1, h, 1) +
                    oy.view(n, 1, 1))
            cols = (torch.arange(w, device=x.device).view(1, 1, w) +
                    ox.view(n, 1, 1))
            bidx = torch.arange(n, device=x.device).view(n, 1, 1)
            out = padded[bidx, :, rows, cols]        # -> (n, h, w, c)
            out = out.permute(0, 3, 1, 2).contiguous().to(x.dtype)
        if self.aug_flip:
            flip = torch.rand(n, device=x.device) < 0.5
            out = torch.where(flip.view(n, 1, 1, 1), out.flip(-1), out)
        return out

    # -- forward ----------------------------------------------------------
    def forward(self, x, return_latent: bool = False):
        x = self.normalise(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        latent = F.adaptive_avg_pool2d(x, 1).flatten(1)
        logits = self.fc_last(latent)
        return (logits, latent) if return_latent else logits

    # -- DEDUCE's GUM needs to know which units it may reinitialise --------
    def gum_blocks(self) -> list[tuple[str, str, str]]:
        """(layer, its norm, the layer its outputs feed) triples, by dotted name.

        Only the FIRST convolution of each residual block is offered. Its output
        channels are consumed by exactly one place -- that block's conv2 -- so
        zeroing an outgoing channel there is a local edit with the same meaning
        it has in the MLP. The second convolution is deliberately withheld: its
        output enters the residual addition, so a channel there is also carried
        forward by the identity path, and reinitialising it would leave that
        path intact while claiming the unit had been reset.
        """
        out = []
        for stage in ("layer1", "layer2", "layer3", "layer4"):
            for i, _ in enumerate(getattr(self, stage)):
                out.append((f"{stage}.{i}.conv1", f"{stage}.{i}.bn1",
                            f"{stage}.{i}.conv2"))
        return out
