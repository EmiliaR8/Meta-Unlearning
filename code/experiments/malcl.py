"""MalCL baseline (Park, Ji, Park, Rahman & Oh, AAAI 2025; arXiv:2501.01110).

Generative replay: a GAN is trained alongside the classifier, and each batch is
augmented with synthetic samples of old classes selected by nearest-neighbour
matching in the classifier's feature space. Upstream:
https://github.com/MalwareReplayGAN/MalCL (MIT).

=========================================================================
FAITHFULNESS
=========================================================================
Generator, Discriminator and Classifier are structurally as released. The
GAN/classifier update order, the feature-matching loss, the three selection
schemes, k=3, z_dim=62, 50 epochs per task, Adam for G/D and SGD for C, and the
per-batch replay regeneration are all as published.

TWO KNOWN DEFECTS ARE DELIBERATELY PRESERVED, because this row reports MalCL as
published rather than a repaired version of it. Both are recorded per task so
the caveat travels with the numbers:

  (1) DOUBLE SOFTMAX. Classifier.forward ends in nn.Softmax (after a ReLU, so
      every pre-softmax value is non-negative) and that output is fed to
      nn.CrossEntropyLoss, which applies log_softmax again.

  (2) FROZEN CLASSIFICATION HEAD. The classifier optimiser is built once from
      C.parameters(). expand_output_layer then REPLACES fc1 and fc1_bn1 with new
      modules whose parameters are not in the optimiser's param groups. From
      task 1 onward the head -- including the rows for every newly introduced
      family -- receives gradients but is never stepped. `head_in_optimizer` is
      logged each task.

  (3) mean_logits skips classes with no collected logits via `continue`, which
      shifts row indices relative to class indices. Preserved; the skipped
      classes are counted and logged.

=========================================================================
WHAT DIFFERS FROM THE UPSTREAM SCRIPT, AND WHY
=========================================================================
  * INPUT DIMENSION IS PARAMETERISED. Upstream hardcodes 2381. Running this
    baseline on the other corpora requires d to follow the data. See the memory
    warning below -- this is not a free change.

  * SCALING IS THE HARNESS'S, NOT MalCL's. Upstream refits a StandardScaler by
    partial_fit each task; this runner scales once on task 0 for every method,
    so that is what MalCL gets here. This is the harness convention MalCL's own
    script exposes as `--scaler task0_fixed`. It makes the row comparable with
    the rows beside it, at the cost of not being MalCL's own preprocessing.
    Recorded as `scaler: task0_fixed` so the substitution is visible.

  * PROTOCOL IS THIS PROJECT'S: the >=200-sample family filter, top-N by
    frequency, largest-family-first deterministic class order, and this
    project's test split. Upstream uses >400 samples and a random class
    permutation. A baseline row has to sit in the same protocol as its
    neighbours or the comparison is not a comparison.

=========================================================================
MEMORY -- READ BEFORE RUNNING ON ANYTHING BUT EMBER
=========================================================================
Discriminator.fc[0] is Linear(256*d, 1024), so its parameter count grows
LINEARLY with input dimensionality:

    d = 2381 (EMBER 2018)   ->  0.62B params,  ~2.5 GB fp32,  ~10 GB with Adam
    d = 2568 (EMBER 2024)   ->  0.67B params,  ~2.7 GB fp32,  ~11 GB with Adam
    d = 4561 (LAMDA)        ->  1.20B params,  ~4.8 GB fp32,  ~19 GB with Adam

Upstream used a 48 GB A6000 for the 2381 case. LAMDA is roughly double that
again and may simply not fit. The estimate is printed at construction and
`--malcl-epochs 1 --n-tasks 2` is the cheap way to find out before committing a
seed.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from .base import Experiment, register


# ---------------------------------------------------------------- models
class Generator(nn.Module):
    """Structurally as released; output width follows the corpus."""

    def __init__(self, output_features: int, z_dim: int = 62):
        super().__init__()
        self.input_dim = z_dim
        self.channel_c, self.channel_d = 256, 512
        self.channel_e, self.channel_f, self.channel_g = 1024, 2048, 4096
        self.output_features = output_features
        self.conv = nn.Sequential(
            nn.Conv1d(self.input_dim, self.channel_c, 3, padding=1),
            nn.BatchNorm1d(self.channel_c), nn.ReLU(),
            nn.Conv1d(self.channel_c, self.channel_e, 3, padding=1),
            nn.BatchNorm1d(self.channel_e), nn.ReLU(),
            nn.Conv1d(self.channel_e, self.channel_g, 3, padding=1),
            nn.BatchNorm1d(self.channel_g), nn.ReLU(),
            nn.Conv1d(self.channel_g, self.channel_e, 3, padding=1),
            nn.BatchNorm1d(self.channel_e), nn.ReLU())
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.channel_e, self.channel_f),
            nn.BatchNorm1d(self.channel_f), nn.ReLU(),
            nn.Linear(self.channel_f, self.channel_g),
            nn.BatchNorm1d(self.channel_g), nn.ReLU())
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(self.channel_g, self.channel_e, 3, padding=1),
            nn.BatchNorm1d(self.channel_e), nn.ReLU(),
            nn.ConvTranspose1d(self.channel_e, self.channel_d, 3, padding=1),
            nn.BatchNorm1d(self.channel_d), nn.ReLU(),
            nn.ConvTranspose1d(self.channel_d, self.output_features, 3, padding=1),
            nn.Sigmoid())
        self.apply(self.weights_init)

    def reinit(self):
        self.apply(self.weights_init)

    def forward(self, x):
        x = x.view(-1, self.input_dim, 1)
        x = self.conv(x)
        x = self.fc(x)
        x = x.view(-1, self.channel_g, 1)
        x = self.deconv(x)
        return x.view(-1, self.output_features)

    @staticmethod
    def weights_init(m):
        name = m.__class__.__name__
        if name.find("Conv") != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif name.find("BatchNorm") != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


class Discriminator(nn.Module):
    def __init__(self, input_features: int):
        super().__init__()
        self.input_channel, self.output_dim = 1, 1
        self.channel_c, self.channel_d = 256, 512
        self.input_features = input_features
        self.latent_dim = 1024
        self.conv = nn.Sequential(
            nn.Conv1d(self.input_channel, self.channel_d, 3, padding=1), nn.ReLU(),
            nn.Conv1d(self.channel_d, self.channel_c, 3, padding=1), nn.ReLU(),
            nn.BatchNorm1d(self.channel_c))
        self.fc = nn.Sequential(
            nn.Linear(self.channel_c * self.input_features, self.latent_dim),
            nn.ReLU(), nn.BatchNorm1d(self.latent_dim),
            nn.Linear(self.latent_dim, self.output_dim), nn.Sigmoid())
        self.apply(Generator.weights_init)

    def reinit(self):
        self.apply(Generator.weights_init)

    def forward(self, x):
        x = x.view(-1, self.input_channel, self.input_features)
        x = self.conv(x)
        feature = x.view(-1, self.channel_c * self.input_features)
        return self.fc(feature).view(-1, 1), feature


class Classifier(nn.Module):
    def __init__(self, input_features: int, init_classes: int, drop_prob: float = 0.5):
        super().__init__()
        self.input_features = input_features
        self.output_dim = init_classes
        self.block1 = nn.Sequential(
            nn.Conv1d(self.input_features, 512, 3, stride=3, padding=1),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.Conv1d(512, 256, 3, 3, 1),
            nn.BatchNorm1d(256), nn.Dropout(drop_prob), nn.ReLU(),
            nn.MaxPool1d(3, 3, 1))
        self.block2 = nn.Sequential(
            nn.Conv1d(256, 128, 3, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.Dropout(drop_prob), nn.ReLU())
        self.fc1_f = nn.Flatten()
        self.fc1 = nn.Linear(128, self.output_dim)
        self.fc1_bn1 = nn.BatchNorm1d(self.output_dim)
        self.fc1_drop1 = nn.Dropout(drop_prob)
        self.fc1_act1 = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)   # PRESERVED DEFECT (1): CE re-softmaxes

    def _trunk(self, x):
        shape = x.size()
        batch = shape[0] if len(shape) == 2 else shape[0] * shape[1]
        x = x.reshape(batch, self.input_features, -1)
        x = self.block1(x)
        x = self.block2(x)
        return self.fc1_f(x), shape

    def forward(self, x):
        h, shape = self._trunk(x)
        h = self.fc1(h)
        h = self.fc1_bn1(h)
        h = self.fc1_drop1(h)
        h = self.fc1_act1(h)
        out = self.softmax(h)
        return out.view(shape[0], shape[1], -1) if len(shape) == 3 else out

    def predict(self, x):
        return self.forward(x)

    def get_logits(self, x):
        h, _ = self._trunk(x)
        return h.detach()

    def expand_output_layer(self, init_classes, nb_inc, task):
        old_fc1, old_bn = self.fc1, self.fc1_bn1
        self.output_dim = init_classes + nb_inc * task
        self.fc1 = nn.Linear(old_fc1.in_features, self.output_dim)
        self.fc1_bn1 = nn.BatchNorm1d(self.output_dim)
        with torch.no_grad():
            self.fc1.weight[:old_fc1.out_features].copy_(old_fc1.weight.data)
            self.fc1.bias[:old_fc1.out_features].copy_(old_fc1.bias.data)
            self.fc1_bn1.weight[:old_bn.num_features].copy_(old_bn.weight.data)
            self.fc1_bn1.bias[:old_bn.num_features].copy_(old_bn.bias.data)
        return self


# ---------------------------------------------------------------- selection
def _compute_l1(A, B):
    return torch.mean(torch.abs(A.unsqueeze(1) - B.unsqueeze(0)), dim=2)


def _compute_l2(A, B):
    return torch.norm(A.unsqueeze(1) - B.unsqueeze(0), dim=2)


def _knn_sel(arr, k):
    idx = []
    for row in arr:
        for _ in range(k):
            _, ind = torch.min(row, dim=0)
            idx.append(ind.item())
            arr[:, ind] = torch.inf
    return idx


def _glob_sel(arr, sample_num):
    _, sorted_index = torch.sort(torch.flatten(arr))
    cols = sorted_index % len(arr[0])
    uni, _ = torch.unique(cols, sorted=False, return_inverse=True)
    return uni.flip(0)[:sample_num]


def _l2_one_hot(n_class, n_inc, k, device, synthetic, pred_label, logits_gen, logits_real):
    one_hot = torch.eye(n_class - n_inc).to(device)
    idx = _knn_sel(_compute_l2(one_hot, pred_label).cpu(), k)
    labels = torch.arange(0, n_class - n_inc).repeat_interleave(k)[:len(idx)]
    return synthetic[idx].to(device), nn.functional.one_hot(
        labels.to(torch.int64), num_classes=n_class).to(device)


def _l1_b_mean(n_class, n_inc, k, device, synthetic, pred_label, logits_gen, logits_real):
    arr = _compute_l1(logits_real.to(device), logits_gen)
    num_syn = len(arr[0])
    sample_num = min(num_syn, (n_class - n_inc) * k)
    idx = _glob_sel(arr.cpu(), sample_num)
    labels = torch.Tensor([list(i).index(max(i)) for i in pred_label[idx]])
    return synthetic[idx].to(device), nn.functional.one_hot(
        labels.to(torch.int64), num_classes=n_class).to(device)


def _l1_c_mean(n_class, n_inc, k, device, synthetic, pred_label, logits_gen, logits_real):
    arr = _compute_l1(logits_real.to(device), logits_gen)
    idx = _knn_sel(arr.cpu(), k)
    labels = torch.arange(0, n_class - n_inc).repeat_interleave(k)[:len(idx)]
    return synthetic[idx].to(device), nn.functional.one_hot(
        labels.to(torch.int64), num_classes=n_class).to(device)


SELECTORS = {"L2_One_Hot": _l2_one_hot, "L1_B_Mean": _l1_b_mean,
             "L1_C_Mean": _l1_c_mean}


# ---------------------------------------------------------------- experiment
@register("malcl")
class MalCLExperiment(Experiment):
    uses_buffer = False
    uses_si = False

    def __init__(self, ctx):
        super().__init__(ctx)
        d = self.ctx.input_dim
        self.G = Generator(d, self.hp["malcl_z_dim"]).to(self.device)
        self.D = Discriminator(d).to(self.device)
        self.G.reinit(); self.D.reinit()
        self.G_opt = optim.Adam(self.G.parameters(), lr=self.hp["task0_lr"])
        self.D_opt = optim.Adam(self.D.parameters(), lr=self.hp["task0_lr"])
        # PRESERVED DEFECT (2): built ONCE, never rebuilt after the head expands.
        self.C_opt = optim.SGD(self.model.parameters(), lr=self.hp["task0_lr"],
                               momentum=self.hp["momentum"],
                               weight_decay=self.hp["weight_decay"])
        self.criterion = nn.CrossEntropyLoss()
        self.bce = nn.BCELoss()
        self.past_G = self.past_C = None
        self.logits_real = None
        self._warn_memory()

    def build_model(self):
        """MalCL's own classifier, sized to task 0's class count."""
        return Classifier(self.ctx.input_dim, self.schedule.task0).to(self.device)

    def model_info(self) -> dict:
        n = lambda m: sum(p.numel() for p in m.parameters())
        return {"model": "malcl", "input_dim": self.ctx.input_dim,
                "num_classes": self.schedule.n_classes,
                "n_params": n(self.model) + n(self.G) + n(self.D),
                "n_params_classifier": n(self.model),
                "n_params_generator": n(self.G),
                "n_params_discriminator": n(self.D),
                "note": "MalCL: classifier + GAN; head expands per task"}

    def _warn_memory(self):
        d_params = sum(p.numel() for p in self.D.parameters())
        gb = d_params * 4 / 1e9
        print(f"  MalCL discriminator: {d_params/1e9:.2f}B params "
              f"(~{gb:.1f} GB fp32, ~{gb * 4:.0f} GB with Adam moments) "
              f"at d={self.ctx.input_dim}")
        if gb * 4 > 20:
            print("  WARNING: this is likely to exhaust a 24 GB GPU. Upstream used "
                  "48 GB at d=2381. Try --malcl-epochs 1 --n-tasks 2 first.")

    # -- loaders ----------------------------------------------------------
    def _onehot_loader(self, tid: int, n_class: int):
        """MalCL's loader: class-balanced sampler and one-hot targets."""
        X, y = self.ctx.X_train, self.ctx.y_train
        m = torch.isin(y, torch.as_tensor(self.schedule.classes_for(tid)))
        xs, ys = X[m], y[m]
        y_oh = nn.functional.one_hot(ys.long(), num_classes=n_class).float()
        ds = data.TensorDataset(xs, y_oh)
        counts = torch.bincount(ys, minlength=int(ys.max()) + 1).float()
        w = torch.where(counts > 0, 1.0 / counts.clamp(min=1), torch.zeros_like(counts))
        sampler = data.WeightedRandomSampler(w[ys], len(ys), replacement=True)
        return data.DataLoader(ds, batch_size=self.hp["batch_size"],
                               sampler=sampler, drop_last=True)

    # -- GAN steps (as released) -----------------------------------------
    def _batch(self, x, y):
        z = torch.rand((x.size(0), self.hp["malcl_z_dim"]), device=self.device)
        real = torch.ones(x.size(0), 1, device=self.device)
        fake = torch.zeros(x.size(0), 1, device=self.device)
        return x.to(self.device), y.to(self.device), z, real, fake

    def _update_d(self, x, z, real, fake):
        self.D_opt.zero_grad()
        d_real, _ = self.D(x)
        d_fake, _ = self.D(self.G(z))
        loss = self.bce(d_real, real[:x.size(0)]) + self.bce(d_fake, fake[:x.size(0)])
        loss.backward(); self.D_opt.step()

    def _update_g_fml(self, x, z):
        self.G_opt.zero_grad()
        _, f_fake = self.D(self.G(z))
        _, f_real = self.D(x)
        loss = torch.mean(torch.abs(f_real.mean(0) - f_fake.mean(0)))
        loss.backward(); self.G_opt.step()

    def _update_c(self, x, y):
        # PRESERVED DEFECT (1): C(x) is already softmaxed; CE softmaxes again.
        self.C_opt.zero_grad()
        loss = self.criterion(self.model(x), y)
        loss.backward(); self.C_opt.step()
        return float(loss.detach())

    # -- replay generation ------------------------------------------------
    def _generate_replay(self, n_class: int):
        k, n_inc = self.hp["malcl_k"], self.schedule.step
        bs, z_dim = self.hp["batch_size"], self.hp["malcl_z_dim"]
        syn_size = math.ceil(((n_class - n_inc) * k) / float(bs))
        synthetic = torch.tensor([], device=self.device)
        self.past_G.eval()
        with torch.no_grad():
            for _ in range(syn_size):
                z = torch.rand((bs, z_dim), device=self.device)
                synthetic = torch.cat((synthetic, self.past_G(z)), dim=0)
        self.past_C.eval()
        pred_label, logits_gen = None, torch.tensor([], device=self.device)
        with torch.no_grad():
            if self.cfg["selector"] in ("L2_One_Hot", "L1_B_Mean"):
                pred_label = self.past_C.predict(synthetic).detach()
            if self.cfg["selector"] in ("L1_B_Mean", "L1_C_Mean"):
                for i in range(syn_size):
                    sb = synthetic[i * bs:(i + 1) * bs]
                    logits_gen = torch.cat(
                        (logits_gen, self.past_C.get_logits(sb).to(self.device)), dim=0)
        return synthetic, pred_label, logits_gen

    def _mean_logits(self, collected):
        """PRESERVED DEFECT (3): empty classes are skipped, shifting row indices
        relative to class indices. Counted and logged rather than repaired."""
        rows, skipped = [], []
        for i, row in enumerate(collected):
            if len(row) == 0:
                skipped.append(i); continue
            rows.append(torch.mean(torch.stack(row).float(), dim=0))
        return (torch.stack(rows) if rows else None), skipped

    def _head_in_optimizer(self) -> bool:
        owned = {id(p) for g in self.C_opt.param_groups for p in g["params"]}
        return id(self.model.fc1.weight) in owned and id(self.model.fc1.bias) in owned

    # -- the task ---------------------------------------------------------
    def train_task(self, tid: int, loader) -> dict:
        n_class = self.schedule.active_count(tid)
        if tid > 0:
            self.model = self.model.expand_output_layer(
                self.schedule.task0, self.schedule.step, tid).to(self.device)

        owned = self._head_in_optimizer()
        if tid > 0 and not owned:
            print("    [preserved defect] classification head is NOT in the "
                  "optimizer; fc1/fc1_bn1 will not be updated this task.")

        train_loader = self._onehot_loader(tid, n_class)
        selector = SELECTORS[self.cfg["selector"]]
        collected = [[] for _ in range(n_class)]
        losses, steps = [], 0

        for _ in range(self.hp["malcl_epochs"]):
            for xb, yb in train_loader:
                xb, yb = xb.float().to(self.device), yb.float().to(self.device)
                if tid > 0:
                    syn, pred, lg = self._generate_replay(n_class)
                    replay, re_label = selector(
                        n_class, self.schedule.step, self.hp["malcl_k"],
                        self.device, syn, pred, lg, self.logits_real)
                    xb = torch.cat((xb, replay), 0)
                    yb = torch.cat((yb, re_label.float()), 0)
                self.model.train(); self.G.train(); self.D.train()
                x, y, z, real, fake = self._batch(xb, yb)
                self._update_g_fml(x, z)
                z = torch.rand((x.size(0), self.hp["malcl_z_dim"]), device=self.device)
                self._update_d(x, z, real, fake)
                losses.append(self._update_c(x, y))
                steps += 1
                with torch.no_grad():
                    self.model.eval()
                    vec = self.model.get_logits(x).cpu()
                    for j, lab in enumerate(y):
                        collected[int(torch.max(lab, dim=0)[1])].append(vec[j])

        self.past_G = copy.deepcopy(self.G)
        self.past_C = copy.deepcopy(self.model)
        self.logits_real, skipped = self._mean_logits(collected)
        if skipped:
            print(f"    [preserved defect] {len(skipped)} class(es) had no logits "
                  f"and were skipped, shifting row indices: {skipped[:10]}")

        return {"grad_steps": steps, "phase": "malcl",
                "head_in_optimizer": bool(owned),
                "empty_logit_classes": skipped,
                "replay_per_batch": 0 if tid == 0 else
                                    (n_class - self.schedule.step) * self.hp["malcl_k"],
                "classifier_loss": float(np.mean(losses)) if losses else None,
                "preserved_defects": ["double_softmax", "frozen_classifier_head",
                                      "mean_logits_index_shift"],
                "scaler": "task0_fixed (harness convention, not MalCL's partial_fit)"}
