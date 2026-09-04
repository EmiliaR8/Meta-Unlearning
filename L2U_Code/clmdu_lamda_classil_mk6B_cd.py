"""
Usage:
    python clmdu_lamda_classil_mk6B_cd.py --seed 42                      # Option B, alpha=0.2
    python clmdu_lamda_classil_mk6B_cd.py --seed 42 --alpha 0.0          # control
    python clmdu_lamda_classil_mk6B_cd.py --seed 42 --cl-iters 8000      # robustness point
"""

import argparse
import math
import copy
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

from lamda_data import build_arrays_lamda

# --- Parse Command Line Seed ---
parser = argparse.ArgumentParser(description="LAMDA Class-IL (MADAR-IF + ER/KD/SI + Unlearning)")
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--cache', type=str, default='./Datasets/lamda_class_il_cache.npz')
parser.add_argument('--split-mode', choices=['random', 'temporal'], default='random')
parser.add_argument('--temporal-cut', type=int, default=2020)
parser.add_argument('--min-test-samples', type=int, default=0)
parser.add_argument('--num-classes', type=int, default=80)
parser.add_argument('--task0-classes', type=int, default=30)
parser.add_argument('--step-classes', type=int, default=5)
parser.add_argument('--min-family-samples', type=int, default=200)
parser.add_argument('--task0-epochs', type=int, default=30)
parser.add_argument('--cl-iters', type=int, default=4000,
                    help='elbow of the macro-vs-budget curve; the epoch invariant '
                         )
parser.add_argument('--mem-size', type=int, default=2500,
                    help='replay budget; 1.85%% of task 0, matching the EMBER ratio')
parser.add_argument('--si-c', type=float, default=100.0,
                    help='SI strength. 100 per EMBER 2024 (SI is inert below ~100)')
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--scale', choices=['none', 'standard'], default='none')
parser.add_argument('--family-cap', type=int, default=0)
parser.add_argument('--cap-seed', type=int, default=12345)
parser.add_argument('--target-fams', type=str, default='')
parser.add_argument('--split-option', type=str, default='b', choices=['a', 'b'],
                    help="'b' (latent donut hole) is the protocol default; 'a' fits IF in "
                         "raw 4561-d binary space and ties heavily on LAMDA")
parser.add_argument('--alpha', type=float, default=0.2,
                    help='forget-objective weight. alpha=0 is the extra-training CONTROL')
parser.add_argument('--unlearn-ratio', type=float, default=0.10,
                    help='fraction of the current task forgotten (Option B)')
parser.add_argument('--unlearn-epochs', type=int, default=3)
parser.add_argument('--unlearn-lr', type=float, default=1e-4)
parser.add_argument('--rnt-mode', choices=['inverse', 'floor', 'sqrt', 'fixed'],
                    default='inverse',
                    help="plasticity/stability weight schedule. 'inverse' = 1/(tid+1), "
                         "the original van de Ven-style per-task-equal weighting and the "
                         "default so nothing changes silently. 'floor' = "
                         "max(1/(tid+1), --rnt-floor). 'sqrt' = 1/sqrt(tid+1). "
                         "'fixed' = --rnt-value.")
parser.add_argument('--rnt-floor', type=float, default=0.25,
                    help="lower bound for --rnt-mode floor. 0.25 gives new families "
                         "12.5%% of the gradient, the level at which family 40 reached "
                         "50%% recall; at the unfloored 4.5%% family 75 was never "
                         "predicted at all.")
parser.add_argument('--rnt-value', type=float, default=0.5,
                    help='constant rnt for --rnt-mode fixed')
parser.add_argument('--out-prefix', type=str, default='lamda_classil')
args = parser.parse_args()

SEED = args.seed

# --- Enforce Determinism (identical block in all conditions) ---
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
#      HYPERPARAMETERS & CONFIGURATION
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TASK0_EPOCHS = args.task0_epochs
CL_ITERS = args.cl_iters
BATCH_SIZE = args.batch_size

MEM_SIZE = args.mem_size
# NOTE: contamination only shifts IsolationForest's decision threshold; the
# selection below is purely rank-based on decision_function scores, so this
# value does not affect which samples are chosen. Kept for API compatibility.
MADAR_CONTAMINATION = 0.1

TASK_1_FAMILIES = args.task0_classes         # 30 (EMBER: 50)
SUBSEQUENT_FAMILIES = args.step_classes      # 5  (same as EMBER)
NUM_CLASSES = args.num_classes               # 80 (EMBER: 100)

KD_TEMP = 2.0
SI_C = args.si_c                             # 100 (EMBER 2018: 1.0)
SI_EPS = 0.1

SPLIT_OPTION = args.split_option
UNLEARN_ALPHA = args.alpha
UNLEARN_RATIO = args.unlearn_ratio
UNLEARN_EPOCHS = args.unlearn_epochs
UNLEARN_LR = args.unlearn_lr
# ==========================================



def rnt_for(tid):
    """Weight on the current-task loss: total = rnt*loss_cur + (1-rnt)*loss_replay.

    `loss_cur` is the ONLY term that can raise a new family's logit (loss_replay
    is KD over [:prev_active], i.e. old classes only), and new families are half
    of its batch. So a new family's share of the objective is rnt/2.

    The original schedule 1/(tid+1) weights every task seen so far equally. That
    is principled when tasks are interchangeable, but here task 0 has 30 families
    and 135k samples while task 10 has 5 families and 1k, and the headline metric
    (macro accuracy) weights FAMILIES equally, not tasks. Under the original
    schedule the final task's families get 4.5% of the gradient and are never
    learned at all.
    """
    if args.rnt_mode == 'inverse':
        return 1.0 / (tid + 1)
    if args.rnt_mode == 'floor':
        return max(1.0 / (tid + 1), args.rnt_floor)
    if args.rnt_mode == 'sqrt':
        return 1.0 / math.sqrt(tid + 1)
    return args.rnt_value


def rnt_tag():
    """Suffix so a changed schedule cannot overwrite an --rnt-mode inverse run."""
    if args.rnt_mode == 'inverse':
        return ''
    if args.rnt_mode == 'floor':
        return f'_rntfloor{args.rnt_floor:g}'
    if args.rnt_mode == 'sqrt':
        return '_rntsqrt'
    return f'_rntfix{args.rnt_value:g}'

# --- 2. Model Definition (identical architecture to the EMBER conditions) ---
class EmberNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(EmberNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.fc3_bn = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128)
        self.fc4_bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.fc_last = nn.Linear(128, num_classes)

    def forward(self, x, return_latent=False):
        x = self.relu(self.fc1_bn(self.fc1(x)))
        x = self.relu(self.fc2_bn(self.fc2(x)))
        x = self.relu(self.fc3_bn(self.fc3(x)))
        latent = self.relu(self.fc4_bn(self.fc4(x)))
        logits = self.fc_last(latent)
        return (logits, latent) if return_latent else logits


# --- 3. CL & Replay Functions ---
replay_buffer = []
family_buffers = {}


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def get_loader(X, y, families, drop_last=False):
    mask = torch.isin(y, torch.tensor(families))
    if not mask.any():
        return None
    return data.DataLoader(data.TensorDataset(X[mask], y[mask]),
                           batch_size=BATCH_SIZE, shuffle=True, drop_last=drop_last)


def loss_fn_kd(scores, target_scores, T=2.0):
    log_scores_norm = F.log_softmax(scores / T, dim=1)
    targets_norm = F.softmax(target_scores / T, dim=1)
    return F.kl_div(log_scores_norm, targets_norm, reduction='batchmean') * (T ** 2)


def update_buffer_madar(loader, model):
    """MADAR-IF buffer update. IsolationForest is fit in LATENT space (128-d),
    NOT raw feature space -- this matters far more on LAMDA than on EMBER, since
    raw LAMDA features are 4561-d binary with many constant columns per family,
    which would give heavily tied scores and near-arbitrary argsort ordering.
    The 128-d latent is continuous regardless of input representation."""
    global family_buffers, replay_buffer
    model.eval()
    # iterate the full underlying dataset with drop_last=False so buffer
    # selection sees every sample, even when the training loader drops its
    # last (partial) batch.
    full_loader = data.DataLoader(loader.dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_vecs, all_labs, all_latents = [], [], []
    with torch.no_grad():
        for v, l in full_loader:
            v = v.to(DEVICE)
            _, latent = model(v, return_latent=True)
            all_vecs.append(v.cpu())
            all_labs.append(l.cpu())
            all_latents.append(latent.cpu())

    X_np = torch.cat(all_vecs).numpy()
    Y_np = torch.cat(all_labs).numpy()
    L_np = torch.cat(all_latents).numpy()
    current_families = np.unique(Y_np)
    budget_per_family = MEM_SIZE // (len(family_buffers) + len(current_families))

    for fam in family_buffers:
        if len(family_buffers[fam]) > budget_per_family:
            current_buffer = family_buffers[fam]
            half_budget = budget_per_family // 2
            anomalies = current_buffer[::2]
            inliers = current_buffer[1::2]
            new_buffer = [val for pair in zip(anomalies[:half_budget], inliers[:half_budget])
                          for val in pair]
            if budget_per_family % 2 != 0 and len(anomalies) > half_budget:
                new_buffer.append(anomalies[half_budget])
            family_buffers[fam] = new_buffer

    for fam in current_families:
        fam_mask = (Y_np == fam)
        X_fam, Y_fam, L_fam = X_np[fam_mask], Y_np[fam_mask], L_np[fam_mask]
        n_select = min(budget_per_family, len(X_fam))
        if n_select == 0:
            continue

        iso = IsolationForest(contamination=MADAR_CONTAMINATION, n_jobs=-1, random_state=SEED)
        iso.fit(L_fam)
        scores = iso.decision_function(L_fam)
        sorted_idx = np.argsort(scores)

        half = n_select // 2
        anomalies_idx = sorted_idx[:half]
        inliers_idx = sorted_idx[-(n_select - half):]

        interleaved_idx = [idx for pair in zip(anomalies_idx, inliers_idx) for idx in pair]
        if n_select % 2 != 0:
            interleaved_idx.append(inliers_idx[-1])

        family_buffers[fam] = [(torch.tensor(X_fam[i]), torch.tensor(Y_fam[i]))
                               for i in interleaved_idx]

    replay_buffer.clear()
    for fam_data in family_buffers.values():
        replay_buffer.extend(fam_data)


def train_cl_er(model, teacher_model, optimizer, loader, iters, active_count,
                prev_active_count, W, omega, p_old_task, tid, si_c):
    model.train()

    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.eval()

    mask = torch.full((model.fc_last.out_features,), -1e9).to(DEVICE)
    mask[:active_count] = 0.0

    loader_iter = iter(cycle(loader))

    assert replay_buffer, "train_cl_er requires a non-empty replay buffer"
    b_v = torch.stack([i[0] for i in replay_buffer])
  
    b_l = torch.stack([i[1] for i in replay_buffer])
    buf_loader = data.DataLoader(data.TensorDataset(b_v, b_l),
                                 batch_size=BATCH_SIZE, shuffle=True)
    buf_iter = iter(cycle(buf_loader))

    for step in range(iters):
        v, l = next(loader_iter)
        v, l = v.to(DEVICE), l.to(DEVICE)
        optimizer.zero_grad()

        mem_v, mem_l = next(buf_iter)
        mem_v, mem_l = mem_v.to(DEVICE), mem_l.to(DEVICE)

        combined_v = torch.cat([v, mem_v], dim=0)
        combined_l = torch.cat([l, mem_l], dim=0)
        loss_cur = nn.CrossEntropyLoss()(model(combined_v) + mask, combined_l)

        rnt = rnt_for(tid)
        outputs_mem = model(mem_v)
        with torch.no_grad():
            teacher_logits = teacher_model(mem_v)[:, :prev_active_count]

        loss_replay = loss_fn_kd(outputs_mem[:, :prev_active_count], teacher_logits, T=KD_TEMP)
        loss_main = (rnt * loss_cur) + ((1.0 - rnt) * loss_replay)

        si_loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                n_key = n.replace('.', '__')
                si_loss += (omega[n_key] * (p - p_old_task[n_key]) ** 2).sum()

        total_loss = loss_main + (si_c * si_loss)
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Non-finite loss at task {tid}, step {step} - training diverged")
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        snap = {}
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                n_key = n.replace('.', '__')
                snap[n_key] = (p.grad.detach().clone(), p.detach().clone())

        optimizer.step()

        for n, p in model.named_parameters():
            if p.requires_grad:
                n_key = n.replace('.', '__')
                if n_key in snap:
                    g, p_before = snap[n_key]
                    W[n_key].add_(-g * (p.detach() - p_before))


# --- Eval (shared with the naive / joint conditions) ---
def eval_acc(model, X, y, families, active_count):
    """Micro accuracy (% correct) over the given families."""
    model.eval()
    mask = torch.isin(y, torch.tensor(families))
    if not mask.any():
        return 0.0
    eval_loader = data.DataLoader(data.TensorDataset(X[mask], y[mask]), batch_size=BATCH_SIZE)
    correct = total = 0
    with torch.no_grad():
        for bv, bl in eval_loader:
            bv, bl = bv.to(DEVICE), bl.to(DEVICE)
            _, predicted = torch.max(model(bv)[:, :active_count], 1)
            total += bl.size(0)
            correct += (predicted == bl).sum().item()
    return (correct / total) * 100


def eval_acc_macro(model, X, y, families, active_count):
    
    model.eval()
    mask = torch.isin(y, torch.tensor(families))
    if not mask.any():
        return 0.0
    eval_loader = data.DataLoader(data.TensorDataset(X[mask], y[mask]), batch_size=BATCH_SIZE)
    correct = torch.zeros(active_count)
    total = torch.zeros(active_count)
    with torch.no_grad():
        for bv, bl in eval_loader:
            bv = bv.to(DEVICE)
            _, predicted = torch.max(model(bv)[:, :active_count], 1)
            bl_cpu = bl.cpu()
            if int(bl_cpu.max()) >= active_count:
                raise RuntimeError(
                    f"label {int(bl_cpu.max())} >= active_count {active_count}; "
                    f"the prefix-mask assumption is broken")
            total += torch.bincount(bl_cpu, minlength=active_count).float()
            correct += torch.bincount(bl_cpu[(predicted.cpu() == bl_cpu)],
                                      minlength=active_count).float()
    present = total > 0
    if not present.any():
        return 0.0
    return (correct[present] / total[present]).mean().item() * 100


def eval_family_metrics(model, X_test, y_test, target_families, active_count):
    """Per-family recall and false-positive rate over all seen families."""
    model.eval()
    seen_mask = y_test < active_count
    if not seen_mask.any():
        return {fam: {'acc': 0.0, 'fpr': 0.0} for fam in target_families}
    eval_loader = data.DataLoader(
        data.TensorDataset(X_test[seen_mask], y_test[seen_mask]), batch_size=BATCH_SIZE)
    counts = {fam: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0} for fam in target_families}
    with torch.no_grad():
        for bv, bl in eval_loader:
            bv, bl = bv.to(DEVICE), bl.to(DEVICE)
            _, predicted = torch.max(model(bv)[:, :active_count], 1)
            for fam in target_families:
                counts[fam]['tp'] += ((predicted == fam) & (bl == fam)).sum().item()
                counts[fam]['fp'] += ((predicted == fam) & (bl != fam)).sum().item()
                counts[fam]['fn'] += ((predicted != fam) & (bl == fam)).sum().item()
                counts[fam]['tn'] += ((predicted != fam) & (bl != fam)).sum().item()
    results = {}
    for fam in target_families:
        tp, fp, fn, tn = (counts[fam][k] for k in ('tp', 'fp', 'fn', 'tn'))
        results[fam] = {
            'acc': (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0,
            'fpr': (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0.0,
        }
    return results


def cap_per_family(X_np, y_tensor, cap, cap_seed):
    if not cap or cap <= 0:
        return X_np, y_tensor
    rng = np.random.default_rng(cap_seed)
    y_np = y_tensor.numpy()
    keep = []
    for fam in np.unique(y_np):
        idx = np.flatnonzero(y_np == fam)
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    return X_np[keep], y_tensor[keep]


def default_target_fams(task0, num_classes, step):
    t0 = {0, task0 // 6, task0 // 3, (2 * task0) // 3, task0 - 1}
    later = [task0 + step * 2, num_classes - step]
    return sorted({c for c in list(t0) + later if 0 <= c < num_classes})


# --- 4. Unlearning Modules ---

def split_option_a_scrub_leftovers(X, y, families, budget_per_family):
    
    mask = torch.isin(y, torch.tensor(families))
    if not mask.any():
        return None, None

    X_subset, y_subset = X[mask], y[mask]
    retain_idx_all, forget_idx_all = [], []
    X_np = X_subset.numpy()
    y_np = y_subset.numpy()

    for fam in families:
        fam_mask = (y_np == fam)
        if not fam_mask.any():
            continue
        fam_indices = np.where(fam_mask)[0]
        X_fam = X_np[fam_mask]
        n_select = min(budget_per_family, len(X_fam))
        if n_select == len(X_fam):
            retain_idx_all.extend(fam_indices)
            continue

        iso = IsolationForest(contamination=MADAR_CONTAMINATION, n_jobs=-1, random_state=SEED)
        iso.fit(X_fam)
        scores = iso.decision_function(X_fam)
        sorted_local_idx = np.argsort(scores)

        half = n_select // 2
        anomalies_local = sorted_local_idx[:half]
        inliers_local = sorted_local_idx[-(n_select - half):]
        selected_local = np.concatenate((anomalies_local, inliers_local))
        unselected_local = np.setdiff1d(np.arange(len(X_fam)), selected_local)

        retain_idx_all.extend(fam_indices[selected_local])
        forget_idx_all.extend(fam_indices[unselected_local])

    if not forget_idx_all:
        retain_loader = data.DataLoader(data.TensorDataset(X_subset, y_subset),
                                        batch_size=BATCH_SIZE, shuffle=True)
        return retain_loader, None

    retain_loader = data.DataLoader(
        data.TensorDataset(X_subset[retain_idx_all], y_subset[retain_idx_all]),
        batch_size=BATCH_SIZE, shuffle=True)
    forget_loader = data.DataLoader(
        data.TensorDataset(X_subset[forget_idx_all], y_subset[forget_idx_all]),
        batch_size=BATCH_SIZE, shuffle=True)
    return retain_loader, forget_loader


def split_option_b_donut_hole(X, y, families, model, forget_ratio=0.1):
    model.eval()
    mask = torch.isin(y, torch.tensor(families))
    if not mask.any():
        return None, None

    X_subset, y_subset = X[mask], y[mask]
    all_latents = []
    with torch.no_grad():
        loader = data.DataLoader(data.TensorDataset(X_subset, y_subset), batch_size=BATCH_SIZE)
        for v, _ in loader:
            _, latent = model(v.to(DEVICE), return_latent=True)
            all_latents.append(latent.cpu())
    L_np = torch.cat(all_latents).numpy()

    iso = IsolationForest(contamination=MADAR_CONTAMINATION, n_jobs=-1, random_state=SEED)
    iso.fit(L_np)
    scores = iso.decision_function(L_np)

    sorted_idx = np.argsort(scores)
    n_total = len(sorted_idx)
    n_forget = int(n_total * forget_ratio)

    if n_forget == 0:
        retain_loader = data.DataLoader(data.TensorDataset(X_subset, y_subset),
                                        batch_size=BATCH_SIZE, shuffle=True)
        return retain_loader, None

    mid_start = (n_total // 2) - (n_forget // 2)
    mid_end = mid_start + n_forget
    forget_idx = sorted_idx[mid_start:mid_end]
    retain_idx = np.concatenate((sorted_idx[:mid_start], sorted_idx[mid_end:]))

    forget_fams, forget_counts = np.unique(y_subset[forget_idx].numpy(), return_counts=True)
    print(f"    [Split B] Forget set per family: "
          f"{dict(zip(forget_fams.tolist(), forget_counts.tolist()))}")

    forget_loader = data.DataLoader(
        data.TensorDataset(X_subset[forget_idx], y_subset[forget_idx]),
        batch_size=BATCH_SIZE, shuffle=True)
    retain_loader = data.DataLoader(
        data.TensorDataset(X_subset[retain_idx], y_subset[retain_idx]),
        batch_size=BATCH_SIZE, shuffle=True)
    return retain_loader, forget_loader


def unlearn_teacher_guided(model, teacher_model, forget_loader, retain_loader, active_count,
                           prev_active_count, omega, p_old_task, si_c, epochs=3, lr=1e-4,
                           alpha=0.2):
    """Flattens the forget set's current-task predictions toward uniform,
    anchored by ground truth CE + KD on retain/replay data and an SI penalty
    protecting past-task-important parameters.
    With alpha=0 this becomes the extra-training CONTROL: identical steps,
    optimizer and data flow, but no forget objective."""
    global replay_buffer
    model.train()
    teacher_model.eval()

    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.eval()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    mask = torch.full((model.fc_last.out_features,), -1e9).to(DEVICE)
    mask[:active_count] = 0.0

    retain_iter = iter(cycle(retain_loader))
    if replay_buffer:
        b_v = torch.stack([i[0] for i in replay_buffer])
        b_l = torch.stack([i[1] for i in replay_buffer])
        buf_loader = data.DataLoader(data.TensorDataset(b_v, b_l),
                                     batch_size=BATCH_SIZE, shuffle=True)
        buf_iter = iter(cycle(buf_loader))
    else:
        buf_iter = None

    for epoch in range(epochs):
        for f_v, f_l in forget_loader:
            f_v, f_l = f_v.to(DEVICE), f_l.to(DEVICE)
            optimizer.zero_grad()

            # --- 1. TARGETED ENTROPY (Forget) ---
            logits = model(f_v)
            current_task_logits = logits[:, prev_active_count:active_count]
            log_probs = F.log_softmax(current_task_logits, dim=1)
            num_current_classes = active_count - prev_active_count
            uniform_probs = torch.ones_like(log_probs) / num_current_classes
            forget_loss = F.kl_div(log_probs, uniform_probs, reduction='batchmean')

            # --- 2. GROUND TRUTH + DISTILLATION (Retain + Replay) ---
            r_v, r_l = next(retain_iter)
            r_v, r_l = r_v.to(DEVICE), r_l.to(DEVICE)
            if buf_iter:
                mem_v, mem_l = next(buf_iter)
                mem_v, mem_l = mem_v.to(DEVICE), mem_l.to(DEVICE)
                r_v = torch.cat([r_v, mem_v], dim=0)
                r_l = torch.cat([r_l, mem_l], dim=0)

            with torch.no_grad():
                teacher_logits = teacher_model(r_v)[:, :active_count]

            student_logits_raw = model(r_v)
            student_logits_masked = student_logits_raw + mask
            student_logits_active = student_logits_raw[:, :active_count]

            retain_ce = nn.CrossEntropyLoss()(student_logits_masked, r_l)
            retain_kd = loss_fn_kd(student_logits_active, teacher_logits, T=KD_TEMP)
            retain_loss = (0.5 * retain_ce) + (0.5 * retain_kd)

            # --- 3. SYNAPTIC INTELLIGENCE (Protect Past Tasks) ---
            # omega already includes the just-finished task's contribution (updated
            # before unlearning), and p_old_task is still the START-OF-TASK anchor.
            # So unlearning is penalised for drifting important parameters away from
            # where the task began, except through the forget objective. Intentional.
            si_loss = 0
            for n, p in model.named_parameters():
                if p.requires_grad:
                    n_key = n.replace('.', '__')
                    if n_key in p_old_task:
                        si_loss += (omega[n_key] * (p - p_old_task[n_key]) ** 2).sum()

            # --- 4. COMBINED UPDATE ---
            total_loss = (alpha * forget_loss) + ((1.0 - alpha) * retain_loss) + (si_c * si_loss)
            if not torch.isfinite(total_loss):
                raise RuntimeError(
                    f"Non-finite loss during unlearning (epoch {epoch}) - diverged")
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()


def measure_unlearning_efficacy(model_before, model_after, loader, active_count):
    """Accuracy BOTH before and after unlearning; entropy raw and normalised by
    log(active_count) (raw entropy grows mechanically as classes are added, so only
    the normalised value is comparable across tasks); latent shift as mean/median/max
    (a handful of extreme samples can inflate the mean)."""
    model_before.eval()
    model_after.eval()
    acc_before = acc_after = 0.0
    entropies, shifts = [], []
    total_samples = 0

    with torch.no_grad():
        for v, l in loader:
            v, l = v.to(DEVICE), l.to(DEVICE)
            logits_before, latent_before = model_before(v, return_latent=True)
            logits_before = logits_before[:, :active_count]
            logits_after, latent_after = model_after(v, return_latent=True)
            logits_after = logits_after[:, :active_count]
            probs = F.softmax(logits_after, dim=1)

            acc_before += (torch.argmax(logits_before, dim=1) == l).sum().item()
            acc_after += (torch.argmax(probs, dim=1) == l).sum().item()
            entropies.append(-(probs * torch.log(probs + 1e-9)).sum(dim=1).cpu())
            shifts.append(torch.norm(latent_after - latent_before, p=2, dim=1).cpu())
            total_samples += l.size(0)

    entropies = torch.cat(entropies)
    shifts = torch.cat(shifts)
    mean_entropy = entropies.mean().item()
    return {
        'acc_before': (acc_before / total_samples) * 100,
        'acc_after': (acc_after / total_samples) * 100,
        'entropy': mean_entropy,
        'entropy_norm': mean_entropy / float(np.log(active_count)),
        'shift_mean': shifts.mean().item(),
        'shift_median': shifts.median().item(),
        'shift_max': shifts.max().item(),
    }


# --- 5. Main Execution ---
if __name__ == "__main__":
    if not os.path.exists(args.cache):
        raise SystemExit(f"cache not found: {args.cache}\nRun build_lamda_cache.py first.")

    arrays = build_arrays_lamda(
        args.cache,
        split_mode=args.split_mode,
        temporal_cut=args.temporal_cut if args.split_mode == 'temporal' else None,
        min_family_samples=args.min_family_samples,
        num_classes=NUM_CLASSES,
        task0_classes=TASK_1_FAMILIES,
        step_classes=SUBSEQUENT_FAMILIES,
        min_test_samples=args.min_test_samples,
    )
    INPUT_DIM = arrays['input_dim']
    task_families = arrays['task_families']
    TOTAL_TASKS = len(task_families)

    X_train_np, X_test_np = arrays['X_train'], arrays['X_test']
    if args.scale == 'standard':
        from sklearn.preprocessing import StandardScaler
        FEATURE_CLIP = 10.0
        scaler = StandardScaler()
        t0m = np.isin(arrays['y_train'], task_families[0])
        scaler.fit(X_train_np[t0m])
        X_train_np = np.clip(scaler.transform(X_train_np), -FEATURE_CLIP, FEATURE_CLIP)
        X_test_np = np.clip(scaler.transform(X_test_np), -FEATURE_CLIP, FEATURE_CLIP)
        print("[scale] standard + clip +/-10 applied (ABLATION, not the default)")
    else:
        print("[scale] none -- binary features used as-is (protocol default)")

    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_train = torch.tensor(arrays['y_train'], dtype=torch.long)
    y_test = torch.tensor(arrays['y_test'], dtype=torch.long)
    del X_train_np, X_test_np

    n_before = len(y_train)
    X_train, y_train = cap_per_family(X_train, y_train, args.family_cap, args.cap_seed)
    if args.family_cap:
        print(f"[cap] {args.family_cap}/family: {n_before:,} -> {len(y_train):,} train")
    else:
        print("[cap] none (protocol default)")

    TARGET_FAMS = ([int(x) for x in args.target_fams.split(',') if x.strip()]
                   if args.target_fams
                   else default_target_fams(TASK_1_FAMILIES, NUM_CLASSES, SUBSEQUENT_FAMILIES))
    print(f"[track] per-family recall/FPR for classes {TARGET_FAMS}")
    print(f"[madar] MEM_SIZE={MEM_SIZE}  CL_ITERS={CL_ITERS}  SI_C={SI_C}  KD_TEMP={KD_TEMP}")
    _r = [rnt_for(t) for t in range(1, TOTAL_TASKS)]
    print(f"[rnt] mode={args.rnt_mode}"
          + (f" floor={args.rnt_floor:g}" if args.rnt_mode == 'floor' else "")
          + (f" value={args.rnt_value:g}" if args.rnt_mode == 'fixed' else "")
          + f" | rnt task1..{TOTAL_TASKS-1}: "
          + " ".join(f"{x:.3f}" for x in _r)
          + f" | new-family gradient weight {_r[0]*50:.1f}% -> {_r[-1]*50:.1f}%")
    print(f"[unlearn] option={SPLIT_OPTION.upper()}  alpha={UNLEARN_ALPHA}  "
          f"ratio={UNLEARN_RATIO}  epochs={UNLEARN_EPOCHS}  lr={UNLEARN_LR}"
          + ("   <-- alpha=0: EXTRA-TRAINING CONTROL (no forget objective)"
             if UNLEARN_ALPHA == 0 else ""))
    if SPLIT_OPTION == 'a':
        print("[unlearn] !! Option A fits IsolationForest in RAW 4561-d BINARY space; "
              "scores tie heavily on LAMDA. Option B (latent) is the protocol default.")

    model = EmberNN(INPUT_DIM, NUM_CLASSES).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] EmberNN({INPUT_DIM} -> 1024 -> 512 -> 256 -> 128 -> {NUM_CLASSES})"
          f"  params={n_params:,}")

    W = {n.replace('.', '__'): torch.zeros_like(p).to(DEVICE)
         for n, p in model.named_parameters() if p.requires_grad}
    p_old_task = {n.replace('.', '__'): p.detach().clone().to(DEVICE)
                  for n, p in model.named_parameters() if p.requires_grad}
    omega = {n.replace('.', '__'): torch.zeros_like(p).to(DEVICE)
             for n, p in model.named_parameters() if p.requires_grad}

    teacher_model = None
    prev_active_count = 0
    history = {'Avg_Acc': [], 'Avg_Acc_Macro': [], 'Task0_Acc': [],
               'New_Fam_Acc': [], 'New_Fam_Acc_Macro': [],
               'Train_Samples': [], 'Grad_Steps': [], 'Buffer_Size': [],
               'Forget_Acc_Before': [], 'Forget_Acc_After': [],
               'Retain_Acc_Before': [], 'Retain_Acc_After': [],
               'Forget_Entropy_Norm': [], 'Forget_Shift_Median': [],
               'Forget_Set_Size': []}
    fam_history = {fam: {'acc': [], 'fpr': []} for fam in TARGET_FAMS}

    for tid, current_fams in enumerate(task_families):
        print(f"\n=== Training Task {tid} ({len(current_fams)} Families) ===")
        active_count = TASK_1_FAMILIES + (tid * SUBSEQUENT_FAMILIES)
        seen_fams = [f for sublist in task_families[:tid + 1] for f in sublist]
        assert active_count == len(seen_fams), "active_count / task split mismatch"


        loader = get_loader(X_train, y_train, current_fams, drop_last=True)
        if loader is None:
            raise RuntimeError(f"task {tid} has no training samples")
        if len(loader) == 0:
            raise RuntimeError(
                f"task {tid} has {len(loader.dataset)} samples but 0 batches at "
                f"batch_size={BATCH_SIZE} with drop_last=True")

        n_samples = len(loader.dataset)

        if tid == 0:
            # --- PURE BASE LEARNING: Adam, Epochs, No SI, No KD ---
            grad_steps = TASK0_EPOCHS * len(loader)
            print(f"  Samples: {n_samples} | Gradient steps: {grad_steps}")
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            model.train()
            mask = torch.full((model.fc_last.out_features,), -1e9).to(DEVICE)
            mask[:active_count] = 0.0

            for epoch in range(TASK0_EPOCHS):
                for v, l in loader:
                    v, l = v.to(DEVICE), l.to(DEVICE)
                    optimizer.zero_grad()
                    loss = nn.CrossEntropyLoss()(model(v) + mask, l)
                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            f"Non-finite loss at task {tid}, epoch {epoch} - training diverged")
                    loss.backward()
                    optimizer.step()

            for k in ('Forget_Acc_Before', 'Forget_Acc_After', 'Retain_Acc_Before',
                      'Retain_Acc_After', 'Forget_Entropy_Norm', 'Forget_Shift_Median',
                      'Forget_Set_Size'):
                history[k].append(None)

        else:
            # --- CONTINUAL LEARNING: SGD, Iters, SI, KD ---
            grad_steps = CL_ITERS
            print(f"  Samples: {n_samples} (+{len(replay_buffer)} replay) | "
                  f"Gradient steps: {grad_steps}")
            optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-6)
            train_cl_er(model, teacher_model, optimizer, loader, CL_ITERS, active_count,
                        prev_active_count, W, omega, p_old_task, tid, si_c=SI_C)

            # --- 2. SI OMEGA UPDATE: immediately after CL, BEFORE unlearning ---

            for n, p in model.named_parameters():
                if p.requires_grad:
                    n_key = n.replace('.', '__')
                    p_post_cl = p.detach().clone()
                    omega[n_key] += W[n_key] / ((p_post_cl - p_old_task[n_key]) ** 2 + SI_EPS)
                    W[n_key].zero_()

            # --- 3. UNLEARNING PHASE ---
            print(f"  -> Unlearning (option {SPLIT_OPTION.upper()}, alpha={UNLEARN_ALPHA})")
            if SPLIT_OPTION == 'a':
                budget_per_family = MEM_SIZE // (len(family_buffers) + len(current_fams))
                retain_loader, forget_loader = split_option_a_scrub_leftovers(
                    X_train, y_train, current_fams, budget_per_family)
            else:
                retain_loader, forget_loader = split_option_b_donut_hole(
                    X_train, y_train, current_fams, model, forget_ratio=UNLEARN_RATIO)

            if forget_loader is not None and retain_loader is not None:
                model_pre_unlearn = copy.deepcopy(model)
                unlearn_teacher_guided(
                    model=model, teacher_model=model_pre_unlearn,
                    forget_loader=forget_loader, retain_loader=retain_loader,
                    active_count=active_count, prev_active_count=prev_active_count,
                    omega=omega, p_old_task=p_old_task, si_c=SI_C,
                    epochs=UNLEARN_EPOCHS, lr=UNLEARN_LR, alpha=UNLEARN_ALPHA)

                # account the extra optimisation honestly
                grad_steps += UNLEARN_EPOCHS * len(forget_loader)

                f_m = measure_unlearning_efficacy(model_pre_unlearn, model,
                                                  forget_loader, active_count)
                r_m = measure_unlearning_efficacy(model_pre_unlearn, model,
                                                  retain_loader, active_count)
                for tag, m in (('Forget', f_m), ('Retain', r_m)):
                    print(f"    [{tag}] Acc {m['acc_before']:.2f}% -> {m['acc_after']:.2f}%"
                          f" | Entropy {m['entropy']:.3f} (norm {m['entropy_norm']:.3f})"
                          f" | Latent shift mean/med/max "
                          f"{m['shift_mean']:.3f}/{m['shift_median']:.3f}/{m['shift_max']:.1f}")

                history['Forget_Acc_Before'].append(f_m['acc_before'])
                history['Forget_Acc_After'].append(f_m['acc_after'])
                history['Retain_Acc_Before'].append(r_m['acc_before'])
                history['Retain_Acc_After'].append(r_m['acc_after'])
                history['Forget_Entropy_Norm'].append(f_m['entropy_norm'])
                history['Forget_Shift_Median'].append(f_m['shift_median'])
                history['Forget_Set_Size'].append(len(forget_loader.dataset))

                # buffer must be rebuilt from the RETAIN set only
                loader = retain_loader
            else:
                print("    [unlearn] split produced no forget set; skipping")
                for k in ('Forget_Acc_Before', 'Forget_Acc_After', 'Retain_Acc_Before',
                          'Retain_Acc_After', 'Forget_Entropy_Norm',
                          'Forget_Shift_Median', 'Forget_Set_Size'):
                    history[k].append(None)

        # --- 4. POST-TASK UPDATES (all tasks) ---
    
        for n, p in model.named_parameters():
            if p.requires_grad:
                p_old_task[n.replace('.', '__')] = p.detach().clone()
        teacher_model = copy.deepcopy(model)
        teacher_model.eval()
        prev_active_count = active_count
      
        update_buffer_madar(loader, model)

        avg_acc = eval_acc(model, X_test, y_test, seen_fams, active_count)
        macro_acc = eval_acc_macro(model, X_test, y_test, seen_fams, active_count)
        t0_acc = eval_acc(model, X_test, y_test, task_families[0], active_count)
        new_acc = eval_acc(model, X_test, y_test, current_fams, active_count)
        new_macro = eval_acc_macro(model, X_test, y_test, current_fams, active_count)

        history['Avg_Acc'].append(avg_acc)
        history['Avg_Acc_Macro'].append(macro_acc)
        history['Task0_Acc'].append(t0_acc)
        history['New_Fam_Acc'].append(new_acc)
        history['New_Fam_Acc_Macro'].append(new_macro)
        history['Train_Samples'].append(n_samples)
        history['Grad_Steps'].append(grad_steps)
        history['Buffer_Size'].append(len(replay_buffer))
        print(f"  -> Avg: {avg_acc:.2f}% (macro {macro_acc:.2f}%) | Task 0: {t0_acc:.2f}%"
              f" | New fams: {new_acc:.2f}% (macro {new_macro:.2f}%)"
              f" | Buffer: {len(replay_buffer)}")

        fam_metrics = eval_family_metrics(model, X_test, y_test, TARGET_FAMS, active_count)
        for fam in TARGET_FAMS:
            fam_history[fam]['acc'].append(fam_metrics[fam]['acc'])
            fam_history[fam]['fpr'].append(fam_metrics[fam]['fpr'])
        print("  Per-family recall: " +
              " ".join(f"f{fam}:{fam_metrics[fam]['acc']:.1f}%" for fam in TARGET_FAMS))

    # Output tag includes any hyperparameter that differs from the protocol
    # default, so sweep runs cannot silently overwrite one another.
    tag = f"{args.out_prefix}_mu_opt{SPLIT_OPTION.upper()}_a{UNLEARN_ALPHA:g}_r{UNLEARN_RATIO:g}"
    if args.cl_iters != parser.get_default('cl_iters'):
        tag += f"_it{args.cl_iters}"
    if args.si_c != parser.get_default('si_c'):
        tag += f"_sic{args.si_c:g}"
    if args.mem_size != parser.get_default('mem_size'):
        tag += f"_mem{args.mem_size}"
    tag += rnt_tag()
    if args.split_mode == 'temporal':
        tag += f"_temporal{args.temporal_cut}"
    # Schedule/floor go in the tag too: a >=400-floor run has a DIFFERENT family
    # set and task count from the >=200 protocol, and without this it would
    # silently overwrite those results.
    if args.min_family_samples != parser.get_default('min_family_samples'):
        tag += f"_min{args.min_family_samples}"
    if (args.num_classes != parser.get_default('num_classes')
            or args.task0_classes != parser.get_default('task0_classes')
            or args.step_classes != parser.get_default('step_classes')):
        tag += f"_sched{args.task0_classes}+{args.step_classes}x" \
               f"{(args.num_classes - args.task0_classes) // args.step_classes}"
    meta = {'dataset': 'lamda_classil', 'condition': 'madar_mu',
            'rnt_mode': args.rnt_mode,
            'rnt_floor': args.rnt_floor, 'rnt_value': args.rnt_value,
            'split_mode': args.split_mode,
            'temporal_cut': args.temporal_cut if args.split_mode == 'temporal' else None,
            'num_classes': NUM_CLASSES, 'task0_classes': TASK_1_FAMILIES,
            'step_classes': SUBSEQUENT_FAMILIES, 'total_tasks': TOTAL_TASKS,
            'input_dim': INPUT_DIM, 'scale': args.scale,
            'family_cap': args.family_cap or None, 'cap_seed': args.cap_seed,
            'mem_size': MEM_SIZE, 'cl_iters': CL_ITERS, 'si_c': SI_C,
            'split_option': SPLIT_OPTION, 'alpha': UNLEARN_ALPHA,
            'unlearn_ratio': UNLEARN_RATIO, 'unlearn_epochs': UNLEARN_EPOCHS,
            'unlearn_lr': UNLEARN_LR,
            'si_eps': SI_EPS, 'kd_temp': KD_TEMP,
            'task0_epochs': TASK0_EPOCHS, 'batch_size': BATCH_SIZE,
            'seed': SEED, 'n_params': n_params, 'target_fams': TARGET_FAMS}
    history['Family_Metrics'] = {str(f): fam_history[f] for f in TARGET_FAMS}
    with open(f'{tag}_history_seed{SEED}.json', 'w') as f:
        json.dump({'meta': meta, **history}, f, indent=2)

    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes[0]
    ax.plot(history['Avg_Acc'], marker='o', label='Avg (seen families, micro)')
    ax.plot(history['Avg_Acc_Macro'], marker='s', label='Avg (macro, per-family mean)')
    ax.plot(history['Task0_Acc'], marker='x', label='Task 0 (retention)')
    ax.set_xlabel('Task'); ax.set_ylabel('Accuracy (%)')
    ax.set_title('Mean accuracy over seen families')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(history['New_Fam_Acc'], marker='^', color='#B4342A',
            label=f'newest {SUBSEQUENT_FAMILIES} families (micro)')
    ax.plot(history['New_Fam_Acc_Macro'], marker='v', color='#8C3A78',
            label=f'newest {SUBSEQUENT_FAMILIES} families (macro)')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylim(bottom=-2)
    ax.set_xlabel('Task'); ax.set_ylabel('Accuracy on families just added (%)')
    ax.set_title('Acquisition: the families introduced by each task')
    ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle(f'LAMDA Class-IL MADAR+MU (seed {SEED}, opt{SPLIT_OPTION.upper()}, alpha={UNLEARN_ALPHA:g}, {TASK_1_FAMILIES}+{SUBSEQUENT_FAMILIES}x{TOTAL_TASKS-1}, min {args.min_family_samples})', fontsize=12)
    fig.tight_layout()
    fig.savefig(f'{tag}_seed{SEED}.png', dpi=120, bbox_inches='tight')

    plt.figure(figsize=(10, 6))
    for fam in TARGET_FAMS:
        plt.plot(fam_history[fam]['acc'], marker='o', label=f'family {fam}')
    plt.xlabel('Task')
    plt.ylabel('Per-family recall (%)')
    plt.title(f'LAMDA Class-IL MADAR+MU: tracked family retention (seed {SEED})')
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.savefig(f'{tag}_families_seed{SEED}.png', dpi=120, bbox_inches='tight')

    print(f"\nwrote {tag}_history_seed{SEED}.json, {tag}_seed{SEED}.png, "
          f"{tag}_families_seed{SEED}.png")