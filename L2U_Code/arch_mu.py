"""ARCH STUDY -- MADAR + machine unlearning.

Derived from clmdu_ember18_mk6B_cd.py. Identical training logic; the only changes are that the
network architecture is selectable via --arch and that all output goes to a
dedicated results directory with arch-tagged filenames. At --arch full the
network is numerically identical to the original.

Purpose: measure how the unlearning gain (MADAR+MU - MADAR) varies with
classifier capacity. Run the full condition set at each architecture --
joint is required as the denominator of the recovery ratio
    recovery = (MU - MADAR) / (joint - MADAR)
which is the capacity-comparable statistic; raw pp gaps are not comparable
across architectures because a weaker baseline opens more headroom.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler  # FIX: unused RobustScaler import removed
import ember
import os
import random
import json
import csv
import argparse

from model import EmberNN, get_dims, count_params

# --- Parse Command Line Arguments ---

parser = argparse.ArgumentParser(description="Continual Learning Experiment (MADAR-IF + ER/KD/SI + Unlearning)")
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--split_option', type=str, default='b', choices=['a', 'b'],
                    help="Forget/retain split: 'a' = scrub leftovers (per-family, raw-feature IF), 'b' = donut hole (joint, latent-space IF)")
parser.add_argument('--alpha', type=float, default=0.2,
                    help='Weight of the forget loss. 0.0 gives the extra-training control ablation.')
parser.add_argument('--unlearn_ratio', type=float, default=0.10,
                    help='Fraction of current-task samples to forget (Option B only)')
parser.add_argument('--arch', type=str, default='full',
                    help="Architecture: preset (full/trunk/half/quarter/tiny) or four "
                         "widths, e.g. '512 256 128 64'. 'full' reproduces the original "
                         "network exactly. 'trunk' shrinks capacity while HOLDING the "
                         "128-d latent fixed -- that latent is the space MADAR buffer "
                         "selection and the donut selector operate in, so shrinking it "
                         "confounds capacity with selection geometry.")
parser.add_argument('--out-dir', dest='out_dir', type=str, default='results_arch',
                    help='Directory for this study\'s JSON and PNG output. Kept separate '
                         'from the original stage-1 filenames by design.')
parser.add_argument('--si_c', type=float, default=1.0,
                    help='Synaptic Intelligence strength. si_loss is a plain sum over '
                         'parameters, so at fixed si_c a smaller network gets a weaker '
                         'effective penalty. This is the primary confound of the arch '
                         'study -- it moves in the same direction as the hypothesis. '
                         'Vary it at the smallest arch as a sensitivity check.')
args = parser.parse_args()

SEED = args.seed
SPLIT_OPTION = args.split_option
UNLEARN_ALPHA = args.alpha
UNLEARN_RATIO = args.unlearn_ratio
# --- Architecture + output naming -------------------------------------------
ARCH_DIMS = get_dims(args.arch)
ARCH_TAG = ''.join(c if c.isalnum() else 'x' for c in args.arch).strip('x')
OUT_DIR = args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f'{OUT_DIR}/scatter', exist_ok=True)   # per-task misclassification plots (11 per run)
RUN_TAG = f"opt{SPLIT_OPTION.upper()}_a{UNLEARN_ALPHA}_r{UNLEARN_RATIO}_{ARCH_TAG}_seed{SEED}"

# --- Enforce Determinism (identical block in all four conditions) ---
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
#       HYPERPARAMETERS & CONFIGURATION
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./Datasets/ember2018"
EMBERSIM_DIR = "./Datasets/embersim-databank"

INPUT_DIM = 2381
FEATURE_CLIP = 10.0        # clip standardized features to [-10, 10]
TASK0_EPOCHS = 30
CL_ITERS = 2000
BATCH_SIZE = 256

MEM_SIZE = 5000
# NOTE: contamination only shifts IsolationForest's decision threshold; the
# selection is purely rank-based on decision_function scores, so this value
# does not affect which samples are chosen. Kept for API compatibility.
MADAR_CONTAMINATION = 0.1

TASK_1_FAMILIES = 50
SUBSEQUENT_FAMILIES = 5
TOTAL_TASKS = 11           # 50 base + (10 increments * 5 families) = 100 families total

KD_TEMP = 2.0
SI_C = args.si_c
SI_EPS = 0.1

UNLEARN_EPOCHS = 3
UNLEARN_LR = 1e-4
# ==========================================

# --- 1. Data Loading ---
def build_hash_to_family_map(embersim_dir):
    hash_to_fam = {}
    label_file = os.path.join(embersim_dir, "data", "raw", "ember_original_metadata.csv")
    if not os.path.exists(label_file): return hash_to_fam
    with open(label_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fam = row.get('avclass', '').strip()
            if fam and fam != 'SINGLETON':
                hash_to_fam[row['sha256']] = fam
    return hash_to_fam

def load_and_filter_features(data_dir, subset, hash_to_fam, family_to_id, X_raw):
    jsonl_files = [f for f in os.listdir(data_dir) if f.startswith(f"{subset}_features") and f.endswith(".jsonl")]
    jsonl_files.sort()
    filtered_X, filtered_y = [], []
    row_idx = 0
    for j_file in jsonl_files:
        with open(os.path.join(data_dir, j_file), 'r') as f:
            for line in f:
                data_json = json.loads(line)
                h = data_json['sha256']
                if h in hash_to_fam:
                    fam_name = hash_to_fam[h]
                    if fam_name not in family_to_id: family_to_id[fam_name] = len(family_to_id)
                    filtered_X.append(X_raw[row_idx])
                    filtered_y.append(family_to_id[fam_name])
                row_idx += 1
    return np.array(filtered_X), np.array(filtered_y)

def load_ember_data(data_dir, embersim_dir):
    if not os.path.exists(os.path.join(data_dir, "X_train.dat")):
        ember.create_vectorized_features(data_dir, feature_version=2)
    X_train_raw, _, X_test_raw, _ = ember.read_vectorized_features(data_dir, feature_version=2)
    hash_to_fam = build_hash_to_family_map(embersim_dir)
    family_to_id = {}
    X_train_np, y_train_np = load_and_filter_features(data_dir, "train", hash_to_fam, family_to_id, X_train_raw)
    X_test_np, y_test_np = load_and_filter_features(data_dir, "test", hash_to_fam, family_to_id, X_test_raw)
    return X_train_np, y_train_np, X_test_np, y_test_np

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

# --- 2. Model Definition ---
# EmberNN lives in model.py so the architecture is a controllable variable.
# At --arch full its state_dict is identical to the original definition.

# --- 3. CL & Replay Functions ---
replay_buffer = []
family_buffers = {}

def get_loader(X, y, families, drop_last=False):
    mask = torch.isin(y, torch.tensor(families))
    if not mask.any(): return None
    return data.DataLoader(data.TensorDataset(X[mask], y[mask]),
                           batch_size=BATCH_SIZE, shuffle=True, drop_last=drop_last)

def loss_fn_kd(scores, target_scores, T=2.0):
    log_scores_norm = F.log_softmax(scores / T, dim=1)
    targets_norm = F.softmax(target_scores / T, dim=1)
    return F.kl_div(log_scores_norm, targets_norm, reduction='batchmean') * (T**2)

def update_buffer_madar(loader, model):
    global family_buffers, replay_buffer
    model.eval()
    
    full_loader = data.DataLoader(loader.dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_vecs, all_labs, all_latents = [], [], []
    with torch.no_grad():
        for v, l in full_loader:
            v = v.to(DEVICE)
            _, latent = model(v, return_latent=True)
            all_vecs.append(v.cpu()); all_labs.append(l.cpu()); all_latents.append(latent.cpu())

    X_np, Y_np, L_np = torch.cat(all_vecs).numpy(), torch.cat(all_labs).numpy(), torch.cat(all_latents).numpy()
    current_families = np.unique(Y_np)
    budget_per_family = MEM_SIZE // (len(family_buffers) + len(current_families))

    for fam in family_buffers:
        if len(family_buffers[fam]) > budget_per_family:
            current_buffer = family_buffers[fam]
            half_budget = budget_per_family // 2
            anomalies = current_buffer[::2]
            inliers = current_buffer[1::2]
            new_buffer = [val for pair in zip(anomalies[:half_budget], inliers[:half_budget]) for val in pair]
            if budget_per_family % 2 != 0 and len(anomalies) > half_budget:
                new_buffer.append(anomalies[half_budget])
            family_buffers[fam] = new_buffer

    for fam in current_families:
        fam_mask = (Y_np == fam)
        X_fam, Y_fam, L_fam = X_np[fam_mask], Y_np[fam_mask], L_np[fam_mask]
        n_select = min(budget_per_family, len(X_fam))
        if n_select == 0: continue

        iso = IsolationForest(contamination=MADAR_CONTAMINATION, n_jobs=-1, random_state=SEED)
        iso.fit(L_fam)
        scores = iso.decision_function(L_fam)
        sorted_idx = np.argsort(scores)

        half = n_select // 2
        anomalies_idx = sorted_idx[:half]
        inliers_idx = sorted_idx[-(n_select - half):]

        interleaved_idx = [idx for pair in zip(anomalies_idx, inliers_idx) for idx in pair]
        if n_select % 2 != 0: interleaved_idx.append(inliers_idx[-1])

        family_buffers[fam] = [(torch.tensor(X_fam[i]), torch.tensor(Y_fam[i])) for i in interleaved_idx]

    replay_buffer.clear()
    for fam_data in family_buffers.values(): replay_buffer.extend(fam_data)

def train_cl_er(model, teacher_model, optimizer, loader, iters, active_count, prev_active_count, W, omega, p_old_task, tid, si_c):
    model.train()

    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.eval()

    mask = torch.full((model.fc_last.out_features,), -1e9).to(DEVICE)
    mask[:active_count] = 0.0

    loader_iter = iter(cycle(loader))

    
    assert replay_buffer, "train_cl_er requires a non-empty replay buffer"
    b_v = torch.stack([i[0] for i in replay_buffer])
    b_l = torch.tensor([i[1] for i in replay_buffer])
    buf_loader = data.DataLoader(data.TensorDataset(b_v, b_l), batch_size=BATCH_SIZE, shuffle=True)
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

        rnt = 1.0 / (tid + 1)
        outputs_mem = model(mem_v)
        with torch.no_grad():
            teacher_logits = teacher_model(mem_v)[:, :prev_active_count]

        loss_replay = loss_fn_kd(outputs_mem[:, :prev_active_count], teacher_logits, T=KD_TEMP)
        loss_main = (rnt * loss_cur) + ((1.0 - rnt) * loss_replay)

        si_loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                n_key = n.replace('.', '__')
                si_loss += (omega[n_key] * (p - p_old_task[n_key])**2).sum()

        total_loss = loss_main + (si_c * si_loss)
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

def eval_acc(model, X, y, families, active_count):
    model.eval()
    mask = torch.isin(y, torch.tensor(families))
    if not mask.any(): return 0.0
    v_subset, l_subset = X[mask].to(DEVICE), y[mask].to(DEVICE)
    eval_loader = data.DataLoader(data.TensorDataset(v_subset, l_subset), batch_size=BATCH_SIZE)
    correct = total = 0
    with torch.no_grad():
        for bv, bl in eval_loader:
            _, predicted = torch.max(model(bv)[:, :active_count], 1)
            total += bl.size(0); correct += (predicted == bl).sum().item()
    return (correct / total) * 100

# --- 4. Unlearning Modules ---

def split_option_a_scrub_leftovers(X, y, families, budget_per_family):
    """Per-family split in RAW FEATURE space: keep the MADAR-style selection
    (extreme anomalies + strongest inliers), forget everything else.
    NOTE: forgets a potentially large fraction of the task (everything not
    selected), unlike Option B's fixed ratio. Document this asymmetry."""
    mask = torch.isin(y, torch.tensor(families))
    if not mask.any(): return None, None

    X_subset, y_subset = X[mask], y[mask]

    retain_idx_all = []
    forget_idx_all = []

    X_np = X_subset.numpy()
    y_np = y_subset.numpy()

    for fam in families:
        fam_mask = (y_np == fam)
        if not fam_mask.any(): continue

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
        retain_loader = data.DataLoader(data.TensorDataset(X_subset, y_subset), batch_size=BATCH_SIZE, shuffle=True)
        return retain_loader, None

    retain_loader = data.DataLoader(data.TensorDataset(X_subset[retain_idx_all], y_subset[retain_idx_all]), batch_size=BATCH_SIZE, shuffle=True)
    forget_loader = data.DataLoader(data.TensorDataset(X_subset[forget_idx_all], y_subset[forget_idx_all]), batch_size=BATCH_SIZE, shuffle=True)

    return retain_loader, forget_loader

def split_option_b_donut_hole(X, y, families, model, forget_ratio=0.1):
    """Joint split in LATENT space: forget the middle band of the IF score
    distribution ('moderately typical' samples), keep strong inliers and
    anomalies. Fit is joint over all current-task families, so the forgotten
    band can be family-asymmetric — worth logging per-family forget counts."""
    model.eval()
    mask = torch.isin(y, torch.tensor(families))
    if not mask.any(): return None, None

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
        retain_loader = data.DataLoader(data.TensorDataset(X_subset, y_subset), batch_size=BATCH_SIZE, shuffle=True)
        return retain_loader, None

    mid_start = (n_total // 2) - (n_forget // 2)
    mid_end = mid_start + n_forget

    forget_idx = sorted_idx[mid_start:mid_end]
    retain_idx = np.concatenate((sorted_idx[:mid_start], sorted_idx[mid_end:]))

    # Log per-family forget counts so family asymmetry of the joint fit is visible
    forget_fams, forget_counts = np.unique(y_subset[forget_idx].numpy(), return_counts=True)
    print(f"    [Split B] Forget set per family: {dict(zip(forget_fams.tolist(), forget_counts.tolist()))}")

    forget_loader = data.DataLoader(data.TensorDataset(X_subset[forget_idx], y_subset[forget_idx]), batch_size=BATCH_SIZE, shuffle=True)
    retain_loader = data.DataLoader(data.TensorDataset(X_subset[retain_idx], y_subset[retain_idx]), batch_size=BATCH_SIZE, shuffle=True)

    return retain_loader, forget_loader

def unlearn_teacher_guided(model, teacher_model, forget_loader, retain_loader, active_count, prev_active_count, omega, p_old_task, si_c, epochs=3, lr=1e-4, alpha=0.2):
    """
    Flattens the forget set's current-task predictions toward uniform,
    anchored by ground truth CE + KD on retain/replay data and an SI penalty
    protecting past-task-important parameters.
    With alpha=0 this becomes the extra-training CONTROL: identical steps,
    optimizer and data flow, but no forget objective.
    """
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
        b_l = torch.tensor([i[1] for i in replay_buffer])
        buf_loader = data.DataLoader(data.TensorDataset(b_v, b_l), batch_size=BATCH_SIZE, shuffle=True)
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
            # NOTE: omega here already includes the just-finished task's
            # contribution (updated before unlearning), and p_old_task is the
            # start-of-task anchor. So unlearning is penalized for drifting
            # important parameters away from where the task began, except
            # through the forget objective. Intentional;
            si_loss = 0
            for n, p in model.named_parameters():
                if p.requires_grad:
                    n_key = n.replace('.', '__')
                    if n_key in p_old_task:
                        si_loss += (omega[n_key] * (p - p_old_task[n_key])**2).sum()

            # --- 4. COMBINED UPDATE ---
            total_loss = (alpha * forget_loss) + ((1.0 - alpha) * retain_loss) + (si_c * si_loss)
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

def measure_unlearning_efficacy(model_before, model_after, loader, active_count):
   
    model_before.eval()
    model_after.eval()

    acc_before = 0.0
    acc_after = 0.0
    entropies = []
    shifts = []
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

def plot_global_misclassifications_scatter(model, X_test, y_test, seen_fams, active_count, tid):
    """Jittered scatter of True vs. Predicted labels for misclassified test samples."""
    model.eval()
    mask = torch.isin(y_test, torch.tensor(seen_fams))
    if not mask.any(): return

    v_subset, l_subset = X_test[mask].to(DEVICE), y_test[mask].to(DEVICE)
    eval_loader = data.DataLoader(data.TensorDataset(v_subset, l_subset), batch_size=BATCH_SIZE)

    all_true = []
    all_preds = []

    with torch.no_grad():
        for bv, bl in eval_loader:
            _, predicted = torch.max(model(bv)[:, :active_count], 1)
            all_true.extend(bl.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    all_true = np.array(all_true)
    all_preds = np.array(all_preds)

    wrong_idx = all_true != all_preds
    wrong_true = all_true[wrong_idx]
    wrong_preds = all_preds[wrong_idx]

    if len(wrong_true) == 0:
        print(f"    [Visualizer] No misclassifications on global test set for Task {tid}.")
        return

    plt.figure(figsize=(10, 10))

    jitter_x = wrong_true + np.random.uniform(-0.35, 0.35, size=len(wrong_true))
    jitter_y = wrong_preds + np.random.uniform(-0.35, 0.35, size=len(wrong_preds))

    plt.scatter(jitter_x, jitter_y, alpha=0.3, s=15, c='purple', marker='o', edgecolors='none')
    plt.plot([0, active_count-1], [0, active_count-1], color='gray', linestyle='-', alpha=0.3, label='Correct Prediction Line')
    plt.axvline(x=TASK_1_FAMILIES - 0.5, color='green', linestyle='--', alpha=0.8, label='Task 0 Boundary')
    plt.axhline(y=TASK_1_FAMILIES - 0.5, color='green', linestyle='--', alpha=0.8)

    for i in range(1, tid + 1):
        bound = TASK_1_FAMILIES + (i * SUBSEQUENT_FAMILIES) - 0.5
        plt.axvline(x=bound, color='orange', linestyle=':', alpha=0.5)
        plt.axhline(y=bound, color='orange', linestyle=':', alpha=0.5)

    plt.title(f'Global Misclassifications After Task {tid}\n(Evaluating all {active_count} seen families)')
    plt.xlabel('True Family ID (What it actually is)')
    plt.ylabel('Predicted Family ID (What the model guessed)')
    plt.xlim(-1, active_count)
    plt.ylim(-1, active_count)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/scatter/global_scatter_{RUN_TAG}_T{tid}.png', dpi=150)
    plt.close()

# --- 5. Main Execution ---
if __name__ == "__main__":
    X_train_np, y_train_np, X_test_np, y_test_np = load_ember_data(DATA_DIR, EMBERSIM_DIR)

    y_train_tensor = torch.tensor(y_train_np, dtype=torch.long)
    unique, counts = torch.unique(y_train_tensor, return_counts=True)

    counts_dict = dict(zip(unique.tolist(), counts.tolist()))
    eligible = sorted([f for f, c in counts_dict.items() if c >= 200],
                      key=lambda f: counts_dict[f], reverse=True)[:100]
    id_map = {old_id: new_id for new_id, old_id in enumerate(eligible)}

    def remap_and_filter(X_np, y_np):
        y_tensor = torch.tensor([id_map.get(int(i), -1) for i in y_np], dtype=torch.long)
        mask = y_tensor != -1
        return X_np[mask], y_tensor[mask]

    X_train_np, y_train = remap_and_filter(X_train_np, y_train_np)
    X_test_np, y_test = remap_and_filter(X_test_np, y_test_np)

    all_fam_list = list(range(100))
    task_families = [all_fam_list[:TASK_1_FAMILIES]] + [all_fam_list[TASK_1_FAMILIES + (i*SUBSEQUENT_FAMILIES):TASK_1_FAMILIES + ((i+1)*SUBSEQUENT_FAMILIES)] for i in range(TOTAL_TASKS - 1)]


    scaler = StandardScaler()
    task0_mask = torch.isin(y_train, torch.tensor(task_families[0])).numpy()
    scaler.fit(X_train_np[task0_mask])

   
    X_train_scaled = torch.tensor(np.clip(scaler.transform(X_train_np), -FEATURE_CLIP, FEATURE_CLIP), dtype=torch.float32)
    X_test_scaled = torch.tensor(np.clip(scaler.transform(X_test_np), -FEATURE_CLIP, FEATURE_CLIP), dtype=torch.float32)

    model = EmberNN(INPUT_DIM, 100, dims=ARCH_DIMS).to(DEVICE)
    print(f"[arch study] {args.arch} {ARCH_DIMS} | "
          f"{count_params(INPUT_DIM, 100, ARCH_DIMS):,} params | si_c {SI_C} | out {OUT_DIR}")
    W = {n.replace('.','__'): torch.zeros_like(p).to(DEVICE) for n, p in model.named_parameters() if p.requires_grad}
    p_old_task = {n.replace('.','__'): p.detach().clone().to(DEVICE) for n, p in model.named_parameters() if p.requires_grad}
    omega = {n.replace('.','__'): torch.zeros_like(p).to(DEVICE) for n, p in model.named_parameters() if p.requires_grad}

    teacher_model = None
    prev_active_count = 0
    history = {'Avg_Acc': [], 'Task0_Acc': [], 'Train_Samples': [], 'Grad_Steps': [],
               'Forget_Acc_Before': [], 'Forget_Acc_After': [], 'Retain_Acc_Before': [], 'Retain_Acc_After': [],
               'SI_Penalty': []}

    for tid, current_fams in enumerate(task_families):
        print(f"\n=== Training Task {tid} ({len(current_fams)} Families) ===")
        active_count = TASK_1_FAMILIES + (tid * SUBSEQUENT_FAMILIES)
        seen_fams = [f for sublist in task_families[:tid+1] for f in sublist]
        assert active_count == len(seen_fams), "active_count / task split mismatch"

        loader = get_loader(X_train_scaled, y_train, current_fams, drop_last=True)
        if not loader: continue

        n_samples = len(loader.dataset)

        if tid == 0:
            # --- PURE BASE LEARNING ---
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
                        raise RuntimeError(f"Non-finite loss at task {tid}, epoch {epoch} - training diverged")
                    loss.backward()
                    optimizer.step()

            history['Forget_Acc_Before'].append(None); history['Forget_Acc_After'].append(None)
            history['Retain_Acc_Before'].append(None); history['Retain_Acc_After'].append(None)
            history['SI_Penalty'].append(None)
        else:
            # --- 1. CONTINUAL LEARNING ---
            grad_steps = CL_ITERS
            print(f"  Samples: {n_samples} (+{len(replay_buffer)} replay) | Gradient steps: {grad_steps}")
            optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-6)
            train_cl_er(model, teacher_model, optimizer, loader, CL_ITERS, active_count, prev_active_count, W, omega, p_old_task, tid, si_c=SI_C)


           
            with torch.no_grad():
                si_val = sum((omega[n.replace('.', '__')] *
                              (p - p_old_task[n.replace('.', '__')])**2).sum()
                             for n, p in model.named_parameters() if p.requires_grad)
            history['SI_Penalty'].append(float(si_val))

            # --- 2. SI OMEGA UPDATE (FIX: moved to IMMEDIATELY after CL training) ---
           
            for n, p in model.named_parameters():
                if p.requires_grad:
                    n_key = n.replace('.', '__')
                    p_post_cl = p.detach().clone()
                    omega[n_key] += W[n_key] / ((p_post_cl - p_old_task[n_key])**2 + SI_EPS)
                    W[n_key].zero_()

            # --- 3. UNLEARNING PHASE ---
            print(f" -> Initiating Unlearning Phase for Task {tid} (option {SPLIT_OPTION.upper()}, alpha={UNLEARN_ALPHA})...")

            if SPLIT_OPTION == 'a':
                budget_per_family = MEM_SIZE // (len(family_buffers) + len(current_fams))
                retain_loader, forget_loader = split_option_a_scrub_leftovers(
                    X_train_scaled, y_train, current_fams, budget_per_family
                )
            else:
                retain_loader, forget_loader = split_option_b_donut_hole(
                    X_train_scaled, y_train, current_fams, model, forget_ratio=UNLEARN_RATIO
                )

            if forget_loader and retain_loader:
                model_pre_unlearn = copy.deepcopy(model)

                unlearn_teacher_guided(
                    model=model,
                    teacher_model=model_pre_unlearn,
                    forget_loader=forget_loader,
                    retain_loader=retain_loader,
                    active_count=active_count,
                    prev_active_count=prev_active_count,
                    omega=omega,
                    p_old_task=p_old_task,
                    si_c=SI_C,
                    epochs=UNLEARN_EPOCHS,
                    lr=UNLEARN_LR,
                    alpha=UNLEARN_ALPHA
                )

                # Account the extra optimization honestly: forget-loader
                # batches drive the step count.
                grad_steps += UNLEARN_EPOCHS * len(forget_loader)

                f_m = measure_unlearning_efficacy(model_pre_unlearn, model, forget_loader, active_count)
                r_m = measure_unlearning_efficacy(model_pre_unlearn, model, retain_loader, active_count)

                for tag, m in (('Forget Set', f_m), ('Retain Set', r_m)):
                    print(f"    [{tag}] Acc: {m['acc_before']:.2f}% -> {m['acc_after']:.2f}%, "
                          f"Entropy: {m['entropy']:.3f} (norm: {m['entropy_norm']:.3f}), "
                          f"Latent Shift mean/med/max: {m['shift_mean']:.3f}/{m['shift_median']:.3f}/{m['shift_max']:.1f}")

                history['Forget_Acc_Before'].append(f_m['acc_before']); history['Forget_Acc_After'].append(f_m['acc_after'])
                history['Retain_Acc_Before'].append(r_m['acc_before']); history['Retain_Acc_After'].append(r_m['acc_after'])

                loader = retain_loader
            else:
                history['Forget_Acc_Before'].append(None); history['Forget_Acc_After'].append(None)
                history['Retain_Acc_Before'].append(None); history['Retain_Acc_After'].append(None)

        # --- 4. POST-TASK UPDATES (all tasks) ---
        for n, p in model.named_parameters():
            if p.requires_grad: p_old_task[n.replace('.', '__')] = p.detach().clone()
        teacher_model = copy.deepcopy(model)
        teacher_model.eval()
        prev_active_count = active_count

        # Buffer strictly picks from the post-unlearning retain data
        # (loader was swapped to retain_loader above when unlearning ran).
        update_buffer_madar(loader, model)

        avg_acc = eval_acc(model, X_test_scaled, y_test, seen_fams, active_count)
        t0_acc = eval_acc(model, X_test_scaled, y_test, task_families[0], active_count)
        history['Avg_Acc'].append(avg_acc); history['Task0_Acc'].append(t0_acc)
        history['Train_Samples'].append(n_samples); history['Grad_Steps'].append(grad_steps)
        print(f"  -> Avg: {avg_acc:.2f}% | Task 0: {t0_acc:.2f}%")

        plot_global_misclassifications_scatter(model, X_test_scaled, y_test, seen_fams, active_count, tid)


    # Self-describing output: every run records the configuration that produced
    # it, so the compiler never has to infer anything from a filename.
    history['config'] = {
        'condition': 'mu', 'arch': args.arch, 'arch_dims': list(ARCH_DIMS),
        'n_params': count_params(INPUT_DIM, 100, ARCH_DIMS), 'seed': SEED,
        'input_dim': INPUT_DIM, 'num_classes': 100, 'si_c': SI_C, 'alpha': UNLEARN_ALPHA,
        'unlearn_ratio': UNLEARN_RATIO, 'split_option': SPLIT_OPTION,
    }
    out_json = f'{OUT_DIR}/mu_{RUN_TAG}.json'
    with open(out_json, 'w') as f:
        json.dump(history, f, indent=2)
    print(f'  wrote {out_json}')
    plt.figure(figsize=(10, 6))
    plt.plot(history['Avg_Acc'], marker='o', label='Avg (seen families)')
    plt.plot(history['Task0_Acc'], marker='x', label='Task 0')
    plt.xlabel('Task'); plt.ylabel('Accuracy (%)')
    plt.title(f'MADAR + Unlearning - arch {args.arch} ({RUN_TAG})')
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(f'{OUT_DIR}/mu_{RUN_TAG}.png')