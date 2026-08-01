import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import thrember
import os
import json
import random
import argparse

# --- Parse Command Line Arguments ---
parser = argparse.ArgumentParser(description="EMBER 2024 Continual Learning Experiment (MADAR-IF + ER/KD/SI + Unlearning)")
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--family-cap', type=int, default=10000,
                    help='Max TRAINING samples per family (0 = no cap). Default 10000 keeps '
                         '~44%% of the data (task 0 ~454K samples), compressing the 94:1 family '
                         'imbalance to roughly 1.7:1 while staying comparable to EMBER 2018 in '
                         'compute. MUST be identical across all four conditions.')
parser.add_argument('--cap-seed', type=int, default=12345,
                    help='Seed for the capped subsample. Deliberately INDEPENDENT of --seed so '
                         'every condition and every run trains on the identical subset.')
parser.add_argument('--split_option', type=str, default='b', choices=['a', 'b'],
                    help="Forget-set selection. 'b' (default): donut hole - joint latent-space "
                         "Isolation Forest, forget the middle band of the score distribution. "
                         "'a': per-family raw-space scrub - forget everything the MADAR-style "
                         "selection would not keep (much larger, budget-dependent forget sets).")
parser.add_argument('--unlearn_ratio', type=float, default=0.10,
                    help='Fraction of current-task samples to forget (option b). 2018 mainline '
                         'used 0.10; the legacy 2024 script used 0.05.')
parser.add_argument('--unlearn_epochs', type=int, default=3,
                    help='Epochs over the forget set during the unlearning phase.')
parser.add_argument('--unlearn_lr', type=float, default=1e-4, help='Adam lr for unlearning.')
parser.add_argument('--alpha', type=float, default=0.2,
                    help='Weight of the forgetting objective. alpha=0 is the extra-training '
                         'CONTROL: identical steps and data flow, no forget term. 2018 mainline '
                         'used 0.2; the legacy 2024 script used 0.1.')
parser.add_argument('--weak-multiplier', type=float, default=1.0,
                    help='LEGACY targeted-replay option: replay-budget multiplier for '
                         '--weak-families. Default 1.0 = OFF, giving the same uniform budgeting '
                         'as the MADAR baseline. Enabling this confounds the MADAR-vs-MADAR+U '
                         'comparison (it changes the replay policy, not just unlearning), so use '
                         'only for dedicated targeted-replay experiments.')
parser.add_argument('--weak-families', type=int, nargs='+', default=[9, 14, 26, 43],
                    help='Families receiving --weak-multiplier x replay budget (post-remap ids).')
parser.add_argument('--task0-checkpoint', type=str, default=None,
                    help='Path to cache the trained task-0 model + replay buffer. If the file '
                         'exists it is loaded instead of retraining task 0 (~53k gradient steps), '
                         'making CL-phase ablations much faster. The checkpoint records arch / cap '
                         '/ cap-seed / seed and refuses to load under a mismatch.')
parser.add_argument('--cl-iters', type=int, default=2000,
                    help='CL iterations per incremental task. NOTE: this is a FIXED budget, so '
                         'larger incremental tasks get less exposure per sample. EMBER 2024 tasks '
                         'are ~4x larger than 2018 (27k vs 7k samples), so 2000 iters gives new '
                         'families ~4x fewer effective epochs - raise this to test plasticity.')
parser.add_argument('--si-c', type=float, default=2.0,
                    help='Synaptic Intelligence penalty strength (2018 used 1.0). Higher = more '
                         'stability, less plasticity on new families.')
parser.add_argument('--arch', type=str, default='mlp', choices=['mlp', 'resnet'],
                    help="Backbone. 'mlp' matches EMBER 2018 and the 2024 naive/joint baselines "
                         "(use this for the main results). 'resnet' is the deeper residual variant "
                         "- only valid if ALL four conditions are re-run with it.")
args = parser.parse_args()

SEED = args.seed

# --- Enforce Determinism (identical block in all four conditions) ---
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
DATA_DIR = "./Datasets/ember2024"

INPUT_DIM = 2568           # EMBER 2024 feature dimensionality
FEATURE_CLIP = 10.0        # clip standardized features to [-10, 10]
TASK0_EPOCHS = 30
CL_ITERS = args.cl_iters   # CLI-configurable (see --cl-iters)
BATCH_SIZE = 256

# NOTE: 2024 uses MEM_SIZE 10000 (2018 used 5000). With 100 families that is
# 100 replay slots/family vs 2018's 50 - proportional to 2024 having roughly
# twice the capped training data. Document this difference in the writeup.
MEM_SIZE = 10000
# NOTE: contamination only shifts IsolationForest's decision threshold; the
# selection below is purely rank-based on decision_function scores, so this
# value does not affect which samples are chosen. Kept for API compatibility.
MADAR_CONTAMINATION = 0.1

TASK_1_FAMILIES = 50
SUBSEQUENT_FAMILIES = 5
TOTAL_TASKS = 11
MIN_FAMILY_SAMPLES = 200
NUM_CLASSES = TASK_1_FAMILIES + (TOTAL_TASKS - 1) * SUBSEQUENT_FAMILIES   # = 100

KD_TEMP = 2.0
SI_C = args.si_c           # CLI-configurable; default 2.0 (2018 used 1.0)
SI_EPS = 0.1

SPLIT_OPTION = args.split_option
UNLEARN_RATIO = args.unlearn_ratio
UNLEARN_EPOCHS = args.unlearn_epochs
UNLEARN_LR = args.unlearn_lr
UNLEARN_ALPHA = args.alpha
WEAK_FAMILIES = set(args.weak_families)
WEAK_MULTIPLIER = args.weak_multiplier

# Families tracked individually (post-remap ids = frequency ranks).
TARGET_FAMS = [6, 9, 10, 14, 26, 43]
# ==========================================

# --- 1. Data Loading ---
def load_ember_data(data_dir):
    """thrember exposes family labels directly, so no hash->AVClass join is
    needed (unlike the EMBER 2018 pipeline)."""
    if not os.path.exists(os.path.join(data_dir, "X_train.dat")):
        thrember.create_vectorized_features(data_dir, label_type="family")
    X_train_np, y_train_np = thrember.read_vectorized_features(data_dir, subset="train")
    X_test_np, y_test_np = thrember.read_vectorized_features(data_dir, subset="test")
    return X_train_np, y_train_np, X_test_np, y_test_np

def cap_per_family(X_np, y_tensor, cap, cap_seed):
    """Randomly subsample each family down to `cap` TRAINING samples.
    Uses a FIXED seed so all conditions/seeds see exactly the same data."""
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

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

# --- 2. Model Definitions ---
class EmberNN(nn.Module):
    """Plain MLP backbone - identical to EMBER 2018 and the 2024 naive/joint
    baselines. 128-d penultimate layer is the latent space used by MADAR's
    Isolation Forests."""
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

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.gelu(self.norm1(self.linear1(x)))
        out = self.dropout(out)
        out = self.norm2(self.linear2(out))
        out = out + residual
        return self.gelu(out)

class EmberResNet(nn.Module):
    """Deeper residual backbone (LayerNorm/GELU/Dropout, 256-d latent).
    ONLY use if every condition is re-run with --arch resnet."""
    def __init__(self, input_dim, num_classes, hidden_dim=1024, latent_dim=256):
        super(EmberResNet, self).__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.norm_proj = nn.LayerNorm(hidden_dim)
        self.gelu = nn.GELU()
        self.res1 = ResidualBlock(hidden_dim)
        self.res2 = ResidualBlock(hidden_dim)
        self.res3 = ResidualBlock(hidden_dim)
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)
        self.norm_latent = nn.LayerNorm(latent_dim)
        self.fc_last = nn.Linear(latent_dim, num_classes)

    def forward(self, x, return_latent=False):
        x = self.gelu(self.norm_proj(self.proj(x)))
        x = self.res1(x); x = self.res2(x); x = self.res3(x)
        latent = self.gelu(self.norm_latent(self.fc_latent(x)))
        logits = self.fc_last(latent)
        return (logits, latent) if return_latent else logits

def build_model():
    if args.arch == 'mlp':
        return EmberNN(INPUT_DIM, NUM_CLASSES).to(DEVICE)
    return EmberResNet(INPUT_DIM, NUM_CLASSES).to(DEVICE)

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

    
    all_tracked = set(family_buffers.keys()) | set(int(f) for f in current_families)
    def fam_weight(f):
        return WEAK_MULTIPLIER if f in WEAK_FAMILIES else 1.0
    total_weight = sum(fam_weight(f) for f in all_tracked)
    base_budget = int(MEM_SIZE / max(1.0, total_weight))
    def fam_budget(f):
        return int(base_budget * fam_weight(f))

    for fam in family_buffers:
        budget_f = fam_budget(fam)
        if len(family_buffers[fam]) > budget_f:
            current_buffer = family_buffers[fam]
            half_budget = budget_f // 2
            anomalies = current_buffer[::2]
            inliers = current_buffer[1::2]
            new_buffer = [val for pair in zip(anomalies[:half_budget], inliers[:half_budget]) for val in pair]
            if budget_f % 2 != 0 and len(anomalies) > half_budget:
                new_buffer.append(anomalies[half_budget])
            family_buffers[fam] = new_buffer

    for fam in current_families:
        fam_mask = (Y_np == fam)
        X_fam, Y_fam, L_fam = X_np[fam_mask], Y_np[fam_mask], L_np[fam_mask]
        n_select = min(fam_budget(int(fam)), len(X_fam))
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

        family_buffers[fam] = [(torch.tensor(X_fam[i]), torch.tensor(int(Y_fam[i]))) for i in interleaved_idx]

    replay_buffer.clear()
    for fam_data in family_buffers.values(): replay_buffer.extend(fam_data)

def train_cl_er(model, teacher_model, optimizer, loader, iters, active_count,
                prev_active_count, W, omega, p_old_task, tid, si_c):
    model.train()

    
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.eval()

    mask = torch.full((model.fc_last.out_features,), -1e9).to(DEVICE)
    mask[:active_count] = 0.0

    loader_iter = iter(cycle(loader))

    
    si_diag = {'steps': [], 'loss_main': [], 'si_term': []}
    probe_steps = {0, iters // 4, iters // 2, (3 * iters) // 4, max(iters - 1, 0)}

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
        if step in probe_steps:
            
            _lm = loss_main.detach() if torch.is_tensor(loss_main) else loss_main
            _st = si_c * si_loss
            _st = _st.detach() if torch.is_tensor(_st) else _st
            si_diag['steps'].append(step)
            si_diag['loss_main'].append(float(_lm))
            si_diag['si_term'].append(float(_st))
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

    return si_diag

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
            if not torch.isfinite(total_loss):
                raise RuntimeError("Non-finite loss during unlearning - diverged")
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
    plt.savefig(f'global_scatter_{RUN_TAG}_T{tid}.png', dpi=150)
    plt.close()

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

def eval_acc_macro(model, X, y, families, active_count):
    """Mean per-family recall (macro accuracy). The EMBER 2024 TEST set keeps
    its natural 94:1 imbalance, so sample-weighted (micro) accuracy is
    dominated by a few large families."""
    model.eval()
    mask = torch.isin(y, torch.tensor(families))
    if not mask.any(): return 0.0
    v_subset, l_subset = X[mask].to(DEVICE), y[mask].to(DEVICE)
    eval_loader = data.DataLoader(data.TensorDataset(v_subset, l_subset), batch_size=BATCH_SIZE)
    correct = torch.zeros(active_count)
    total = torch.zeros(active_count)
    with torch.no_grad():
        for bv, bl in eval_loader:
            _, predicted = torch.max(model(bv)[:, :active_count], 1)
            bl_cpu = bl.cpu()
            total += torch.bincount(bl_cpu, minlength=active_count).float()
            correct += torch.bincount(bl_cpu[(predicted == bl).cpu()], minlength=active_count).float()
    present = total > 0
    if not present.any(): return 0.0
    return (correct[present] / total[present]).mean().item() * 100

def eval_family_metrics(model, X_test, y_test, target_families, active_count):
    """Per-family recall ('acc') and false-positive rate over all seen families."""
    model.eval()
    seen_mask = y_test < active_count
    if not seen_mask.any():
        return {fam: {'acc': 0.0, 'fpr': 0.0} for fam in target_families}

    v_subset, l_subset = X_test[seen_mask].to(DEVICE), y_test[seen_mask].to(DEVICE)
    eval_loader = data.DataLoader(data.TensorDataset(v_subset, l_subset), batch_size=BATCH_SIZE)
    counts = {fam: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0} for fam in target_families}

    with torch.no_grad():
        for bv, bl in eval_loader:
            _, predicted = torch.max(model(bv)[:, :active_count], 1)
            for fam in target_families:
                counts[fam]['tp'] += ((predicted == fam) & (bl == fam)).sum().item()
                counts[fam]['fp'] += ((predicted == fam) & (bl != fam)).sum().item()
                counts[fam]['fn'] += ((predicted != fam) & (bl == fam)).sum().item()
                counts[fam]['tn'] += ((predicted != fam) & (bl != fam)).sum().item()

    results = {}
    for fam in target_families:
        tp, fp, fn, tn = counts[fam]['tp'], counts[fam]['fp'], counts[fam]['fn'], counts[fam]['tn']
        results[fam] = {
            'acc': (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0,
            'fpr': (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0.0
        }
    return results

# --- 4. Main Execution ---
if __name__ == "__main__":
    X_train_np, y_train_np, X_test_np, y_test_np = load_ember_data(DATA_DIR)

    
    y_train_all = np.asarray(y_train_np)
    n_neg = int((y_train_all < 0).sum())
    if n_neg:
        print(f"NOTE: dropping {n_neg} training samples with negative (sentinel) labels")
    keep = y_train_all >= 0
    X_train_np, y_train_np = X_train_np[keep], y_train_all[keep]
    y_test_all = np.asarray(y_test_np)
    keep_te = y_test_all >= 0
    X_test_np, y_test_np = X_test_np[keep_te], y_test_all[keep_te]

    y_train_tensor = torch.tensor(y_train_np, dtype=torch.long)
    unique, counts = torch.unique(y_train_tensor, return_counts=True)

    
    counts_dict = dict(zip(unique.tolist(), counts.tolist()))
    eligible_all = sorted([f for f, c in counts_dict.items() if c >= MIN_FAMILY_SAMPLES],
                          key=lambda f: counts_dict[f], reverse=True)
    assert len(eligible_all) >= NUM_CLASSES, (
        f"Only {len(eligible_all)} families have >= {MIN_FAMILY_SAMPLES} samples, "
        f"but the task schedule needs {NUM_CLASSES}.")
    eligible = eligible_all[:NUM_CLASSES]
    id_map = {old_id: new_id for new_id, old_id in enumerate(eligible)}

    sel_counts = [counts_dict[f] for f in eligible]
    print(f"Families with >= {MIN_FAMILY_SAMPLES} samples: {len(eligible_all)} "
          f"(using top {NUM_CLASSES})")
    print(f"  selected family sizes: max {max(sel_counts)}, min {min(sel_counts)}, "
          f"total {sum(sel_counts)} samples")

    def remap_and_filter(X_np, y_np):
        y_tensor = torch.tensor([id_map.get(int(i), -1) for i in y_np], dtype=torch.long)
        mask = y_tensor != -1
        return X_np[mask], y_tensor[mask]

    X_train_np, y_train = remap_and_filter(X_train_np, y_train_np)
    X_test_np, y_test = remap_and_filter(X_test_np, y_test_np)

    # FIX: cap TRAINING samples per family (test set left untouched).
    n_before = len(y_train)
    X_train_np, y_train = cap_per_family(X_train_np, y_train, args.family_cap, args.cap_seed)
    if args.family_cap:
        print(f"Family cap {args.family_cap}: {n_before:,} -> {len(y_train):,} training samples "
              f"(cap seed {args.cap_seed}, test set unchanged at {len(y_test):,})")
    print(f"Backbone: {args.arch}  |  MEM_SIZE {MEM_SIZE}  |  SI_C {SI_C}  |  CL_ITERS {CL_ITERS}")

    all_fam_list = list(range(NUM_CLASSES))
    task_families = [all_fam_list[:TASK_1_FAMILIES]] + [
        all_fam_list[TASK_1_FAMILIES + (i * SUBSEQUENT_FAMILIES):
                     TASK_1_FAMILIES + ((i + 1) * SUBSEQUENT_FAMILIES)]
        for i in range(TOTAL_TASKS - 1)]

    
    scaler = StandardScaler()
    task0_mask = torch.isin(y_train, torch.tensor(task_families[0])).numpy()
    scaler.fit(X_train_np[task0_mask])

    
    X_train_scaled = torch.tensor(np.clip(scaler.transform(X_train_np), -FEATURE_CLIP, FEATURE_CLIP), dtype=torch.float32)
    X_test_scaled = torch.tensor(np.clip(scaler.transform(X_test_np), -FEATURE_CLIP, FEATURE_CLIP), dtype=torch.float32)

    model = build_model()
    W = {n.replace('.', '__'): torch.zeros_like(p).to(DEVICE) for n, p in model.named_parameters() if p.requires_grad}
    p_old_task = {n.replace('.', '__'): p.detach().clone().to(DEVICE) for n, p in model.named_parameters() if p.requires_grad}
    omega = {n.replace('.', '__'): torch.zeros_like(p).to(DEVICE) for n, p in model.named_parameters() if p.requires_grad}

    teacher_model = None
    prev_active_count = 0
    history = {'Avg_Acc': [], 'Task0_Acc': [], 'Avg_Acc_Macro': [],
               'New_Fam_Acc': [], 'New_Fam_Acc_Macro': [], 'SI_Diag': [],
               'Train_Samples': [], 'Grad_Steps': [],
               'config': {'family_cap': args.family_cap, 'cap_seed': args.cap_seed,
                          'seed': SEED, 'arch': args.arch, 'mem_size': MEM_SIZE,
                          'si_c': SI_C, 'cl_iters': CL_ITERS,
                          'split_option': SPLIT_OPTION, 'alpha': UNLEARN_ALPHA,
                          'unlearn_ratio': UNLEARN_RATIO, 'unlearn_epochs': UNLEARN_EPOCHS,
                          'weak_multiplier': WEAK_MULTIPLIER},
               'Unlearn_Efficacy': []}
    fam_history = {fam: {'acc': [], 'fpr': []} for fam in TARGET_FAMS}

    for tid, current_fams in enumerate(task_families):
        print(f"\n=== Training Task {tid} ({len(current_fams)} Families) ===")
        active_count = TASK_1_FAMILIES + (tid * SUBSEQUENT_FAMILIES)
        seen_fams = [f for sublist in task_families[:tid + 1] for f in sublist]
        assert active_count == len(seen_fams), "active_count / task split mismatch"

        loader = get_loader(X_train_scaled, y_train, current_fams, drop_last=True)
        if not loader: continue

        n_samples = len(loader.dataset)

        if tid == 0:
            # --- PURE BASE LEARNING: Adam, epochs, no SI, no KD ---
            grad_steps = TASK0_EPOCHS * len(loader)
            ckpt_key = {'arch': args.arch, 'family_cap': args.family_cap,
                        'cap_seed': args.cap_seed, 'seed': SEED}
            if args.task0_checkpoint and os.path.exists(args.task0_checkpoint):
                ck = torch.load(args.task0_checkpoint, map_location='cpu', weights_only=False)
                assert ck['key'] == ckpt_key, (
                    f"checkpoint was built with {ck['key']} but this run is {ckpt_key}")
                model.load_state_dict(ck['model'])
                family_buffers.update(ck['family_buffers'])
                replay_buffer.clear()
                for fd in family_buffers.values(): replay_buffer.extend(fd)
                for n, p in model.named_parameters():
                    if p.requires_grad: p_old_task[n.replace('.', '__')] = p.detach().clone()
                teacher_model = copy.deepcopy(model); teacher_model.eval()
                prev_active_count = active_count
                grad_steps = 0
                print(f"  Loaded task-0 checkpoint {args.task0_checkpoint} "
                      f"(skipped {TASK0_EPOCHS * len(loader):,} gradient steps)")
                avg_acc = eval_acc(model, X_test_scaled, y_test, seen_fams, active_count)
                t0_acc = avg_acc
                macro_acc = eval_acc_macro(model, X_test_scaled, y_test, seen_fams, active_count)
                new_acc, new_macro = avg_acc, macro_acc
                history['Avg_Acc'].append(avg_acc); history['Task0_Acc'].append(t0_acc)
                history['Avg_Acc_Macro'].append(macro_acc)
                history['New_Fam_Acc'].append(new_acc); history['New_Fam_Acc_Macro'].append(new_macro)
                history['Train_Samples'].append(n_samples); history['Grad_Steps'].append(grad_steps)
                print(f"  -> Avg: {avg_acc:.2f}% (macro {macro_acc:.2f}%) | Task 0: {t0_acc:.2f}%"
                      f" | New fams: {new_acc:.2f}% (macro {new_macro:.2f}%)")
                fam_metrics = eval_family_metrics(model, X_test_scaled, y_test, TARGET_FAMS, active_count)
                for fam in TARGET_FAMS:
                    fam_history[fam]['acc'].append(fam_metrics[fam]['acc'])
                    fam_history[fam]['fpr'].append(fam_metrics[fam]['fpr'])
                print("  Per-family recall: " +
                      " ".join(f"f{fam}:{fam_metrics[fam]['acc']:.1f}%" for fam in TARGET_FAMS))
                continue
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

            
            for n, p in model.named_parameters():
                if p.requires_grad: p_old_task[n.replace('.', '__')] = p.detach().clone()
            teacher_model = copy.deepcopy(model)
            teacher_model.eval()
            prev_active_count = active_count
            update_buffer_madar(loader, model)
            if args.task0_checkpoint:
                torch.save({'model': model.state_dict(),
                            'family_buffers': family_buffers,
                            'key': ckpt_key}, args.task0_checkpoint)
                print(f"  Saved task-0 checkpoint to {args.task0_checkpoint}")

        else:
            # --- CONTINUAL LEARNING: SGD, fixed iters, SI, KD ---
            grad_steps = CL_ITERS
            print(f"  Samples: {n_samples} (+{len(replay_buffer)} replay) | Gradient steps: {grad_steps}")
            optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-6)
            si_diag = train_cl_er(model, teacher_model, optimizer, loader, CL_ITERS, active_count,
                                  prev_active_count, W, omega, p_old_task, tid, si_c=SI_C)

            mean_main = float(np.mean(si_diag['loss_main']))
            mean_si = float(np.mean(si_diag['si_term']))
            share = mean_si / (mean_main + mean_si) if (mean_main + mean_si) > 0 else 0.0
            omega_flat = torch.cat([omega[k].flatten() for k in omega])
            om_mean, om_max = float(omega_flat.mean()), float(omega_flat.max())
            print(f"  [SI diag] loss_main {mean_main:.4f} | si_term {mean_si:.3e} "
                  f"| SI share {share*100:.4f}% | omega mean {om_mean:.3e} max {om_max:.3e}")
            history['SI_Diag'].append({'task': tid, 'loss_main': mean_main, 'si_term': mean_si,
                                       'si_share': share, 'omega_mean': om_mean, 'omega_max': om_max})

            for n, p in model.named_parameters():
                if p.requires_grad:
                    n_key = n.replace('.', '__')
                    p_current = p.detach().clone()
                    omega[n_key] += W[n_key] / ((p_current - p_old_task[n_key])**2 + SI_EPS)
                    W[n_key].zero_()

            # --- UNLEARNING PHASE ---
            print(f" -> Unlearning (option {SPLIT_OPTION.upper()}, alpha={UNLEARN_ALPHA}, "
                  f"ratio={UNLEARN_RATIO})...")
            if SPLIT_OPTION == 'a':
                budget_per_family = MEM_SIZE // (len(family_buffers) + len(current_fams))
                retain_loader, forget_loader = split_option_a_scrub_leftovers(
                    X_train_scaled, y_train, current_fams, budget_per_family)
            else:
                retain_loader, forget_loader = split_option_b_donut_hole(
                    X_train_scaled, y_train, current_fams, model, forget_ratio=UNLEARN_RATIO)

            if forget_loader and retain_loader:
                model_pre_unlearn = copy.deepcopy(model)
                unlearn_teacher_guided(
                    model=model, teacher_model=model_pre_unlearn,
                    forget_loader=forget_loader, retain_loader=retain_loader,
                    active_count=active_count, prev_active_count=prev_active_count,
                    omega=omega, p_old_task=p_old_task, si_c=SI_C,
                    epochs=UNLEARN_EPOCHS, lr=UNLEARN_LR, alpha=UNLEARN_ALPHA)

                f_m = measure_unlearning_efficacy(model_pre_unlearn, model, forget_loader, active_count)
                r_m = measure_unlearning_efficacy(model_pre_unlearn, model, retain_loader, active_count)
                print(f"    [Forget] acc {f_m['acc_before']:.2f}->{f_m['acc_after']:.2f}% | "
                      f"H_norm {f_m['entropy_norm']:.3f} | shift mean/med/max "
                      f"{f_m['shift_mean']:.2f}/{f_m['shift_median']:.2f}/{f_m['shift_max']:.2f}")
                print(f"    [Retain] acc {r_m['acc_before']:.2f}->{r_m['acc_after']:.2f}% | "
                      f"H_norm {r_m['entropy_norm']:.3f} | shift mean/med/max "
                      f"{r_m['shift_mean']:.2f}/{r_m['shift_median']:.2f}/{r_m['shift_max']:.2f}")
                history['Unlearn_Efficacy'].append({'task': tid, 'forget': f_m, 'retain': r_m})
                # Forgotten samples are excluded from the buffer forever:
                loader = retain_loader

            # advance the SI anchor only now, after unlearning
            for n, p in model.named_parameters():
                if p.requires_grad:
                    p_old_task[n.replace('.', '__')] = p.detach().clone()

            teacher_model = copy.deepcopy(model)
            teacher_model.eval()
            prev_active_count = active_count
            update_buffer_madar(loader, model)

        avg_acc = eval_acc(model, X_test_scaled, y_test, seen_fams, active_count)
        t0_acc = eval_acc(model, X_test_scaled, y_test, task_families[0], active_count)
        macro_acc = eval_acc_macro(model, X_test_scaled, y_test, seen_fams, active_count)
        new_acc = eval_acc(model, X_test_scaled, y_test, current_fams, active_count)
        new_macro = eval_acc_macro(model, X_test_scaled, y_test, current_fams, active_count)
        history['Avg_Acc'].append(avg_acc); history['Task0_Acc'].append(t0_acc)
        history['Avg_Acc_Macro'].append(macro_acc)
        history['New_Fam_Acc'].append(new_acc); history['New_Fam_Acc_Macro'].append(new_macro)
        history['Train_Samples'].append(n_samples); history['Grad_Steps'].append(grad_steps)
        print(f"  -> Avg: {avg_acc:.2f}% (macro {macro_acc:.2f}%) | Task 0: {t0_acc:.2f}%"
              f" | New fams: {new_acc:.2f}% (macro {new_macro:.2f}%)")

        fam_metrics = eval_family_metrics(model, X_test_scaled, y_test, TARGET_FAMS, active_count)
        for fam in TARGET_FAMS:
            fam_history[fam]['acc'].append(fam_metrics[fam]['acc'])
            fam_history[fam]['fpr'].append(fam_metrics[fam]['fpr'])
        print("  Per-family recall: " +
              " ".join(f"f{fam}:{fam_metrics[fam]['acc']:.1f}%" for fam in TARGET_FAMS))

    
    history['Family_Metrics'] = {str(f): fam_history[f] for f in TARGET_FAMS}
    tag = f"cap{args.family_cap}_{args.arch}_{SPLIT_OPTION}_a{UNLEARN_ALPHA}_r{UNLEARN_RATIO}_seed{SEED}"
    if args.cl_iters != 2000: tag += f"_it{args.cl_iters}"
    if args.si_c != 2.0: tag += f"_si{args.si_c}"
    if WEAK_MULTIPLIER != 1.0: tag += f"_wm{WEAK_MULTIPLIER}"
    with open(f'ember24_madaru_history_{tag}.json', 'w') as f:
        json.dump(history, f, indent=2)

    plt.figure(figsize=(10, 6))
    plt.plot(history['Avg_Acc'], marker='o', label='Avg (seen families, micro)')
    plt.plot(history['Avg_Acc_Macro'], marker='s', label='Avg (macro, per-family mean)')
    plt.plot(history['Task0_Acc'], marker='x', label='Task 0')
    plt.plot(history['New_Fam_Acc'], marker='^', label='Newest 5 families')
    plt.xlabel('Task'); plt.ylabel('Accuracy (%)')
    plt.title(f'EMBER 2024 MADAR-IF + Unlearning ({args.arch}, {SPLIT_OPTION}/a={UNLEARN_ALPHA}, seed {SEED})')
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(f'ember24_cl_madaru_{tag}.png')

    plt.figure(figsize=(10, 6))
    for fam in TARGET_FAMS:
        plt.plot(fam_history[fam]['acc'], marker='o', label=f'family {fam}')
    plt.xlabel('Task'); plt.ylabel('Per-family recall (%)')
    plt.title(f'EMBER 2024 MADAR+U: tracked family retention (seed {SEED})')
    plt.legend(fontsize=9); plt.grid(alpha=0.3)
    plt.savefig(f'ember24_madaru_families_{tag}.png')