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
import ember
import os
import random
import json
import csv
import argparse

# --- Parse Command Line Seed ---
parser = argparse.ArgumentParser(description="Continual Learning Experiment (MADAR-IF + ER/KD/SI)")
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
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
DATA_DIR = "./Datasets/ember2018"
EMBERSIM_DIR = "./Datasets/embersim-databank"

INPUT_DIM = 2381
FEATURE_CLIP = 10.0        # clip standardized features to [-10, 10]
TASK0_EPOCHS = 30
CL_ITERS = 2000
BATCH_SIZE = 256

MEM_SIZE = 5000
# NOTE: contamination only shifts IsolationForest's decision threshold; the
# selection below is purely rank-based on decision_function scores, so this
# value does not affect which samples are chosen. Kept for API compatibility.
MADAR_CONTAMINATION = 0.1

TASK_1_FAMILIES = 50
SUBSEQUENT_FAMILIES = 5
TOTAL_TASKS = 11           # 50 base + (10 increments * 5 families) = 100 families total

KD_TEMP = 2.0
SI_C = 1.0
SI_EPS = 0.1
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

def get_loader(X, y, families, drop_last=False):
    .
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

# --- 4. Main Execution ---
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

    model = EmberNN(INPUT_DIM, 100).to(DEVICE)
    W = {n.replace('.','__'): torch.zeros_like(p).to(DEVICE) for n, p in model.named_parameters() if p.requires_grad}
    p_old_task = {n.replace('.','__'): p.detach().clone().to(DEVICE) for n, p in model.named_parameters() if p.requires_grad}
    omega = {n.replace('.','__'): torch.zeros_like(p).to(DEVICE) for n, p in model.named_parameters() if p.requires_grad}

    teacher_model = None
    prev_active_count = 0
    history = {'Avg_Acc': [], 'Task0_Acc': [], 'Train_Samples': [], 'Grad_Steps': []}

    for tid, current_fams in enumerate(task_families):
        print(f"\n=== Training Task {tid} ({len(current_fams)} Families) ===")
        active_count = TASK_1_FAMILIES + (tid * SUBSEQUENT_FAMILIES)
        seen_fams = [f for sublist in task_families[:tid+1] for f in sublist]
        assert active_count == len(seen_fams), "active_count / task split mismatch"

        # drop_last=True matters for task 0 (BN in train mode); harmless for
        # CL tasks (BN frozen to eval). Buffer selection is unaffected since
        # update_buffer_madar re-iterates the full dataset.
        loader = get_loader(X_train_scaled, y_train, current_fams, drop_last=True)
        if not loader: continue

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
                        raise RuntimeError(f"Non-finite loss at task {tid}, epoch {epoch} - training diverged")
                    loss.backward()
                    optimizer.step()

            # Initialize the SI anchors and Replay Buffer only after Task 0 finishes
            for n, p in model.named_parameters():
                if p.requires_grad: p_old_task[n.replace('.', '__')] = p.detach().clone()
            teacher_model = copy.deepcopy(model)
            teacher_model.eval()
            prev_active_count = active_count
            update_buffer_madar(loader, model)

        else:
            # --- CONTINUAL LEARNING: SGD, Iters, SI, KD ---
            grad_steps = CL_ITERS
            print(f"  Samples: {n_samples} (+{len(replay_buffer)} replay) | Gradient steps: {grad_steps}")
            optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-6)
            train_cl_er(model, teacher_model, optimizer, loader, CL_ITERS, active_count, prev_active_count, W, omega, p_old_task, tid, si_c=SI_C)

            # SI Omega Update
            for n, p in model.named_parameters():
                if p.requires_grad:
                    n_key = n.replace('.', '__')
                    p_current = p.detach().clone()
                    omega[n_key] += W[n_key] / ((p_current - p_old_task[n_key])**2 + SI_EPS)
                    W[n_key].zero_()
                    p_old_task[n_key] = p_current

            teacher_model = copy.deepcopy(model)
            teacher_model.eval()
            prev_active_count = active_count
            update_buffer_madar(loader, model)

        avg_acc = eval_acc(model, X_test_scaled, y_test, seen_fams, active_count)
        t0_acc = eval_acc(model, X_test_scaled, y_test, task_families[0], active_count)
        history['Avg_Acc'].append(avg_acc); history['Task0_Acc'].append(t0_acc)
        history['Train_Samples'].append(n_samples); history['Grad_Steps'].append(grad_steps)
        print(f"  -> Avg: {avg_acc:.2f}% | Task 0: {t0_acc:.2f}%")

    with open(f'ember_madar_history_seed{SEED}.json', 'w') as f:
        json.dump(history, f, indent=2)
    plt.figure(figsize=(10, 6))
    plt.plot(history['Avg_Acc'], marker='o', label='Avg (seen families)')
    plt.plot(history['Task0_Acc'], marker='x', label='Task 0')
    plt.xlabel('Task'); plt.ylabel('Accuracy (%)')
    plt.title(f'MADAR-IF + ER/KD/SI (seed {SEED})')
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(f'ember_cl_madar_seed{SEED}.png')