import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import ember
import os
import csv
import json
import random
import argparse

# --- Parse Command Line Seed ---
# FIX: added seed handling so the naive baseline runs under the identical
# reproducibility regime as the other three conditions.
parser = argparse.ArgumentParser(description="Continual Learning Experiment (Naive Baseline)")
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
#       HYPERPARAMETERS & CONFIGURATION
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./Datasets/ember2018"
EMBERSIM_DIR = "./Datasets/embersim-databank"

INPUT_DIM = 2381
FEATURE_CLIP = 10.0        # clip standardized features to [-10, 10]
EPOCHS_PER_TASK = 30
BATCH_SIZE = 256

TASK_1_FAMILIES = 50
SUBSEQUENT_FAMILIES = 5
TOTAL_TASKS = 11
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

# --- 3. Training & Eval Functions ---
def get_loader(X, y, families, drop_last=False):
    # FIX: drop_last option added. Training loaders use drop_last=True so a
    # final batch of size 1 can never crash BatchNorm in train mode.
    mask = torch.isin(y, torch.tensor(families))
    if not mask.any(): return None
    return data.DataLoader(data.TensorDataset(X[mask], y[mask]),
                           batch_size=BATCH_SIZE, shuffle=True, drop_last=drop_last)

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

    # FIX: readable, behavior-identical construction of the top-100 family map.
    # Families with >= 200 samples, sorted by descending frequency (ties keep
    # ascending original-id order, matching the original one-liner).
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
    # FIX: history now also tracks compute cost per task for the
    # efficiency comparison across conditions.
    history = {'Avg_Acc': [], 'Task0_Acc': [], 'Train_Samples': [], 'Grad_Steps': []}

    for tid, current_fams in enumerate(task_families):
        print(f"\n=== Training Task {tid} ({len(current_fams)} Families) ===")
        active_count = TASK_1_FAMILIES + (tid * SUBSEQUENT_FAMILIES)
        seen_fams = [f for sublist in task_families[:tid+1] for f in sublist]
        # FIX: sanity check that masking arithmetic and task splits stay in sync.
        assert active_count == len(seen_fams), "active_count / task split mismatch"

        loader = get_loader(X_train_scaled, y_train, current_fams, drop_last=True)
        if not loader: continue

        n_samples = len(loader.dataset)
        grad_steps = EPOCHS_PER_TASK * len(loader)
        print(f"  Samples: {n_samples} | Gradient steps: {grad_steps}")

        # NAIVE LEARNING: Always use Adam, only train on current task loader
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        mask = torch.full((model.fc_last.out_features,), -1e9).to(DEVICE)
        mask[:active_count] = 0.0

        for epoch in range(EPOCHS_PER_TASK):
            for v, l in loader:
                v, l = v.to(DEVICE), l.to(DEVICE)
                optimizer.zero_grad()
                loss = nn.CrossEntropyLoss()(model(v) + mask, l)
                
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at task {tid}, epoch {epoch} - training diverged")
                loss.backward()
                optimizer.step()

        avg_acc = eval_acc(model, X_test_scaled, y_test, seen_fams, active_count)
        t0_acc = eval_acc(model, X_test_scaled, y_test, task_families[0], active_count)
        history['Avg_Acc'].append(avg_acc); history['Task0_Acc'].append(t0_acc)
        history['Train_Samples'].append(n_samples); history['Grad_Steps'].append(grad_steps)
        print(f"  -> Avg: {avg_acc:.2f}% | Task 0: {t0_acc:.2f}%")

    # FIX: legend now renders; axis labels added; seed in filename; history dumped
    # to JSON so multi-seed aggregation does not require re-parsing stdout.
    with open(f'ember_naive_history_seed{SEED}.json', 'w') as f:
        json.dump(history, f, indent=2)
    plt.figure(figsize=(10, 6))
    plt.plot(history['Avg_Acc'], marker='o', label='Avg (seen families)')
    plt.plot(history['Task0_Acc'], marker='x', label='Task 0')
    plt.xlabel('Task'); plt.ylabel('Accuracy (%)')
    plt.title(f'Naive Baseline (seed {SEED})')
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(f'ember_baseline_naive_seed{SEED}.png')