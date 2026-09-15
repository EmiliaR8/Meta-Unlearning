import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import thrember
import os
import json
import random
import argparse

# --- Parse Command Line Seed ---
parser = argparse.ArgumentParser(description="EMBER 2024 Continual Learning Experiment (Joint Retraining Oracle)")
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--family-cap', type=int, default=10000,
                    help='Max TRAINING samples per family (0 = no cap). Default 10000 keeps '
                         '~44%% of the data (task 0 ~454K samples), compressing the 94:1 family '
                         'imbalance to roughly 1.7:1 while staying comparable to EMBER 2018 in '
                         'compute. MUST be identical across all four conditions.')
parser.add_argument('--cap-seed', type=int, default=12345,
                    help='Seed for the capped subsample. Deliberately INDEPENDENT of --seed so '
                         'every condition and every run trains on the identical subset.')
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
DATA_DIR = "./Datasets/ember2024"

INPUT_DIM = 2568           # EMBER 2024 feature dimensionality
FEATURE_CLIP = 10.0        # clip standardized features to [-10, 10]
EPOCHS_PER_TASK = 30
BATCH_SIZE = 256

TASK_1_FAMILIES = 50
SUBSEQUENT_FAMILIES = 5
TOTAL_TASKS = 11
MIN_FAMILY_SAMPLES = 200
# Total classes derived from the task structure (was hard-coded 175, of which
# 75 outputs were permanently masked and unused).
NUM_CLASSES = TASK_1_FAMILIES + (TOTAL_TASKS - 1) * SUBSEQUENT_FAMILIES   # = 100

# Families tracked individually (post-remap ids, i.e. frequency ranks:
# id 6 = 7th most common family, id 43 = 44th, ...). All < 50, so they are
# introduced in task 0 and their retention is tracked across every later task.
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

    EMBER 2024 is far more imbalanced than 2018 (largest family 256,844 vs
    smallest 2,730 - a 94x ratio), which both skews learning and inflates
    compute ~4.6x. Capping compresses the imbalance, makes task 0 comparable
    in size to EMBER 2018, and leaves the smaller (later-task) families
    untouched. The subsample uses a FIXED seed so all conditions/seeds see
    exactly the same data."""
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
    .
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

def eval_acc_macro(model, X, y, families, active_count):
    """Mean per-family recall (macro accuracy). The EMBER 2024 TEST set keeps
    its natural 94:1 imbalance, so sample-weighted (micro) accuracy is
    dominated by a few large families; macro shows whether small families are
    actually being served."""
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
    """Per-family recall ('acc') and false-positive rate over all seen families.
    EMBER 2024 only - lets us watch specific families degrade task by task."""
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
    # FIX: fail early and clearly if the dataset cannot fill the task schedule.
    assert len(eligible_all) >= NUM_CLASSES, (
        f"Only {len(eligible_all)} families have >= {MIN_FAMILY_SAMPLES} samples, "
        f"but the task schedule needs {NUM_CLASSES}. Lower MIN_FAMILY_SAMPLES or "
        f"reduce TOTAL_TASKS / SUBSEQUENT_FAMILIES.")
    eligible = eligible_all[:NUM_CLASSES]
    id_map = {old_id: new_id for new_id, old_id in enumerate(eligible)}

    # Diagnostic: sanity-check that the selected families look like real families
    # (e.g. no single dominant "benign" bucket swallowing task 0).
    sel_counts = [counts_dict[f] for f in eligible]
    print(f"Families with >= {MIN_FAMILY_SAMPLES} samples: {len(eligible_all)} "
          f"(using top {NUM_CLASSES})")
    print(f"  selected family sizes: max {max(sel_counts)}, min {min(sel_counts)}, "
          f"total {sum(sel_counts)} samples")
    print(f"  top 5 original ids -> new ids: "
          f"{[(f, id_map[f], counts_dict[f]) for f in eligible[:5]]}")

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

    # NOTE: no model here - the oracle instantiates a fresh model each task.
    # FIX: history now tracks compute cost per task, and the per-family
    # metrics are actually persisted (previously computed and discarded).
    history = {'Avg_Acc': [], 'Task0_Acc': [], 'Avg_Acc_Macro': [],
               # NEW: accuracy restricted to the 5 families introduced by THIS
               # task. Separates forgetting (Task0_Acc falling) from intransigence
               # (New_Fam_Acc low) - the two failure modes look identical in Avg_Acc.
               'New_Fam_Acc': [], 'New_Fam_Acc_Macro': [],
               'Train_Samples': [], 'Grad_Steps': [],
               'config': {'family_cap': args.family_cap, 'cap_seed': args.cap_seed,
                          'seed': SEED, 'epochs_per_task': EPOCHS_PER_TASK}}
    fam_history = {fam: {'acc': [], 'fpr': []} for fam in TARGET_FAMS}

    for tid, current_fams in enumerate(task_families):
        seen_fams = [f for sublist in task_families[:tid + 1] for f in sublist]
        print(f"\n=== Training Task {tid} ({len(seen_fams)} Families Cumulative) ===")
        active_count = TASK_1_FAMILIES + (tid * SUBSEQUENT_FAMILIES)
        
        assert active_count == len(seen_fams), "active_count / task split mismatch"

        # JOINT LEARNING: load ALL data seen up to this point
        joint_loader = get_loader(X_train_scaled, y_train, seen_fams, drop_last=True)
        if not joint_loader: continue

        n_samples = len(joint_loader.dataset)
        grad_steps = EPOCHS_PER_TASK * len(joint_loader)
        print(f"  Samples: {n_samples} | Gradient steps: {grad_steps}")

        # Throw away the old model and start fresh
        model = EmberNN(INPUT_DIM, NUM_CLASSES).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        mask = torch.full((model.fc_last.out_features,), -1e9).to(DEVICE)
        mask[:active_count] = 0.0

        for epoch in range(EPOCHS_PER_TASK):
            for v, l in joint_loader:
                v, l = v.to(DEVICE), l.to(DEVICE)
                optimizer.zero_grad()
                loss = nn.CrossEntropyLoss()(model(v) + mask, l)
                
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at task {tid}, epoch {epoch} - training diverged")
                loss.backward()
                optimizer.step()

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
    with open(f'ember24_joint_history_cap{args.family_cap}_seed{SEED}.json', 'w') as f:
        json.dump(history, f, indent=2)

    .
    plt.figure(figsize=(10, 6))
    plt.plot(history['Avg_Acc'], marker='o', label='Avg (seen families, micro)')
    plt.plot(history['Avg_Acc_Macro'], marker='s', label='Avg (macro, per-family mean)')
    plt.plot(history['Task0_Acc'], marker='x', label='Task 0')
    plt.plot(history['New_Fam_Acc'], marker='^', label='Newest 5 families')
    plt.xlabel('Task'); plt.ylabel('Accuracy (%)')
    plt.title(f'EMBER 2024 Joint Retraining Oracle (seed {SEED})')
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(f'ember24_baseline_joint_cap{args.family_cap}_seed{SEED}.png')

    
    plt.figure(figsize=(10, 6))
    for fam in TARGET_FAMS:
        plt.plot(fam_history[fam]['acc'], marker='o', label=f'family {fam}')
    plt.xlabel('Task'); plt.ylabel('Per-family recall (%)')
    plt.title(f'EMBER 2024 Joint: tracked family retention (seed {SEED})')
    plt.legend(fontsize=9); plt.grid(alpha=0.3)
    plt.savefig(f'ember24_joint_families_cap{args.family_cap}_seed{SEED}.png')