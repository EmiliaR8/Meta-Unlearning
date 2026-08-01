"""
Usage:
    python clmd_lamda_classil_naive_cd.py --seed 42
    python clmd_lamda_classil_naive_cd.py --seed 42 --split-mode temporal \
        --temporal-cut 2020 --num-classes 60 --min-test-samples 10
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lamda_data import build_arrays_lamda

# --- Parse Command Line Seed ---
parser = argparse.ArgumentParser(description="LAMDA Class-IL (Naive Baseline)")
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--cache', type=str, default='./Datasets/lamda_class_il_cache.npz')
parser.add_argument('--split-mode', choices=['random', 'temporal'], default='random')
parser.add_argument('--temporal-cut', type=int, default=2020)
parser.add_argument('--min-test-samples', type=int, default=0,
                    help='require this much test support per family (temporal runs)')
parser.add_argument('--num-classes', type=int, default=80)
parser.add_argument('--task0-classes', type=int, default=30)
parser.add_argument('--step-classes', type=int, default=5)
parser.add_argument('--min-family-samples', type=int, default=200)
parser.add_argument('--epochs-per-task', type=int, default=30)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--scale', choices=['none', 'standard'], default='none',
                    help="LAMDA features are binary; 'none' is the documented default")
parser.add_argument('--family-cap', type=int, default=0,
                    help='cap TRAIN samples per family (0 = off; protocol default is OFF '
                         )
parser.add_argument('--cap-seed', type=int, default=12345,
                    help='fixed seed for the family cap so all conditions see the same subset')
parser.add_argument('--target-fams', type=str, default='',
                    help='comma-separated class ids to track per-family recall/FPR for; '
                         'default derives a spread from the schedule')
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
#       HYPERPARAMETERS & CONFIGURATION
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS_PER_TASK = args.epochs_per_task
BATCH_SIZE = args.batch_size

TASK_1_FAMILIES = args.task0_classes        # 30 (EMBER: 50)
SUBSEQUENT_FAMILIES = args.step_classes     # 5  (same as EMBER)
NUM_CLASSES = args.num_classes              # 80 (EMBER: 100)
# INPUT_DIM is derived from the cache, not hardcoded (expect 4561)
# ==========================================


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


# --- 3. Training & Eval Functions ---
def get_loader(X, y, families, drop_last=False):
    
    mask = torch.isin(y, torch.tensor(families))
    if not mask.any():
        return None
    return data.DataLoader(data.TensorDataset(X[mask], y[mask]),
                           batch_size=BATCH_SIZE, shuffle=True, drop_last=drop_last)


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
 "
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
    """Per-family recall and false-positive rate over all seen families.
    Tracked across tasks to visualise WHICH families are forgotten and whether a
    family becomes an attractor (high FPR) as the model degrades."""
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
    """Spread of tracked classes: several from task 0 (measures FORGETTING) plus
    two from later tasks (measures ACQUISITION then retention)."""
    t0 = {0, task0 // 6, task0 // 3, (2 * task0) // 3, task0 - 1}
    later = [task0 + step * 2, num_classes - step]
    return sorted({c for c in list(t0) + later if 0 <= c < num_classes})


# --- 4. Main Execution ---
if __name__ == "__main__":
    if not os.path.exists(args.cache):
        raise SystemExit(f"cache not found: {args.cache}\n"
                         f"Run build_lamda_cache.py first.")

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

    # --- scaling: 'none' is the documented LAMDA default (binary features) ---
    X_train_np, X_test_np = arrays['X_train'], arrays['X_test']
    if args.scale == 'standard':
        # provided for ablation only; NOT the protocol default
        from sklearn.preprocessing import StandardScaler
        FEATURE_CLIP = 10.0
        scaler = StandardScaler()
        t0 = np.isin(arrays['y_train'], task_families[0])
        scaler.fit(X_train_np[t0])
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

    # family cap: OFF by default for LAMDA (protocol decision -- any cap tight
    # enough to dent the 128:1 imbalance would discard most of the corpus)
    n_before = len(y_train)
    X_train, y_train = cap_per_family(X_train, y_train, args.family_cap, args.cap_seed)
    if args.family_cap:
        print(f"[cap] {args.family_cap}/family: {n_before:,} -> {len(y_train):,} train "
              f"(cap seed {args.cap_seed}; test unchanged at {len(y_test):,})")
    else:
        print("[cap] none (protocol default)")

    TARGET_FAMS = ([int(x) for x in args.target_fams.split(',') if x.strip()]
                   if args.target_fams
                   else default_target_fams(TASK_1_FAMILIES, NUM_CLASSES, SUBSEQUENT_FAMILIES))
    print(f"[track] per-family recall/FPR for classes {TARGET_FAMS}")

    model = EmberNN(INPUT_DIM, NUM_CLASSES).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] EmberNN({INPUT_DIM} -> 1024 -> 512 -> 256 -> 128 -> {NUM_CLASSES})"
          f"  params={n_params:,}")

    history = {'Avg_Acc': [], 'Avg_Acc_Macro': [], 'Task0_Acc': [],
               'New_Fam_Acc': [], 'New_Fam_Acc_Macro': [],
               'Train_Samples': [], 'Grad_Steps': []}
    fam_history = {fam: {'acc': [], 'fpr': []} for fam in TARGET_FAMS}

    for tid, current_fams in enumerate(task_families):
        print(f"\n=== Training Task {tid} ({len(current_fams)} Families) ===")
        active_count = TASK_1_FAMILIES + (tid * SUBSEQUENT_FAMILIES)
        seen_fams = [f for sublist in task_families[:tid + 1] for f in sublist]
        # sanity check that masking arithmetic and task splits stay in sync
        assert active_count == len(seen_fams), "active_count / task split mismatch"

        loader = get_loader(X_train, y_train, current_fams, drop_last=True)
        if loader is None:
            raise RuntimeError(f"task {tid} has no training samples")
        if len(loader) == 0:
            raise RuntimeError(
                f"task {tid} has {len(loader.dataset)} samples but 0 batches at "
                f"batch_size={BATCH_SIZE} with drop_last=True -- it would train "
                f"on nothing silently")

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
                # fail loudly on divergence instead of silently producing a
                # collapsed model
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite loss at task {tid}, epoch {epoch} - training diverged")
                loss.backward()
                optimizer.step()

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
        print(f"  -> Avg: {avg_acc:.2f}% (macro {macro_acc:.2f}%) | Task 0: {t0_acc:.2f}%"
              f" | New fams: {new_acc:.2f}% (macro {new_macro:.2f}%)")

        fam_metrics = eval_family_metrics(model, X_test, y_test, TARGET_FAMS, active_count)
        for fam in TARGET_FAMS:
            fam_history[fam]['acc'].append(fam_metrics[fam]['acc'])
            fam_history[fam]['fpr'].append(fam_metrics[fam]['fpr'])
        print("  Per-family recall: " +
              " ".join(f"f{fam}:{fam_metrics[fam]['acc']:.1f}%" for fam in TARGET_FAMS))

    tag = f"{args.out_prefix}_naive"
    if args.split_mode == 'temporal':
        tag += f"_temporal{args.temporal_cut}"
    meta = {'dataset': 'lamda_classil', 'split_mode': args.split_mode,
            'temporal_cut': args.temporal_cut if args.split_mode == 'temporal' else None,
            'num_classes': NUM_CLASSES, 'task0_classes': TASK_1_FAMILIES,
            'step_classes': SUBSEQUENT_FAMILIES, 'total_tasks': TOTAL_TASKS,
            'input_dim': INPUT_DIM, 'scale': args.scale,
            'family_cap': args.family_cap or None, 'cap_seed': args.cap_seed,
            'epochs_per_task': EPOCHS_PER_TASK,
            'batch_size': BATCH_SIZE, 'seed': SEED, 'n_params': n_params,
            'target_fams': TARGET_FAMS}
    history['Family_Metrics'] = {str(f): fam_history[f] for f in TARGET_FAMS}
    with open(f'{tag}_history_seed{SEED}.json', 'w') as f:
        json.dump({'meta': meta, **history}, f, indent=2)

    plt.figure(figsize=(10, 6))
    plt.plot(history['Avg_Acc'], marker='o', label='Avg (seen families, micro)')
    plt.plot(history['Avg_Acc_Macro'], marker='s', label='Avg (macro, per-family mean)')
    plt.plot(history['Task0_Acc'], marker='x', label='Task 0')
    plt.plot(history['New_Fam_Acc'], marker='^', label='Newest families')
    plt.xlabel('Task')
    plt.ylabel('Accuracy (%)')
    plt.title(f'LAMDA Class-IL Naive Baseline (seed {SEED}, {args.split_mode} split)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f'{tag}_seed{SEED}.png', dpi=120, bbox_inches='tight')

    plt.figure(figsize=(10, 6))
    for fam in TARGET_FAMS:
        plt.plot(fam_history[fam]['acc'], marker='o', label=f'family {fam}')
    plt.xlabel('Task')
    plt.ylabel('Per-family recall (%)')
    plt.title(f'LAMDA Class-IL Naive: tracked family retention (seed {SEED})')
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.savefig(f'{tag}_families_seed{SEED}.png', dpi=120, bbox_inches='tight')

    print(f"\nwrote {tag}_history_seed{SEED}.json, {tag}_seed{SEED}.png, "
          f"{tag}_families_seed{SEED}.png")