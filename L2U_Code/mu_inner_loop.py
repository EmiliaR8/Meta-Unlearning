"""
Inner loop for meta-learning forget-set selection.

This is the validated MADAR+MU pipeline, refactored to be called many times
cheaply by an outer optimizer (ES) or standalone for baselines:

  * --cache_data:   filtered EMBER arrays are saved/loaded as one .npz,
                    skipping the slow jsonl scan on every run.
  * --checkpoint:   task-0 model + scaler + replay buffer are trained once
                    and reused, skipping ~33k gradient steps per run.
  * --selector:     donut (hand-crafted baseline) | random | nn1 (learned).
  * --scorer_weights: flat parameter vector for nn1_scorer (npz, key 'params').
  * mini-run knobs: --n_tasks, --cl_iters, --unlearn_epochs.
  * --out:          reward JSON {mean_acc, final_acc, per_task} for the driver.

Examples:
  # one-time per permutation: build cache + task-0 checkpoint
  python mu_inner_loop.py --cache_data ember18.npz --checkpoint t0_p0.pt --make_checkpoint
  # baseline evaluations (mini config)
  python mu_inner_loop.py --cache_data ember18.npz --checkpoint t0_p0.pt \
      --selector donut --n_tasks 5 --cl_iters 500 --out reward_donut.json
  # one ES candidate
  python mu_inner_loop.py --cache_data ember18.npz --checkpoint t0_p0.pt \
      --selector nn1 --scorer_weights cand_003.npz --n_tasks 5 --cl_iters 500 --out r3.json
"""

import argparse
import copy
import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import nn1_scorer

# --- CLI ---
parser = argparse.ArgumentParser(description="MU inner loop for meta-learned forget-set selection")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--perm_seed', type=int, default=None,
                    help='If set, shuffle which families form task 0 / increments (a "data distribution")')
parser.add_argument('--dataset', type=str, default='ember2018',
                    choices=['ember2018', 'ember2024', 'lamda_classil'],
                    help='Which corpus to run on. Selects the array builder used when the cache '
                         'is absent AND the protocol defaults (memory size, SI strength, CL '
                         'iterations). Everything downstream is dataset-agnostic, so a scorer '
                         'trained on one dataset can be evaluated on the other unchanged.')
parser.add_argument('--cache_data', type=str, default=None,
                    help='Filtered arrays .npz (default: <dataset>_cache.npz)')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='Task-0 model + buffer (default: task0_ckpt.pt for ember2018, task0_ember2024.pt for ember2024)')
parser.add_argument('--make_checkpoint', action='store_true',
                    help='Train task 0 and save the checkpoint, then exit')
parser.add_argument('--mode', type=str, default='a', choices=['a', 'b'],
                    help="a: select/unlearn CURRENT-task samples (original design). "
                         "b: select/unlearn OLD replay-buffer samples that conflict with the new task.")
parser.add_argument('--selector', type=str, default='donut',
                    choices=['donut', 'random', 'nn1', 'none', 'gradconflict', 'densityratio'],
                    help="mode a: donut|random|nn1. mode b: none|random|gradconflict|densityratio|nn1.")
parser.add_argument('--scorer_weights', type=str, default=None, help='npz with key "params" (selector=nn1)')
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--forget_ratio', type=float, default=0.10)
parser.add_argument('--n_tasks', type=int, default=10, help='CL tasks to run after task 0 (max 10)')
parser.add_argument('--cl_iters', type=int, default=None,
                    help='CL iterations per task (default: 2000 for ember2018, 8000 for '
                         'ember2024, which matches epochs-per-sample given 2024s larger tasks)')
parser.add_argument('--mem_size', type=int, default=None,
                    help='Replay buffer size (default: 5000 / 10000)')
parser.add_argument('--si_c', type=float, default=None,
                    help='Synaptic Intelligence strength (default: 1.0 / 100.0)')
parser.add_argument('--family_cap', type=int, default=None,
                    help='Max TRAINING samples per family, applied when building the cache '
                         '(default: 0 = none for ember2018, 10000 for ember2024)')
parser.add_argument('--cap_seed', type=int, default=12345,
                    help='Fixed seed for the capped subsample, independent of --seed')
parser.add_argument('--unlearn_epochs', type=int, default=3)
parser.add_argument('--out', type=str, default=None, help='Write reward JSON here')
parser.add_argument('--zero_features', type=str, nargs='*', default=[],
                    help="ABLATION: zero these NN-1 features so they cannot contribute "
                         "to the score. Names, not indices: "
                         "iso_latent iso_raw ce_loss entropy margin centroid_dist "
                         "log_family_size grad_conflict density_ratio. Applied AFTER "
                         "z-scoring, which is per-column, so the remaining features are "
                         "unaffected. Only the nn1 selector is touched -- the "
                         "gradconflict/densityratio BASELINES read f_conf/f_dens "
                         "directly and stay comparable across ablations.")
parser.add_argument('--quiet', action='store_true')
args = parser.parse_args()

# Protocol defaults per dataset (explicit flags always win).
_DEFAULTS = {
    'ember2018': dict(cl_iters=2000, mem_size=5000,  si_c=1.0,   family_cap=0),
    'ember2024': dict(cl_iters=8000, mem_size=10000, si_c=100.0, family_cap=10000),
    # LAMDA: CL_ITERS from the elbow of the macro-vs-budget curve (the 121-epoch
    # invariant did NOT transfer); mem 2500 = 1.85% of task 0, matching the EMBER
    # ratio; si_c 100 for cross-corpus consistency (measured near-inert on LAMDA);
    # no family cap (any cap tight enough to dent 128:1 discards most of the corpus).
    'lamda_classil': dict(cl_iters=4000, mem_size=2500, si_c=100.0, family_cap=0),
}
for _k, _v in _DEFAULTS[args.dataset].items():
    if getattr(args, _k) is None:
        setattr(args, _k, _v)
_LEGACY_PATHS = {   # keep the existing EMBER 2018 filenames so prior runs stay valid
    'ember2018': ('ember18_cache.npz', 'task0_ckpt.pt'),
    'ember2024': ('ember2024_cache.npz', 'task0_ember2024.pt'),
    'lamda_classil': ('lamda_classil_meta_cache.npz', 'task0_lamda_classil.pt'),
}
if args.cache_data is None:
    args.cache_data = _LEGACY_PATHS[args.dataset][0]
if args.checkpoint is None:
    args.checkpoint = _LEGACY_PATHS[args.dataset][1]

_A = {'donut', 'random', 'nn1'}
_B = {'none', 'random', 'gradconflict', 'densityratio', 'nn1'}
assert (args.mode == 'a' and args.selector in _A) or (args.mode == 'b' and args.selector in _B), \
    f"selector '{args.selector}' not valid for mode '{args.mode}'"

SEED = args.seed
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Config (matching the validated pipeline) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./Datasets/ember2018"
EMBERSIM_DIR = "./Datasets/embersim-databank"
DATA_DIR_2024 = "./Datasets/ember2024"
LAMDA_CACHE = "./Datasets/lamda_class_il_cache.npz"   # raw Option-A join
INPUT_DIM = None            # derived from the loaded arrays (2381 / 2568)
FEATURE_CLIP = 10.0
TASK0_EPOCHS = 30
BATCH_SIZE = 256
MEM_SIZE = args.mem_size
MADAR_CONTAMINATION = 0.1
MIN_FAMILY_SAMPLES = 200
# Schedule is dataset-dependent. LAMDA has only 80 families clearing the (identical)
# >=200 train bar, so it runs the pre-registered 30 + 5x10 fallback. Task COUNT and
# increment SIZE match EMBER, which is what keeps the corpora comparable.
_SCHEDULE = {
    'ember2018':    dict(task0=50, step=5, n_classes=100, scale='standard'),
    'ember2024':    dict(task0=50, step=5, n_classes=100, scale='standard'),
    'lamda_classil': dict(task0=30, step=5, n_classes=80,  scale='none'),
}
TASK_1_FAMILIES = _SCHEDULE[args.dataset]['task0']
SUBSEQUENT_FAMILIES = _SCHEDULE[args.dataset]['step']
NUM_CLASSES = _SCHEDULE[args.dataset]['n_classes']
SCALE_MODE = _SCHEDULE[args.dataset]['scale']
KD_TEMP = 2.0
SI_C = args.si_c
SI_EPS = 0.1
UNLEARN_LR = 1e-4

def log(*a):
    if not args.quiet: print(*a, flush=True)

# --- Data (with npz cache) ---
# Only these two builders are dataset-specific; everything downstream operates
# on the cached arrays, so the two corpora share one code path.
def build_arrays_2018():
    import ember
    def build_hash_to_family_map(embersim_dir):
        hash_to_fam = {}
        label_file = os.path.join(embersim_dir, "data", "raw", "ember_original_metadata.csv")
        if not os.path.exists(label_file): return hash_to_fam
        with open(label_file, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                fam = row.get('avclass', '').strip()
                if fam and fam != 'SINGLETON':
                    hash_to_fam[row['sha256']] = fam
        return hash_to_fam

    def load_and_filter(subset, hash_to_fam, family_to_id, X_raw):
        jsonl_files = sorted(f for f in os.listdir(DATA_DIR)
                             if f.startswith(f"{subset}_features") and f.endswith(".jsonl"))
        fx, fy, row_idx = [], [], 0
        for jf in jsonl_files:
            with open(os.path.join(DATA_DIR, jf), 'r') as f:
                for line in f:
                    h = json.loads(line)['sha256']
                    if h in hash_to_fam:
                        fam = hash_to_fam[h]
                        if fam not in family_to_id: family_to_id[fam] = len(family_to_id)
                        fx.append(X_raw[row_idx]); fy.append(family_to_id[fam])
                    row_idx += 1
        return np.array(fx), np.array(fy)

    if not os.path.exists(os.path.join(DATA_DIR, "X_train.dat")):
        ember.create_vectorized_features(DATA_DIR, feature_version=2)
    X_train_raw, _, X_test_raw, _ = ember.read_vectorized_features(DATA_DIR, feature_version=2)
    h2f, f2i = build_hash_to_family_map(EMBERSIM_DIR), {}
    Xtr, ytr = load_and_filter("train", h2f, f2i, X_train_raw)
    Xte, yte = load_and_filter("test", h2f, f2i, X_test_raw)

    counts = {}
    for v in ytr: counts[int(v)] = counts.get(int(v), 0) + 1
    eligible = sorted([f for f, c in sorted(counts.items()) if c >= 200],
                      key=lambda f: counts[f], reverse=True)[:100]
    id_map = {old: new for new, old in enumerate(eligible)}

    def remap(X, y):
        y2 = np.array([id_map.get(int(i), -1) for i in y])
        m = y2 != -1
        return X[m].astype(np.float32), y2[m].astype(np.int64)

    return remap(Xtr, ytr) + remap(Xte, yte)


def build_arrays_2024():
    """EMBER 2024 via thrember: family labels are native, so no hash->AVClass
    join is needed. Two extra steps relative to 2018: negative sentinel labels
    (benign/unlabelled) are dropped, and the training set is capped per family
    to bound the 94:1 class imbalance."""
    import thrember
    if not os.path.exists(os.path.join(DATA_DIR_2024, "X_train.dat")):
        thrember.create_vectorized_features(DATA_DIR_2024, label_type="family")
    Xtr, ytr = thrember.read_vectorized_features(DATA_DIR_2024, subset="train")
    Xte, yte = thrember.read_vectorized_features(DATA_DIR_2024, subset="test")

    ytr, yte = np.asarray(ytr), np.asarray(yte)
    n_neg = int((ytr < 0).sum())
    if n_neg:
        log(f"  dropping {n_neg} training samples with negative (sentinel) labels")
    ktr, kte = ytr >= 0, yte >= 0
    Xtr, ytr, Xte, yte = Xtr[ktr], ytr[ktr], Xte[kte], yte[kte]

    counts = {}
    for v in ytr: counts[int(v)] = counts.get(int(v), 0) + 1
    eligible = sorted([f for f, c in sorted(counts.items()) if c >= MIN_FAMILY_SAMPLES],
                      key=lambda f: counts[f], reverse=True)
    assert len(eligible) >= NUM_CLASSES, (
        f"only {len(eligible)} families have >= {MIN_FAMILY_SAMPLES} samples, need {NUM_CLASSES}")
    id_map = {old: new for new, old in enumerate(eligible[:NUM_CLASSES])}

    def remap(X, y):
        y2 = np.array([id_map.get(int(i), -1) for i in y])
        m = y2 != -1
        return X[m].astype(np.float32), y2[m].astype(np.int64)

    Xtr, ytr = remap(Xtr, ytr)
    Xte, yte = remap(Xte, yte)

    # Cap TRAINING samples per family (test set untouched), with a fixed seed so
    # every condition and every run sees the identical subset.
    if args.family_cap and args.family_cap > 0:
        rng = np.random.default_rng(args.cap_seed)
        keep = []
        for fam in np.unique(ytr):
            idx = np.flatnonzero(ytr == fam)
            if len(idx) > args.family_cap:
                idx = rng.choice(idx, args.family_cap, replace=False)
            keep.append(idx)
        keep = np.sort(np.concatenate(keep))
        log(f"  family cap {args.family_cap}: {len(ytr):,} -> {len(keep):,} training samples")
        Xtr, ytr = Xtr[keep], ytr[keep]
    return Xtr, ytr, Xte, yte

def build_arrays_lamda():
    """LAMDA Class-IL arrays via the shared lamda_data module, so the meta pipeline
    and the CL scripts cannot drift apart on family selection or task construction."""
    from lamda_data import build_arrays_lamda as _bal
    r = _bal(LAMDA_CACHE, split_mode='random',
             min_family_samples=MIN_FAMILY_SAMPLES, num_classes=NUM_CLASSES,
             task0_classes=TASK_1_FAMILIES, step_classes=SUBSEQUENT_FAMILIES,
             verbose=not args.quiet)
    return r['X_train'], r['y_train'], r['X_test'], r['y_test']


if os.path.exists(args.cache_data):
    z = np.load(args.cache_data)
    X_train_np, y_train_np, X_test_np, y_test_np = z['Xtr'], z['ytr'], z['Xte'], z['yte']
    # Caches written before this field existed are EMBER 2018 by construction.
    cached_ds = str(z['dataset']) if 'dataset' in z.files else 'ember2018'
    assert cached_ds == args.dataset, (
        f"cache {args.cache_data} holds {cached_ds} but --dataset is {args.dataset}")
    if 'family_cap' in z.files and int(z['family_cap']) != args.family_cap:
        raise SystemExit(
            f"cache was built with --family_cap {int(z['family_cap'])} but this run asks for "
            f"{args.family_cap}; use a different --cache_data file")
    log(f"Loaded cached arrays from {args.cache_data}")
else:
    builder = {'ember2018': build_arrays_2018,
               'ember2024': build_arrays_2024,
               'lamda_classil': build_arrays_lamda}[args.dataset]
    log(f"Building {args.dataset} arrays (no cache found)...")
    X_train_np, y_train_np, X_test_np, y_test_np = builder()
    np.savez(args.cache_data, Xtr=X_train_np, ytr=y_train_np, Xte=X_test_np, yte=y_test_np,
             dataset=args.dataset, family_cap=args.family_cap, cap_seed=args.cap_seed)
    log(f"Built and cached arrays to {args.cache_data}")

# Feature dimensionality follows the data (2381 for 2018, 2568 for 2024).
INPUT_DIM = int(X_train_np.shape[1])
log(f"Dataset {args.dataset}: d={INPUT_DIM}, train={len(y_train_np):,}, test={len(y_test_np):,} "
    f"| mem {MEM_SIZE}, si_c {SI_C}, cl_iters {args.cl_iters}")

y_train = torch.tensor(y_train_np, dtype=torch.long)
y_test = torch.tensor(y_test_np, dtype=torch.long)

# --- Task assignment (optionally permuted = a different "distribution") ---
fam_order = list(range(NUM_CLASSES))
if args.perm_seed is not None:
    fam_order = list(np.random.default_rng(args.perm_seed).permutation(NUM_CLASSES))
task_families = [fam_order[:TASK_1_FAMILIES]] + \
    [fam_order[TASK_1_FAMILIES + i*SUBSEQUENT_FAMILIES:TASK_1_FAMILIES + (i+1)*SUBSEQUENT_FAMILIES]
     for i in range(10)]
# In-model class ids follow presentation order, so remap labels to that order.
present_order = [f for t in task_families for f in t]
fam_to_class = {fam: i for i, fam in enumerate(present_order)}
y_train = torch.tensor([fam_to_class[int(v)] for v in y_train], dtype=torch.long)
y_test = torch.tensor([fam_to_class[int(v)] for v in y_test], dtype=torch.long)
task_classes = [list(range(TASK_1_FAMILIES))] + \
    [list(range(TASK_1_FAMILIES + i*SUBSEQUENT_FAMILIES, TASK_1_FAMILIES + (i+1)*SUBSEQUENT_FAMILIES))
     for i in range(10)]

# --- Model ---
class EmberNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024); self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512);       self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256);        self.fc3_bn = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128);        self.fc4_bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(); self.fc_last = nn.Linear(128, num_classes)
    def forward(self, x, return_latent=False):
        x = self.relu(self.fc1_bn(self.fc1(x)))
        x = self.relu(self.fc2_bn(self.fc2(x)))
        x = self.relu(self.fc3_bn(self.fc3(x)))
        latent = self.relu(self.fc4_bn(self.fc4(x)))
        logits = self.fc_last(latent)
        return (logits, latent) if return_latent else logits

def cycle(it):
    while True:
        for x in it: yield x

def get_loader(X, y, classes, drop_last=False):
    m = torch.isin(y, torch.tensor(classes))
    if not m.any(): return None
    return data.DataLoader(data.TensorDataset(X[m], y[m]), batch_size=BATCH_SIZE,
                           shuffle=True, drop_last=drop_last)

def loss_fn_kd(scores, targets, T=2.0):
    return F.kl_div(F.log_softmax(scores/T, 1), F.softmax(targets/T, 1),
                    reduction='batchmean') * (T**2)

def eval_acc(model, X, y, classes, active_count):
    model.eval()
    m = torch.isin(y, torch.tensor(classes))
    if not m.any(): return 0.0
    ld = data.DataLoader(data.TensorDataset(X[m].to(DEVICE), y[m].to(DEVICE)), batch_size=BATCH_SIZE)
    correct = total = 0
    with torch.no_grad():
        for bv, bl in ld:
            _, pred = torch.max(model(bv)[:, :active_count], 1)
            total += bl.size(0); correct += (pred == bl).sum().item()
    return correct / total * 100

def eval_acc_macro(model, X, y, classes, active_count):
    """Mean per-family recall. LAMDA's test set is 128:1 imbalanced, so micro
    accuracy is dominated by a few huge families; macro is the headline metric
    in the 2024-era protocol. Vectorised with bincount."""
    model.eval()
    m = torch.isin(y, torch.tensor(classes))
    if not m.any(): return 0.0
    ld = data.DataLoader(data.TensorDataset(X[m].to(DEVICE), y[m].to(DEVICE)), batch_size=BATCH_SIZE)
    correct = torch.zeros(active_count); total = torch.zeros(active_count)
    with torch.no_grad():
        for bv, bl in ld:
            _, pred = torch.max(model(bv)[:, :active_count], 1)
            blc = bl.cpu()
            total += torch.bincount(blc, minlength=active_count).float()
            correct += torch.bincount(blc[(pred.cpu() == blc)], minlength=active_count).float()
    present = total > 0
    return float((correct[present] / total[present]).mean()) * 100 if present.any() else 0.0


# --- Replay buffer (MADAR-IF, identical to validated pipeline) ---
family_buffers, replay_buffer = {}, []

def update_buffer_madar(loader, model):
    global family_buffers, replay_buffer
    model.eval()
    full = data.DataLoader(loader.dataset, batch_size=BATCH_SIZE, shuffle=False)
    vs, ls, lats = [], [], []
    with torch.no_grad():
        for v, l in full:
            _, lat = model(v.to(DEVICE), return_latent=True)
            vs.append(v.cpu()); ls.append(l.cpu()); lats.append(lat.cpu())
    X, Y, L = torch.cat(vs).numpy(), torch.cat(ls).numpy(), torch.cat(lats).numpy()
    fams = np.unique(Y)
    budget = MEM_SIZE // (len(family_buffers) + len(fams))
    for fam in family_buffers:
        buf = family_buffers[fam]
        if len(buf) > budget:
            half = budget // 2
            an, inl = buf[::2], buf[1::2]
            nb = [v for p in zip(an[:half], inl[:half]) for v in p]
            if budget % 2 and len(an) > half: nb.append(an[half])
            family_buffers[fam] = nb
    for fam in fams:
        m = Y == fam
        Xf, Yf, Lf = X[m], Y[m], L[m]
        n_sel = min(budget, len(Xf))
        if n_sel == 0: continue
        iso = IsolationForest(contamination=MADAR_CONTAMINATION, n_jobs=-1, random_state=SEED)
        iso.fit(Lf)
        order = np.argsort(iso.decision_function(Lf))
        half = n_sel // 2
        an, inl = order[:half], order[-(n_sel - half):]
        idx = [i for p in zip(an, inl) for i in p]
        if n_sel % 2: idx.append(inl[-1])
        family_buffers[fam] = [(torch.tensor(Xf[i]), torch.tensor(int(Yf[i]))) for i in idx]
    replay_buffer.clear()
    for fd in family_buffers.values(): replay_buffer.extend(fd)

# --- CL training (with corrected SI accumulation) ---
def train_cl_er(model, teacher, optimizer, loader, iters, active_count, prev_active,
                W, omega, p_old, tid):
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d): m.eval()
    mask = torch.full((model.fc_last.out_features,), -1e9).to(DEVICE)
    mask[:active_count] = 0.0
    li = iter(cycle(loader))
    assert replay_buffer
    bv = torch.stack([i[0] for i in replay_buffer])
    bl = torch.tensor([i[1] for i in replay_buffer])
    bi = iter(cycle(data.DataLoader(data.TensorDataset(bv, bl), batch_size=BATCH_SIZE, shuffle=True)))
    for _ in range(iters):
        v, l = next(li); v, l = v.to(DEVICE), l.to(DEVICE)
        optimizer.zero_grad()
        mv, ml = next(bi); mv, ml = mv.to(DEVICE), ml.to(DEVICE)
        loss_cur = nn.CrossEntropyLoss()(model(torch.cat([v, mv])) + mask, torch.cat([l, ml]))
        rnt = 1.0 / (tid + 1)
        with torch.no_grad():
            tlog = teacher(mv)[:, :prev_active]
        loss_rep = loss_fn_kd(model(mv)[:, :prev_active], tlog, T=KD_TEMP)
        si = sum((omega[n.replace('.', '__')] * (p - p_old[n.replace('.', '__')])**2).sum()
                 for n, p in model.named_parameters() if p.requires_grad)
        total = rnt * loss_cur + (1 - rnt) * loss_rep + SI_C * si
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        snap = {n.replace('.', '__'): (p.grad.detach().clone(), p.detach().clone())
                for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}
        optimizer.step()
        for n, p in model.named_parameters():
            k = n.replace('.', '__')
            if p.requires_grad and k in snap:
                g, pb = snap[k]
                W[k].add_(-g * (p.detach() - pb))

# --- Unlearning (identical loss structure to validated pipeline) ---
def unlearn(model, teacher, forget_loader, retain_loader, active_count, prev_active,
            omega, p_old, epochs, alpha, flat_lo=None, flat_hi=None):
   
    if flat_lo is None: flat_lo, flat_hi = prev_active, active_count
    model.train(); teacher.eval()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d): m.eval()
    opt = optim.Adam(model.parameters(), lr=UNLEARN_LR)
    mask = torch.full((model.fc_last.out_features,), -1e9).to(DEVICE)
    mask[:active_count] = 0.0
    ri = iter(cycle(retain_loader))
    bi = None
    if replay_buffer:
        bv = torch.stack([i[0] for i in replay_buffer])
        bl = torch.tensor([i[1] for i in replay_buffer])
        bi = iter(cycle(data.DataLoader(data.TensorDataset(bv, bl), batch_size=BATCH_SIZE, shuffle=True)))
    for _ in range(epochs):
        for fv, fl in forget_loader:
            fv = fv.to(DEVICE)
            opt.zero_grad()
            cur = model(fv)[:, flat_lo:flat_hi]
            logp = F.log_softmax(cur, 1)
            unif = torch.ones_like(logp) / (flat_hi - flat_lo)
            floss = F.kl_div(logp, unif, reduction='batchmean')
            rv, rl = next(ri); rv, rl = rv.to(DEVICE), rl.to(DEVICE)
            if bi:
                mv, ml = next(bi)
                rv = torch.cat([rv, mv.to(DEVICE)]); rl = torch.cat([rl, ml.to(DEVICE)])
            with torch.no_grad():
                tl = teacher(rv)[:, :active_count]
            sr = model(rv)
            rce = nn.CrossEntropyLoss()(sr + mask, rl)
            rkd = loss_fn_kd(sr[:, :active_count], tl, T=KD_TEMP)
            rloss = 0.5 * rce + 0.5 * rkd
            si = sum((omega[n.replace('.', '__')] * (p - p_old[n.replace('.', '__')])**2).sum()
                     for n, p in model.named_parameters() if p.requires_grad)
            (alpha * floss + (1 - alpha) * rloss + SI_C * si).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()

# --- Forget/retain selectors ---
def _apply_feature_ablation(feats, names):
    """Zero the named feature columns in an ALREADY z-scored matrix.

    For a linear scorer, score = feats @ w + b, so a zeroed column contributes
    exactly 0 no matter what weight ES assigns it. Zeroing after z-scoring is
    equivalent to zeroing before (a constant column z-scores to 0) but leaves
    the other columns' statistics visibly untouched.
    """
    if not args.zero_features:
        return feats
    valid = nn1_scorer.FEATURE_NAMES_B
    bad = [n for n in args.zero_features if n not in valid]
    if bad:
        raise SystemExit(f"--zero_features: unknown feature(s) {bad}. Valid: {valid}")
    feats = feats.copy()
    nf = feats.shape[1]
    for n in args.zero_features:
        i = valid.index(n)
        if i < nf:
            feats[:, i] = 0.0
        else:
            raise SystemExit(
                f"--zero_features {n!r} is index {i}, but this run's scorer has only "
                f"{nf} features (mode {args.mode}). grad_conflict/density_ratio exist "
                f"only in mode b.")
    return feats


def compute_nn1_features(X_sub, y_sub, model, active_count):
    """Per-sample features for NN-1 (see nn1_scorer docstring)."""
    model.eval()
    lats, logits_all, losses = [], [], []
    ld = data.DataLoader(data.TensorDataset(X_sub, y_sub), batch_size=BATCH_SIZE)
    with torch.no_grad():
        for v, l in ld:
            v, l = v.to(DEVICE), l.to(DEVICE)
            lo, la = model(v, return_latent=True)
            lo = lo[:, :active_count]
            lats.append(la.cpu()); logits_all.append(lo.cpu())
            losses.append(F.cross_entropy(lo, l, reduction='none').cpu())
    L = torch.cat(lats).numpy()
    logits = torch.cat(logits_all)
    loss = torch.cat(losses).numpy()
    probs = F.softmax(logits, 1).numpy()
    ent = -(probs * np.log(probs + 1e-9)).sum(1) / np.log(active_count)
    top2 = np.sort(probs, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    iso_l = IsolationForest(contamination=MADAR_CONTAMINATION, n_jobs=-1, random_state=SEED).fit(L)
    f_iso_lat = iso_l.decision_function(L)
    Xn = X_sub.numpy()
    iso_r = IsolationForest(contamination=MADAR_CONTAMINATION, n_jobs=-1, random_state=SEED).fit(Xn)
    f_iso_raw = iso_r.decision_function(Xn)
    yn = y_sub.numpy()
    cent_dist = np.zeros(len(yn)); fam_size = np.zeros(len(yn))
    for fam in np.unique(yn):
        m = yn == fam
        c = L[m].mean(0)
        cent_dist[m] = np.linalg.norm(L[m] - c, axis=1)
        fam_size[m] = np.log10(m.sum())
    feats = np.stack([f_iso_lat, f_iso_raw, loss, ent, margin, cent_dist, fam_size], axis=1)
    return _apply_feature_ablation(nn1_scorer.zscore(feats), args.zero_features)

def split_forget_retain(X, y, classes, model, active_count, tid):
    m = torch.isin(y, torch.tensor(classes))
    X_sub, y_sub = X[m], y[m]
    n = len(y_sub)
    n_forget = int(n * args.forget_ratio)
    if n_forget == 0:
        return data.DataLoader(data.TensorDataset(X_sub, y_sub), batch_size=BATCH_SIZE, shuffle=True), None

    if args.selector == 'donut':
        model.eval()
        lats = []
        with torch.no_grad():
            for v, _ in data.DataLoader(data.TensorDataset(X_sub, y_sub), batch_size=BATCH_SIZE):
                _, la = model(v.to(DEVICE), return_latent=True)
                lats.append(la.cpu())
        L = torch.cat(lats).numpy()
        iso = IsolationForest(contamination=MADAR_CONTAMINATION, n_jobs=-1, random_state=SEED).fit(L)
        order = np.argsort(iso.decision_function(L))
        mid = (n // 2) - (n_forget // 2)
        forget_idx = order[mid:mid + n_forget]
        retain_idx = np.concatenate([order[:mid], order[mid + n_forget:]])
    elif args.selector == 'random':
        rng = np.random.default_rng(SEED * 1000 + tid)
        perm = rng.permutation(n)
        forget_idx, retain_idx = perm[:n_forget], perm[n_forget:]
    else:  # nn1
        assert args.scorer_weights, "--scorer_weights required for selector=nn1"
        params, hidden, nf = nn1_scorer.load(args.scorer_weights)
        _trained_zf = nn1_scorer.zero_features_of(args.scorer_weights)
     
        _missing = [f for f in _trained_zf if f not in args.zero_features]
        if _missing:
            raise SystemExit(
                f"ABLATION MISMATCH: {args.scorer_weights} was trained with "
                f"zero_features={_trained_zf} but this run does not zero {_missing}. "
                f"Those weights drifted during ES without affecting anything, so "
                f"activating them now would evaluate a policy that was never trained. "
                f"Pass --zero_features {' '.join(_trained_zf)}.")
        _extra = [f for f in args.zero_features if f not in _trained_zf]
        if _extra and not args.quiet:
            log(f"[ablation] zeroing {_extra} on a scorer trained WITH them active "
                f"-- measuring how much the trained policy relies on them.")
        assert nf == nn1_scorer.N_FEATURES, f"mode a expects {nn1_scorer.N_FEATURES}-feature scorer, got {nf}"
        feats = compute_nn1_features(X_sub, y_sub, model, active_count)
        forget_idx, retain_idx = nn1_scorer.select_forget(params, feats, args.forget_ratio, hidden=hidden)

    fl = data.DataLoader(data.TensorDataset(X_sub[forget_idx], y_sub[forget_idx]),
                         batch_size=BATCH_SIZE, shuffle=True)
    rl = data.DataLoader(data.TensorDataset(X_sub[retain_idx], y_sub[retain_idx]),
                         batch_size=BATCH_SIZE, shuffle=True)
    return rl, fl


# --- Mode B: score OLD (buffer) points against the NEW task's distribution ---
def _last_layer_grads(model, X, y, active_count, max_n=None):
    
    model.eval()
    if max_n is not None and len(X) > max_n:
        idx = np.random.default_rng(SEED).choice(len(X), max_n, replace=False)
        X, y = X[idx], y[idx]
    outs = []
    with torch.no_grad():
        for v, l in data.DataLoader(data.TensorDataset(X, y), batch_size=BATCH_SIZE):
            v = v.to(DEVICE)
            logits, lat = model(v, return_latent=True)
            p = F.softmax(logits[:, :active_count], 1)
            err = p.clone()
            err[torch.arange(len(l)), l.to(DEVICE)] -= 1.0          # (b, active)
            g = torch.einsum('bc,bd->bcd', err, lat).reshape(len(l), -1)  # (b, active*128)
            outs.append(g.cpu())
    return torch.cat(outs).numpy().astype(np.float32)

def compute_buffer_features(model, X_new, y_new, active_count, prev_active):
    """9 features per buffer point, z-scored over the buffer population."""
    b_X = torch.stack([e[0] for e in replay_buffer])
    b_y = torch.tensor([int(e[1]) for e in replay_buffer])
    model.eval()
    lats, logits_all, losses = [], [], []
    with torch.no_grad():
        for v, l in data.DataLoader(data.TensorDataset(b_X, b_y), batch_size=BATCH_SIZE):
            v, l = v.to(DEVICE), l.to(DEVICE)
            lo, la = model(v, return_latent=True)
            lo = lo[:, :active_count]
            lats.append(la.cpu()); logits_all.append(lo.cpu())
            losses.append(F.cross_entropy(lo, l, reduction='none').cpu())
    L = torch.cat(lats).numpy()
    logits = torch.cat(logits_all)
    loss = torch.cat(losses).numpy()
    probs = F.softmax(logits, 1).numpy()
    ent = -(probs * np.log(probs + 1e-9)).sum(1) / np.log(active_count)
    top2 = np.sort(probs, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    iso_l = IsolationForest(contamination=MADAR_CONTAMINATION, n_jobs=-1, random_state=SEED).fit(L)
    f_iso_lat = iso_l.decision_function(L)
    Xn = b_X.numpy()
    iso_r = IsolationForest(contamination=MADAR_CONTAMINATION, n_jobs=-1, random_state=SEED).fit(Xn)
    f_iso_raw = iso_r.decision_function(Xn)
    yn = b_y.numpy()
    cent = np.zeros(len(yn)); fsz = np.zeros(len(yn))
    for fam in np.unique(yn):
        m = yn == fam
        c = L[m].mean(0)
        cent[m] = np.linalg.norm(L[m] - c, axis=1)
        fsz[m] = np.log10(max(m.sum(), 1))
    # grad_conflict: cos( g_i(buffer), mean g(new task batch) )
    G_buf = _last_layer_grads(model, b_X, b_y, active_count)
    G_new = _last_layer_grads(model, X_new, y_new, active_count, max_n=2048)
    g_ref = G_new.mean(0)
    g_ref = g_ref / (np.linalg.norm(g_ref) + 1e-12)
    f_conf = (G_buf @ g_ref) / (np.linalg.norm(G_buf, axis=1) + 1e-12)
    # density_ratio: latent-space logistic discriminator old(0) vs new(1); log-odds of "new"
    with torch.no_grad():
        n_lats = []
        idx = np.random.default_rng(SEED + 1).choice(len(X_new), min(len(X_new), 5000), replace=False)
        for v, _ in data.DataLoader(data.TensorDataset(X_new[idx], y_new[idx]), batch_size=BATCH_SIZE):
            _, la = model(v.to(DEVICE), return_latent=True)
            n_lats.append(la.cpu())
    Ln = torch.cat(n_lats).numpy()
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=300)
    clf.fit(np.vstack([L, Ln]), np.concatenate([np.zeros(len(L)), np.ones(len(Ln))]))
    f_dens = clf.decision_function(L)          # low = unlikely under the new distribution
    feats = np.stack([f_iso_lat, f_iso_raw, loss, ent, margin, cent, fsz, f_conf, f_dens], axis=1)
    return (_apply_feature_ablation(nn1_scorer.zscore(feats), args.zero_features),
            b_X, b_y, f_conf, f_dens)

def select_buffer_forget(model, X_new, y_new, active_count, prev_active, tid):
    feats, b_X, b_y, f_conf, f_dens = compute_buffer_features(model, X_new, y_new, active_count, prev_active)
    n = len(b_y)
    n_forget = int(n * args.forget_ratio)
    if args.selector == 'none' or n_forget == 0:
        return None, None, None
    if args.selector == 'random':
        rng = np.random.default_rng(SEED * 1000 + tid)
        forget_idx = rng.permutation(n)[:n_forget]
    elif args.selector == 'gradconflict':
        forget_idx = np.argsort(f_conf)[:n_forget]          # most negative cosine
    elif args.selector == 'densityratio':
        forget_idx = np.argsort(f_dens)[:n_forget]          # least likely under new dist
    else:                                                    # nn1, 9 features
        params, hidden, nf = nn1_scorer.load(args.scorer_weights)
        _trained_zf = nn1_scorer.zero_features_of(args.scorer_weights)
     
        _missing = [f for f in _trained_zf if f not in args.zero_features]
        if _missing:
            raise SystemExit(
                f"ABLATION MISMATCH: {args.scorer_weights} was trained with "
                f"zero_features={_trained_zf} but this run does not zero {_missing}. "
                f"Those weights drifted during ES without affecting anything, so "
                f"activating them now would evaluate a policy that was never trained. "
                f"Pass --zero_features {' '.join(_trained_zf)}.")
        _extra = [f for f in args.zero_features if f not in _trained_zf]
        if _extra and not args.quiet:
            log(f"[ablation] zeroing {_extra} on a scorer trained WITH them active "
                f"-- measuring how much the trained policy relies on them.")
        assert nf == nn1_scorer.N_FEATURES_B, f"mode b expects {nn1_scorer.N_FEATURES_B}-feature scorer, got {nf}"
        forget_idx, _ = nn1_scorer.select_forget(params, feats, args.forget_ratio, hidden=hidden)
    fmask = np.zeros(n, dtype=bool); fmask[forget_idx] = True
    return b_X, b_y, fmask

def remove_from_buffer(b_y, fmask):
    """Delete the flagged points from family_buffers (they never replay again).
    NOTE: removal breaks the anomaly/inlier interleave used by later trimming;
    trimming degrades to approximately-stratified truncation. Documented caveat."""
    global replay_buffer
    pos = 0
    for fam in list(family_buffers.keys()):
        k = len(family_buffers[fam])
        keep = [e for j, e in enumerate(family_buffers[fam]) if not fmask[pos + j]]
        family_buffers[fam] = keep
        pos += k
    replay_buffer.clear()
    for fd in family_buffers.values(): replay_buffer.extend(fd)

# --- Task 0: train-or-load checkpoint ---
if SCALE_MODE == 'standard':
    scaler = StandardScaler()
    t0_mask = torch.isin(y_train, torch.tensor(task_classes[0])).numpy()
    scaler.fit(X_train_np[t0_mask])
    X_train = torch.tensor(np.clip(scaler.transform(X_train_np), -FEATURE_CLIP, FEATURE_CLIP),
                           dtype=torch.float32)
    X_test = torch.tensor(np.clip(scaler.transform(X_test_np), -FEATURE_CLIP, FEATURE_CLIP),
                          dtype=torch.float32)
else:
    
    log(f"Scaling: none ({args.dataset} features are binary)")
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)

model = EmberNN(INPUT_DIM, NUM_CLASSES).to(DEVICE)

if os.path.exists(args.checkpoint):
    # map_location='cpu' is deliberate: the replay buffer must stay on CPU
    # (new entries from update_buffer_madar are CPU tensors; batches are
    # moved to DEVICE per step). Loading to DEVICE created a mixed-device
    # buffer that crashed torch.stack at task 2. load_state_dict copies CPU
    # tensors into the CUDA model correctly.
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    assert ck['perm_seed'] == args.perm_seed, \
        f"checkpoint perm_seed={ck['perm_seed']} != --perm_seed {args.perm_seed}"
    # Checkpoints written before this field existed are EMBER 2018 by construction.
    _ck_ds = ck.get('dataset', 'ember2018')
    assert _ck_ds == args.dataset, \
        f"checkpoint was built on {_ck_ds} but --dataset is {args.dataset}"
    model.load_state_dict(ck['model'])
    family_buffers.update(ck['family_buffers'])
    for fd in family_buffers.values(): replay_buffer.extend(fd)
    log(f"Loaded task-0 checkpoint {args.checkpoint}")
else:
    log("Training task 0 (no checkpoint found)...")
    loader0 = get_loader(X_train, y_train, task_classes[0], drop_last=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    mask0 = torch.full((NUM_CLASSES,), -1e9).to(DEVICE); mask0[:TASK_1_FAMILIES] = 0.0
    for epoch in range(TASK0_EPOCHS):
        for v, l in loader0:
            v, l = v.to(DEVICE), l.to(DEVICE)
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(model(v) + mask0, l)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss in task 0, epoch {epoch}")
            loss.backward(); optimizer.step()
    update_buffer_madar(loader0, model)
    torch.save({'model': model.state_dict(), 'family_buffers': family_buffers,
                'perm_seed': args.perm_seed, 'dataset': args.dataset,
                'family_cap': args.family_cap}, args.checkpoint)
    log(f"Saved task-0 checkpoint to {args.checkpoint}")

if args.make_checkpoint:
    # Report task-0 quality so a bad checkpoint is caught here rather than
    # after hours of downstream runs built on top of it.
    t0_acc = eval_acc(model, X_test, y_test, task_classes[0], TASK_1_FAMILIES)
    log(f"Task-0 accuracy on held-out test: {t0_acc:.2f}% "
        f"(buffer {len(replay_buffer)} samples over {len(family_buffers)} families)")
    log("Checkpoint ready; exiting (--make_checkpoint).")
    raise SystemExit(0)

# --- CL + unlearning over n_tasks ---
W = {n.replace('.', '__'): torch.zeros_like(p).to(DEVICE)
     for n, p in model.named_parameters() if p.requires_grad}
omega = {k: torch.zeros_like(v) for k, v in W.items()}
p_old = {n.replace('.', '__'): p.detach().clone()
         for n, p in model.named_parameters() if p.requires_grad}
teacher = copy.deepcopy(model); teacher.eval()
prev_active = TASK_1_FAMILIES
per_task_acc = []
# Acquisition vs retention, mirroring the EMBER 2024 two-panel diagnostic:
# new-family accuracy isolates INTRANSIGENCE, task-0 accuracy isolates
# FORGETTING. The aggregate 'seen families' number hides which one moved.
per_task_new = []
per_task_task0 = []
per_task_macro = []

for tid in range(1, args.n_tasks + 1):
    classes = task_classes[tid]
    active = TASK_1_FAMILIES + tid * SUBSEQUENT_FAMILIES
    loader = get_loader(X_train, y_train, classes, drop_last=True)
    if not loader:
        continue
    opt = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-6)
    train_cl_er(model, teacher, opt, loader, args.cl_iters, active, prev_active,
                W, omega, p_old, tid)
    # SI omega update at end of CL (before unlearning moves the weights)
    for n, p in model.named_parameters():
        k = n.replace('.', '__')
        if p.requires_grad:
            omega[k] += W[k] / ((p.detach() - p_old[k])**2 + SI_EPS)
            W[k].zero_()
    # Unlearning with the chosen selector
    if args.mode == 'a':
        retain_loader, forget_loader = split_forget_retain(X_train, y_train, classes, model, active, tid)
        if forget_loader is not None:
            pre = copy.deepcopy(model)
            unlearn(model, pre, forget_loader, retain_loader, active, prev_active,
                    omega, p_old, args.unlearn_epochs, args.alpha)
            loader = retain_loader
    else:
        # MODE B: unlearn OLD buffer points that conflict with the new task.
        Xt, yt = loader.dataset.tensors
        b_X, b_y, fmask = select_buffer_forget(model, Xt, yt, active, prev_active, tid)
        if fmask is not None and fmask.any():
            fam_f, cnt_f = np.unique(b_y.numpy()[fmask], return_counts=True)
            log(f"    [Mode B/{args.selector}] forgetting {int(fmask.sum())} buffer points "
                f"across {len(fam_f)} old families")
            # Remove from the buffer FIRST so the retain/replay stream inside
            # unlearn() no longer contains the points being forgotten.
            remove_from_buffer(b_y, fmask)
            forget_loader = data.DataLoader(
                data.TensorDataset(b_X[fmask], b_y[fmask]), batch_size=BATCH_SIZE, shuffle=True)
            pre = copy.deepcopy(model)
            # Flatten the OLD-knowledge logit slice [0, prev_active) for these points;
            # retain anchor = full current-task loader + cleaned replay buffer.
            unlearn(model, pre, forget_loader, loader, active, prev_active,
                    omega, p_old, args.unlearn_epochs, args.alpha,
                    flat_lo=0, flat_hi=prev_active)
        # current-task data fully retained in mode b: loader unchanged
    # Post-task updates
    for n, p in model.named_parameters():
        if p.requires_grad: p_old[n.replace('.', '__')] = p.detach().clone()
    teacher = copy.deepcopy(model); teacher.eval()
    prev_active = active
    update_buffer_madar(loader, model)

    seen = [c for t in task_classes[:tid + 1] for c in t]
    acc = eval_acc(model, X_test, y_test, seen, active)
    per_task_acc.append(acc)
    acc_new = eval_acc(model, X_test, y_test, classes, active)          # acquisition
    acc_t0 = eval_acc(model, X_test, y_test, task_classes[0], active)   # retention
    acc_macro = eval_acc_macro(model, X_test, y_test, seen, active)
    per_task_new.append(acc_new)
    per_task_task0.append(acc_t0)
    per_task_macro.append(acc_macro)
    log(f"Task {tid}: Avg {acc:.2f}% (macro {acc_macro:.2f}%) | "
        f"new {acc_new:.2f}% | task-0 {acc_t0:.2f}%")

reward = {
    'mean_acc': float(np.mean(per_task_acc)),
    'final_acc': float(per_task_acc[-1]),
    'per_task': [float(a) for a in per_task_acc],
    # Added for the acquisition/retention diagnostic. Runs produced before this
    # change lack these keys; downstream tooling must treat them as optional.
    'per_task_new_fam': [float(a) for a in per_task_new],
    'per_task_task0': [float(a) for a in per_task_task0],
    'per_task_macro': [float(a) for a in per_task_macro],
    # run identity -- lets a driver detect a stale/mismatched cached result
    # instead of silently mixing corpora or task counts into one comparison.
    'config': {'dataset': args.dataset, 'mode': args.mode, 'selector': args.selector,
               'n_tasks': args.n_tasks, 'cl_iters': args.cl_iters,
               'mem_size': args.mem_size, 'si_c': args.si_c, 'alpha': args.alpha,
               'forget_ratio': args.forget_ratio, 'unlearn_epochs': args.unlearn_epochs,
               'seed': args.seed, 'perm_seed': args.perm_seed,
               'zero_features': list(args.zero_features),
               'scorer_weights': args.scorer_weights},
    'selector': args.selector, 'seed': SEED, 'perm_seed': args.perm_seed,
    'n_tasks': args.n_tasks, 'cl_iters': args.cl_iters,
    'forget_ratio': args.forget_ratio, 'alpha': args.alpha,
}
log(f"Mean acc over tasks 1..{args.n_tasks}: {reward['mean_acc']:.2f}% (final {reward['final_acc']:.2f}%)")
if args.out:
    with open(args.out, 'w') as f:
        json.dump(reward, f, indent=2)