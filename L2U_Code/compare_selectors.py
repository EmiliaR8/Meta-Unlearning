"""
Multi-seed comparison of forget-set selectors under an identical config.

Runs donut, random, and nn1 (a trained scorer) across a set of seeds in
parallel over your GPUs, then prints mean +/- std and per-seed paired
differences (nn1 - donut, nn1 - random). This is the statistical check the
ES log itself cannot provide: the ES baselines are evaluated at one seed
while theta moves across seeds, so their gap needs same-seed pairing.

Usage:
    python compare_selectors.py --gpus 0 1 2 \
        --scorer_weights es_work_mini/theta_final.npz \
        --seeds 42 43 44 45 46 47 48 49 50 51 \
        --n_tasks 5 --cl_iters 500 --unlearn_epochs 1
"""

import argparse
import json
import os
import queue
import subprocess
import sys
import threading

import numpy as np

parser = argparse.ArgumentParser(description="Multi-seed selector comparison")
parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2])
parser.add_argument('--mode', type=str, default='a', choices=['a', 'b'])
parser.add_argument('--scorer_weights', type=str, default=None,
                    help='NN-1 weights .npz. Required only if the selector list '
                         "includes 'nn1'; not needed for a baselines-only run "
                         '(e.g. --selectors donut random).')
parser.add_argument('--seeds', type=int, nargs='+', default=list(range(42, 52)))
parser.add_argument('--n_tasks', type=int, default=5)
parser.add_argument('--cl_iters', type=int, default=None,
                    help='Default: the inner loop dataset default (2000 / 8000)')
parser.add_argument('--unlearn_epochs', type=int, default=1)
parser.add_argument('--forget_ratio', type=float, default=0.10)
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--dataset', type=str, default='ember2018',
                    choices=['ember2018', 'ember2024', 'lamda_classil'],
                    help='Corpus for the inner loop. Also selects the protocol defaults '
                         '(memory size, SI strength, CL iterations) unless overridden.')
parser.add_argument('--cache_data', type=str, default=None,
                    help='Default: <dataset>_cache.npz')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='Default: task0_<dataset>.pt')
parser.add_argument('--perm_seed', type=int, default=None)
parser.add_argument('--treatment', type=str, default=None,
                    help="selector used as the TREATMENT in paired differences. "
                         "Default: 'nn1' if present, else the first selector. Needed "
                         'because a baselines-only run (e.g. --selectors donut random) '
                         'has no nn1 to difference against.')
parser.add_argument('--selectors', type=str, nargs='+', default=None,
                    help='override the selector list for this mode. e.g. '
                         '--selectors donut random  (skips nn1, saving one run per seed)')
parser.add_argument('--zero_features', type=str, nargs='*', default=[],
                    help='ABLATION: pass through to mu_inner_loop --zero_features. '
                         'Zeroes those NN-1 feature columns so they cannot contribute '
                         'to the learned score.')
parser.add_argument('--workdir', type=str, default=None,
                    help='default: compare_work_<dataset>_mode<mode>. The old shared '
                         '"compare_work" caused cross-corpus contamination: cache files '
                         'were named {selector}_s{seed}.json with no dataset/mode/n_tasks, '
                         'so an EMBER run at seed 42 was silently reused for LAMDA.')
args = parser.parse_args()

_LEGACY_PATHS = {   # keep the existing EMBER 2018 filenames so prior runs stay valid
    'ember2018': ('ember18_cache.npz', 'task0_ckpt.pt'),
    'ember2024': ('ember2024_cache.npz', 'task0_ember2024.pt'),
    'lamda_classil': ('lamda_classil_meta_cache.npz', 'task0_lamda_classil.pt'),
}
if args.cache_data is None:
    args.cache_data = _LEGACY_PATHS[args.dataset][0]
if args.checkpoint is None:
    args.checkpoint = _LEGACY_PATHS[args.dataset][1]

if args.workdir is None:
    args.workdir = f'compare_work_{args.dataset}_mode{args.mode}'
os.makedirs(args.workdir, exist_ok=True)
_DEFAULT_SELECTORS = (['donut', 'random', 'nn1'] if args.mode == 'a'
                      else ['none', 'random', 'gradconflict', 'densityratio', 'nn1'])
SELECTORS = args.selectors or _DEFAULT_SELECTORS
_valid = set(_DEFAULT_SELECTORS)
bad = [x for x in SELECTORS if x not in _valid]
if bad:
    raise SystemExit(f"selector(s) {bad} are not valid for mode '{args.mode}'; "
                     f"valid: {sorted(_valid)}")
if 'nn1' in SELECTORS and not args.scorer_weights:
    raise SystemExit("--scorer_weights is required when 'nn1' is in the selector list. "
                     "Either pass it, or drop nn1 (e.g. --selectors donut random).")
if args.scorer_weights and not os.path.exists(args.scorer_weights):
    raise SystemExit(f"--scorer_weights not found: {args.scorer_weights}")

gpu_q = queue.Queue()
for g in args.gpus:
    gpu_q.put(g)

def run_one(selector, seed, out_path):
    gpu = gpu_q.get()
    try:
        cmd = [sys.executable, 'mu_inner_loop.py', '--mode', args.mode,
               '--selector', selector, '--seed', str(seed), '--out', out_path,
               '--cache_data', args.cache_data, '--checkpoint', args.checkpoint,
               '--dataset', args.dataset,
               '--n_tasks', str(args.n_tasks),
               '--unlearn_epochs', str(args.unlearn_epochs),
               '--forget_ratio', str(args.forget_ratio), '--alpha', str(args.alpha),
               '--quiet']
        if args.zero_features:
            cmd += ['--zero_features'] + list(args.zero_features)
        if args.cl_iters is not None:
            cmd += ['--cl_iters', str(args.cl_iters)]
        if args.perm_seed is not None:
            cmd += ['--perm_seed', str(args.perm_seed)]
        if selector == 'nn1':
            cmd += ['--scorer_weights', args.scorer_weights]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        res = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stdout[-2000:], res.stderr[-2000:], file=sys.stderr)
            raise RuntimeError(f"failed: {selector} seed {seed}")
    finally:
        gpu_q.put(gpu)

jobs, threads = [], []
for sel in SELECTORS:
    for seed in args.seeds:
        out = (f'{args.workdir}/{sel}_s{seed}_t{args.n_tasks}'
               f'_a{args.alpha:g}_r{args.forget_ratio:g}.json')
        if not os.path.exists(out):          # resumable: skip already-done runs
            jobs.append((sel, seed, out))
print(f"Running {len(jobs)} inner loops over GPUs {args.gpus} "
      f"({len(SELECTORS)} selectors x {len(args.seeds)} seeds, cached runs skipped)...")
for job in jobs:
    t = threading.Thread(target=run_one, args=job); t.start(); threads.append(t)
for t in threads:
    t.join()

# --- Aggregate (per-task, so train/validation task splits can be scored) ---
P = {}
for sel in SELECTORS:
    rows = []
    for s in args.seeds:
        fp = (f'{args.workdir}/{sel}_s{s}_t{args.n_tasks}'
              f'_a{args.alpha:g}_r{args.forget_ratio:g}.json')
        d = json.load(open(fp))
        cfg = d.get('config')
        if cfg:   # written by mu_inner_loop; older files have none
            for k, want in (('dataset', args.dataset), ('mode', args.mode),
                            ('n_tasks', args.n_tasks), ('selector', sel)):
                if cfg.get(k) != want:
                    raise SystemExit(
                        f"STALE CACHE: {fp} has {k}={cfg.get(k)!r} but this run wants "
                        f"{want!r}. Delete it (or use a fresh --workdir) and re-run.")
        if len(d['per_task']) != args.n_tasks:
            raise SystemExit(
                f"STALE CACHE: {fp} has {len(d['per_task'])} tasks, expected "
                f"{args.n_tasks}. Delete it (or use a fresh --workdir) and re-run.")
        rows.append(d['per_task'])
    P[sel] = np.array(rows)                            # (n_seeds, n_tasks)
# Treatment selector for the paired differences. The original code hardcoded
# 'nn1'; that breaks a baselines-only run, and silently assumes nn1 is always
# the thing under test.
TREAT = args.treatment or ('nn1' if 'nn1' in P else SELECTORS[0])
if TREAT not in P:
    raise SystemExit(f"--treatment {TREAT!r} is not among the loaded selectors {sorted(P)}")
n_tasks = P[TREAT].shape[1]
half = min(5, n_tasks)
if len(P) < 2:
    print(f"\n(only one selector loaded: {sorted(P)} - no paired differences to report)")
print(f"\nTreatment selector for paired differences: {TREAT}")                                 # tasks 1..5 = meta-training horizon

def report(name, sl):
    print(f"\n=== {name}, {len(args.seeds)} seeds ===")
    for sel in [x for x in SELECTORS if x in P]:
        v = P[sel][:, sl].mean(axis=1)
        print(f"  {sel:8s} {v.mean():.3f} +/- {v.std():.3f}")
    for base in [x for x in SELECTORS if x != TREAT and x in P]:
        d = P[TREAT][:, sl].mean(axis=1) - P[base][:, sl].mean(axis=1)
        per_seed = ', '.join(f"{s}: {v:+.2f}" for s, v in zip(args.seeds, d))
        print(f"  {TREAT} - {base}: mean {d.mean():+.3f} +/- {d.std():.3f} | "
              f"wins {(d > 0).sum()}/{len(d)} | [{per_seed}]")

report(f"ALL tasks 1..{n_tasks} (mean acc)", slice(0, n_tasks))
if n_tasks > half:
    report(f"TRAIN-HORIZON tasks 1..{half}", slice(0, half))
    report(f"HELD-OUT tasks {half+1}..{n_tasks} (validation)", slice(half, n_tasks))