"""

MINI run (quick signal, ~1-2 h on 2-4 GPUs):
    python es_meta_train.py --gpus 0 1 --pop 8 --generations 10 \
        --n_tasks 5 --cl_iters 500 --unlearn_epochs 1 --tag mini

FULL run (after the mini looks sane):
    python es_meta_train.py --gpus 0 1 2 3 --pop 16 --generations 40 \
        --n_tasks 10 --cl_iters 2000 --unlearn_epochs 3 --tag full

Prerequisites (one-time):
    python mu_inner_loop.py --cache_data ember18_cache.npz \
        --checkpoint task0_ckpt.pt --make_checkpoint
"""

import argparse
import json
import os
import queue
import subprocess
import sys
import threading

import numpy as np

import nn1_scorer

parser = argparse.ArgumentParser(description="ES meta-training for NN-1 forget-set selection")
parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='GPU ids to spread candidates over')
parser.add_argument('--pop', type=int, default=8, help='Population size (must be even; mirrored pairs)')
parser.add_argument('--generations', type=int, default=10)
parser.add_argument('--sigma', type=float, default=0.1, help='Perturbation std')
parser.add_argument('--lr', type=float, default=0.05, help='ES learning rate')
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
parser.add_argument('--inner_seed', type=int, default=42, help='Base inner-loop seed (rotates per generation)')
parser.add_argument('--tag', type=str, default='es')
parser.add_argument('--mode', type=str, default='a', choices=['a', 'b'],
                    help='a: current-task selection (7 features). b: buffer-vs-new-task selection (9 features).')
parser.add_argument('--hidden', type=int, default=0,
                    help='NN-1 hidden units. 0 = linear (8 params, recommended for small budgets)')
parser.add_argument('--resume', type=str, default=None, help='npz with key "params" to resume from')
parser.add_argument('--zero_features', type=str, nargs='*', default=[],
                    help='ABLATION: pass through to mu_inner_loop --zero_features. '
                         'Zeroes those NN-1 feature columns so they cannot contribute '
                         'to the learned score.')
parser.add_argument('--skip_baselines', action='store_true')
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

assert args.pop % 2 == 0, "--pop must be even (mirrored sampling)"
WORK = f'es_work_{args.tag}'
os.makedirs(WORK, exist_ok=True)

# --- Subprocess evaluation with a GPU queue ---
gpu_q = queue.Queue()
for g in args.gpus:
    gpu_q.put(g)

def run_inner(selector, out_path, seed, scorer_path=None):
    # NOTE: `selector` and `seed` ARE parameters here, so the failure
    # message below is well-scoped.
    gpu = gpu_q.get()
    try:
        cmd = [sys.executable, 'mu_inner_loop.py', '--mode', args.mode,
               '--selector', selector, '--out', out_path, '--seed', str(seed),
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
        if scorer_path:
            cmd += ['--scorer_weights', scorer_path]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        res = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stdout[-2000:], res.stderr[-2000:], file=sys.stderr)
            raise RuntimeError(f"inner loop failed ({selector}, seed {seed}, out={out_path})")
        with open(out_path) as f:
            return json.load(f)['mean_acc']
    finally:
        gpu_q.put(gpu)

def run_parallel(jobs):
    """jobs: list of (selector, out_path, seed, scorer_path). Returns rewards in order."""
    rewards = [None] * len(jobs)
    threads = []
    def work(i, job):
        rewards[i] = run_inner(*job)
    for i, job in enumerate(jobs):
        t = threading.Thread(target=work, args=(i, job)); t.start(); threads.append(t)
    for t in threads: t.join()
    return rewards

# --- Baselines under the SAME mini config (crucial for fair comparison) ---
BASELINE_SELECTORS = ['donut', 'random'] if args.mode == 'a' else \
                     ['none', 'random', 'gradconflict', 'densityratio']
baselines = {}
if not args.skip_baselines:
    print(f"Evaluating baselines {BASELINE_SELECTORS} under the current config...")
    jobs = [(b, f'{WORK}/base_{b}.json', args.inner_seed, None) for b in BASELINE_SELECTORS]
    r = run_parallel(jobs)
    baselines = dict(zip(BASELINE_SELECTORS, r))
    for k, v in baselines.items():
        print(f"  {k:14s} {v:.2f}%")
    with open(f'{WORK}/baselines.json', 'w') as f:
        json.dump(baselines, f, indent=2)

# --- ES main loop ---
N_FEAT = nn1_scorer.N_FEATURES if args.mode == 'a' else nn1_scorer.N_FEATURES_B
if args.resume:
    theta, resumed_hidden, resumed_nf = nn1_scorer.load(args.resume)
    assert resumed_hidden == args.hidden, "resume file architecture != --hidden"
    if resumed_nf != N_FEAT:
        # warm-start across feature sets: keep shared weights, zero-init new features
        assert args.hidden == 0 and resumed_nf < N_FEAT, "can only grow a linear scorer"
        grown = np.zeros(N_FEAT + 1)
        grown[:resumed_nf] = theta[:resumed_nf]          # shared feature weights
        grown[-1] = theta[-1]                            # bias
        theta = grown
        print(f"Warm-start: grew scorer {resumed_nf} -> {N_FEAT} features (new weights start at 0)")
else:
    theta = nn1_scorer.init_params(hidden=args.hidden, seed=0, scale=0.3, n_features=N_FEAT)
D = nn1_scorer.n_params(args.hidden, N_FEAT)
rng = np.random.default_rng(123)
# Full run provenance next to the log, so an ablation is identifiable from the
# artifacts rather than only from the directory name someone chose.
with open(f'{WORK}/run_config.json', 'w') as f:
    json.dump({'tag': args.tag, 'dataset': args.dataset, 'mode': args.mode,
               'zero_features': list(args.zero_features),
               'n_features': N_FEAT, 'hidden': args.hidden,
               'pop': args.pop, 'generations': args.generations,
               'sigma': args.sigma, 'lr': args.lr,
               'n_tasks': args.n_tasks, 'cl_iters': args.cl_iters,
               'unlearn_epochs': args.unlearn_epochs, 'alpha': args.alpha,
               'forget_ratio': args.forget_ratio, 'inner_seed': args.inner_seed,
               'resume': args.resume, 'baseline_selectors': BASELINE_SELECTORS},
              f, indent=2)

log_path = f'{WORK}/es_log.csv'
_zf = '+'.join(args.zero_features) if args.zero_features else 'none'
with open(log_path, 'a') as f:
    f.write('generation,theta_reward,pop_mean,pop_max,sigma,lr,zero_features\n')

for gen in range(args.generations):
    gen_seed = args.inner_seed + gen          # common random numbers within a generation
    eps = rng.standard_normal((args.pop // 2, D))
    eps = np.concatenate([eps, -eps])          # mirrored pairs

    jobs = []
    for i in range(args.pop):
        cand = theta + args.sigma * eps[i]
        cpath = f'{WORK}/cand_g{gen}_i{i}.npz'
        nn1_scorer.save(cpath, cand, hidden=args.hidden, n_features=N_FEAT,
                        zero_features=args.zero_features)
        jobs.append(('nn1', f'{WORK}/r_g{gen}_i{i}.json', gen_seed, cpath))
    rewards = np.array(run_parallel(jobs), dtype=float)

    # Rank-shaped fitness (robust to reward scale/outliers)
    ranks = np.empty(args.pop); ranks[np.argsort(rewards)] = np.arange(args.pop)
    shaped = (ranks / (args.pop - 1)) - 0.5
    grad = (shaped[:, None] * eps).mean(axis=0) / args.sigma
    theta = theta + args.lr * grad

    # Evaluate current theta itself (same seed) for a clean learning curve
    tpath = f'{WORK}/theta_g{gen}.npz'
    nn1_scorer.save(tpath, theta, hidden=args.hidden, n_features=N_FEAT,
                    zero_features=args.zero_features)
    theta_reward = run_inner('nn1', f'{WORK}/r_theta_g{gen}.json', gen_seed, tpath)

    with open(log_path, 'a') as f:
        f.write(f'{gen},{theta_reward:.4f},{rewards.mean():.4f},{rewards.max():.4f},{args.sigma},{args.lr},{_zf}\n')
    base_str = ' | '.join(f'{k} {v:.2f}' for k, v in baselines.items())
    print(f"Gen {gen:3d} | theta {theta_reward:.2f}% | pop mean {rewards.mean():.2f}% "
          f"max {rewards.max():.2f}% | baselines: {base_str}")

nn1_scorer.save(f'{WORK}/theta_final.npz', theta, hidden=args.hidden,
                n_features=N_FEAT, zero_features=args.zero_features)
if args.hidden == 0:
    names = nn1_scorer.FEATURE_NAMES if args.mode == 'a' else nn1_scorer.FEATURE_NAMES_B
    print('Learned linear feature weights (positive => more likely forgotten):')
    for name, w in zip(names, theta[:N_FEAT]):
        print(f'    {name:18s} {w:+.3f}')
    print(f'    {"bias":18s} {theta[N_FEAT]:+.3f}')
print(f"\nDone. Final NN-1 weights: {WORK}/theta_final.npz | log: {log_path}")
print("Evaluate on held-out config with, e.g.:")
print(f"  python mu_inner_loop.py --selector nn1 --scorer_weights {WORK}/theta_final.npz "
      f"--n_tasks 10 --cl_iters 2000 --unlearn_epochs 3 --seed 7 --out eval_full.json")