"""EMBER 2018 paper compiler.

Discovery stays here (2018 has its own filename conventions); every figure and
table is rendered by paper_figs.py so the three datasets cannot drift.

Produces, all named {dataset}_{category}_{content}:

  ember2018_stage1_accuracy.png       + .csv   naive / joint / MADAR / MADAR+Unlearning
  ember2018_stage1_accuracy_zoom.png          same, naive omitted
  ember2018_stage1_decomposition.png  + .csv   families just added, and task-0
  ember2018_meta_accuracy.png         + .csv   every selector + paired differences
  ember2018_meta_decomposition.png    + .csv   families just added, and task-0
  ember2018_results.md                         every figure with its table

Only the better alpha arm is reported for MADAR+Unlearning, with no alpha in the
label; which arm won, and by how much, is printed and written into the markdown
so the choice is visible rather than silent.

Usage:
    python compile_all_2018.py
    python compile_all_2018.py --root ~/Thesis --out-dir paper_figs
"""

import argparse
import glob
import json
import os
import re

import paper_figs as pf

DATASET = 'ember2018'
TITLE = 'EMBER 2018 Class-IL'

p = argparse.ArgumentParser(description="Compile EMBER 2018 results for the paper")
p.add_argument('--root', default='.')
p.add_argument('--out-dir', default='.')
p.add_argument('--stage1-seeds', type=int, nargs='+', default=None,
               help='Default: every seed found')
p.add_argument('--meta-seeds', type=int, nargs='+', default=None)
p.add_argument('--meta-dirs', nargs='+', default=None,
               help="Meta workdirs to pool. Default: the mode-A and mode-B full-config "
                    "sweeps. Mode A and mode B selectors are pooled into one figure "
                    "and one table.")
p.add_argument('--meta-metric', choices=['micro', 'macro'], default='micro')
p.add_argument('--meta-reference', default='none',
               help='Selector used as the paired-difference reference')
args = p.parse_args()

R = args.root

# ---------------------------------------------------------------- discovery
# Stage-1 filenames are fixed for this corpus. MU files carry the alpha in the
# name, which is how the two arms are separated before one is chosen.
STAGE1_PATTERNS = {
    'naive': ['ember_naive_history_seed*.json'],
    'joint': ['ember_joint_history_seed*.json'],
    'madar': ['ember_madar_history_seed*.json'],
    'mu':    ['ember_mu_history_optB_a*_r*_seed*.json'],
}

DEFAULT_META_DIRS = ['compare_a_full', 'compare_b_fullES']


def alpha_of(path, rec):
    c = rec.get('config') or {}
    if 'alpha' in c:
        return c['alpha']
    m = re.search(r'_a(\d+(?:\.\d+)?)_', os.path.basename(path))
    return float(m.group(1)) if m else None


def seed_of(path, rec):
    c = rec.get('config') or {}
    if 'seed' in c:
        return c['seed']
    m = re.search(r'seed(\d+)', os.path.basename(path))
    return int(m.group(1)) if m else None


records = {}
for cond, pats in STAGE1_PATTERNS.items():
    found = []
    for pat in pats:
        for path in sorted(glob.glob(os.path.join(R, pat))):
            try:
                with open(path) as f:
                    rec = json.load(f)
            except Exception as e:
                print(f"  ! {os.path.basename(path)}: {e}")
                continue
            if not rec.get('Avg_Acc'):
                continue
            if args.stage1_seeds and seed_of(path, rec) not in args.stage1_seeds:
                continue
            found.append((path, rec))
    records[cond] = found
    print(f"  stage-1 {cond:<6} {len(found):>3} run(s)")

stage = pf.stage_from_records(records, alpha_of=alpha_of)
mu_runs, mu_alpha, mu_note = pf.pick_mu_arm(stage, reference='madar')
if mu_note:
    print(f"  {mu_note}")

s1 = pf.stage1_figures(stage, DATASET, TITLE, out_dir=args.out_dir,
                       mu_runs=mu_runs, mu_note=mu_note)

# ---------------------------------------------------------------- meta
meta_dirs = [os.path.join(R, d) for d in (args.meta_dirs or DEFAULT_META_DIRS)]
meta_dirs = [d for d in meta_dirs if os.path.isdir(d)]
if not meta_dirs:
    print(f"  ! no meta workdirs found (looked for "
          f"{args.meta_dirs or DEFAULT_META_DIRS})")
    mres = {'figures': {}, 'tables': {}, 'notes': ['no meta workdirs found']}
else:
    data, info = pf.load_meta(meta_dirs, seeds=args.meta_seeds)
    for wd in meta_dirs:
        sels = sorted({s for w, s in data if w == wd})
        c = info.get(wd) or {}
        print(f"  meta {os.path.basename(wd):<22} mode={c.get('mode', '?')} "
              f"alpha={c.get('alpha')} epochs={c.get('unlearn_epochs')} {sels}")
    mres = pf.meta_figures(data, info, DATASET,
                           f"{TITLE} \u2014 learned forget-set selection",
                           out_dir=args.out_dir, metric=args.meta_metric,
                           reference=args.meta_reference)

# ---------------------------------------------------------------- markdown
notes = []
if mu_alpha is not None:
    notes.append(f"MADAR + Unlearning is the alpha={mu_alpha} arm, selected as the "
                 f"better of the two on mean accuracy. Selecting the arm on its own "
                 f"outcome is a researcher degree of freedom; both arms' scores are "
                 f"given in the stage-1 note below.")
notes.append("Stage-1 runs retrain task 0 per seed and are plotted from task 0. "
             "Meta runs share one task-0 checkpoint and vary only the CL-phase seed, "
             "so they start at task 1. The two are not seed-paired with each other.")

md = pf.markdown(DATASET, "EMBER 2018 \u2014 results",
                 [("Stage 1", s1), ("Meta: forget-set selection", mres)],
                 out_dir=args.out_dir, extra_notes=notes)

print(f"\nWrote {md}")
for res in (s1, mres):
    for fp in res.get('figures', {}).values():
        print(f"Wrote {fp}")
    for t in res.get('tables', {}):
        print(f"Wrote {os.path.join(args.out_dir, f'{DATASET}_{t}.csv')}")
for res in (s1, mres):
    for n in res.get('notes', []):
        print(f"  note: {n}")