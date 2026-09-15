"""EMBER 2024 paper compiler.

Discovery stays here -- 2024 tags every history file with cap/arch/iters/si and
several protocol variants coexist on disk, so the compiler must filter to one
operating point before anything is averaged. Rendering is delegated to
paper_figs.py, shared with the 2018 and LAMDA compilers.

Produces, all named {dataset}_{category}_{content}:

  ember2024_stage1_accuracy.png       + .csv   naive / joint / MADAR / MADAR+Unlearning
  ember2024_stage1_accuracy_zoom.png          same, naive omitted
  ember2024_stage1_decomposition.png  + .csv   families just added, and task-0
  ember2024_meta_accuracy.png         + .csv   every selector + paired differences
  ember2024_meta_decomposition.png    + .csv   families just added, and task-0
  ember2024_results.md                         every figure with its table

Usage:
    python compile_all_2024.py
    python compile_all_2024.py --root ~/Thesis --cap 10000 --arch mlp --iters 8000 --si 100.0
"""

import argparse
import glob
import json
import os

import paper_figs as pf

DATASET = 'ember2024'
TITLE = 'EMBER 2024 Class-IL'

p = argparse.ArgumentParser(description="Compile EMBER 2024 results for the paper")
p.add_argument('--root', default='.')
p.add_argument('--out-dir', default='.')
p.add_argument('--cap', type=int, default=10000, help='Per-family training cap')
p.add_argument('--arch', default='mlp', help='Backbone')
p.add_argument('--iters', type=int, default=8000, help='CL iterations (madar/madaru)')
p.add_argument('--si', type=float, default=100.0, help='SI_C (madar/madaru)')
p.add_argument('--meta-dirs', nargs='+', default=None,
               help="Meta workdirs to pool. Default: the mode-B transfer sweep and the "
                    "mode-A sweep at the same alpha, so the two modes appear in one "
                    "figure and one table without mixing operating points.")
p.add_argument('--meta-seeds', type=int, nargs='+', default=None)
p.add_argument('--meta-metric', choices=['micro', 'macro'], default='micro')
p.add_argument('--meta-reference', default='none')
args = p.parse_args()

R = args.root

# compare_2024_a is alpha=0.0 like the mode-B sweeps; compare_2024_a_a0.2 is the
# alpha=0.2 arm. Defaulting to the matched pair keeps the pooled figure at one
# operating point instead of silently mixing two.
DEFAULT_META_DIRS = ['compare_2024_transfer', 'compare_2024_a']


def parse_tag(tag):
    """cap10000_mlp_b_a0.2_r0.1_seed1_it8000_si100.0 -> dict.

    Kept identical to the original compiler so the protocol filter selects the
    same files it always did.
    """
    d = dict(cap=None, arch=None, split=None, alpha=None, ratio=None,
             seed=None, iters=None, si=None, wm=None)
    for tok in tag.split('_'):
        if not tok:
            continue
        if tok.startswith('cap'):        d['cap'] = int(tok[3:])
        elif tok in ('mlp', 'resnet'):   d['arch'] = tok
        elif tok in ('a', 'b'):          d['split'] = tok
        elif tok.startswith('seed'):     d['seed'] = int(tok[4:])
        elif tok.startswith('it'):       d['iters'] = int(tok[2:])
        elif tok.startswith('si'):       d['si'] = float(tok[2:])
        elif tok.startswith('wm'):       d['wm'] = float(tok[2:])
        elif tok.startswith('a'):        d['alpha'] = float(tok[1:])
        elif tok.startswith('r'):        d['ratio'] = float(tok[1:])
    if d['iters'] is None:
        d['iters'] = 2000               # the run scripts omit tokens at default
    if d['si'] is None:
        d['si'] = 2.0
    return d


def in_protocol(cond, meta):
    if meta['cap'] != args.cap:
        return False
    if cond in ('naive', 'joint'):
        return True                     # epoch-based: no iters/si to match
    return (meta['arch'] == args.arch and meta['iters'] == args.iters
            and abs(meta['si'] - args.si) < 1e-9)


# ---------------------------------------------------------------- discovery
# 'madaru' is this corpus's name for MADAR + unlearning.
COND_MAP = {'naive': 'naive', 'joint': 'joint', 'madar': 'madar', 'madaru': 'mu'}

records = {k: [] for k in ('naive', 'joint', 'madar', 'mu')}
alphas = {}
skipped = 0
for path in sorted(glob.glob(os.path.join(R, 'ember24_*_history_*.json'))):
    base = os.path.basename(path)[:-5]
    try:
        raw, tag = base.split('_history_')
        cond = COND_MAP.get(raw.replace('ember24_', ''))
        meta = parse_tag(tag)
        with open(path) as f:
            hist = json.load(f)
    except Exception as e:
        print(f"  ! {base}: {e}")
        continue
    if cond is None or not hist.get('Avg_Acc'):
        continue
    if not in_protocol(raw.replace('ember24_', ''), meta):
        skipped += 1
        continue
    records[cond].append((path, hist))
    alphas[path] = meta['alpha']

for c, v in records.items():
    print(f"  stage-1 {c:<6} {len(v):>3} run(s)")
print(f"  ({skipped} file(s) outside the requested protocol "
      f"cap={args.cap} arch={args.arch} iters={args.iters} si={args.si})")

stage = pf.stage_from_records(records, alpha_of=lambda path, rec: alphas.get(path))
mu_runs, mu_alpha, mu_note = pf.pick_mu_arm(stage, reference='madar')
if mu_note:
    print(f"  {mu_note}")

s1 = pf.stage1_figures(stage, DATASET, TITLE, out_dir=args.out_dir,
                       mu_runs=mu_runs, mu_note=mu_note)

# ---------------------------------------------------------------- meta
meta_dirs = [os.path.join(R, d) for d in (args.meta_dirs or DEFAULT_META_DIRS)]
meta_dirs = [d for d in meta_dirs if os.path.isdir(d)]
if not meta_dirs:
    print(f"  ! no meta workdirs found ({args.meta_dirs or DEFAULT_META_DIRS})")
    mres = {'figures': {}, 'tables': {}, 'notes': ['no meta workdirs found']}
else:
    data, info = pf.load_meta(meta_dirs, seeds=args.meta_seeds)
    for wd in meta_dirs:
        sels = sorted({s for w, s in data if w == wd})
        c = info.get(wd) or {}
        print(f"  meta {os.path.basename(wd):<26} mode={c.get('mode', '?')} "
              f"alpha={c.get('alpha')} epochs={c.get('unlearn_epochs')} {sels}")
    mres = pf.meta_figures(data, info, DATASET,
                           f"{TITLE} \u2014 learned forget-set selection",
                           out_dir=args.out_dir, metric=args.meta_metric,
                           reference=args.meta_reference)

# ---------------------------------------------------------------- markdown
notes = [f"Protocol: cap={args.cap}, arch={args.arch}, cl_iters={args.iters}, "
         f"si_c={args.si}. Files outside it are excluded, not averaged in."]
if mu_alpha is not None:
    notes.append(f"MADAR + Unlearning is the alpha={mu_alpha} arm, chosen as the better "
                 f"of the two on mean accuracy; both arms' scores are in the stage-1 "
                 f"note below.")

md = pf.markdown(DATASET, "EMBER 2024 \u2014 results",
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