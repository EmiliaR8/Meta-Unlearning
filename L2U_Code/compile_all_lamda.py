"""LAMDA paper compiler.

LAMDA differs from the other two corpora in one way that shapes this file: stage 1
was run at MORE THAN ONE CL_ITERS budget, and the paper reports both. Each budget
therefore gets its own complete set of figures and tables, tagged lamda_it{N}, so
nothing from two training budgets is ever averaged into one line.

Naive and joint train by EPOCHS_PER_TASK and carry no _it tag, so the same runs
serve every budget. They are re-used rather than duplicated, and the markdown
says so.

Produces, per budget N:

  lamda_it{N}_stage1_accuracy.png       + .csv
  lamda_it{N}_stage1_accuracy_zoom.png
  lamda_it{N}_stage1_decomposition.png  + .csv

and once for the meta runs (a single budget on disk):

  lamda_meta_accuracy.png               + .csv
  lamda_meta_decomposition.png          + .csv
  lamda_results.md                       every figure with its table

Usage:
    python compile_all_lamda.py
    python compile_all_lamda.py --root ~/Thesis --iters 4000 8000
"""

import argparse
import glob
import json
import os
import re

import paper_figs as pf

TITLE = 'LAMDA Class-IL'

p = argparse.ArgumentParser(description="Compile LAMDA results for the paper")
p.add_argument('--root', default='.')
p.add_argument('--out-dir', default='.')
p.add_argument('--iters', type=int, nargs='+', default=None,
               help='CL_ITERS budgets to report. Default: every budget found on disk.')
p.add_argument('--default-iters', type=int, default=4000,
               help='Budget that untagged filenames correspond to. The run scripts omit '
                    'the _it tag at the protocol default.')
p.add_argument('--stage1-seeds', type=int, nargs='+', default=None)
p.add_argument('--meta-dirs', nargs='+', default=None)
p.add_argument('--meta-seeds', type=int, nargs='+', default=None)
p.add_argument('--meta-metric', choices=['micro', 'macro'], default='macro',
               help="LAMDA's test set is heavily imbalanced, so macro (mean per-family "
                    "recall) is the default here, unlike the EMBER corpora.")
p.add_argument('--meta-reference', default='none')
args = p.parse_args()

R = args.root
DEFAULT_META_DIRS = ['compare_work_lamda_classil_modeb',
                     'compare_work_lamda_classil_modea']

# ---------------------------------------------------------------- discovery
# naive/joint are epoch-based: no _it tag, and the same files serve every budget.
UNTAGGED = {'naive', 'joint'}
PATTERNS = {
    'naive': 'lamda_classil_naive*_history_seed*.json',
    'joint': 'lamda_classil_joint*_history_seed*.json',
    'madar': 'lamda_classil_madar*_history_seed*.json',
    'mu':    'lamda_classil_mu_optB_a*_r*_history_seed*.json',
}


def iters_of(path):
    m = re.search(r'_it(\d+)_history_', os.path.basename(path))
    return int(m.group(1)) if m else args.default_iters


def alpha_of(path, rec):
    c = rec.get('config') or {}
    if 'alpha' in c:
        return c['alpha']
    # note a=0 is written 'a0', not 'a0.0', by the LAMDA run scripts
    m = re.search(r'_a(\d+(?:\.\d+)?)_', os.path.basename(path))
    return float(m.group(1)) if m else None


def seed_of(path, rec):
    c = rec.get('config') or {}
    if 'seed' in c:
        return c['seed']
    m = re.search(r'seed(\d+)', os.path.basename(path))
    return int(m.group(1)) if m else None


by_cond = {}
for cond, pat in PATTERNS.items():
    rows = []
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
        rows.append((path, rec))
    by_cond[cond] = rows

budgets = sorted({iters_of(pth) for c in ('madar', 'mu') for pth, _ in by_cond[c]})
if args.iters:
    missing = [b for b in args.iters if b not in budgets]
    if missing:
        print(f"  ! requested budget(s) {missing} not found on disk; have {budgets}")
    budgets = [b for b in args.iters if b in budgets]
if not budgets:
    raise SystemExit("No CL_ITERS budgets found. Check --root and --default-iters.")
print(f"  budgets found: {budgets}  (untagged files treated as "
      f"{args.default_iters})")

blocks, notes = [], []
for b in budgets:
    tag = f"lamda_it{b}"
    records = {}
    for cond in ('naive', 'joint', 'madar', 'mu'):
        if cond in UNTAGGED:
            records[cond] = by_cond[cond]          # epoch-based: shared across budgets
        else:
            records[cond] = [(pth, r) for pth, r in by_cond[cond] if iters_of(pth) == b]
        print(f"  [it{b}] {cond:<6} {len(records[cond]):>3} run(s)"
              + ("  (epoch-based, shared across budgets)" if cond in UNTAGGED else ""))

    stage = pf.stage_from_records(records, alpha_of=alpha_of)
    mu_runs, mu_alpha, mu_note = pf.pick_mu_arm(stage, reference='madar')
    if mu_note:
        print(f"  [it{b}] {mu_note}")
    res = pf.stage1_figures(stage, tag, f"{TITLE} \u2014 CL_ITERS={b}",
                            out_dir=args.out_dir, mu_runs=mu_runs, mu_note=mu_note)
    blocks.append((f"Stage 1 \u2014 CL_ITERS={b}", res, tag))
    if mu_alpha is not None:
        notes.append(f"CL_ITERS={b}: MADAR + Unlearning is the alpha={mu_alpha} arm, "
                     f"chosen as the better of the two on mean accuracy.")

# ---------------------------------------------------------------- meta
meta_dirs = [os.path.join(R, d) for d in (args.meta_dirs or DEFAULT_META_DIRS)]
meta_dirs = [d for d in meta_dirs if os.path.isdir(d)]
if not meta_dirs:
    print(f"  ! no meta workdirs found ({args.meta_dirs or DEFAULT_META_DIRS})")
else:
    data, info = pf.load_meta(meta_dirs, seeds=args.meta_seeds)
    m_iters = set()
    for wd in meta_dirs:
        sels = sorted({s for w, s in data if w == wd})
        c = info.get(wd) or {}
        m_iters.add(c.get('cl_iters'))
        print(f"  meta {os.path.basename(wd):<36} mode={c.get('mode', '?')} "
              f"cl_iters={c.get('cl_iters')} alpha={c.get('alpha')} {sels}")
    mres = pf.meta_figures(data, info, 'lamda',
                           f"{TITLE} \u2014 learned forget-set selection",
                           out_dir=args.out_dir, metric=args.meta_metric,
                           reference=args.meta_reference)
    it_txt = "/".join(str(i) for i in sorted(x for x in m_iters if x))
    blocks.append((f"Meta: forget-set selection (CL_ITERS={it_txt or '?'})",
                   mres, 'lamda'))
    if len([i for i in m_iters if i]) > 1:
        mres['notes'].append("Meta workdirs pooled here span more than one CL_ITERS "
                             "budget; state which in the caption.")
    sels_all = {s for _, s in data}
    if 'nn1' not in {s for w, s in data if (info.get(w) or {}).get('mode') == 'a'}:
        mres['notes'].append("Mode A has no learned NN-1 run on this corpus, so only "
                             "donut and random appear for that mode.")

notes.append("Naive and joint train by EPOCHS_PER_TASK and carry no CL_ITERS tag, so "
             "the same runs appear under every budget. Only MADAR and "
             "MADAR + Unlearning differ between the budget sections.")
notes.append(f"Meta accuracy is reported as {args.meta_metric}; LAMDA's test set is "
             f"heavily imbalanced, so mean per-family recall is the more "
             f"representative aggregate.")

md = pf.markdown('lamda', "LAMDA \u2014 results", blocks,
                 out_dir=args.out_dir, extra_notes=notes)

print(f"\nWrote {md}")
for _, res, tag in blocks:
    for fp in res.get('figures', {}).values():
        print(f"Wrote {fp}")
    for t in res.get('tables', {}):
        print(f"Wrote {os.path.join(args.out_dir, f'{tag}_{t}.csv')}")
for _, res, _t in blocks:
    for n in res.get('notes', []):
        print(f"  note: {n}")