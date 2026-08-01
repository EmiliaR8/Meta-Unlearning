"""

Usage:
    python compile_arch_study.py
    python compile_arch_study.py --results-dir results_arch --tasks 1 10
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

p = argparse.ArgumentParser(description="Compile the architecture study")
p.add_argument('--results-dir', default='results_arch')
p.add_argument('--out-prefix', default='arch_study')
p.add_argument('--tasks', type=int, nargs=2, default=[1, 10], metavar=('LO', 'HI'),
               help='Inclusive task range to average over. Default 1 10 (excludes '
                    'task 0, which is identical across conditions by construction).')
p.add_argument('--alpha-main', type=float, default=0.2, help='Primary unlearning alpha')
p.add_argument('--meta-dir', dest='meta_dir', nargs='+', default=None,
               help="Meta (selector) workdirs to render alongside the stage-1 study. "
                    "Default: auto-discover compare_arch_b*. Pass 'none' to skip.")
p.add_argument('--meta-metric', choices=['micro', 'macro'], default='micro',
               help="Aggregate for the meta figure. macro = per_task_macro.")
p.add_argument('--order', nargs='+',
               default=['full', 'trunk', 'half', 'quarter', 'tiny'],
               help='Architecture display order, largest first')
args = p.parse_args()

LO, HI = args.tasks
INK, MUT, GRID = "#23213B", "#5F5E5A", "#E5E3DA"
COLORS = {'madar': "#185FA5", 'mu': "#534AB7", 'mu_a0': "#8C8A82",
          'joint': "#0F6E56", 'naive': "#B4342A"}

# ------------------------------------------------------------------ load
runs = defaultdict(dict)      # (arch, condition_key) -> {seed: mean_acc}
curves = defaultdict(dict)    # (arch, condition_key) -> {seed: full curve}
nparams, si_pen = {}, defaultdict(list)

files = sorted(glob.glob(os.path.join(args.results_dir, '*.json')))
if not files:
    raise SystemExit(f"No JSON found in {args.results_dir}/")

for path in files:
    with open(path) as f:
        d = json.load(f)
    cfg = d.get('config')
    if cfg is None:
        print(f"  ! {os.path.basename(path)}: no config block, skipping "
              f"(pre-arch-study file?)")
        continue
    cond, arch, seed = cfg['condition'], cfg['arch'], cfg['seed']
    if cond == 'mu':                       # separate the alpha arms
        cond = 'mu' if abs(cfg.get('alpha', args.alpha_main) - args.alpha_main) < 1e-9 else 'mu_a0'
    acc = d['Avg_Acc']
    if len(acc) <= HI:
        print(f"  ! {os.path.basename(path)}: only {len(acc)} tasks, skipping")
        continue
    runs[(arch, cond)][seed] = float(np.mean(acc[LO:HI + 1]))
    curves[(arch, cond)][seed] = acc
    nparams[arch] = cfg['n_params']
    if d.get('SI_Penalty'):
        si_pen[arch] += [v for v in d['SI_Penalty'] if v is not None]

archs = [a for a in args.order if any(k[0] == a for k in runs)]
archs += sorted({k[0] for k in runs} - set(archs))
print(f"Loaded {len(files)} files | architectures: {archs}\n")


def cell(arch, cond):
    """Return (seeds, values) sorted by seed."""
    d = runs.get((arch, cond), {})
    s = sorted(d)
    return s, np.array([d[k] for k in s], dtype=float)


# ------------------------------------------------------------------ per-arch
summary = {}
print(f"{'arch':>8} {'params':>10} | {'madar':>14} {'mu':>14} {'joint':>8} | "
      f"{'d = mu-madar':>18} {'p':>7} | {'headroom':>9} {'recovery':>16} | "
      f"{'d0 (a=0)':>10} {'d-d0':>8}")
print('-' * 146)

for a in archs:
    s_m, v_m = cell(a, 'madar')
    s_u, v_u = cell(a, 'mu')
    s_j, v_j = cell(a, 'joint')
    if not len(v_m) or not len(v_u):
        print(f"{a:>8} {'':>10} | incomplete (madar={len(v_m)}, mu={len(v_u)})")
        continue

    shared = sorted(set(s_m) & set(s_u))
    if not shared:
        print(f"{a:>8} | no shared seeds between madar and mu")
        continue
    dm = {k: v for k, v in zip(s_m, v_m)}
    du = {k: v for k, v in zip(s_u, v_u)}
    d = np.array([du[k] - dm[k] for k in shared])

    if len(d) > 1:
        t, pv = stats.ttest_1samp(d, 0)
        pstr, dstr = f"{pv:.3f}", f"{d.mean():+.3f} +/- {d.std(ddof=1):.3f}"
    else:
        pv, pstr, dstr = np.nan, "  n/a", f"{d.mean():+.3f} (n=1)"

    # alpha=0 keeps the curation and the step count but removes the forget
    # objective, so d0 is the curation-only gain and d - d0 is what the KL
    # term adds on top. Paired by seed against madar, exactly as d is.
    s_0, v_0 = cell(a, 'mu_a0')
    if len(v_0):
        d0map = {k: v for k, v in zip(s_0, v_0)}
        sh0 = sorted(set(s_m) & set(d0map))
        d0 = np.array([d0map[k] - dm[k] for k in sh0]) if sh0 else np.array([])
    else:
        d0 = np.array([])

    head = (v_j.mean() - v_m.mean()) if len(v_j) else np.nan
    if np.isfinite(head) and head > 0:
        rec = d / head
        rstr = f"{rec.mean():.3f} +/- {rec.std(ddof=1):.3f}" if len(rec) > 1 else f"{rec.mean():.3f}"
    else:
        rec, rstr = np.array([]), "  (no joint)"

    summary[a] = dict(d=d, d0=d0, seeds=shared, recovery=rec, headroom=head,
                      madar=v_m.mean(), mu=v_u.mean(),
                      joint=v_j.mean() if len(v_j) else np.nan, p=pv,
                      n_m=len(v_m), n_u=len(v_u), n_j=len(v_j))

    print(f"{a:>8} {nparams.get(a, 0):>10,} | "
          f"{v_m.mean():>7.3f}(n={len(v_m)}) {v_u.mean():>7.3f}(n={len(v_u)}) "
          f"{(v_j.mean() if len(v_j) else float('nan')):>8.3f} | "
          f"{dstr:>18} {pstr:>7} | {head:>9.3f} {rstr:>16} | "
          f"{(f'{d0.mean():+.3f}' if len(d0) else '     -'):>10} "
          f"{(f'{d.mean()-d0.mean():+.3f}' if len(d0) else '   -'):>8}")

# ------------------------------------------------------------------ interaction
# `present` is not defined until the figure section, so filter locally.
_has0 = [a for a in archs if a in summary and len(summary[a].get('d0', []))]
if _has0:
    print(f"\n{'='*146}\nCURATION vs FORGET OBJECTIVE")
    print("  d0 = MU(alpha=0) - MADAR. At alpha=0 the forget-set is still selected and")
    print("  removed and the step count is unchanged; only the KL-toward-uniform term is")
    print("  gone. So d0 is the gain from CURATION alone and d - d0 is what the forget")
    print("  objective adds. If d0 carries most of d, the method is memory curation.\n")
    for a in _has0:
        d, d0 = summary[a]['d'], summary[a]['d0']
        frac = d0.mean() / d.mean() if abs(d.mean()) > 1e-9 else float('nan')
        print(f"    {a:>8}  d {d.mean():+.3f}  d0 {d0.mean():+.3f}  "
              f"curation share {frac:5.1%}  n={len(d0)}")

print(f"\n{'='*122}\nINTERACTION TEST -- does d differ between architectures?")
print("(two-sample: no pairing exists across architectures)\n")
have = [a for a in archs if a in summary and len(summary[a]['d']) > 1]
if len(have) < 2:
    print("  need >=2 architectures with >=2 seeds each")
else:
    print(f"  {'comparison':>22} {'delta_d':>12} {'t':>7} {'p':>8}   {'delta_recovery':>16}")
    base = have[0]
    for a in have[1:]:
        d0, d1 = summary[base]['d'], summary[a]['d']
        t, pv = stats.ttest_ind(d1, d0, equal_var=False)
        r0, r1 = summary[base]['recovery'], summary[a]['recovery']
        if len(r0) > 1 and len(r1) > 1:
            _, pr = stats.ttest_ind(r1, r0, equal_var=False)
            rs = f"{r1.mean()-r0.mean():+.3f} (p={pr:.3f})"
        else:
            rs = "n/a"
        print(f"  {a+' vs '+base:>22} {d1.mean()-d0.mean():>+12.3f} "
              f"{t:>7.2f} {pv:>8.3f}   {rs:>16}")

# ------------------------------------------------------------------ SI check
if si_pen:
    print(f"\n{'='*122}\nSI PENALTY MAGNITUDE (confound check)")
    print("  si_loss is a raw sum over parameters, so its scale varies with architecture.")
    print("  Compare against a cross-entropy loss of order 1.\n")
    for a in archs:
        if si_pen.get(a):
            v = np.array(si_pen[a])
            verdict = "INERT" if v.max() < 1e-2 else "active -- treat as covariate"
            print(f"  {a:>8}  mean {v.mean():.3e}  max {v.max():.3e}   {verdict}")

# ------------------------------------------------------------------ figure
present = [a for a in archs if a in summary]

# ------------------------------------------------------------------ joint warning
if not any(s['n_j'] for s in summary.values()):
    print(f"\n{'='*122}\nNO JOINT RUNS FOUND -- recovery is undefined, raw d only.")
    print("  A smaller network forgets more, so the MADAR baseline drops, headroom")
    print("  opens, and any intervention gains more percentage points. A growing d")
    print("  across widths is therefore the EXPECTED NULL, not a finding: without")
    print("  joint you cannot separate 'unlearning matters more at low capacity'")
    print("  from 'there was simply more room'.")
    _ext = [a for a in present if a in (archs[0], archs[-1])] or present[:2]
    print(f"  {3*len(_ext)} runs fix it (extremes only, 3 seeds; joint is stable at sd 0.30pp):")
    print("    for S in 1 2 3; do")
    for _a in _ext:
        print(f"      python arch_joint.py --seed $S --arch {_a:<8} --out-dir {args.results_dir}")
    print("    done")

if present:
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))
    tasks = np.arange(LO, HI + 1)

    ax = axes[0]
    for a, ls in zip(present, ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]):
        for cond in ('madar', 'mu'):
            cs = curves.get((a, cond), {})
            if not cs:
                continue
            m = np.array([v[LO:HI + 1] for v in cs.values()], dtype=float).mean(axis=0)
            ax.plot(tasks, m, ls, lw=1.7, color=COLORS[cond], marker='o', ms=3,
                    label=f"{a} {cond}")
    ax.set_xlabel("Task"); ax.set_ylabel("Average accuracy (%)")
    ax.set_title("Accuracy by task and architecture", fontsize=11)
    ax.set_xticks(tasks); ax.grid(alpha=0.3, color=GRID); ax.legend(fontsize=7.5, ncol=2)

    ax = axes[1]
    x = np.arange(len(present))
    mu_d = [summary[a]['d'].mean() for a in present]
    sd_d = [summary[a]['d'].std(ddof=1) if len(summary[a]['d']) > 1 else 0 for a in present]
    ax.bar(x, mu_d, yerr=sd_d, capsize=4, color=COLORS['mu'], alpha=0.85)
    ax.axhline(0, color='black', lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels([f"{a}\n{nparams.get(a,0)/1e6:.2f}M" for a in present],
                                         fontsize=9)
    ax.set_ylabel("d = MU - MADAR (pp)")
    ax.set_title("Raw gain\n(NOT comparable across architectures)", fontsize=11)
    ax.grid(alpha=0.3, color=GRID, axis='y')

    ax = axes[2]
    mu_r = [summary[a]['recovery'].mean() if len(summary[a]['recovery']) else np.nan
            for a in present]
    sd_r = [summary[a]['recovery'].std(ddof=1) if len(summary[a]['recovery']) > 1 else 0
            for a in present]
    ax.bar(x, mu_r, yerr=sd_r, capsize=4, color=COLORS['joint'], alpha=0.85)
    ax.axhline(0, color='black', lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels([f"{a}\n{nparams.get(a,0)/1e6:.2f}M" for a in present],
                                         fontsize=9)
    ax.set_ylabel("recovery = d / (joint - madar)")
    ax.set_title("Headroom-normalised gain\n(the comparable statistic)", fontsize=11)
    ax.grid(alpha=0.3, color=GRID, axis='y')

    fig.suptitle(f"EMBER 2018 - unlearning gain vs classifier capacity "
                 f"(tasks {LO}-{HI})", fontsize=13)
    fig.tight_layout()
    fp = f"{args.out_prefix}_fig.png"
    fig.savefig(fp, dpi=150)
    print(f"\nWrote {fp}")

# ------------------------------------------------------------------ csv
csv_path = f"{args.out_prefix}_summary.csv"
with open(csv_path, 'w') as f:
    f.write("arch,n_params,n_madar,n_mu,n_joint,madar,mu,joint,headroom,"
            "d_mean,d_sd,d_p,recovery_mean,recovery_sd,"
            "d0_mean,d0_sd,n_mu_a0,curation_share\n")
    for a in present:
        s = summary[a]
        d, r, d0 = s['d'], s['recovery'], s.get('d0', np.array([]))
        f.write(f"{a},{nparams.get(a,0)},{s['n_m']},{s['n_u']},{s['n_j']},"
                f"{s['madar']:.4f},{s['mu']:.4f},{s['joint']:.4f},{s['headroom']:.4f},"
                f"{d.mean():.4f},{d.std(ddof=1) if len(d)>1 else 0:.4f},{s['p']:.4f},"
                f"{r.mean() if len(r) else float('nan'):.4f},"
                f"{r.std(ddof=1) if len(r)>1 else 0:.4f},"
                f"{d0.mean() if len(d0) else float('nan'):.4f},"
                f"{d0.std(ddof=1) if len(d0)>1 else 0:.4f},{len(d0)},"
                f"{(d0.mean()/d.mean()) if len(d0) and abs(d.mean())>1e-9 else float('nan'):.4f}\n")
print(f"Wrote {csv_path}")

# ------------------------------------------------------------------ meta figures
# The selector view of the meta runs. Additive: if no meta workdirs exist yet
# (or --meta-dir none), everything above is unchanged and nothing is written.
if args.meta_dir != ['none']:
    try:
        import meta_curves
    except ImportError:
        meta_curves = None
        print("\n  (meta_curves.py not found -- skipping the selector figures)")
    if meta_curves is not None:
        mdirs = (args.meta_dir if args.meta_dir
                 else meta_curves.discover(['compare_arch_b', 'compare_arch_b_*']))
        if mdirs:
            print(f"\n{'='*122}\nMETA (SELECTOR) FIGURES: {', '.join(mdirs)}")
            # Joint at the reference arch, as a dotted ceiling. Different protocol
            # (stage-1 retrains task 0 per seed), so it is labelled as such.
            ceil = None
            _cs = curves.get((archs[0], 'joint'), {}) if archs else {}
            if _cs:
                ceil = np.array(list(_cs.values()), dtype=float).mean(axis=0)
            rows = meta_curves.render(mdirs, f"{args.out_prefix}_meta",
                                      metric=args.meta_metric, ceiling=ceil)
            if rows:
                mp = f"{args.out_prefix}_meta_summary.csv"
                with open(mp, 'w') as f:
                    for r in rows:
                        f.write(",".join(str(c) for c in r) + "\n")
                print(f"Wrote {mp}")
        else:
            print("\n  (no compare_arch_b* workdirs yet -- selector figures skipped)")