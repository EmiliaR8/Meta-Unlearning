"""How much each method loses when the model shrinks.

    python3 capacity.py <log_root>/<group> [--out fig.png] [--csv out.csv]
                        [--metric mean_acc|final_acc|task0_final]

THE QUESTION. param_gain.py asks what unlearning buys at a fixed model size.
This asks the complementary one: across a size sweep, which methods hold their
accuracy as capacity falls and which collapse? A method whose advantage only
exists at full width is a different proposition from one that survives at 0.32M
parameters, and the size study answers that for every method in it, not just the
two that have an unlearning arm.

PAIRED BY SEED, ACROSS SIZES. Seeds are shared across model sizes, so

    d = acc(method, small, seed) - acc(method, largest, seed)

cancels the seed term the same way the ordering term cancels elsewhere here.
Seed-to-seed spread in absolute accuracy is larger than the capacity effect, so
comparing unpaired means would mostly measure which seeds happened to land well.

WHY THIS IS NOT param_gain.py. That script divides a gain by a parameter count,
which orders itself by size whenever the numerator is flat -- an arithmetic
result wearing an empirical costume. This one never divides: it reports the
accuracy a method actually reaches at each width, and the paired drop between
widths. Nothing here is a ratio.

A t is withheld below three pairs: with fewer, its sd has under two degrees of
freedom and the number is noise wearing a statistic's clothes.
"""
import argparse, csv, json, statistics as st
from pathlib import Path

NICE = {"naive": "Naive", "joint": "Joint", "si_only": "Synaptic Intelligence",
        "er_only": "Experience Replay", "agem": "A-GEM", "malcl": "MalCL",
        "xder": "X-DER", "star_xder": "X-DER + STAR",
        "deduce_star_xder": "DEDUCE", "gcr": "GCR", "madar": "MADAR",
        "madar_unlearn": "MADAR + Unlearning", "meta_unlearn": "Meta-Unlearning"}
ORDER = ["joint", "naive", "si_only", "er_only", "malcl", "agem", "xder",
         "star_xder", "deduce_star_xder", "gcr", "madar", "madar_unlearn",
         "meta_unlearn"]


def metric_of(rec: dict, which: str):
    per = rec.get("per_task") or []
    if not per:
        return None
    if which == "mean_acc":
        return st.mean(t["metrics"]["seen"]["accuracy"] for t in per)
    if which == "final_acc":
        return per[-1]["metrics"]["seen"]["accuracy"]
    if which == "task0_final":
        return per[-1]["metrics"]["task0"]["accuracy"]
    raise SystemExit(f"unknown --metric {which!r}")


def collect(group: Path, which: str):
    runs, params, dupes = {}, {}, set()
    for f in sorted(group.glob("jsons/*.json")):
        try:
            r = json.loads(f.read_text())
        except (OSError, ValueError):
            print(f"  ! {f.name}: unreadable; skipped"); continue
        v = metric_of(r, which)
        if v is None:
            continue
        cfg = r["config"]
        key = (cfg["method"], cfg.get("model"), cfg.get("seed"))
        if key in runs:
            # Two records for one (method, size, seed) means a second axis --
            # an alpha=0 control, a different si_c -- and silently keeping
            # whichever sorted last would compare two different arms.
            dupes.add(key[:2])
            continue
        runs[key] = v
        n = (r.get("model") or {}).get("n_params")
        if n:
            params[cfg.get("model")] = int(n)
    if dupes:
        print("  ! more than one run per (method, size, seed) for: "
              + ", ".join(f"{m}/{s}" for m, s in sorted(dupes))
              + "\n    the first by filename was kept. Separate the extra arm "
                "into its own\n    --group, or this compares two different "
                "configurations as one.")
    return runs, params


def paired(a: list, b: list):
    """mean, sd, positives, t  for a - b elementwise. t only at n >= 3.

    The sd guard is RELATIVE, not `if sd`. When every pair moves by the same
    amount the residual sd is not 0 but ~1e-16 of floating-point dust, and
    dividing by it yields a t of 1e15 -- a number that looks like overwhelming
    significance and means the opposite: there is no spread left to measure.
    Below the tolerance the difference is effectively deterministic, and a t is
    not the right thing to report about it.
    """
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if not n:
        return None
    mean = st.mean(d)
    sd = st.stdev(d) if n > 1 else 0.0
    scale = max(abs(mean), max((abs(x) for x in d), default=0.0), 1.0)
    t = (mean / (sd / n ** 0.5)) if (n >= 3 and sd > 1e-9 * scale) else None
    return {"n": n, "mean": mean, "sd": sd, "pos": sum(x > 0 for x in d),
            "t": t, "degenerate": n > 1 and sd <= 1e-9 * scale}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("group")
    ap.add_argument("--metric", default="mean_acc",
                    choices=["mean_acc", "final_acc", "task0_final"])
    ap.add_argument("--out", default="capacity.png")
    ap.add_argument("--csv")
    a = ap.parse_args()

    group = Path(a.group)
    runs, params = collect(group, a.metric)
    if not runs:
        raise SystemExit(f"no usable records under {group}/jsons")

    models = sorted({m for _, m, _ in runs}, key=lambda m: -params.get(m, 0))
    methods = sorted({m for m, _, _ in runs},
                     key=lambda m: ORDER.index(m) if m in ORDER else len(ORDER))
    if len(models) < 2:
        raise SystemExit(f"only one model size present ({models}); "
                         f"there is no capacity sweep here")
    big = models[0]

    print(f"metric: {a.metric}   sizes: "
          + ", ".join(f"{m} ({params.get(m,0)/1e6:.2f}M)" for m in models))
    head = f"{'method':>24}" + "".join(f"{m:>16}" for m in models)
    print(head)
    print("-" * len(head))

    data, rows = {}, []
    for meth in methods:
        cells, series = [], {}
        for m in models:
            seeds = sorted(s for mm, md, s in runs if mm == meth and md == m)
            vals = [runs[(meth, m, s)] for s in seeds]
            series[m] = (seeds, vals)
            if vals:
                sd = st.stdev(vals) if len(vals) > 1 else 0.0
                cells.append(f"{st.mean(vals):9.2f}+-{sd:<4.2f}")
            else:
                cells.append(f"{'--':>16}")
        print(f"{NICE.get(meth, meth):>24}" + "".join(f"{c:>16}" for c in cells))
        data[meth] = series
        rows.append({"method": meth, "label": NICE.get(meth, meth),
                     **{f"{m}_mean": (st.mean(series[m][1]) if series[m][1] else None)
                        for m in models}})

    # -- the paired drop, which is the actual claim --------------------------
    print(f"\npaired drop from {big}, matched on seed "
          f"(negative = worse at the smaller width):")
    print(f"{'method':>24}" + "".join(f"{m + ' - ' + big:>30}" for m in models[1:]))
    for meth in methods:
        big_seeds, big_vals = data[meth][big]
        cells = []
        for m in models[1:]:
            seeds, vals = data[meth][m]
            common = [s for s in seeds if s in big_seeds]
            if not common:
                cells.append(f"{'--':>30}"); continue
            p = paired([runs[(meth, m, s)] for s in common],
                       [runs[(meth, big, s)] for s in common])
            t = (f" t{p['t']:+7.2f}" if p["t"] is not None
                 else ("  identical" if p["degenerate"] else "          "))
            cells.append(f"{p['mean']:+7.2f}+-{p['sd']:<5.2f} "
                         f"{p['pos']}/{p['n']}{t}".rjust(30))
            for r in rows:
                if r["method"] == meth:
                    r[f"drop_{m}"] = p["mean"]
                    r[f"drop_{m}_sd"] = p["sd"]
        print(f"{NICE.get(meth, meth):>24}" + "".join(cells))

    if a.csv:
        fields = list(rows[0].keys())
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {a.csv}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available; the numbers above are the result")
        return

    xs = [params.get(m, 0) / 1e6 for m in models]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cmap = plt.get_cmap("tab10")
    marks = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "8"]

    for i, meth in enumerate(methods):
        ys = [st.mean(data[meth][m][1]) if data[meth][m][1] else None
              for m in models]
        es = [st.stdev(data[meth][m][1]) if len(data[meth][m][1]) > 1 else 0.0
              for m in models]
        pts = [(x, y, e) for x, y, e in zip(xs, ys, es) if y is not None]
        if not pts:
            continue
        px, py, pe = zip(*pts)
        axes[0].errorbar(px, py, yerr=pe, marker=marks[i % len(marks)],
                         capsize=3, lw=1.6, ms=5, color=cmap(i % 10),
                         label=NICE.get(meth, meth))
    axes[0].set_xscale("log")
    axes[0].set_xlabel("parameters (millions, log scale)")
    axes[0].set_ylabel({"mean_acc": "mean accuracy over tasks (%)",
                        "final_acc": "accuracy after the final task (%)",
                        "task0_final": "task-0 accuracy after the final task (%)"
                        }[a.metric])
    axes[0].set_title("Accuracy vs capacity")
    axes[0].grid(alpha=0.3)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7, frameon=False)

    small = models[-1]
    labels, vals, errs = [], [], []
    for meth in methods:
        seeds_s, _ = data[meth][small]
        seeds_b, _ = data[meth][big]
        common = [s for s in seeds_s if s in seeds_b]
        if not common:
            continue
        p = paired([runs[(meth, small, s)] for s in common],
                   [runs[(meth, big, s)] for s in common])
        labels.append(NICE.get(meth, meth)); vals.append(p["mean"])
        errs.append(p["sd"])
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    axes[1].barh([labels[i] for i in order], [vals[i] for i in order],
                 xerr=[errs[i] for i in order], capsize=3, color="#2e8b73")
    axes[1].axvline(0, color="#444", lw=0.8)
    axes[1].set_xlabel(f"accuracy at {small} minus {big} (pp, paired by seed)")
    axes[1].set_title(f"What shrinking to {small} costs")
    axes[1].tick_params(labelsize=8)
    axes[1].grid(axis="x", alpha=0.3)
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
