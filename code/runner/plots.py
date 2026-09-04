"""Figures for a log group.

    python -m runner.plots --group madar_fidelity

Writes one PNG per metric to Logs/<group>/figures/:
accuracy, accuracy on task 0, accuracy on the most recent task, precision,
recall -- each as accuracy-by-task with a mean +/- std band across seeds.

FIGURE CONVENTIONS, and why

  * One y-axis per figure, never two. Two measures on one pair of axes lets the
    reader infer any relationship the author chooses by rescaling.

  * Colour is assigned from a FIXED slot order keyed on the condition, so a
    figure showing three conditions paints them exactly as the figure showing
    six does. Colour follows the entity, not its rank in the current plot;
    otherwise dropping one condition silently repaints the rest between two
    figures in the same document.

  * The palette is the validated categorical set (blue, orange, aqua, yellow,
    magenta, green, violet, red). Two of its light-mode steps fall below 3:1
    against the surface, which obligates relief: every figure ships with a
    matching table from `aggregate.py`, and series are directly labelled at the
    line end when there are four or fewer. Identity is therefore never carried
    by colour alone.

  * Protocols are never drawn on the same axes: stage1 and meta runs differ in
    whether task 0 is shared, so a gap between them is protocol, not method.
    Each gets its own file.

  * Series whose curves COINCIDE exactly are drawn as interleaved dashes on one
    shared track, so all of them are visible and the legend matches the picture.
    Nothing is nudged: the values plotted are the values computed.

  * KNOWN GAP -- NEAR-coincidence. Two conditions differing by a fraction of a
    point overlap visually but are not equal, so they are drawn plainly and one
    hides the other. Widening the tolerance would print "identical" over curves
    that are not, which is worse. The right instrument for sub-point gaps is a
    PAIRED-DIFFERENCE panel -- each condition minus a reference on the SAME
    seed, which cancels the run-to-run variation the two share -- and that is
    not implemented here yet. Until it is, read small gaps off the tables from
    `aggregate.py`, not off these curves.

  * These are print figures for a document, so a single light surface is a
    deliberate commitment rather than an omission.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from runner import paths as paths_mod
from runner.aggregate import (CURVES, condition_of, condition_label,
                              label_conditions, load_group, short_label,
                              split_keys)

# Validated categorical palette, in fixed slot order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Canonical series order: a known condition always takes the same slot, so
# subsetting the figure cannot repaint the survivors.
# Fixed slot order. A condition keeps its hue whichever subset is plotted, so a
# figure showing three methods paints them as the figure showing eight does.
# Eight slots exist; past that the caller facets rather than reusing a hue.
CANONICAL = ["naive", "joint", "madar", "madar_unlearn + donut",
             "er_only", "si_only", "malcl", "agem",
             "madar_unlearn + random", "madar_unlearn + leftover"]

# Model presets, recognised so they can be stripped from the colour key.
MODEL_SUFFIXES = {"full", "trunk", "half", "quarter", "tiny"}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e5e3da"

TITLES = {
    "accuracy": ("Accuracy over families seen so far", "Accuracy (%)"),
    "accuracy_task0": ("Retention: accuracy on the task-0 families", "Accuracy (%)"),
    "accuracy_recent": ("Acquisition: accuracy on the families just added",
                        "Accuracy (%)"),
    "precision": ("Macro precision over families seen so far", "Precision (%)"),
    "recall": ("Macro recall over families seen so far", "Recall (%)"),
    "f1": ("Macro F1 over families seen so far", "F1 (%)"),
}


def colour_key(label: str) -> str:
    """The ENTITY colour follows: the method and its selector.

    Model width is a second dimension. Folding it into the colour key would mean
    `madar` at two widths took two unrelated hues, and would push the palette
    past its slot count on a study that varies width. Width is carried by line
    style instead, so `madar` is one colour in every figure it appears in.
    """
    return _strip_model(label)


def _strip_model(label: str) -> str:
    parts = [p.strip() for p in label.split(" + ")]
    keep = [p for p in parts if p not in MODEL_SUFFIXES]
    return " + ".join(keep) if keep else label


def colour_for(label: str, present: list[str]) -> str:
    """Fixed-slot assignment. Canonical entities first, then any others in a
    stable sorted order. NEVER cycles: past the slot count the caller folds or
    facets, because a repeated hue is a false identity claim."""
    order = list(CANONICAL)
    for lab in sorted({colour_key(p) for p in present}):
        if lab not in order:
            order.append(lab)
    idx = order.index(colour_key(label))
    if idx >= len(PALETTE):
        raise IndexError(
            f"{idx + 1} distinct conditions exceeds the {len(PALETTE)}-slot palette; "
            f"facet the figure rather than reusing a hue")
    return PALETTE[idx]


def style_for(label: str, models: list[str]) -> str:
    """Line style as the secondary channel when one figure spans model widths."""
    if len(models) <= 1:
        return "-"
    order = ["-", "--", "-.", ":"]
    model = next((p.strip() for p in label.split(" + ")
                  if p.strip() in MODEL_SUFFIXES), None)
    if model is None:
        return "-"
    return order[sorted(models).index(model) % len(order)]


def coincident_groups(means: dict, tol: float = 1e-6) -> list[list[str]]:
    """Group series whose curves coincide to within `tol` at every point.

    Overlapping lines are drawn one on top of another, so only the last is
    visible while the legend still claims all of them. That is a false picture:
    the reader sees two conditions where five were plotted. Detecting the
    coincidence lets it be DRAWN rather than hidden -- never by nudging the
    values, which would misstate the data to make the picture convenient.
    """
    labels = list(means)
    groups, assigned = [], set()
    for i, a in enumerate(labels):
        if a in assigned:
            continue
        group = [a]
        assigned.add(a)
        for b in labels[i + 1:]:
            if b in assigned:
                continue
            va, vb = means[a], means[b]
            if len(va) == len(vb) and np.allclose(va, vb, atol=tol, rtol=0):
                group.append(b)
                assigned.add(b)
        groups.append(group)
    return groups


def dash_for(rank: int, size: int, segment: float = 5.0):
    """Interleaved dash phases so coincident lines tile one visible track.

    Each member draws the same on/off period at a different phase, so together
    they alternate along the shared path and every one of them is visible at
    full 2px weight. Preferred over thinning or offsetting: nothing about the
    plotted values changes.
    """
    if size <= 1:
        return "-"
    return (rank * segment, (segment, segment * (size - 1)))


def declutter(positions: list[float], min_gap: float) -> list[float]:
    """Nudge overlapping direct labels apart, preserving their vertical order.

    Without this, converging curves stack their labels into an unreadable blot --
    which is worse than no direct label at all, since the reader cannot tell
    which text belongs to which line."""
    order = sorted(range(len(positions)), key=lambda i: positions[i])
    out = list(positions)
    for rank, i in enumerate(order):
        if rank == 0:
            continue
        prev = out[order[rank - 1]]
        if out[i] - prev < min_gap:
            out[i] = prev + min_gap
    return out


def series_for(records: list[dict], metric: str):
    """-> {label: (n_tasks array, stacked curves (n_seeds, n_tasks))}"""
    key = CURVES[metric]
    keys = split_keys(records)
    by_cond = defaultdict(list)
    for rec in records:
        curve = [v for v in (rec["curves"].get(key) or []) if v is not None]
        if curve:
            by_cond[condition_of(rec, keys)].append(curve)
    # Label per CONDITION, not per short name: two conditions differing only in a
    # hyperparameter must not be merged into one line.
    labels = label_conditions(by_cond, keys)
    groups = {labels[c]: v for c, v in by_cond.items()}

    out = {}
    for label, curves in groups.items():
        n = min(len(c) for c in curves)
        out[label] = np.asarray([c[:n] for c in curves], dtype=float)
    return out


def draw(series: dict, metric: str, title: str, out_path: Path,
         x_offset: int = 0) -> Path:
    fig, ax = plt.subplots(figsize=(8.4, 5.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    present = list(series)
    ordered = sorted(present, key=lambda l: (CANONICAL.index(colour_key(l))
                                             if colour_key(l) in CANONICAL else 99, l))
    models = sorted({p.strip() for l in present for p in l.split(" + ")
                     if p.strip() in MODEL_SUFFIXES})
    direct_label = len(ordered) <= 4

    means = {l: series[l].mean(axis=0) for l in ordered}
    groups = coincident_groups(means)
    dash, overlaps = {}, []
    for group in groups:
        if len(group) > 1:
            overlaps.append(group)
        for rank, label in enumerate(sorted(group, key=ordered.index)):
            dash[label] = (dash_for(rank, len(group))
                           if len(group) > 1 else style_for(label, models))

    ends = []
    for label in ordered:
        arr = series[label]
        x = np.arange(x_offset, x_offset + arr.shape[1])
        mean = means[label]
        colour = colour_for(label, present)
        ax.plot(x, mean, linestyle=dash[label], lw=2.0, marker="o", ms=4.5,
                color=colour, label=f"{label} (n={arr.shape[0]})",
                markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=3)
        if arr.shape[0] > 1:
            sd = arr.std(axis=0, ddof=1)
            ax.fill_between(x, mean - sd, mean + sd, color=colour, alpha=0.15,
                            lw=0, zorder=2)
        ends.append((float(x[-1]), float(mean[-1]), label))

    heading, ylabel = TITLES.get(metric, (metric, metric))
    ax.set_title(f"{heading}\n{title}", fontsize=11, color=INK, loc="left")
    ax.set_xlabel("Task", fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=10, color=INK_SECONDARY)
    n_ticks = max(arr.shape[1] for arr in series.values())
    ax.set_xticks(np.arange(x_offset, x_offset + n_ticks))
    ax.grid(True, color=GRID, lw=0.8, alpha=0.9, zorder=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)

    if overlaps:
        joined = "; ".join(" = ".join(g) for g in overlaps)
        ax.text(0.0, -0.16, f"identical curves drawn as interleaved dashes: {joined}",
                transform=ax.transAxes, fontsize=7.5, color=INK_SECONDARY,
                va="top", ha="left")

    if direct_label:
        # Direct labels are the relief the contrast WARN obligates. Space is
        # reserved on the right and overlapping labels are nudged apart, so
        # converging curves do not stack their names into an unreadable blot.
        ax.margins(x=0.02)
        lo, hi = ax.get_ylim()
        ax.set_xlim(right=x_offset + n_ticks - 1 + 0.05 * max(1, n_ticks - 1))
        gap = (hi - lo) * 0.045
        placed = declutter([e[1] for e in ends], gap)
        for (xe, _ye, label), y in zip(ends, placed):
            ax.annotate(label, xy=(xe, y), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=8.5,
                        color=INK_SECONDARY, annotation_clip=False, zorder=4)

    # A legend is always present for >= 2 series; a lone series is named by the title.
    if len(ordered) >= 2:
        ax.legend(fontsize=8.5, frameon=False, loc="lower left",
                  labelcolor=INK_SECONDARY)

    fig.tight_layout()
    if direct_label:
        fig.subplots_adjust(right=0.74)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Plot a log group")
    p.add_argument("--group", required=True)
    p.add_argument("--metrics", nargs="+", default=list(TITLES),
                   choices=list(TITLES))
    p.add_argument("--log-root"); p.add_argument("--data-root")
    p.add_argument("--cache-root")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args(argv)

    store = paths_mod.resolve(args.data_root, args.log_root, args.cache_root)
    records = load_group(store, args.group)
    out_dir = Path(args.out_dir) if args.out_dir else store.figure_dir(args.group)

    # One file per (protocol, dataset, task_setup): these cannot share axes.
    blocks = defaultdict(list)
    for rec in records:
        cfg = rec["config"]
        blocks[(cfg.get("protocol"), cfg.get("dataset"),
                cfg.get("task_setup"))].append(rec)

    written = []
    for (protocol, dataset, setup), recs in sorted(blocks.items(),
                                                   key=lambda kv: str(kv[0])):
        stem = f"{dataset}_{setup}_{protocol}".replace("+", "-")
        subtitle = f"{dataset} · {setup} · protocol {protocol}"
        for metric in args.metrics:
            series = series_for(recs, metric)
            if not series:
                continue
            n_entities = len({colour_key(l) for l in series})
            if n_entities > len(PALETTE):
                keep = [l for l in sorted(series)
                        if colour_key(l) in sorted({colour_key(x) for x in series})
                        [:len(PALETTE)]]
                print(f"  ! {n_entities} distinct conditions exceeds the "
                      f"{len(PALETTE)}-slot palette. Plotting {len(keep)}; facet the "
                      f"rest rather than reusing a hue.")
                series = {k: series[k] for k in keep}
            path = draw(series, metric, subtitle, out_dir / f"{stem}_{metric}.png")
            written.append(path)

    if len(blocks) > 1:
        print(f"note: {len(blocks)} (protocol, dataset, task setup) block(s); "
              f"each gets its own files and they are never drawn on shared axes.")
    for path in written:
        print(f"  wrote {path}")
    if not written:
        print("no figures written (no usable curves found)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
