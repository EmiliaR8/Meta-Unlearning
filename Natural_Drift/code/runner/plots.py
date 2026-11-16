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
from runner.aggregate import (CONDITION_KEYS, CURVES, condition_of,
                              condition_label, label_conditions, load_group,
                              short_label, split_keys)
from runner.config import default_hyperparams

# Validated categorical palette, in fixed slot order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Canonical series order: a known condition always takes the same slot, so
# subsetting the figure cannot repaint the survivors.
# Fixed slot order. A condition keeps its hue whichever subset is plotted, so a
# figure showing three methods paints them as the figure showing eight does.
# Eight slots exist; past that the caller facets rather than reusing a hue.
CANONICAL = ["naive", "joint", "madar", "madar_unlearn + donut",
             "meta_unlearn + nn1", "er_only", "si_only", "malcl", "agem",
             "gcr", "xder", "star_xder", "deduce_star_xder",
             "madar_unlearn + nn1",
             "madar_unlearn + random", "madar_unlearn + leftover"]

# Bounds, not competitors. Joint is trained on everything at once and Naive
# replays nothing, so they bracket the range every method in between lives in.
# Drawing them in ink rather than in palette colour says that in the figure, and
# it buys back two of the eight hue slots for methods that are actually being
# compared -- which is what makes a ten-series figure fit a palette built for
# eight, without a single hue being claimed by two identities.
REFERENCE = {
    "joint": ("#52514e", (0, (6, 3))),
    "naive": ("#9a9892", (0, (1.5, 2.5))),
}

# Model presets, recognised so they can be stripped from the colour key. The
# convolutional names belong here for the same reason the MLP widths do: on an
# image corpus EVERY condition carries one, so without them no label would match
# CANONICAL and the whole figure would fall through to alphabetical ordering --
# `madar` would take a different hue on Tiny ImageNet than on EMBER for no
# reason a reader could see.
MODEL_SUFFIXES = {"full", "trunk", "half", "quarter", "tiny",
                  "resnet18", "resnet18_half", "resnet18_quarter", "resnet10"}

# Display names. Renaming happens AFTER colour assignment is keyed, via
# COLOUR_ALIAS below, so a condition keeps its hue when it is renamed: the
# figure in the paper and the figure in the appendix paint the same method the
# same colour even though only one of them spells it "Heuristic L2U".
# (method, selector) -> the name the paper prints. "*" matches any selector.
# The key has to carry the selector because Heuristic-L2U and Meta-L2U ARE one
# method under two selectors: keyed on the method alone, the two rows of the
# results table would collapse into one line on the figure.
#
# A name matched on an exact selector is COMPLETE -- it already says which
# selector it is -- so the ` + <selector>` suffix comes off. A name matched on
# "*" is not, and keeps whatever suffix distinguishes it.
PAPER_RENAME = {
    ("joint", "*"): "Joint",
    ("naive", "*"): "Naive",
    ("si_only", "*"): "SI",
    ("gcr", "*"): "GCR",
    ("malcl", "*"): "MalCL",
    ("madar", "*"): "MADAR",
    ("agem", "*"): "A-GEM",
    ("deduce_star_xder", "*"): "DEDUCE",
    ("madar_unlearn", "donut"): "Heuristic-L2U",
    ("madar_unlearn", "nn1"): "Meta-L2U",
    ("meta_unlearn", "nn1"): "Meta-L2U",
    ("meta_unlearn", "*"): "Meta-L2U",
}

# Display name -> the CANONICAL key its colour slot is keyed on. Without this a
# rename would move the series to a different palette slot, which is precisely
# the repainting the fixed slot order exists to prevent.
COLOUR_ALIAS = {
    "Joint": "joint", "Naive": "naive", "SI": "si_only", "GCR": "gcr",
    "MalCL": "malcl", "MADAR": "madar", "A-GEM": "agem",
    "DEDUCE": "deduce_star_xder",
    "Heuristic-L2U": "madar_unlearn + donut",
    "Meta-L2U": "meta_unlearn + nn1",
}

# Conditions the `paper` selection does not draw, as (method, field, value).
# These are ablation floors and unreported variants: every one of them stays in
# the logs and in the tables, which is where an ablation belongs. Dropping them
# from the CURVES is a presentation choice, not a deletion -- `--select all`
# still draws them, and the CSVs never lose a row either way.
PAPER_DROP = (
    ("madar_unlearn", "alpha", 0),     # unlearning strength off
    ("si_only", "si_c", 0),            # SI penalty off
)

# The methods the paper's figure carries, in the order its table lists them.
# A WHITELIST rather than a drop list: the log group also holds er_only, xder
# and star_xder, and a drop list would have to be extended every time a method
# is run that the figure does not report -- which fails silently, by drawing
# something nobody asked for, rather than loudly.
PAPER_METHODS = ("joint", "naive", "si_only", "gcr", "malcl", "madar",
                 "agem", "deduce_star_xder", "madar_unlearn", "meta_unlearn")

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

    A renamed series resolves back through COLOUR_ALIAS first, so the display
    name it carries cannot move it to a different slot.
    """
    stripped = _strip_model(label)
    return COLOUR_ALIAS.get(stripped, stripped)


def _strip_model(label: str) -> str:
    parts = [p.strip() for p in label.split(" + ")]
    keep = [p for p in parts if p not in MODEL_SUFFIXES]
    return " + ".join(keep) if keep else label


def colour_for(label: str, present: list[str]) -> str:
    """Slot assignment over the entities PRESENT in this figure, ordered
    canonically. NEVER cycles: past the slot count the caller folds or facets,
    because a repeated hue is a false identity claim.

    Slots are taken among the present entities rather than at each entity's
    absolute position in CANONICAL. The absolute scheme raised IndexError for
    anything listed past the eighth slot even when it was the ONLY series in the
    figure, which made the 9th canonical method unplottable rather than merely
    the 9th simultaneous one. CANONICAL now fixes the ORDER hues are handed out
    in, not the hue itself.

    The cost is real and worth stating: a method's colour depends on which other
    methods share its figure, so `madar` is not guaranteed the same hue across
    two figures with different series sets. Within one figure, and across
    figures drawn from the same set of conditions -- which is what the aggregate
    emits -- the assignment is stable. Where a colour must hold across
    differently-populated figures, pass the same `present` list to every call.
    """
    key = colour_key(label)
    if key in REFERENCE:
        return REFERENCE[key][0]
    canon = [lab for lab in CANONICAL]
    here = {k for k in ({colour_key(p) for p in present} | {key})
            if k not in REFERENCE}
    order = [lab for lab in canon if lab in here]
    order += sorted(lab for lab in here if lab not in canon)
    idx = order.index(key)
    if idx >= len(PALETTE):
        raise IndexError(
            f"{len(order)} distinct conditions exceeds the {len(PALETTE)}-slot "
            f"palette; facet the figure rather than reusing a hue")
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


def field_of(rec: dict, key: str):
    """A run's value for `key`, wherever it is recorded.

    Method-defining fields live in `config`, tuned knobs in `hyperparams`, and a
    run written before a knob existed carries neither -- it ran with that knob's
    default, so the default is its true value. Reading it as None instead would
    make an old run look like a different condition than an identical new one.
    """
    cfg = rec["config"]
    if cfg.get(key) is not None:
        return cfg[key]
    hp = rec.get("hyperparams") or {}
    if hp.get(key) is not None:
        return hp[key]
    return default_hyperparams(cfg.get("dataset", ""), cfg.get("method")).get(key)


def same_value(got, want) -> bool:
    """Compare numerically when both sides are numbers, textually otherwise.

    `alpha` reaches here as 0, 0.0 or "0.0" depending on when the run was
    written; a string compare would match one spelling of zero and silently keep
    the other two on the figure.
    """
    try:
        return float(got) == float(want)
    except (TypeError, ValueError):
        return str(got).strip().lower() == str(want).strip().lower()


def drop_for_paper(rec: dict) -> str | None:
    """-> why this run is not on the paper figure, or None to keep it."""
    method = rec["config"].get("method")
    if method not in PAPER_METHODS:
        return "not a method the paper figure reports"
    for m, key, val in PAPER_DROP:
        if method == m and same_value(field_of(rec, key), val):
            return f"{key}={val} (ablation floor)"
    return None


def display_for(cond: tuple) -> tuple[str, bool]:
    """-> (name, complete). `complete` means the name already encodes the
    selector, so the ` + <selector>` suffix would only repeat it."""
    method = cond[CONDITION_KEYS.index("method")]
    selector = cond[CONDITION_KEYS.index("selector")]
    exact = PAPER_RENAME.get((method, str(selector)))
    if exact is not None:
        return exact, True
    return PAPER_RENAME.get((method, "*"), method), False


def apply_rename(labels: dict) -> dict:
    """Display names, plus: drop a suffix that no longer distinguishes anything.

    `short_label` appends the selector because two selectors of one method are
    two conditions. Once a selection leaves a method with exactly ONE condition
    on the figure -- or the display name already names the selector -- that
    suffix separates nothing, and prints a hyperparameter the caption does not
    discuss.

    It comes off only when it is safe. If two surviving conditions would land on
    the SAME display name, both keep their full labels: a figure that merges two
    conditions into one line is worse than a figure with an ugly legend, because
    the reader cannot see that it happened.
    """
    per_method = defaultdict(list)
    for cond in labels:
        per_method[cond[CONDITION_KEYS.index("method")]].append(cond)

    proposed = {}
    for method, conds in per_method.items():
        for cond in conds:
            name, complete = display_for(cond)
            lab = labels[cond]
            if complete or len(conds) == 1:
                # Model width is a real second dimension, shared across methods,
                # so it is never the redundant suffix -- it stays.
                models = [q.strip() for q in lab.split(" + ")
                          if q.strip() in MODEL_SUFFIXES]
                proposed[cond] = " + ".join([name] + models)
            else:
                proposed[cond] = lab.replace(method, name, 1)

    clashes = defaultdict(list)
    for cond, lab in proposed.items():
        clashes[lab].append(cond)
    return {cond: (proposed[cond] if len(clashes[proposed[cond]]) == 1
                   else labels[cond])
            for cond in labels}


def pick_top_seeds(groups: dict, rules: list[str], announce: bool) -> dict:
    """Keep only the N best-scoring seeds of the named series.

    This is a REPORTING choice with a cost, and the cost is stated rather than
    hidden: ranking seeds by their own score and keeping the top N biases the
    mean upward and shrinks the spread, so a series thinned this way is not
    measured the same way as one that reports every seed it ran. What is kept
    and what is dropped is printed, so the caption can say so.
    """
    out = dict(groups)
    for rule in rules:
        pattern, _, n = rule.rpartition("=")
        if not pattern or not n.isdigit():
            raise SystemExit(f"--top-seeds wants LABEL=N, got {rule!r}")
        n = int(n)
        for label in list(out):
            if pattern.lower() not in label.lower():
                continue
            runs = out[label]
            if len(runs) <= n:
                continue
            width = min(len(c) for _s, c in runs)
            ranked = sorted(runs, key=lambda sc: -sum(sc[1][:width]) / width)
            out[label] = ranked[:n]
            if announce:
                kept = [s for s, _ in ranked[:n]]
                cut = [s for s, _ in ranked[n:]]
                print(f"  ! {label}: kept the {n} best-scoring seeds {kept}, "
                      f"dropped {cut}. This is selection on the outcome -- the "
                      f"mean is biased upward and the std understated relative "
                      f"to a series that reports every seed.")
    return out


def series_for(records: list[dict], metric: str, select: str = "all",
               only: list[str] | None = None, top_seeds: list[str] | None = None,
               seeds: list[int] | None = None, announce: bool = False):
    """-> {label: stacked curves, shape (n_seeds, n_tasks)}"""
    key = CURVES[metric]
    if seeds is not None:
        # A seed whitelist applies to EVERY series, which is the point: it makes
        # the figure symmetric by construction. Restricting one arm's seeds and
        # not another's is a choice about that arm, and if the choice is made
        # after seeing the curves it is a choice about the result.
        want = set(seeds)
        records = [r for r in records if r["config"].get("seed") in want]
    if select == "paper":
        cut = defaultdict(set)
        kept = []
        for r in records:
            why = drop_for_paper(r)
            if why is None:
                kept.append(r)
            else:
                cfg = r["config"]
                cut[(cfg.get("method"), cfg.get("selector"), why)].add(
                    cfg.get("seed"))
        if announce:
            for (m, sel, why), seeds in sorted(cut.items(), key=str):
                name = f"{m} + {sel}" if sel else m
                print(f"  - dropped {name} ({len(seeds)} run(s)): {why}")
        records = kept
    if not records:
        return {}
    keys = split_keys(records)
    by_cond = defaultdict(list)
    for rec in records:
        curve = [v for v in (rec["curves"].get(key) or []) if v is not None]
        if curve:
            by_cond[condition_of(rec, keys)].append(
                (rec["config"].get("seed"), curve))
    # Label per CONDITION, not per short name: two conditions differing only in a
    # hyperparameter must not be merged into one line.
    labels = label_conditions(by_cond, keys)
    if select == "paper":
        labels = apply_rename(labels)
    if only:
        wanted = [w.lower() for w in only]
        keep = {c for c, lab in labels.items()
                if any(w in lab.lower() for w in wanted)}
        by_cond = {c: v for c, v in by_cond.items() if c in keep}
    groups = {labels[c]: v for c, v in by_cond.items()}
    if top_seeds:
        groups = pick_top_seeds(groups, top_seeds, announce)
    if announce:
        for label in sorted(groups):
            got = sorted(s for s, _c in groups[label] if s is not None)
            print(f"    {label:24} seeds {got}")

    out = {}
    for label, runs in groups.items():
        n = min(len(c) for _s, c in runs)
        out[label] = np.asarray([c[:n] for _s, c in runs], dtype=float)
    return out


def band_for(arr: np.ndarray, band: str):
    """The half-width drawn around a mean curve, per task.

    `seed` is the spread ACROSS SEEDS AT EACH TASK. It is the honest picture of
    where the seeds disagree, and it is what makes a single collapsed seed at
    one task show up as a bulge at that task.

    `run` is the spread across seeds of each seed's TASK-AVERAGED value -- the
    one number the tables print. It is a constant, so it is drawn as a constant
    ribbon. It cannot show WHERE the seeds disagree, because averaging over
    tasks is what destroys that information: a seed that collapses at one task
    out of eleven moves its own average by about a ninth of the drop. Drawn as a
    band it therefore understates the spread at the task where the collapse
    happens, and overstates it everywhere else. Use it when the figure has to
    agree digit-for-digit with the table; read the collapse off `seed`.
    """
    if arr.shape[0] < 2 or band == "none":
        return None
    if band == "run":
        sd = float(arr.mean(axis=1).std(ddof=1))
        return np.full(arr.shape[1], sd)
    return arr.std(axis=0, ddof=1)


def run_sd(arr: np.ndarray) -> float | None:
    """The number the TABLE prints: spread across seeds of each seed's
    task-averaged value. One scalar per series, so it can sit in a legend."""
    if arr.shape[0] < 2:
        return None
    return float(arr.mean(axis=1).std(ddof=1))


def draw(series: dict, metric: str, title: str, out_path: Path,
         x_offset: int = 0, band: str = "seed", label_std: bool = False) -> Path:
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
        key = colour_key(label)
        # A bound is drawn as a bound: ink, its own dash, no marker. The reader
        # should not have to consult the legend to see that Joint and Naive are
        # not competing with the methods between them.
        is_ref = key in REFERENCE
        sd_run = run_sd(arr)
        tag = f"n={arr.shape[0]}"
        if label_std and sd_run is not None:
            tag += f", sd {sd_run:.2f}"
        ax.plot(x, mean, linestyle=(REFERENCE[key][1] if is_ref else dash[label]),
                lw=1.6 if is_ref else 2.0,
                marker="" if is_ref else "o", ms=4.5,
                color=colour, label=f"{label} ({tag})",
                markeredgecolor=SURFACE, markeredgewidth=0.8,
                zorder=2 if is_ref else 3)
        sd = band_for(arr, band)
        if sd is not None:
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
        # Opaque, not frameless. A ten-entry legend is tall enough to sit on
        # top of a curve, and a frameless one lets that curve run through the
        # text -- which costs more legibility than the frame does restraint.
        leg = ax.legend(fontsize=8.5, frameon=len(ordered) > 4,
                        loc="lower left", labelcolor=INK_SECONDARY,
                        facecolor=SURFACE, edgecolor=GRID, framealpha=0.94)
        if leg.get_frame() is not None:
            leg.get_frame().set_linewidth(0.6)

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
    p.add_argument("--select", default="all", choices=("all", "paper"),
                   help="which conditions to draw. `paper` drops the ablation "
                        "floors (alpha 0, si_c 0) and malcl L1_C_Mean, and "
                        "names what is left as the paper names it. Nothing is "
                        "deleted: the tables still carry every condition.")
    p.add_argument("--only", nargs="+", default=None, metavar="TEXT",
                   help="keep only series whose label contains one of these "
                        "(case-insensitive), applied after --select.")
    p.add_argument("--band", default="seed", choices=("seed", "run", "none"),
                   help="the shaded band: `seed` = spread across seeds at each "
                        "task (default); `run` = the constant the TABLE prints, "
                        "i.e. spread across seeds of each seed's task-averaged "
                        "value; `none` = mean curves only.")
    p.add_argument("--top-seeds", nargs="+", default=None, metavar="LABEL=N",
                   help="keep only the N best-scoring seeds of series whose "
                        "label contains LABEL (e.g. Meta-L2U=3). Selection on "
                        "the outcome: it biases the mean up and the std down, "
                        "and what was dropped is printed so the caption can "
                        "say so.")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="restrict EVERY series to these seeds, e.g. --seeds 1 "
                        "2 3. Applies uniformly, so the figure stays symmetric; "
                        "omit it to draw every seed each arm has.")
    p.add_argument("--label-std", default="auto",
                   choices=("auto", "on", "off"),
                   help="put the table's std in the legend. `auto` = on "
                        "whenever no band is drawn to carry it (--band none or "
                        "run).")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args(argv)
    label_std = (args.band != "seed" if args.label_std == "auto"
                 else args.label_std == "on")

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
            series = series_for(recs, metric, args.select, args.only,
                                args.top_seeds, args.seeds,
                                announce=(metric == args.metrics[0]))
            if not series:
                continue
            n_entities = len({colour_key(l) for l in series
                              if colour_key(l) not in REFERENCE})
            if n_entities > len(PALETTE):
                allowed = sorted({colour_key(x) for x in series
                                  if colour_key(x) not in REFERENCE})[:len(PALETTE)]
                keep = [l for l in sorted(series)
                        if colour_key(l) in allowed or colour_key(l) in REFERENCE]
                print(f"  ! {n_entities} distinct conditions exceeds the "
                      f"{len(PALETTE)}-slot palette. Plotting {len(keep)}; facet the "
                      f"rest rather than reusing a hue.")
                series = {k: series[k] for k in keep}
            path = draw(series, metric, subtitle,
                        out_dir / f"{stem}_{metric}.png", band=args.band,
                        label_std=label_std)
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
