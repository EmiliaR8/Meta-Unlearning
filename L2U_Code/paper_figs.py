"""Shared figure/table builders for the paper compilers.

One implementation for EMBER 2018, EMBER 2024 and LAMDA so the three cannot
drift. Each dataset compiler supplies its own file patterns and calls the
builders below.

Requirements this encodes
  R1  every figure has a matching table and vice versa; markdown() emits them paired
  R2  every panel is a mean over all runs found
  R2a stage 1 = naive / joint / MADAR / MADAR+unlearning, one MU line only, the
      better alpha arm, with no alpha in the label
  R2b stage 1 also gets a zoomed copy with naive omitted
  R2c stage 1 and meta each get an acquisition / task-0 decomposition panel
  R3  meta shows every selector, mode A and mode B together, labelled MADAR + <x>
  R4  no dotted lines anywhere; legend colours match the lines
  R5  every table carries mean +/- std of accuracy across tasks, and final task-0
  R6  file names are {dataset}_{category}_{content}

Statistic convention (R5). For each run, accuracy is averaged over tasks; the
table reports mean and std of that quantity ACROSS RUNS. The std therefore
measures run-to-run reproducibility, which is what the shaded bands in the
figures show, rather than drift along the sequence.

Nothing here is dataset-specific. Task indexing differs between the two families
and is handled: stage-1 logs include task 0, meta logs start at task 1.
"""

import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRID = "#E5E3DA"

# ---------------------------------------------------------------- styling
# R4: solid lines only. Colour alone distinguishes conditions, and every legend
# entry is generated from the same plot call that draws the line.
STAGE_STYLE = {                      # condition -> (colour, label, sort key)
    'naive':  ("#B4342A", "Naive", 0),
    'joint':  ("#0F6E56", "Joint retraining (oracle)", 1),
    'madar':  ("#185FA5", "MADAR", 2),
    'mu':     ("#534AB7", "MADAR + Unlearning", 3),
}
META_STYLE = {                       # selector -> (colour, label, sort key)
    'none':         ("#185FA5", "MADAR (no unlearning)", 0),
    'random':       ("#8C8A82", "MADAR + random 10%", 1),
    'donut':        ("#0F6E56", "MADAR + donut", 2),
    'gradconflict': ("#B08A2E", "MADAR + grad-conflict", 3),
    'densityratio': ("#993C1D", "MADAR + density-ratio", 4),
    'nn1':          ("#534AB7", "MADAR + learned NN-1", 5),
}

ACQ_KEYS = ['New_Fam_Acc', 'NewFam_Acc', 'New_Family_Acc', 'Acq_Acc', 'Newest_Acc']
T0_KEYS = ['Task0_Acc', 'Task_0_Acc']
ACC_KEYS = ['Avg_Acc', 'Avg_Accuracy']
RUN_RE = re.compile(r'^(?P<sel>.+?)_s(?P<seed>\d+)(?P<cfg>_.*)?\.json$')


def _first(d, keys):
    for k in keys:
        v = d.get(k)
        if isinstance(v, list) and len(v):
            return k
    return None


def _stack(curves):
    """(n_runs, n_tasks) from a list of equal-length curves; None if ragged."""
    if not curves:
        return None
    lens = {len(c) for c in curves}
    if len(lens) != 1:
        n = min(lens)
        curves = [c[:n] for c in curves]
    return np.array(curves, dtype=float)


# ---------------------------------------------------------------- loading
def load_stage1(patterns, alpha_of=None):
    """patterns: {condition: [glob, ...]}. Returns {condition: {...}}.

    MU files are grouped by alpha so the better arm can be chosen (R2a); every
    other condition collapses to a single group.
    """
    out = {}
    for cond, pats in patterns.items():
        files = []
        for p in pats:
            files += sorted(glob.glob(p))
        recs = []
        for f in sorted(set(files)):
            try:
                with open(f) as fh:
                    recs.append((f, json.load(fh)))
            except Exception:
                continue
        if not recs:
            continue
        if cond == 'mu':
            groups = defaultdict(list)
            for f, r in recs:
                a = alpha_of(f, r) if alpha_of else (r.get('config') or {}).get('alpha')
                groups[a].append((f, r))
            out[cond] = {'by_alpha': dict(groups)}
        else:
            out[cond] = {'runs': recs}
    return out


def stage_from_records(records, alpha_of=None):
    """Build the stage dict from records a compiler has already discovered.

    records: {condition: [(path, dict), ...]}, where condition is one of
    naive / joint / madar / mu. Each compiler keeps its own discovery logic --
    2018 uses fixed filename patterns, 2024 globs and filters on cap/arch/iters/si,
    LAMDA carries an _it{N} tag -- and only the rendering is shared.
    """
    out = {}
    for cond, recs in records.items():
        if not recs:
            continue
        if cond == 'mu':
            groups = defaultdict(list)
            for f, r in recs:
                a = alpha_of(f, r) if alpha_of else (r.get('config') or {}).get('alpha')
                groups[a].append((f, r))
            out['mu'] = {'by_alpha': dict(groups)}
        else:
            out[cond] = {'runs': list(recs)}
    return out


def pick_mu_arm(stage, reference='madar'):
    """R2a: keep whichever alpha improves most over the reference; drop the label.

    Returns (runs, alpha, note). Selecting the arm on its own outcome is a
    researcher degree of freedom, so the chosen alpha is returned for the caption
    rather than silently discarded.
    """
    if 'mu' not in stage or 'by_alpha' not in stage['mu']:
        return [], None, ""
    ref = None
    if reference in stage and 'runs' in stage[reference]:
        rr = _stack([r[_first(r, ACC_KEYS)] for _, r in stage[reference]['runs']])
        ref = rr.mean() if rr is not None else None
    best, best_a, scores = None, None, {}
    for a, runs in stage['mu']['by_alpha'].items():
        arr = _stack([r[_first(r, ACC_KEYS)] for _, r in runs])
        if arr is None:
            continue
        scores[a] = arr.mean() - (ref if ref is not None else 0.0)
        if best is None or scores[a] > scores[best_a]:
            best, best_a = runs, a
    note = ("alpha arms compared on mean accuracy: "
            + ", ".join(f"a={a}: {v:+.3f}" for a, v in sorted(scores.items(),
                                                              key=lambda kv: str(kv[0])))
            + f"; reported arm is alpha={best_a}") if scores else ""
    return best or [], best_a, note


def load_meta(workdirs, seeds=None):
    """-> data[(workdir, selector)] = {seed: record}, plus per-workdir config info."""
    data, info = defaultdict(dict), {}
    for wd in workdirs:
        if not os.path.isdir(wd):
            continue
        for path in sorted(glob.glob(os.path.join(wd, '*_s*.json'))):
            m = RUN_RE.match(os.path.basename(path))
            if not m:
                continue
            sel, seed = m.group('sel'), int(m.group('seed'))
            if seeds and seed not in seeds:
                continue
            try:
                with open(path) as f:
                    rec = json.load(f)
            except Exception:
                continue
            data[(wd, sel)][seed] = rec
            info.setdefault(wd, rec.get('config') or {})
    return data, info


# ---------------------------------------------------------------- helpers
def _summary_row(label, acc, t0, n):
    """R5: mean +/- std of per-run task-averaged accuracy, and final task-0."""
    per_run = acc.mean(axis=1)
    fin_t0 = t0[:, -1].mean() if t0 is not None and t0.size else float('nan')
    return [label, n, f"{per_run.mean():.3f}", f"{per_run.std(ddof=1) if n > 1 else 0:.3f}",
            f"{acc[:, -1].mean():.3f}", f"{fin_t0:.3f}" if np.isfinite(fin_t0) else ""]


SUMMARY_HEADER = ["Condition", "Runs", "Mean acc (%)", "Std (%)",
                  "Final acc (%)", "Final task-0 acc (%)"]


def _plot_band(ax, x, arr, colour, label):
    m, sd = arr.mean(axis=0), arr.std(axis=0)
    ax.plot(x, m, '-', marker='o', ms=3.5, lw=1.8, color=colour, label=label)
    ax.fill_between(x, m - sd, m + sd, color=colour, alpha=0.15, lw=0)
    return m


# ---------------------------------------------------------------- stage 1
def stage1_figures(stage, dataset, title, out_dir='.', mu_runs=None, mu_note=""):
    """Writes {ds}_stage1_accuracy{,_zoom}.png and _decomposition.png + tables.

    Returns {'figures': {...}, 'tables': {...}, 'notes': [...]}.
    """
    os.makedirs(out_dir, exist_ok=True)
    series, notes = {}, []
    if mu_note:
        notes.append(mu_note)

    for cond in ('naive', 'joint', 'madar'):
        if cond in stage and 'runs' in stage[cond]:
            series[cond] = [r for _, r in stage[cond]['runs']]
    if mu_runs:
        series['mu'] = [r for _, r in mu_runs]

    order = sorted(series, key=lambda c: STAGE_STYLE.get(c, ("", "", 99))[2])
    if not order:
        return {'figures': {}, 'tables': {}, 'notes': ['no stage-1 runs found']}

    acc = {c: _stack([r[_first(r, ACC_KEYS)] for r in series[c]]) for c in order}
    t0 = {c: _stack([r[k] for r in series[c] if (k := _first(r, T0_KEYS))]) for c in order}
    acq = {c: _stack([r[k] for r in series[c] if (k := _first(r, ACQ_KEYS))]) for c in order}
    n_tasks = min(a.shape[1] for a in acc.values() if a is not None)
    x = np.arange(n_tasks)          # stage-1 logs include task 0

    figures, tables = {}, {}

    # --- accuracy, full and zoomed (R2b) ---
    for tag, drop_naive in (("accuracy", False), ("accuracy_zoom", True)):
        fig, ax = plt.subplots(figsize=(8.5, 5.6))
        shown = [c for c in order if not (drop_naive and c == 'naive')]
        for c in shown:
            if acc[c] is None:
                continue
            col, lab, _ = STAGE_STYLE.get(c, ("#5F5E5A", c, 99))
            _plot_band(ax, x, acc[c][:, :n_tasks], col, lab)
        ax.set_xlabel("Task"); ax.set_ylabel("Average accuracy over seen families (%)")
        ax.set_title(title + ("\nzoomed: naive omitted" if drop_naive else ""), fontsize=11)
        ax.set_xticks(x); ax.grid(alpha=0.3, color=GRID); ax.legend(fontsize=9)
        fig.tight_layout()
        fp = os.path.join(out_dir, f"{dataset}_stage1_{tag}.png")
        fig.savefig(fp, dpi=150); plt.close(fig)
        figures[f"stage1_{tag}"] = fp

    rows = [SUMMARY_HEADER]
    for c in order:
        if acc[c] is None:
            continue
        rows.append(_summary_row(STAGE_STYLE.get(c, ("", c, 9))[1],
                                 acc[c][:, :n_tasks], t0[c], acc[c].shape[0]))
    tables["stage1_accuracy"] = rows

    # --- decomposition: acquisition + retention (R2c) ---
    have_acq = any(v is not None and v.size for v in acq.values())
    have_t0 = any(v is not None and v.size for v in t0.values())
    if not have_acq and not have_t0:
        notes.append("Neither the families-just-added nor the task-0 series was "
                     "recorded in this corpus's stage-1 logs, so the decomposition "
                     "figure and table are omitted rather than emitted blank.")
        return {'figures': figures, 'tables': tables, 'notes': notes}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    for ax, src, ttl, ylab in (
            (axes[0], acq, "Acquisition: accuracy on the families just added",
             "Accuracy on newest families (%)"),
            (axes[1], t0, "Retention: accuracy on the task-0 families",
             "Task-0 accuracy (%)")):
        for c in order:
            v = src.get(c)
            if v is None or not v.size:
                continue
            col, lab, _ = STAGE_STYLE.get(c, ("#5F5E5A", c, 99))
            _plot_band(ax, np.arange(v.shape[1]), v, col, lab)
        ax.set_xlabel("Task"); ax.set_ylabel(ylab)
        ax.set_title(ttl, fontsize=11)
        ax.grid(alpha=0.3, color=GRID)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=9)
    if not have_acq:
        axes[0].text(0.5, 0.5, "not recorded for this corpus\n(re-run required)",
                     ha='center', va='center', transform=axes[0].transAxes,
                     fontsize=11, color="#5F5E5A")
        notes.append("Acquisition was not recorded in this corpus's stage-1 logs; "
                     "the left panel is empty and the table omits that column.")
    fig.suptitle(title + " \u2014 acquisition and retention", fontsize=13)
    fig.tight_layout()
    fp = os.path.join(out_dir, f"{dataset}_stage1_decomposition.png")
    fig.savefig(fp, dpi=150); plt.close(fig)
    figures["stage1_decomposition"] = fp

    drows = [["Condition", "Runs", "Mean acquisition (%)", "Final acquisition (%)",
              "Mean task-0 (%)", "Final task-0 (%)"]]
    for c in order:
        a, t = acq.get(c), t0.get(c)
        if (a is None or not a.size) and (t is None or not t.size):
            continue
        drows.append([
            STAGE_STYLE.get(c, ("", c, 9))[1],
            (a.shape[0] if a is not None and a.size else
             (t.shape[0] if t is not None and t.size else 0)),
            f"{a.mean():.3f}" if a is not None and a.size else "",
            f"{a[:, -1].mean():.3f}" if a is not None and a.size else "",
            f"{t.mean():.3f}" if t is not None and t.size else "",
            f"{t[:, -1].mean():.3f}" if t is not None and t.size else ""])
    tables["stage1_decomposition"] = drows

    return {'figures': figures, 'tables': tables, 'notes': notes}


# ---------------------------------------------------------------- meta
def meta_figures(data, info, dataset, title, out_dir='.', metric='micro',
                 reference='none'):
    """Writes {ds}_meta_accuracy.png and _decomposition.png + tables.

    Selectors from mode A and mode B workdirs are pooled into one figure and one
    table (R3). Where the same selector appears in both, the mode is appended to
    the label so the two rows stay distinguishable.
    """
    os.makedirs(out_dir, exist_ok=True)
    key = 'per_task_macro' if metric == 'macro' else 'per_task'
    figures, tables, notes = {}, {}, []

    # selector -> {label: (colour, curves, workdir, mode)}
    modes = {wd: (info.get(wd) or {}).get('mode', '?') for wd in {k[0] for k in data}}
    dup = defaultdict(set)
    for (wd, sel) in data:
        dup[sel].add(modes.get(wd, '?'))

    entries = []
    for (wd, sel), runs in sorted(data.items(),
                                  key=lambda kv: META_STYLE.get(kv[0][1],
                                                                ("", "", 99))[2]):
        curves = [r[key] for r in runs.values() if key in r]
        arr = _stack(curves)
        if arr is None:
            continue
        col, lab, sk = META_STYLE.get(sel, ("#5F5E5A", f"MADAR + {sel}", 99))
        mode = modes.get(wd, '?')
        if len(dup[sel]) > 1:
            lab = f"{lab} [mode {mode.upper()}]"
        acq = _stack([r['per_task_new_fam'] for r in runs.values()
                      if 'per_task_new_fam' in r])
        t0 = _stack([r['per_task_task0'] for r in runs.values()
                     if 'per_task_task0' in r])
        entries.append(dict(sel=sel, label=lab, colour=col, sort=sk, mode=mode,
                            acc=arr, acq=acq, t0=t0, n=arr.shape[0], wd=wd))
    if not entries:
        return {'figures': {}, 'tables': {}, 'notes': ['no meta runs found']}
    entries.sort(key=lambda e: (e['sort'], e['mode']))

    n_tasks = min(e['acc'].shape[1] for e in entries)
    x = np.arange(1, n_tasks + 1)      # meta logs start at task 1

    # Two panels, matching the established EMBER 2018 meta figure: accuracy on the
    # left, paired per-seed differences on the right. The right panel is what makes
    # sub-point gaps legible at all -- the curves themselves overlap.
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    for e in entries:
        _plot_band(ax, x, e['acc'][:, :n_tasks], e['colour'], e['label'])
    ax.set_xlabel("Task")
    ax.set_ylabel("Average accuracy over seen families (%)"
                  + ("  [macro]" if metric == 'macro' else ""))
    n_seeds = max(e['n'] for e in entries)
    ax.set_title(f"Accuracy by task\nmean ± std over {n_seeds} seeds", fontsize=11)
    ax.set_xticks(x); ax.grid(alpha=0.3, color=GRID); ax.legend(fontsize=8.5,
                                                                loc='lower left')

    # Pairing is only valid within a workdir: selectors there share the seed AND the
    # task-0 checkpoint. Against a reference from a different workdir the subtraction
    # no longer cancels a common starting point, so that case is flagged.
    ref_by_wd = {}
    for e in entries:
        if e['sel'] == reference:
            ref_by_wd[e['wd']] = e
    any_ref = next(iter(ref_by_wd.values()), None)
    cross = []

    ax = axes[1]
    diffs = {}
    for e in entries:
        r = ref_by_wd.get(e['wd']) or any_ref
        if r is None or r is e:
            continue
        if r['wd'] != e['wd']:
            cross.append(e['label'])
        seeds_e = sorted(set(range(e['acc'].shape[0])))
        n = min(e['acc'].shape[1], r['acc'].shape[1], n_tasks)
        m_ = min(e['acc'].shape[0], r['acc'].shape[0])
        d = e['acc'][:m_, :n] - r['acc'][:m_, :n]
        diffs[e['label']] = d
        _plot_band(ax, np.arange(1, n + 1), d, e['colour'], e['label'])
    ax.axhline(0, color='black', lw=0.9)
    ax.set_xlabel("Task")
    ax.set_ylabel(f"Paired difference vs {META_STYLE.get(reference, ('', reference, 0))[1]} (pp)")
    ax.set_title("Paired per-seed differences\n(same seed, same task-0 checkpoint)",
                 fontsize=11)
    ax.set_xticks(x); ax.grid(alpha=0.3, color=GRID)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8.5, loc='best')
    if cross:
        notes.append("Paired differences for " + ", ".join(sorted(set(cross)))
                     + " use a reference from a different workdir, so they are not "
                       "paired on a common task-0 checkpoint.")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fp = os.path.join(out_dir, f"{dataset}_meta_accuracy.png")
    fig.savefig(fp, dpi=150); plt.close(fig)
    figures["meta_accuracy"] = fp

    rows = [SUMMARY_HEADER + ["Paired diff (pp)", "Wins"]]
    for e in entries:
        row = _summary_row(e['label'], e['acc'][:, :n_tasks], e['t0'], e['n'])
        d = diffs.get(e['label'])
        if d is None:
            row += ["—", "—"]
        else:
            per_seed = d.mean(axis=1)
            row += [f"{per_seed.mean():+.3f} ± {per_seed.std(ddof=1) if len(per_seed) > 1 else 0:.3f}",
                    f"{int((per_seed > 0).sum())}/{len(per_seed)}"]
        rows.append(row)
    tables["meta_accuracy"] = rows

    have_acq = any(e['acq'] is not None and e['acq'].size for e in entries)
    have_t0 = any(e['t0'] is not None and e['t0'].size for e in entries)
    if not have_acq and not have_t0:
        notes.append("Neither per_task_new_fam nor per_task_task0 was recorded in "
                     "these meta workdirs, so the decomposition figure and table are "
                     "omitted for this dataset rather than emitted blank.")
        return {'figures': figures, 'tables': tables, 'notes': notes}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    for ax, fld, ttl, ylab in (
            (axes[0], 'acq', "Acquisition: accuracy on the families just added",
             "Accuracy on newest families (%)"),
            (axes[1], 't0', "Retention: accuracy on the task-0 families",
             "Task-0 accuracy (%)")):
        for e in entries:
            v = e[fld]
            if v is None or not v.size:
                continue
            _plot_band(ax, np.arange(1, v.shape[1] + 1), v, e['colour'], e['label'])
        ax.set_xlabel("Task"); ax.set_ylabel(ylab)
        ax.set_title(ttl, fontsize=11)
        ax.grid(alpha=0.3, color=GRID)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8.5)
    if not have_acq:
        axes[0].text(0.5, 0.5, "not recorded in these workdirs\n(re-run required)",
                     ha='center', va='center', transform=axes[0].transAxes,
                     fontsize=11, color="#5F5E5A")
        notes.append("Acquisition / task-0 were not recorded in these meta workdirs.")
    fig.suptitle(title + " \u2014 acquisition and retention", fontsize=13)
    fig.tight_layout()
    fp = os.path.join(out_dir, f"{dataset}_meta_decomposition.png")
    fig.savefig(fp, dpi=150); plt.close(fig)
    figures["meta_decomposition"] = fp

    drows = [["Condition", "Runs", "Mean acquisition (%)", "Final acquisition (%)",
              "Mean task-0 (%)", "Final task-0 (%)"]]
    for e in entries:
        a, t = e['acq'], e['t0']
        if (a is None or not a.size) and (t is None or not t.size):
            continue
        drows.append([e['label'], e['n'],
                      f"{a.mean():.3f}" if a is not None and a.size else "",
                      f"{a[:, -1].mean():.3f}" if a is not None and a.size else "",
                      f"{t.mean():.3f}" if t is not None and t.size else "",
                      f"{t[:, -1].mean():.3f}" if t is not None and t.size else ""])
    tables["meta_decomposition"] = drows

    cfgs = {(wd, (info.get(wd) or {}).get('alpha'),
             (info.get(wd) or {}).get('unlearn_epochs')) for wd in {k[0] for k in data}}
    if len({(a, e) for _, a, e in cfgs}) > 1:
        notes.append("Workdirs pooled here do not share (alpha, unlearn_epochs): "
                     + "; ".join(f"{os.path.basename(w)} a={a} epochs={e}"
                                 for w, a, e in sorted(cfgs, key=lambda c: str(c[0])))
                     + ". State this in the caption.")
    return {'figures': figures, 'tables': tables, 'notes': notes}


# ---------------------------------------------------------------- markdown
def markdown(dataset, title, blocks, out_dir='.', extra_notes=()):
    """R1: emit every figure immediately followed by its table, and vice versa.

    blocks: list of (section title, result-dict) or (section title, result-dict,
    csv_tag). The optional third element names the CSVs for that block, which is
    needed when one dataset contributes several sets -- LAMDA reports two CL_ITERS
    budgets, and without it the second block's tables would overwrite the first's.
    Writes {dataset}_results.md and one CSV per table.
    """
    os.makedirs(out_dir, exist_ok=True)
    CAPTION = {
        'stage1_accuracy': "Accuracy by task, all stage-1 conditions.",
        'stage1_accuracy_zoom': "As above with naive omitted, to separate the "
                                "remaining conditions.",
        'stage1_decomposition': "Accuracy on the families just added (left) and on "
                                "the task-0 families (right).",
        'meta_accuracy': "Accuracy by task, all forget-set selectors.",
        'meta_decomposition': "Accuracy on the families just added (left) and on "
                              "the task-0 families (right).",
    }
    md = [f"# {title}", ""]
    for note in extra_notes:
        md += [f"> {note}", ""]

    for block in blocks:
        section, res = block[0], block[1]
        csv_tag = block[2] if len(block) > 2 else dataset
        md += [f"## {section}", ""]
        for note in res.get('notes', []):
            md += [f"> {note}", ""]
        figs, tabs = res.get('figures', {}), res.get('tables', {})
        # every figure, each followed by its table (the zoom shares the accuracy table)
        for fkey, fpath in figs.items():
            tkey = 'stage1_accuracy' if fkey == 'stage1_accuracy_zoom' else fkey
            md += [f"### {CAPTION.get(fkey, fkey)}", "",
                   f"![{fkey}]({os.path.basename(fpath)})", ""]
            rows = tabs.get(tkey)
            if rows:
                md += ["| " + " | ".join(str(c) for c in rows[0]) + " |",
                       "|" + "|".join(["---"] * len(rows[0])) + "|"]
                md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows[1:]]
                md += [""]
                if fkey == 'stage1_accuracy_zoom':
                    md += ["*(same data as the figure above)*", ""]
            else:
                md += ["*(no table: the underlying field was not recorded)*", ""]
        # any table with no figure of its own
        for tkey, rows in tabs.items():
            if tkey in figs or tkey == 'stage1_accuracy':
                continue
            md += [f"### {CAPTION.get(tkey, tkey)} (table)", "",
                   "| " + " | ".join(str(c) for c in rows[0]) + " |",
                   "|" + "|".join(["---"] * len(rows[0])) + "|"]
            md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows[1:]]
            md += [""]
        for tkey, rows in tabs.items():
            cp = os.path.join(out_dir, f"{csv_tag}_{tkey}.csv")
            with open(cp, 'w') as f:
                for r in rows:
                    f.write(",".join(str(c) for c in r) + "\n")

    mp = os.path.join(out_dir, f"{dataset}_results.md")
    with open(mp, 'w') as f:
        f.write("\n".join(md) + "\n")
    return mp