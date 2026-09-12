"""Aggregate a log group into tidy CSVs and summary tables.

    python -m runner.aggregate --group madar_fidelity

Reads every run JSON in Logs/<group>/jsons/ and writes to Logs/<group>/tables/:

    runs.csv        one row per run   -- the flat summary scalars
    per_task.csv    one row per (run, task) -- the curves, tidy
    summary.csv     one row per condition, mean +/- std across seeds
    summary.md      the same, formatted

WHAT IS AND IS NOT AVERAGED TOGETHER. A "condition" is the tuple
(dataset, task_setup, method, model, selector, protocol) and the aggregate is
over SEEDS within it. Runs differing in any other recorded hyperparameter are
also separated -- the run id already encodes them, and pooling two operating
points into one mean is not a mean of anything.

PROTOCOL IS A HARD BOUNDARY. stage1 runs retrain task 0 per seed; meta runs share
one task-0 checkpoint. Their accuracies are not comparable, so a group holding
both is reported as separate blocks and never differenced. This is enforced
rather than documented because differencing across it is the single easiest way
to manufacture a large effect that is entirely protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from runner import paths as paths_mod
from runner.config import DATASET_MODEL, default_hyperparams

CONDITION_KEYS = ("dataset", "task_setup", "method", "model", "selector", "protocol")

# The metrics asked for in tables, as (column label, curve key).
CURVES = {
    "accuracy": "accuracy_seen",
    "accuracy_task0": "accuracy_task0",
    "accuracy_recent": "accuracy_recent",
    "precision": "macro_precision_seen",
    "recall": "macro_recall_seen",
    "f1": "macro_f1_seen",
}

# Hyperparameters that must never be pooled, even when every loaded run happens
# to agree on them. Everything else is discovered (see split_keys): a hardcoded
# list is a promise to remember every future knob, and that promise was broken
# twice -- buffer_space pooled MADAR with MADAR-theta, momentum pooled A-GEM's
# faithful arm with the harness default.
SPLIT_ALWAYS = ("cl_iters", "mem_size", "si_c", "alpha", "forget_ratio",
                "unlearn_epochs", "task0_epochs", "family_cap",
                "buffer_space", "buffer_policy", "keep_rnt", "freeze_bn")

# Recorded per run but not method-defining: splitting on these would fragment
# genuine replicates. cap_seed is fixed by design; the seed itself is the
# replicate axis and lives outside the condition.
NEVER_SPLIT = frozenset({"cap_seed"})


def load_group(store, group: str) -> list[dict]:
    directory = store.json_dir(group)
    if not directory.is_dir():
        raise SystemExit(f"no such log group: {directory}")
    records = []
    for path in sorted(directory.glob("*.json")):
        try:
            with path.open() as f:
                rec = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  ! {path.name}: unreadable ({e}); skipped")
            continue
        if "curves" not in rec or not rec.get("per_task"):
            print(f"  ! {path.name}: no completed tasks; skipped")
            continue
        rec["_path"] = str(path)
        records.append(rec)
    if not records:
        raise SystemExit(f"{directory} contains no usable run records")
    return records


def split_keys(records) -> tuple:
    """Which hyperparameters actually distinguish the loaded runs.

    Any parameter whose value VARIES across the group becomes part of the
    condition, so two runs differing in it can never be averaged together. This
    is discovered rather than declared: the failure mode is always a knob nobody
    remembered to add to a list, and a list cannot anticipate one that does not
    exist yet.
    """
    seen = {}
    for rec in records:
        cfg = rec["config"]
        fb = default_hyperparams(cfg.get("dataset", ""), cfg.get("method"))
        hp = rec.get("hyperparams", {})
        for k in set(hp) | set(fb):
            if k in NEVER_SPLIT:
                continue
            seen.setdefault(k, set()).add(str(hp.get(k, fb.get(k))))
    varying = {k for k, v in seen.items() if len(v) > 1}
    return tuple(sorted(varying | set(SPLIT_ALWAYS)))


def condition_of(rec: dict, keys=SPLIT_ALWAYS) -> tuple:
    """Records written before a hyperparameter existed read it as that
    parameter's default, which is the value they in fact ran with. Reading it as
    None instead would split one condition into two and refuse to pool runs that
    are genuinely replicates."""
    cfg = rec["config"]
    base = tuple(str(cfg.get(k)) for k in CONDITION_KEYS)
    hp = rec.get("hyperparams", {})
    fallback = default_hyperparams(cfg.get("dataset", ""), cfg.get("method"))
    return base + tuple(str(hp.get(k, fallback.get(k))) for k in keys)


def condition_label(cond: tuple, keys=SPLIT_ALWAYS) -> dict:
    out = dict(zip(CONDITION_KEYS, cond[:len(CONDITION_KEYS)]))
    out.update(dict(zip(keys, cond[len(CONDITION_KEYS):])))
    return out


def short_label(cond: tuple, keys=SPLIT_ALWAYS) -> str:
    """Method, plus only what distinguishes this condition from the norm.

    The model is named only when it is NOT the corpus's own default. Comparing
    against the literal string "full" instead meant every Tiny ImageNet series
    was labelled `... + resnet18` -- a suffix carried by all of them, so it
    separated nothing, lengthened every legend entry and pushed the figure
    caption off the canvas. A model worth naming is one that differs from what
    the corpus would have used anyway.
    """
    d = condition_label(cond, keys)
    bits = [d["method"]]
    if d["selector"] not in (None, "None"):
        bits.append(d["selector"])
    if str(d["model"]) != DATASET_MODEL.get(d.get("dataset"), "full"):
        bits.append(d["model"])
    return " + ".join(bits)


def label_conditions(conditions, keys=SPLIT_ALWAYS) -> dict:
    """Assign each condition a label that is unique within the set.

    Two conditions differing only in a hyperparameter (si_only at si_c 1.0 vs
    0.0, say) share a short_label. Left alone that makes a table row ambiguous
    and silently MERGES two series into one line in a figure. Where a collision
    occurs, the fields that actually differ are appended -- so the label says
    what distinguishes the rows rather than hiding it.
    """
    conditions = list(conditions)
    base = {c: short_label(c, keys) for c in conditions}
    groups = {}
    for c, lab in base.items():
        groups.setdefault(lab, []).append(c)

    out = {}
    for lab, members in groups.items():
        if len(members) == 1:
            out[members[0]] = lab
            continue
        allk = list(CONDITION_KEYS) + list(keys)
        differing = [k for i, k in enumerate(allk)
                     if len({m[i] for m in members}) > 1]
        for m in members:
            d = condition_label(m, keys)
            extra = ", ".join(f"{k}={d[k]}" for k in differing)
            out[m] = f"{lab} ({extra})" if extra else lab
    return out


def write_csv(path: Path, rows: list[dict]) -> Path:
    if not rows:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    fields, seen = [], set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k); fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def run_rows(records: list[dict]) -> list[dict]:
    rows = []
    for rec in records:
        cfg, hp = rec["config"], rec.get("hyperparams", {})
        row = {"run_id": rec["run_id"]}
        row.update({k: cfg.get(k) for k in CONDITION_KEYS})
        row["seed"] = cfg.get("seed")
        row.update({k: hp.get(k) for k in split_keys(records)})
        row["n_params"] = rec.get("model", {}).get("n_params")
        row.update({k: v for k, v in rec.get("summary", {}).items()})
        row["git_commit"] = (rec.get("environment") or {}).get("git_commit")
        rows.append(row)
    return rows


def per_task_rows(records: list[dict]) -> list[dict]:
    rows = []
    for rec in records:
        cfg = rec["config"]
        curves = rec["curves"]
        for i, task in enumerate(rec["per_task"]):
            row = {"run_id": rec["run_id"]}
            row.update({k: cfg.get(k) for k in CONDITION_KEYS})
            row["seed"] = cfg.get("seed")
            row["task"] = task["task"]
            row["active_count"] = task["active_count"]
            row["n_train"] = task["n_train"]
            row["grad_steps"] = task["grad_steps"]
            for label, key in CURVES.items():
                series = curves.get(key) or []
                row[label] = series[i] if i < len(series) else None
            rows.append(row)
    return rows


def summarise(records: list[dict]) -> list[dict]:
    """One row per condition: mean +/- std ACROSS SEEDS of each per-run scalar.

    The per-run scalar is that run's metric averaged over tasks, so the std
    measures run-to-run reproducibility, not drift along the sequence. Those are
    different quantities and conflating them overstates or understates stability
    depending on which way the curve slopes.
    """
    keys = split_keys(records)
    groups = defaultdict(list)
    for rec in records:
        groups[condition_of(rec, keys)].append(rec)

    labels = label_conditions(groups, keys)
    rows = []
    for cond, recs in sorted(groups.items()):
        row = condition_label(cond, keys)
        row["label"] = labels[cond]
        row["n_runs"] = len(recs)
        row["seeds"] = ",".join(str(r["config"].get("seed")) for r in
                                sorted(recs, key=lambda r: r["config"].get("seed", 0)))
        for label, key in CURVES.items():
            per_run_mean, per_run_final = [], []
            for rec in recs:
                series = [v for v in (rec["curves"].get(key) or []) if v is not None]
                if series:
                    per_run_mean.append(float(np.mean(series)))
                    per_run_final.append(float(series[-1]))
            if per_run_mean:
                row[f"{label}_mean"] = round(float(np.mean(per_run_mean)), 4)
                row[f"{label}_std"] = round(
                    float(np.std(per_run_mean, ddof=1)) if len(per_run_mean) > 1 else 0.0, 4)
                row[f"{label}_final"] = round(float(np.mean(per_run_final)), 4)
        rows.append(row)
    return rows


def markdown_table(rows: list[dict], columns: list[str], headers: list[str]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        cells = []
        for col in columns:
            v = row.get(col)
            cells.append("" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def render_markdown(group: str, summary: list[dict], notes: list[str]) -> str:
    by_protocol = defaultdict(list)
    for row in summary:
        by_protocol[row["protocol"]].append(row)

    doc = [f"# {group}", ""]
    for note in notes:
        doc += [f"> {note}", ""]
    for protocol, rows in sorted(by_protocol.items()):
        doc += [f"## protocol: {protocol}", ""]
        doc += ["Mean +/- std across seeds of each run's task-averaged metric. "
                "`final` is the mean across seeds of the last task's value.", ""]
        cols = ["label", "n_runs", "seeds",
                "accuracy_mean", "accuracy_std", "accuracy_final",
                "accuracy_task0_final", "accuracy_recent_mean",
                "precision_mean", "recall_mean", "f1_mean"]
        heads = ["condition", "runs", "seeds", "acc mean", "acc std", "acc final",
                 "task-0 final", "recent mean", "precision", "recall", "macro F1"]
        doc += [markdown_table(rows, cols, heads), ""]
    if len(by_protocol) > 1:
        doc += ["> **Protocols are reported separately and are not differenced.** "
                "stage1 retrains task 0 per seed; meta shares one task-0 checkpoint, "
                "so a difference between them is dominated by protocol, not method.",
                ""]
    return "\n".join(doc)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Aggregate a log group")
    p.add_argument("--group", required=True)
    p.add_argument("--log-root"); p.add_argument("--data-root")
    p.add_argument("--cache-root")
    p.add_argument("--out-dir", default=None,
                   help="default: Logs/<group>/tables/")
    args = p.parse_args(argv)

    store = paths_mod.resolve(args.data_root, args.log_root, args.cache_root)
    records = load_group(store, args.group)
    out_dir = Path(args.out_dir) if args.out_dir else store.table_dir(args.group)
    out_dir.mkdir(parents=True, exist_ok=True)

    notes = []
    protocols = {r["config"].get("protocol") for r in records}
    if len(protocols) > 1:
        notes.append(f"This group holds more than one protocol ({', '.join(sorted(map(str, protocols)))}). "
                     f"They are tabulated separately and never differenced.")
    synthetic = [r for r in records if r["config"].get("dataset") == "synthetic"]
    if synthetic:
        notes.append(f"{len(synthetic)} run(s) use the SYNTHETIC smoke-test corpus "
                     f"and are not scientific results.")

    summary = summarise(records)
    written = [
        write_csv(out_dir / "runs.csv", run_rows(records)),
        write_csv(out_dir / "per_task.csv", per_task_rows(records)),
        write_csv(out_dir / "summary.csv", summary),
    ]
    md = out_dir / "summary.md"
    md.write_text(render_markdown(args.group, summary, notes))
    written.append(md)

    print(f"{len(records)} run(s), {len(summary)} condition(s)")
    for note in notes:
        print(f"  note: {note}")
    for path in written:
        print(f"  wrote {path}")
    print()
    print(render_markdown(args.group, summary, []))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
