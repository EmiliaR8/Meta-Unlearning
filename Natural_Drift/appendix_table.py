"""Per-seed appendix tables, one LaTeX table per dataset.

    python3 appendix_table.py <log_root>/<group> [<group> ...] [--tex]
                              [--cols seen task0] [--csv out.csv]

ONE ROW PER SEED -- nothing here is averaged across seeds. The four quantities
are the scopes the runs already report, read straight out of the record, so a
cell here equals the same cell in the aggregator's tables by construction rather
than by two code paths agreeing.

  Mean Acc.       mean of `seen` accuracy over ALL tasks
  Acc. Final      `seen` accuracy after the final task
  Task-0 Final    `task0` accuracy after the final task -- the first task's classes
  F1              macro F1 over `seen` after the final task

`--cols` names the two middle scopes, defaulting to `seen task0`; `recent` is
available if a column should be the last task's own classes instead. Nothing is
recomputed from confusion matrices, so runs stored with `--confusion none`
tabulate fine.

F1 is MACRO F1 -- the unweighted mean over classes with support, which is what
the rest of the reporting uses. Precision and recall are kept in the --csv but
are not tabulated, since only F1 is reported.

The caption is a placeholder -- it states the column provenance so a draft is
never ambiguous about it, and is expected to be rewritten in the paper.
"""
import argparse, csv, json, statistics as st
from pathlib import Path

SCOPES = ("seen", "recent", "task0")

NICE = {"naive": "Naive", "joint": "Joint", "si_only": "Synaptic Intelligence",
        "er_only": "Experience Replay", "agem": "A-GEM", "malcl": "MalCL",
        "xder": "X-DER", "star_xder": "X-DER + STAR",
        "deduce_star_xder": "DEDUCE", "gcr": "GCR", "madar": "MADAR",
        "madar_unlearn": "MADAR + Unlearning", "meta_unlearn": "Meta-Unlearning"}
ORDER = ["joint", "naive", "si_only", "er_only", "malcl", "agem", "xder",
         "star_xder", "deduce_star_xder", "gcr", "madar", "madar_unlearn",
         "meta_unlearn"]
DATASET = {"ember2018": "Ember'18", "ember2024": "Ember'24",
           "lamda_classil": "LAMDA", "tinyimagenet": "Tiny ImageNet",
           "synthetic": "Synthetic"}

# Already carried by the table's structure (dataset -> its own table, method ->
# NICE name, model -> bracket, seed -> its own row), so repeating them in the
# disambiguating parenthetical is noise.
SKIP_KEYS = {"seed", "group", "tag", "notes", "quiet", "out", "paths",
             "dataset", "method", "model", "task_setup"}
_ABSENT = object()


def rows_for(groups, cols, protocol):
    out = []
    skipped = {}
    for group in groups:
        for f in sorted(Path(group).glob("jsons/*.json")):
            try:
                r = json.loads(f.read_text())
            except (OSError, ValueError):
                print(f"  ! {f.name}: unreadable; skipped"); continue
            per = r.get("per_task") or []
            if not per:
                print(f"  ! {f.name}: no completed tasks; skipped"); continue
            cfg, last = r["config"], per[-1]
            proto = cfg.get("protocol", "stage1")
            if protocol != "any" and proto != protocol:
                skipped[proto] = skipped.get(proto, 0) + 1
                continue
            m, sched = last["metrics"], r.get("task_setup") or {}
            row = {"dataset": cfg["dataset"], "protocol": proto,
                   "method": cfg["method"],
                   "model": cfg.get("model"), "seed": cfg.get("seed"),
                   "label": NICE.get(cfg["method"], cfg["method"]),
                   "mean_acc": st.mean(t["metrics"]["seen"]["accuracy"] for t in per),
                   "P": m["seen"]["macro_precision"], "R": m["seen"]["macro_recall"],
                   "F1": m["seen"]["macro_f1"], "run_id": r["run_id"],
                   "n_tasks": len(per),
                   "last_task": int(last.get("task", len(per) - 1)),
                   "n_inc": int(sched.get("n_increments", 0)) if sched else 0,
                   "_cfg": cfg, "_hp": r.get("hyperparams", {})}
            for c in cols:
                row[c] = m.get(c, {}).get("accuracy")
            out.append(row)

    # Bracket the model only where it actually varies. Deciding that globally
    # stamps "[mlp]" on every row of a single-model corpus because some OTHER
    # corpus in the same sweep ran a size study.
    for key in {(r["dataset"], r["protocol"]) for r in out}:
        sub = [r for r in out if (r["dataset"], r["protocol"]) == key]
        if len({r["model"] for r in sub}) > 1:
            for r in sub:
                r["label"] = f"{r['label']} [{r['model']}]"
    disambiguate(out)
    out.sort(key=lambda d: (d["dataset"], d["protocol"],
                            ORDER.index(d["method"]) if d["method"] in ORDER
                            else len(ORDER), d["label"], d["seed"]))
    if skipped:
        note = ", ".join(f"{n} {k}" for k, n in sorted(skipped.items()))
        print(f"  (skipped {note} run(s): --protocol {protocol})")
    return out


def disambiguate(out):
    """Label whatever actually separates runs of one method within a dataset.

    A method with more runs than seeds has a second axis -- an alpha=0 control,
    two si_c values -- and merging those under one name lists the same seed
    twice with different numbers and no way to tell them apart. Discovered, not
    hardcoded, so a knob added later is handled without editing this.
    """
    by = {}
    for r in out:
        by.setdefault((r["dataset"], r["protocol"], r["method"]), []).append(r)
    for _, rs in by.items():
        if len(rs) == len({r["seed"] for r in rs}):
            continue
        keys = set()
        for r in rs:
            keys |= {("_cfg", k) for k in r["_cfg"]} | {("_hp", k) for k in r["_hp"]}
        varying = [(src, k) for src, k in sorted(keys)
                   if k not in SKIP_KEYS
                   and len({str(r[src].get(k, _ABSENT)) for r in rs}) > 1]
        for r in rs:
            # An absent key is the default, not the value None: a record written
            # before a knob existed has no entry for it, and printing
            # "alpha=None" beside "alpha=0.0" reads as a third arm that is not
            # there.
            bits = [f"{k}={r[src][k]}" if k in r[src] else f"{k}=default"
                    for src, k in varying]
            if bits:
                r["label"] = f"{r['label']} ({', '.join(bits)})"


_TEX = {"\\": "\\textbackslash{}", "&": "\\&", "%": "\\%", "$": "\\$",
        "#": "\\#", "_": "\\_", "{": "\\{", "}": "\\}",
        "~": "\\textasciitilde{}", "^": "\\textasciicircum{}"}


def tex(s: str) -> str:
    """Escape a label for LaTeX text mode.

    Model and method names carry underscores -- mlp_small, star_xder, si_only --
    and an unescaped one is a hard compile error, so a table that looks right in
    the terminal would not build. Backslash first, or the replacements escape
    each other's output.
    """
    out = str(s).replace("\\", "\x00")
    for k, v in _TEX.items():
        if k != "\\":
            out = out.replace(k, v)
    return out.replace("\x00", _TEX["\\"])


WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
         7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def headers(rows, cols):
    """Column headers for this dataset, named after the task each scope means.

    `recent` after the final task IS that task, so the header carries its index
    rather than the scope's internal name; `task0` is the first task. Where the
    dataset mixes schedule lengths the header would be a lie, so it falls back
    to the scope name.
    """
    last = {r["last_task"] for r in rows}
    out = []
    for c in cols:
        if c == "recent":
            out.append(f"Task {max(last)}" if len(last) == 1 else "Last Task")
        elif c == "task0":
            out.append("Task-0 Final")
        else:
            out.append("Acc. Final")
    return out


def emit_tex(rows, dataset, cols, hdrs, proto=None):
    name = DATASET.get(dataset, dataset)
    # Only when more than one protocol is on the page: otherwise the caption
    # and label carry a distinction the table does not need to make.
    suffix = f" under the {proto} task-0 protocol" if proto else ""
    lab = f"{str(dataset).replace('_', '')}{'_' + proto if proto else ''}"
    a, b = hdrs
    ca, cb = cols
    n_seed = len({r["seed"] for r in rows})
    count = WORDS.get(n_seed, str(n_seed))
    print(f"\n% ======== {dataset} ========")
    print("\\begin{table}[H]")
    print("    \\centering")
    print(f"    \\caption{{Performance across {count} random seed"
          f"{'' if n_seed == 1 else 's'} for the "
          f"\\textbf{{{tex(name)} dataset}}{tex(suffix)}. Mean Acc. denotes mean accuracy "
          f"across all tasks; {tex(a)} and {tex(b)} are the {tex(ca)} and "
          f"{tex(cb)} scopes after the final task. F1 denotes macro F1 over "
          f"seen classes after the final task.}}")
    print(f"    \\label{{tab:seed_results_{lab}}}")
    print("")
    print("    \\small")
    print("    \\setlength{\\tabcolsep}{3pt}")
    print("    \\renewcommand{\\arraystretch}{1.05}")
    print("")
    print("    \\begin{tabular}{llcccc}")
    print("        \\toprule")
    print("        \\textbf{Method} &")
    print("        \\textbf{Seed} &")
    print("        \\textbf{Mean Acc.} &")
    print(f"        \\textbf{{{tex(a)}}} &")
    print(f"        \\textbf{{{tex(b)}}} &")
    print("        \\textbf{F1} \\\\")
    print("        \\midrule")
    by = {}
    for r in rows:
        by.setdefault(r["label"], []).append(r)
    for i, (label, rs) in enumerate(by.items()):
        if i:
            print("        \\midrule")
        print("        ")
        print(f"        \\multirow{{{len(rs)}}}{{*}}{{{tex(label)}}}")
        for r in rs:
            def fmt(v):
                return "--" if v is None else f"{v:.2f}"
            print(f"        & {r['seed']} & {fmt(r['mean_acc'])} & "
                  f"{fmt(r[ca])} & {fmt(r[cb])} & {fmt(r['F1'])} \\\\")
    print("        ")
    print("        \\bottomrule")
    print("    \\end{tabular}")
    print("\\end{table}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="+")
    ap.add_argument("--tex", action="store_true")
    ap.add_argument("--cols", nargs=2, default=["seen", "task0"],
                    choices=SCOPES, metavar=("A", "B"),
                    help=f"scopes for the two accuracy columns, from {SCOPES} "
                         f"(default: seen task0)")
    ap.add_argument("--protocol", default="stage1",
                    choices=["stage1", "meta", "any"],
                    help="which task-0 protocol to tabulate; `any` emits a "
                         "separate table per protocol, never a pooled one "
                         "(default: stage1)")
    ap.add_argument("--csv")
    a = ap.parse_args()

    cols = tuple(a.cols)
    rows = rows_for(a.groups, cols, a.protocol)
    if not rows:
        raise SystemExit("no usable records found")
    ca, cb = cols
    hdr = ["mean_acc", ca, cb, "P", "R", "F1"]

    all_rows = []
    keys = sorted({(r["dataset"], r["protocol"]) for r in rows})
    multi_proto = len({k[1] for k in keys}) > 1
    for ds, proto in keys:
        sub = [r for r in rows if (r["dataset"], r["protocol"]) == (ds, proto)]
        hdrs = headers(sub, cols)
        tag = f"  [protocol {proto}]" if multi_proto else ""
        print(f"\n=== {DATASET.get(ds, ds)}{tag} ===")
        print(f"    columns: {hdrs[0]} = `{ca}` scope, {hdrs[1]} = `{cb}` scope,"
              f" both after the final task")
        print(f"{'method':>36} {'seed':>5} " + " ".join(f"{h:>10}" for h in hdr))
        for r in sub:
            print(f"{r['label']:>36} {str(r['seed']):>5} "
                  + " ".join(("%10s" % "--") if r[h] is None else f"{r[h]:10.3f}"
                             for h in hdr))

        short = [r for r in sub if r["n_tasks"] < r["n_inc"] + 1]
        if short:
            print(f"  ! {len(short)} run(s) stopped before the last task, so "
                  f"their final-task columns are from an earlier task. "
                  f"Run ids: {', '.join(r['run_id'] for r in short[:3])}"
                  f"{' ...' if len(short) > 3 else ''}")
        if len({r["last_task"] for r in sub}) > 1:
            print(f"  ! this dataset mixes schedule lengths, so '{hdrs[0]}' is "
                  f"not one task index across rows; the header says 'Last Task'.")
        if a.tex:
            emit_tex(sub, ds, cols, hdrs, proto if multi_proto else None)
        for r in sub:
            r["col_a"], r["col_b"] = r[ca], r[cb]
        all_rows += sub

    if a.csv:
        # col_a/col_b rather than the scope name: the header a scope carries
        # varies by dataset, and the pair is chosen by --cols.
        fields = ["dataset", "protocol", "method", "label", "model", "seed",
                  "mean_acc", "col_a", "col_b", "P", "R", "F1", "last_task",
                  "run_id"]
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(all_rows)
        print(f"\nwrote {a.csv} (col_a = `{ca}`, col_b = `{cb}`)")


if __name__ == "__main__":
    main()
