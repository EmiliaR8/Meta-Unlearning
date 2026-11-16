"""Bring run records written by an older version of the runner up to date.

    python -m runner.migrate --group ember2018_stage1            # report only
    python -m runner.migrate --group ember2018_stage1 --apply    # do it

WHY THIS EXISTS. The run id is a hash of the configuration, which is what makes
a sweep resumable and stops two different runs sharing a filename. Adding a new
hyperparameter used to change that hash for every run, including ones already
finished -- so completed results stopped matching their own configuration and a
re-run would recompute them from scratch.

The hash now covers only hyperparameters that DIFFER from the dataset's
defaults, so a new knob left at its default no longer perturbs existing ids.
This tool performs the one-off catch-up for records written before that change:

  * fills in hyperparameters that did not exist when the record was written,
    at their current defaults -- which is what those runs actually used, since
    the behaviour those knobs select was the only behaviour available;
  * recomputes the run id and renames the file to match;
  * rewrites `run_id` inside the record so the file and its contents agree.

NOTHING SCIENTIFIC IS TOUCHED. Metrics, curves, per-task records, config and
environment are left exactly as written. If a record already matches, it is left
alone. Refuses to overwrite an existing file, so a genuine collision surfaces
rather than one result quietly replacing another.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.records import make_run_id
from runner import paths as paths_mod
from runner.config import default_hyperparams, overrides_of


def plan(directory: Path):
    """-> list of (path, new_id, added_keys). Reads nothing else."""
    out = []
    for path in sorted(directory.glob("*.json")):
        try:
            with path.open() as f:
                rec = json.load(f)
        except (OSError, ValueError) as e:
            print(f"  ! {path.name}: unreadable ({e}); skipped")
            continue
        cfg = rec.get("config")
        if not cfg:
            print(f"  ! {path.name}: no config block; skipped")
            continue
        hp = dict(rec.get("hyperparams") or {})
        defaults = default_hyperparams(cfg.get("dataset", ""))
        added = sorted(k for k in defaults if k not in hp)
        for k in added:
            hp[k] = defaults[k]
        new_id = make_run_id(cfg, overrides_of(cfg.get("dataset", ""), hp))
        out.append((path, rec, hp, new_id, added))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Migrate run records to the current id scheme")
    p.add_argument("--group", required=True)
    p.add_argument("--apply", action="store_true",
                   help="perform the rename/rewrite (default: report only)")
    p.add_argument("--log-root"); p.add_argument("--data-root")
    p.add_argument("--cache-root")
    args = p.parse_args(argv)

    store = paths_mod.resolve(args.data_root, args.log_root, args.cache_root)
    directory = store.json_dir(args.group)
    if not directory.is_dir():
        raise SystemExit(f"no such log group: {directory}")

    entries = plan(directory)
    if not entries:
        raise SystemExit(f"{directory} holds no readable run records")

    changed = [e for e in entries if e[0].stem != e[3]]
    print(f"{len(entries)} record(s) in {directory}")
    print(f"{len(changed)} need(s) migrating, {len(entries) - len(changed)} already current\n")

    for path, rec, hp, new_id, added in changed:
        print(f"  {path.name}")
        print(f"    -> {new_id}.json")
        if added:
            print(f"    + hyperparameters filled at current defaults: {', '.join(added)}")

    if not changed:
        return 0
    if not args.apply:
        print("\n(report only -- pass --apply to perform the rename)")
        return 0

    for path, rec, hp, new_id, added in changed:
        target = path.with_name(f"{new_id}.json")
        if target.exists():
            raise SystemExit(
                f"refusing to overwrite {target.name}, which already exists. Two "
                f"records claim the same configuration -- inspect them before "
                f"migrating.")
        rec["hyperparams"] = hp
        rec["run_id"] = new_id
        rec.setdefault("notes", []).append(
            "migrated: hyperparameters added after this run were filled at their "
            "defaults and the run id recomputed; metrics untouched")
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(rec, f, indent=2)
        tmp.replace(target)
        if target != path:
            path.unlink()
    print(f"\nmigrated {len(changed)} record(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
