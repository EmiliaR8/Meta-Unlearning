"""Single-run entry point.

    python -m runner.run --dataset ember2018 --task-setup ember \
        --experiment madar_unlearn --model full --seed 1 --group madar_fidelity

Four axes are selectable, as requested -- dataset, task setup, experiment
(method), model -- plus the ones a comparison turns out to need: the forget-set
selector, the seed, the task-0 protocol, and any hyperparameter override.

Writes exactly one JSON to <log_root>/<group>/jsons/. Use --dry-run to resolve
and print the configuration without training, which is also the fastest way to
check that the storage paths point where you think they do.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # make `core` importable

import numpy as np
import torch

import experiments
from core import data as data_registry
from core import selectors
from core.records import RunRecord, environment, make_run_id
from core.seeding import seed_everything
from core.tasks import PRESETS as TASK_PRESETS, build_schedule
from core.models import PRESETS as MODEL_PRESETS, model_info
from runner import paths as paths_mod
from runner.config import (BASE_HYPERPARAMS, PROTOCOLS, describe, hyperparams_for,
                           load_config_file, overrides_of, split_config)

# Two methods take a --selector, and they mean different things by it. Validating
# the pairing stops a MalCL scheme name silently reaching the unlearning selector
# (or vice versa) and being ignored rather than rejected.
MALCL_SELECTORS = ["L1_C_Mean", "L1_B_Mean", "L2_One_Hot"]
SELECTOR_DEFAULTS = {"madar_unlearn": "donut", "malcl": "L1_C_Mean"}


class RunContext:
    """Everything an Experiment needs that is not method-specific."""

    def __init__(self, config, hyperparams, schedule, X_train, y_train,
                 X_test, y_test, device, n_tasks):
        self.config = config
        self.hyperparams = hyperparams
        self.schedule = schedule
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.device = device
        self.n_tasks = n_tasks
        self.input_dim = int(X_train.shape[1])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run one continual-learning experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    sel = p.add_argument_group("what to run")
    sel.add_argument("--dataset", default="ember2018",
                     choices=data_registry.available())
    sel.add_argument("--task-setup", default=None,
                     help=f"preset ({', '.join(sorted(TASK_PRESETS))}) or "
                          f"'<task0>+<step>x<n>'. Default: the dataset's own.")
    sel.add_argument("--experiment", "--method", dest="experiment", default="madar",
                     choices=experiments.available(),
                     help="the training method / condition")
    sel.add_argument("--model", default="full",
                     help=f"preset ({', '.join(sorted(MODEL_PRESETS))}) or four widths")
    sel.add_argument("--selector", default=None,
                     choices=selectors.available() + MALCL_SELECTORS,
                     help="madar_unlearn: forget-set rule (donut|random|leftover, "
                          "default donut). malcl: replay-selection scheme "
                          "(L1_C_Mean|L1_B_Mean|L2_One_Hot, default L1_C_Mean). "
                          "Ignored by every other method.")
    sel.add_argument("--seed", type=int, default=1)
    sel.add_argument("--n-tasks", type=int, default=None,
                     help="stop after this many tasks (default: the whole schedule)")
    sel.add_argument("--split-mode", default="random", choices=["random", "temporal"],
                     help="LAMDA only. 'random' uses the curation's own split "
                          "(primary); 'temporal' trains on year < --temporal-cut "
                          "(the drift-preserving secondary study).")
    sel.add_argument("--temporal-cut", type=int, default=2020,
                     help="LAMDA --split-mode temporal: first held-out year")

    org = p.add_argument_group("logging and organisation")
    org.add_argument("--group", default="default",
                     help="Logs/<group>/ -- the unit the aggregator averages over")
    org.add_argument("--tag", default=None, help="free-form suffix on the run id")
    org.add_argument("--protocol", default="stage1", choices=PROTOCOLS,
                     help="stage1 retrains task 0 per seed; meta shares a task-0 "
                          "checkpoint. Never differenced against each other.")
    org.add_argument("--confusion", default="sparse", choices=["sparse", "none"],
                     help="store the per-task confusion matrix (sparse COO) or not")
    org.add_argument("--notes", default=None, help="free text recorded in the run")

    io = p.add_argument_group("storage")
    io.add_argument("--data-root"); io.add_argument("--log-root")
    io.add_argument("--cache-root")
    io.add_argument("--config", help="JSON/YAML file of defaults; CLI flags win")

    ctl = p.add_argument_group("control")
    ctl.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ctl.add_argument("--dry-run", action="store_true",
                     help="resolve and print the configuration, then exit")
    ctl.add_argument("--force", action="store_true",
                     help="rerun even if this run's JSON already exists")
    ctl.add_argument("--quiet", action="store_true")

    hp = p.add_argument_group(
        "hyperparameter overrides (any of these wins over the dataset protocol)")
    for name, default in sorted(BASE_HYPERPARAMS.items()):
        # argparse type=bool is a trap: bool("False") is True, so every boolean
        # flag would silently be on. Booleans get an explicit parser instead.
        kind = _str2bool if isinstance(default, bool) else type(default)
        hp.add_argument(f"--{name.replace('_', '-')}", dest=f"hp_{name}",
                        type=kind, default=None)
    return p


def _stale_reasons(path, config: dict, hyperparams: dict) -> list:
    """Fields where a stored record disagrees with the run being asked for."""
    try:
        with open(path) as f:
            rec = json.load(f)
    except (OSError, ValueError):
        return [f"{path.name} could not be read as JSON"]
    out = []
    stored_cfg = rec.get("config") or {}
    for k in ("dataset", "task_setup", "method", "model", "selector", "protocol",
              "seed", "n_tasks"):
        if k in stored_cfg and stored_cfg[k] != config.get(k):
            out.append(f"config.{k}: stored {stored_cfg[k]!r} != requested "
                       f"{config.get(k)!r}")
    stored_hp = rec.get("hyperparams") or {}
    for k, v in sorted(hyperparams.items()):
        # A key absent from an older record was not yet a parameter; it ran at
        # what is now the default, so absence is agreement, not disagreement.
        if k in stored_hp and stored_hp[k] != v:
            out.append(f"hyperparams.{k}: stored {stored_hp[k]!r} != requested {v!r}")
    return out


def _str2bool(v):
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {v!r}")


def _resolve_selector(method: str, selector: str | None) -> str | None:
    """Only methods that use a selector record one; the value must be theirs."""
    if method not in SELECTOR_DEFAULTS:
        if selector is not None:
            raise SystemExit(f"--selector is meaningless for --experiment {method}")
        return None
    valid = MALCL_SELECTORS if method == "malcl" else selectors.available()
    if selector is None:
        return SELECTOR_DEFAULTS[method]
    if selector not in valid:
        raise SystemExit(
            f"--selector {selector!r} is not valid for --experiment {method}; "
            f"choose from {', '.join(valid)}")
    return selector


def resolve(args) -> tuple[dict, dict, object]:
    file_cfg = load_config_file(args.config) if args.config else {}
    file_rest, file_hp = split_config(file_cfg)

    dataset = args.dataset or file_rest.get("dataset")
    cli_hp = {n[3:]: getattr(args, n) for n in vars(args) if n.startswith("hp_")}
    overrides = {**file_hp, **{k: v for k, v in cli_hp.items() if v is not None}}
    hyperparams = hyperparams_for(dataset, overrides, method=args.experiment)

    task_setup = (args.task_setup or file_rest.get("task_setup")
                  or data_registry.default_task_setup(dataset))
    schedule = build_schedule(task_setup)

    config = {
        "dataset": dataset,
        "task_setup": schedule.spec,
        "task_setup_name": task_setup,
        "method": args.experiment,
        "model": args.model,
        "selector": _resolve_selector(args.experiment, args.selector),
        "protocol": args.protocol,
        "seed": args.seed,
        "n_tasks": args.n_tasks if args.n_tasks is not None else schedule.n_tasks,
        "group": args.group,
        "tag": args.tag,
        "confusion": args.confusion,
        "quiet": args.quiet,
    }
    # Recorded ONLY for the corpus it applies to. Adding a key for every dataset
    # would change the config hash, and therefore the run id, of every existing
    # EMBER run -- orphaning results already on disk for a field they never used.
    if dataset == "lamda_classil":
        config["split_mode"] = args.split_mode
        if args.split_mode == "temporal":
            config["temporal_cut"] = args.temporal_cut
    return config, hyperparams, schedule


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config, hyperparams, schedule = resolve(args)
    store = paths_mod.resolve(args.data_root, args.log_root, args.cache_root)

    n_tasks = min(config["n_tasks"], schedule.n_tasks)
    config["n_tasks"] = n_tasks
    run_id = make_run_id(config, overrides_of(config["dataset"], hyperparams,
                                              method=config["method"]))

    if not args.quiet:
        print(describe(config, hyperparams))
        print(f"\nrun id      {run_id}")
        print(paths_mod.describe(store))

    out_dir = store.json_dir(config["group"])
    target = out_dir / f"{run_id}.json"
    if target.exists() and not args.force:
        # Before trusting a cached result, check it was produced by THIS
        # configuration. The run id hashes only non-default overrides, so if a
        # protocol default changed since the file was written its name would
        # still match while its contents would not. Refusing here is the price
        # of ids that survive a new knob being added.
        stale = _stale_reasons(target, config, hyperparams)
        if stale:
            raise SystemExit(
                f"\nA result already exists at {target}, but it does not match this "
                f"configuration:\n  " + "\n  ".join(stale) +
                f"\n\nA protocol default probably changed since it was written. "
                f"Delete it, use a different --group, or pass --force to overwrite.")
        print(f"\nalready done: {target}\n(use --force to rerun)")
        return 0
    if args.dry_run:
        print(f"\nwould write {target}\n[dry run: nothing trained]")
        return 0

    paths_mod.ensure_writable(store, config["group"])
    seed_everything(config["seed"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if not args.quiet:
        print(f"\ndevice      {device}\nloading {config['dataset']}...")

    loader_kwargs = {}
    if config["dataset"] == "lamda_classil":
        # lamda_data builds the schedule itself, so it is handed the SAME numbers
        # the runner resolved -- the two cannot disagree about the task layout.
        loader_kwargs = dict(
            split_mode=args.split_mode, temporal_cut=args.temporal_cut,
            min_family_samples=hyperparams["min_family_samples"],
            num_classes=schedule.n_classes, task0_classes=schedule.task0,
            step_classes=schedule.step, verbose=not args.quiet)

    corpus = data_registry.load_corpus(config["dataset"], store, **loader_kwargs)
    corpus = data_registry.apply_family_selection(
        corpus, hyperparams["min_family_samples"], schedule.n_classes)
    Xtr_np, ytr_np, n_capped = data_registry.cap_per_family(
        corpus.X_train, corpus.y_train, hyperparams["family_cap"],
        hyperparams["cap_seed"])
    corpus.X_train, corpus.y_train = Xtr_np, ytr_np
    X_train, X_test, scale_info = data_registry.scale_features(
        corpus, schedule.classes_for(0), hyperparams["feature_clip"])

    dataset_info = corpus.summary()
    dataset_info.update(scale_info)
    if n_capped:
        dataset_info["n_capped_out"] = n_capped
    if not args.quiet:
        print(f"  {dataset_info['n_train']:,} train / {dataset_info['n_test']:,} test"
              f" | d={dataset_info['input_dim']} | {schedule.spec}")

    ctx = RunContext(
        config=config, hyperparams=hyperparams, schedule=schedule,
        X_train=torch.from_numpy(np.ascontiguousarray(X_train)),
        y_train=torch.from_numpy(corpus.y_train).long(),
        X_test=torch.from_numpy(np.ascontiguousarray(X_test)),
        y_test=torch.from_numpy(corpus.y_test).long(),
        device=device, n_tasks=n_tasks)

    record = RunRecord(
        config=config, paths=store.as_dict(),
        model_info=model_info(config["model"], ctx.input_dim, schedule.n_classes),
        dataset_info=dataset_info, schedule_info=schedule.as_dict(),
        hyperparams=hyperparams, run_id=run_id,
        env=environment(store.repo_root))
    if args.notes:
        record.note(args.notes)
    if config["dataset"] == "synthetic":
        record.note("SYNTHETIC CORPUS -- smoke test only, not a scientific result")

    experiment = experiments.build_experiment(config["method"], ctx)
    if not args.quiet:
        print(f"\n{config['method']} | {record.model_info['n_params']:,} params "
              f"| {n_tasks} tasks")
    record = experiment.run(record)

    written = record.write(out_dir)
    summary = record.summary()
    if not args.quiet:
        print(f"\nseen accuracy  mean {summary.get('seen_accuracy_mean', float('nan')):.3f}"
              f"  final {summary.get('seen_accuracy_final', float('nan')):.3f}")
        print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
