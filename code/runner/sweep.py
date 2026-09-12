"""Fan a run matrix out over GPUs.

    python -m runner.sweep --group madar_fidelity \
        --experiment naive joint madar madar_unlearn \
        --seed 1 2 3 4 5 --dataset ember2018 --gpus 0 1 2

Every list-valued axis is crossed; each cell becomes one `run.py` subprocess
pinned to a GPU by CUDA_VISIBLE_DEVICES. Two behaviours are deliberate:

RESUMABLE, BUT LOUD. A cell whose JSON already exists is skipped, and the count
of skipped cells is printed. Silent resumability is how re-running a sweep into
an existing directory quietly does nothing; here you always see how many cells
were served from disk rather than computed.

A CRASHED CHILD IS A FAILURE, NOT A RESULT. If a subprocess exits non-zero the
cell is recorded as failed and the sweep exits non-zero once the other cells
finish. It is never scored, defaulted, or filled with a placeholder: a missing
reward that becomes NaN and then sorts to the top of a ranking is a real failure
mode, and this is the boundary at which to refuse it.
"""

from __future__ import annotations

import argparse
import itertools
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiments
from core import data as data_registry
from core import selectors
from runner import paths as paths_mod
from runner.config import BASE_HYPERPARAMS, DATASET_MODEL, PROTOCOLS

CROSSED = ("dataset", "experiment", "model", "selector", "seed", "task_setup")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cross a run matrix and execute it over GPUs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", nargs="+", default=["ember2018"],
                   choices=data_registry.available())
    p.add_argument("--experiment", "--method", dest="experiment", nargs="+",
                   default=["madar"], choices=experiments.available())
    p.add_argument("--model", nargs="+", default=[None],
                   help="default: the dataset's own (resnet18 for an image "
                        "corpus, full otherwise)")
    p.add_argument("--selector", nargs="+", default=[None],
                   choices=selectors.available() + ["L1_C_Mean", "L1_B_Mean",
                                                     "L2_One_Hot"],
                   help="passed to methods that take one; others get None")
    p.add_argument("--seed", nargs="+", type=int, default=[1])
    p.add_argument("--task-setup", nargs="+", default=[None])
    p.add_argument("--group", default="default")
    p.add_argument("--protocol", default="stage1", choices=PROTOCOLS)
    p.add_argument("--tag", default=None)
    p.add_argument("--n-tasks", type=int, default=None)
    p.add_argument("--split-mode", default=None, choices=["random", "temporal"],
                   help="LAMDA only; passed through to run.py")
    p.add_argument("--temporal-cut", type=int, default=None)
    p.add_argument("--gpus", nargs="+", default=None,
                   help="GPU ids to spread over. Default: one worker, no pinning.")
    p.add_argument("--workers", type=int, default=None,
                   help="parallel workers when --gpus is not given")
    p.add_argument("--data-root"); p.add_argument("--log-root")
    p.add_argument("--cache-root")
    p.add_argument("--config")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-checkpoint", action="store_true",
                   help="passed through to each run; see run.py --no-checkpoint")
    p.add_argument("--dry-run", action="store_true",
                   help="print the matrix and the commands, run nothing")
    for name, default in sorted(BASE_HYPERPARAMS.items()):
        p.add_argument(f"--{name.replace('_', '-')}", dest=f"hp_{name}",
                       type=type(default), default=None)
    return p


def cells(args) -> list[dict]:
    axes = {k: getattr(args, k) for k in CROSSED}
    # The selector axis only means something for a method that unlearns. Without
    # this, --selector a b c would triple every naive/joint/madar cell into
    # identical duplicates differing only in a field those methods ignore.
    out, seen = [], set()
    for combo in itertools.product(*(axes[k] for k in CROSSED)):
        cell = dict(zip(CROSSED, combo))
        # Resolved here, not at the argument default, because the right model
        # depends on the dataset axis: one sweep may span EMBER and Tiny
        # ImageNet, and those cannot share a backbone. Done before the dedup key
        # so that `--model full` and the EMBER default collapse to one cell
        # rather than running the same configuration twice.
        if cell["model"] is None:
            cell["model"] = DATASET_MODEL.get(cell["dataset"], "full")
        if cell["experiment"] not in ("madar_unlearn", "malcl"):
            cell["selector"] = None
        key = tuple(sorted((k, str(v)) for k, v in cell.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(cell)
    return out


def command(cell: dict, args) -> list[str]:
    cmd = [sys.executable, "-m", "runner.run",
           "--dataset", cell["dataset"], "--experiment", cell["experiment"],
           "--model", str(cell["model"]), "--seed", str(cell["seed"]),
           "--group", args.group, "--protocol", args.protocol, "--quiet"]
    if cell["task_setup"]:
        cmd += ["--task-setup", str(cell["task_setup"])]
    if args.no_checkpoint:
        cmd += ["--no-checkpoint"]
    if cell["selector"]:
        cmd += ["--selector", cell["selector"]]
    if args.tag:
        cmd += ["--tag", args.tag]
    if args.n_tasks is not None:
        cmd += ["--n-tasks", str(args.n_tasks)]
    if args.split_mode is not None:
        cmd += ["--split-mode", args.split_mode]
    if args.temporal_cut is not None:
        cmd += ["--temporal-cut", str(args.temporal_cut)]
    for root in ("data_root", "log_root", "cache_root"):
        value = getattr(args, root)
        if value:
            cmd += [f"--{root.replace('_', '-')}", str(value)]
    if args.config:
        cmd += ["--config", args.config]
    if args.force:
        cmd += ["--force"]
    for name in sorted(BASE_HYPERPARAMS):
        value = getattr(args, f"hp_{name}")
        if value is not None:
            cmd += [f"--{name.replace('_', '-')}", str(value)]
    return cmd


def label(cell: dict) -> str:
    bits = [cell["dataset"], cell["experiment"], str(cell["model"]), f"s{cell['seed']}"]
    if cell["selector"]:
        bits.insert(2, cell["selector"])
    if cell["task_setup"]:
        bits.insert(1, str(cell["task_setup"]))
    return "/".join(bits)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    matrix = cells(args)
    store = paths_mod.resolve(args.data_root, args.log_root, args.cache_root)
    repo_code = Path(__file__).resolve().parents[1]

    print(f"matrix: {len(matrix)} cell(s) -> {store.json_dir(args.group)}")
    if args.dry_run:
        for cell in matrix:
            print(f"  {label(cell)}\n    {' '.join(command(cell, args))}")
        return 0

    gpu_q: queue.Queue = queue.Queue()
    if args.gpus:
        for g in args.gpus:
            gpu_q.put(str(g))
        n_workers = len(args.gpus)
    else:
        n_workers = max(1, args.workers or 1)
        for _ in range(n_workers):
            gpu_q.put(None)

    pending: queue.Queue = queue.Queue()
    for cell in matrix:
        pending.put(cell)

    results = {"done": [], "skipped": [], "failed": []}
    lock = threading.Lock()

    def worker():
        while True:
            try:
                cell = pending.get_nowait()
            except queue.Empty:
                return
            gpu = gpu_q.get()
            try:
                env = dict(os.environ)
                if gpu is not None:
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                t0 = time.time()
                res = subprocess.run(command(cell, args), cwd=str(repo_code),
                                     env=env, capture_output=True, text=True)
                dt = time.time() - t0
                tag = label(cell)
                with lock:
                    if res.returncode != 0:
                        results["failed"].append((tag, res.returncode))
                        print(f"  FAILED  {tag}  (exit {res.returncode})")
                        detail = (res.stderr or res.stdout or "").strip()[-1200:]
                        print("    " + detail.replace("\n", "\n    "))
                    elif "already done:" in res.stdout:
                        results["skipped"].append(tag)
                        print(f"  skipped {tag}  (cached)")
                    else:
                        results["done"].append(tag)
                        print(f"  ok      {tag}  ({dt:.0f}s)")
            finally:
                gpu_q.put(gpu)
                pending.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(n_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"\n{len(results['done'])} run, {len(results['skipped'])} skipped "
          f"(already on disk), {len(results['failed'])} failed")
    if results["failed"]:
        print("failed cells:")
        for tag, code in results["failed"]:
            print(f"  {tag} (exit {code})")
        print("\nThese cells produced NO result. They are not averaged, defaulted "
              "or scored -- fix and re-run; completed cells will be skipped.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
