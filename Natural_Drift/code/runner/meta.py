"""Meta-train NN-1 by Evolution Strategies.

    python -m runner.meta --dataset ember2018 --pop 8 --generations 10 \
        --n-tasks 5 --cl-iters 500 --unlearn-epochs 1 --tag mini --gpus 0 1

WHY ES AND NOT A GRADIENT. The thing being optimised is a SELECTION -- which
buffer points to delete -- and the quantity it is judged on is the accuracy of a
continual-learning run that follows. There is no differentiable path from the
scorer's weights to that number: the selection is a sort-and-threshold, and the
reward is the end of a multi-task training trajectory. ES needs only that the
reward be computable, which it is, at the cost of one full inner run per
candidate. That cost is why core/task0.py exists.

EACH CANDIDATE IS A REAL RUN. A candidate's reward comes from `runner.run`
executing the ordinary `meta_unlearn` experiment with that scorer -- the same
code path, evaluation protocol and record format as every baseline in the study.
Nothing here reimplements the inner loop, so a meta-trained scorer cannot end up
being measured against a pipeline that differs from the one it will be reported
in.

THE REWARD EXCLUDES TASK 0. Every candidate shares one task 0, so including it
would add the same constant to every reward and shrink the differences ES is
trying to rank. The reward is the mean seen-class accuracy over tasks 1..N.

BASELINES RUN UNDER THE SAME CONFIG, FIRST, and WHICH baselines depends on the
mode -- see MODES. In mode B `gradconflict` and `densityratio`
each rank on a SINGLE one of NN-1's nine features. If the learned scorer cannot
beat those, it has not learned to combine anything -- and that is a result worth
getting early, before spending a full generation budget. `random` and `none`
bound the comparison from below.

META-TRAIN AND META-TEST ARE DIFFERENT PRESENTATION ORDERS. Without
`--perm-seeds` there is exactly one episode: the corpus in frequency order,
optimised against directly. A scorer tuned on that has no way to show it learned
a transferable rule rather than a sequence of moves fitted to one ordering, and
the rising `theta_reward` column proves only that the search worked.

    --perm-seeds 1 2 3 --holdout-perm-seeds 4 5

trains on three orderings, rotating one per generation, and then runs the final
scorer and every baseline on two orderings ES never saw. Within a generation all
candidates share the ordering as well as the seed -- common random numbers, so
the population comparison is between scorers and not between which families
happened to land in which task -- while rotating across generations is what
stops the scorer from fitting any single one. The held-out table written to
`holdout.json` is the result; everything in `es_log.csv` is training diagnostics.

THE HELD-OUT TABLE IS PAIRED, AND ALSO RAW. Presentation order moves accuracy
far more than any selector does -- measured over eight orderings on EMBER 2018,
seen accuracy ranged 65.9 to 79.4 (sd ~3.6) while unlearning at all was worth
~1.7. A table of raw means over two held-out orderings is therefore mostly a
report of which families landed in which task, and two such tables run on
different machines duly put `none` first and last. Pairing every arm against
`none` on the SAME ordering cancels that term: the same eight orderings give a
paired sd of 1.36 for random-vs-none. Both views are printed, because the raw
means carry the absolute level the deltas discard. See core/paired.py.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from core import nn1, paired
from runner import paths as paths_mod
from runner.config import BASE_HYPERPARAMS, PROTOCOLS

# Per mode: the experiment the inner runs use, the scorer width, and the
# controls the held-out table compares against. Mode A's controls are the
# selectors that already exist for arriving data; mode B's two single-feature
# rankers do not exist there, because the features they rank on do not.
MODES = {
    "b": {"experiment": "meta_unlearn",
          "baselines": ("none", "random", "gradconflict", "densityratio")},
    "a": {"experiment": "madar_unlearn",
          "baselines": ("random", "donut", "leftover")},
}

# Kept for callers that still import it; mode B's set, which is what it always
# meant.
BASELINES = MODES["b"]["baselines"]
DRIVER_OWNED = frozenset({"scorer_weights", "zero_features"})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ES meta-training for NN-1")
    p.add_argument("--dataset", default="ember2018")
    p.add_argument("--task-setup", default=None)
    p.add_argument("--mode", default="b", choices=["a", "b"],
                   help="b: score BUFFER points against the arriving task "
                        "(meta_unlearn, 9 features). a: score the ARRIVING "
                        "task's own samples (madar_unlearn, 7 features). The "
                        "two scorers are not interchangeable.")
    p.add_argument("--experiment", default=None,
                   help="overrides the experiment --mode selects; only needed "
                        "for a variant that is not one of the two modes")
    p.add_argument("--model", default=None)
    p.add_argument("--group", default="meta")
    p.add_argument("--tag", default="es")
    p.add_argument("--protocol", default="meta", choices=PROTOCOLS,
                   help="meta shares one task-0 checkpoint across candidates, "
                        "which is what makes the budget feasible. stage1 is "
                        "accepted but will retrain task 0 for every candidate.")
    p.add_argument("--n-tasks", type=int, default=None)

    es = p.add_argument_group("evolution strategies")
    es.add_argument("--pop", type=int, default=8,
                    help="population size; must be even (mirrored pairs)")
    es.add_argument("--generations", type=int, default=10)
    es.add_argument("--sigma", type=float, default=0.1, help="perturbation std")
    es.add_argument("--lr", type=float, default=0.05)
    es.add_argument("--hidden", type=int, default=0,
                    help="NN-1 hidden units; 0 = linear (10 params in mode B)")
    es.add_argument("--seed", type=int, default=42,
                    help="base inner-loop seed; rotates per generation so a "
                         "candidate is not tuned to one draw")
    es.add_argument("--es-seed", type=int, default=123,
                    help="seed for the perturbations themselves")
    es.add_argument("--perm-seeds", type=int, nargs="*", default=[],
                    metavar="S",
                    help="meta-TRAIN on these presentation orders, rotating one "
                         "per generation (every candidate in a generation gets "
                         "the same one). Unset means the canonical frequency "
                         "order only -- a single episode, from which nothing "
                         "about transfer can be concluded.")
    es.add_argument("--holdout-perm-seeds", type=int, nargs="*", default=[],
                    metavar="S",
                    help="meta-TEST orders, never shown to ES. After the last "
                         "generation the final scorer and every baseline are "
                         "run on each of these; that table is the result.")
    es.add_argument("--resume", default=None, help=".npz to warm-start from")
    es.add_argument("--zero-features", nargs="*", default=[],
                    help="ablation: feature names NN-1 may not use")
    es.add_argument("--skip-baselines", action="store_true")

    rn = p.add_argument_group("execution")
    rn.add_argument("--gpus", nargs="+", default=None)
    rn.add_argument("--workers", type=int, default=None)
    rn.add_argument("--data-root"); rn.add_argument("--log-root")
    rn.add_argument("--cache-root")
    rn.add_argument("--device", default=None,
                    help="passed through to each inner run (cpu, cuda, ...)")
    rn.add_argument("--dry-run", action="store_true")

    # The driver owns these two: the scorer path is written per candidate and
    # the ablation is an ES-level choice recorded in every scorer file. Exposing
    # them again as pass-through hyperparameters would let one run disagree with
    # the population it belongs to.
    for name, default in sorted(BASE_HYPERPARAMS.items()):
        if name in DRIVER_OWNED:
            continue
        rn.add_argument(f"--{name.replace('_', '-')}", dest=f"hp_{name}",
                        type=type(default), default=None)
    return p


def base_command(args, work: Path) -> list[str]:
    cmd = [sys.executable, "-m", "runner.run",
           "--dataset", args.dataset, "--experiment", args.experiment,
           "--protocol", args.protocol, "--quiet", "--force",
           "--no-checkpoint"]        # a candidate is short and never resumed
    if args.task_setup:
        cmd += ["--task-setup", args.task_setup]
    if args.model:
        cmd += ["--model", args.model]
    if args.n_tasks is not None:
        cmd += ["--n-tasks", str(args.n_tasks)]
    if args.device:
        cmd += ["--device", str(args.device)]
    for root in ("data_root", "log_root", "cache_root"):
        v = getattr(args, root)
        if v:
            cmd += [f"--{root.replace('_', '-')}", str(v)]
    for name in sorted(BASE_HYPERPARAMS):
        if name in DRIVER_OWNED:
            continue
        v = getattr(args, f"hp_{name}")
        if v is not None:
            cmd += [f"--{name.replace('_', '-')}", str(v)]
    if args.zero_features:
        cmd += ["--zero-features", " ".join(args.zero_features)]
    return cmd


# A slot token that is NOT a GPU id. `--workers N` throttles concurrency
# without pinning devices, and its tokens used to be plain None -- which made
# "did I take a token?" indistinguishable from "was I given a GPU?", so the
# token was never returned and the pool drained to a deadlock after exactly N
# runs. The sentinel keeps the two questions separate.
NO_GPU = object()


def run_one(args, work: Path, gpu_q, *, selector, seed, out_path, scorer=None,
            perm_seed=None):
    """One inner run. Returns the reward, or raises with the child's output."""
    token = gpu_q.get() if gpu_q is not None else NO_GPU
    gpu = None if token is NO_GPU else token
    try:
        cmd = base_command(args, work) + [
            "--selector", selector, "--seed", str(seed),
            "--group", f"{args.group}/{args.tag}/work",
            "--result-path", str(out_path)]
        if scorer is not None:
            cmd += ["--scorer-weights", str(scorer)]
        if perm_seed is not None:
            cmd += ["--perm-seed", str(perm_seed)]
        env = dict(os.environ)
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        res = subprocess.run(cmd, env=env, capture_output=True, text=True,
                             cwd=str(Path(__file__).resolve().parents[1]))
        if res.returncode != 0:
            raise RuntimeError(
                f"inner run failed (selector={selector}, seed={seed}, "
                f"perm_seed={perm_seed})\n"
                f"{res.stdout[-2000:]}\n{res.stderr[-2000:]}")
        return reward_of(out_path)
    finally:
        # Always return what was taken, whatever it was.
        if gpu_q is not None:
            gpu_q.put(token)


def reward_of(path: Path, scope: str | None = None) -> float:
    """Mean seen-class accuracy over tasks 1..N, in the requested scope.

    THE SEARCH AND THE REPORT READ DIFFERENT SCOPES, deliberately.

    `scope=None` (the search) prefers `val_seen` when a validation split exists,
    so the test set is never queried while ES is choosing a scorer. That is what
    makes the final numbers an estimate rather than the thing that was
    optimised.

    `scope="seen"` (the meta-test) forces the TEST split. Held-out presentation
    orders are the result that gets reported, and on a chronologically split
    malware corpus the test set is the only one carrying the temporal drift the
    method is supposed to survive -- validation, carved from train, does not.
    Scoring the held-out table on validation would quietly report the easy
    distribution under the name of the hard one; it is legitimate here precisely
    because the scorer never saw these orderings.
    """
    rec = json.loads(Path(path).read_text())
    tasks = [t for t in rec["per_task"] if int(t["task"]) > 0]
    if not tasks:
        raise RuntimeError(f"{path} records no task beyond task 0")
    have = tasks[0].get("metrics") or {}
    if scope is None:
        scope = "val_seen" if "val_seen" in have else "seen"
    if scope not in have:
        raise RuntimeError(f"{path} has no {scope!r} scope; got {sorted(have)}")
    return float(np.mean([t["metrics"][scope]["accuracy"] for t in tasks]))


def run_parallel(args, work, jobs):
    gpu_q = None
    if args.gpus:
        gpu_q = queue.Queue()
        for g in args.gpus:
            gpu_q.put(g)
    else:
        # Default to one at a time. Without a throttle every candidate in the
        # population launches at once -- at pop 16 that is 16 concurrent
        # continual-learning runs, which is a way to exhaust a machine rather
        # than a way to use it.
        gpu_q = queue.Queue()
        for _ in range(max(1, int(args.workers or 1))):
            gpu_q.put(NO_GPU)

    out = [None] * len(jobs)
    errs = [None] * len(jobs)

    def work_one(i, job):
        try:
            out[i] = run_one(args, work, gpu_q, **job)
        except Exception as e:                      # noqa: BLE001
            errs[i] = e

    threads = [threading.Thread(target=work_one, args=(i, j))
               for i, j in enumerate(jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for e in errs:
        if e is not None:
            raise e
    return out


def run_warmed(args, work, jobs, perm, warmed: set):
    """Run a batch that shares one presentation order, training task 0 once.

    Everything in `jobs` has the same `perm_seed`, so under `--protocol meta`
    they all consult the same task-0 cache entry. Launched together at an order
    nobody has run yet they ALL miss, and all train the identical task 0 -- the
    exact duplication core/task0.py exists to prevent, reintroduced by the fan
    out. Running one job first populates the entry; the rest then load it.

    The cost is one serialised run per new ordering. On a 16-candidate
    population that buys back fifteen redundant task-0 fits.
    """
    if perm in warmed or len(jobs) <= 1:
        warmed.add(perm)
        return run_parallel(args, work, jobs)
    first = run_parallel(args, work, jobs[:1])
    warmed.add(perm)
    return first + run_parallel(args, work, jobs[1:])


def main() -> int:
    args = build_parser().parse_args()
    if args.pop % 2:
        raise SystemExit("--pop must be even: ES uses mirrored pairs")
    clash = sorted(set(args.perm_seeds) & set(args.holdout_perm_seeds))
    if clash:
        raise SystemExit(
            f"--holdout-perm-seeds {clash} also appear in --perm-seeds. A "
            f"held-out ordering that ES trained on is not held out, and the "
            f"meta-test number it produces means nothing.")
    if args.holdout_perm_seeds and not args.perm_seeds:
        raise SystemExit(
            "--holdout-perm-seeds needs --perm-seeds: with a single training "
            "order there is no family of episodes to generalise from, so "
            "holding one out tests nothing.")
    # The training orders, as a cycle. An empty list means the canonical
    # frequency order, which the runner spells as no --perm-seed at all.
    train_perms = [int(v) for v in args.perm_seeds] or [None]

    store = paths_mod.resolve(data_root=args.data_root, log_root=args.log_root,
                              cache_root=args.cache_root)
    paths_mod.ensure_writable(store, args.group)
    work = store.log_root / args.group / args.tag
    work.mkdir(parents=True, exist_ok=True)

    mode = MODES[args.mode]
    if args.experiment is None:
        args.experiment = mode["experiment"]
    baselines = mode["baselines"]
    n_feat = nn1.N_FEATURES_B if args.mode == "b" else nn1.N_FEATURES
    names = nn1.feature_names(n_feat)
    bad = [f for f in args.zero_features if f not in names]
    if bad:
        raise SystemExit(f"unknown feature(s) {bad}; known: {names}")

    if args.resume:
        theta, hidden, nf = nn1.load(args.resume)
        if hidden != args.hidden:
            raise SystemExit("--resume architecture does not match --hidden")
        if nf != n_feat:
            theta = nn1.grow(theta, nf, n_feat, hidden=hidden)
            print(f"warm start: grew {nf} -> {n_feat} features, new weights zeroed")
    else:
        theta = nn1.init_params(hidden=args.hidden, seed=0, n_features=n_feat)
    dim = nn1.n_params(args.hidden, n_feat)

    (work / "run_config.json").write_text(json.dumps({
        "dataset": args.dataset, "experiment": args.experiment,
        "protocol": args.protocol, "mode": "b", "n_features": n_feat,
        "hidden": args.hidden, "pop": args.pop, "generations": args.generations,
        "sigma": args.sigma, "lr": args.lr, "seed": args.seed,
        "es_seed": args.es_seed, "zero_features": list(args.zero_features),
        "perm_seeds": [int(v) for v in args.perm_seeds],
        "holdout_perm_seeds": [int(v) for v in args.holdout_perm_seeds],
        "n_tasks": args.n_tasks, "baselines": list(baselines),
        "mode": args.mode,
        "overrides": {k: getattr(args, f"hp_{k}")
                      for k in sorted(set(BASE_HYPERPARAMS) - DRIVER_OWNED)
                      if getattr(args, f"hp_{k}") is not None},
    }, indent=2))

    if args.dry_run:
        print(f"work dir     {work}")
        print(f"scorer       {dim} parameters, {n_feat} features, hidden={args.hidden}")
        n_arms = 1 if args.skip_baselines else 1 + len(baselines)
        n_hold = len(args.holdout_perm_seeds) * n_arms
        n_base = 0 if args.skip_baselines else len(baselines)
        # The theta evaluation is one MORE run per generation, on top of the
        # population. Leaving it out of the budget line understates a pop-8 x
        # 10-generation plan by ten full inner runs.
        total = args.pop * args.generations + args.generations + n_base + n_hold
        print(f"budget       {args.pop} x {args.generations} = "
              f"{args.pop * args.generations} candidate runs"
              f" + {args.generations} theta evals"
              f"{f' + {n_base} baselines' if n_base else ''}"
              f"{f' + {n_hold} meta-test runs' if n_hold else ''}"
              f"  =  {total} inner runs")
        n_orders = len(set(train_perms) | set(int(v) for v in args.holdout_perm_seeds))
        print(f"task 0       trained {n_orders}x (once per presentation order), "
              f"then shared")
        print("meta-train   " + (", ".join(str(v) for v in train_perms)
                                 if args.perm_seeds else "canonical order only"))
        if args.holdout_perm_seeds:
            print("meta-test    " + ", ".join(
                str(v) for v in args.holdout_perm_seeds))
        print("example      " + " ".join(
            base_command(args, work) + ["--selector", "nn1", "--seed",
                                        str(args.seed), "--group",
                                        f"{args.group}/{args.tag}/work"]))
        return 0

    # Presentation orders whose task-0 cache entry is known to exist.
    warmed: set = set()

    if not args.skip_baselines:
        print(f"mode {args.mode} ({args.experiment}), {n_feat} features")
        print(f"baselines under this config: {', '.join(baselines)}")
        jobs = [{"selector": b, "seed": args.seed, "perm_seed": train_perms[0],
                 "out_path": work / f"baseline_{b}.json"} for b in baselines]
        vals = run_warmed(args, work, jobs, train_perms[0], warmed)
        base = dict(zip(baselines, vals))
        (work / "baselines.json").write_text(json.dumps(base, indent=2))
        for k, v in base.items():
            print(f"  {k:14s} {v:6.2f}")

    rng = np.random.default_rng(args.es_seed)
    log = work / "es_log.csv"
    if not log.exists():
        log.write_text("generation,perm_seed,theta_reward,pop_mean,pop_max,sigma,lr\n")

    for gen in range(args.generations):
        gen_seed = args.seed + gen        # common random numbers within a generation
        # Common random numbers extend to the ORDERING too: every candidate in
        # a generation is scored on the same permutation, so the comparison
        # between them is a comparison of scorers and not of which families
        # happened to land in which task. Rotating across generations is what
        # stops the scorer from fitting one ordering.
        perm = train_perms[gen % len(train_perms)]
        half = rng.standard_normal((args.pop // 2, dim))
        eps = np.concatenate([half, -half])

        jobs = []
        for i in range(args.pop):
            path = work / f"cand_g{gen}_i{i}.npz"
            nn1.save(path, theta + args.sigma * eps[i], hidden=args.hidden,
                     n_features=n_feat, zero_features=args.zero_features)
            jobs.append({"selector": "nn1", "seed": gen_seed,
                         "perm_seed": perm,
                         "out_path": work / f"r_g{gen}_i{i}.json",
                         "scorer": path})
        rewards = np.asarray(run_warmed(args, work, jobs, perm, warmed),
                             dtype=float)

        # Rank shaping: ES is told the ORDER of the population, not the spread.
        # Accuracy differences between candidates are small and occasionally
        # dominated by one outlier run; ranking makes the step depend on which
        # candidates were better rather than by how much.
        ranks = np.empty(args.pop)
        ranks[np.argsort(rewards)] = np.arange(args.pop)
        shaped = (ranks / (args.pop - 1)) - 0.5
        theta = theta + args.lr * (shaped[:, None] * eps).mean(axis=0) / args.sigma

        nn1.save(work / "theta.npz", theta, hidden=args.hidden,
                 n_features=n_feat, zero_features=args.zero_features)
        nn1.save(work / f"theta_g{gen}.npz", theta, hidden=args.hidden,
                 n_features=n_feat, zero_features=args.zero_features)
        th = run_parallel(args, work, [{"selector": "nn1", "seed": gen_seed,
                                        "perm_seed": perm,
                                        "out_path": work / f"theta_g{gen}.json",
                                        "scorer": work / "theta.npz"}])[0]
        with log.open("a") as f:
            f.write(f"{gen},{'' if perm is None else perm},{th:.4f},"
                    f"{rewards.mean():.4f},{rewards.max():.4f},"
                    f"{args.sigma},{args.lr}\n")
        print(f"gen {gen:3d}  perm {('--' if perm is None else perm)!s:>4s}  "
              f"theta {th:6.2f}  pop mean {rewards.mean():6.2f}  "
              f"max {rewards.max():6.2f}", flush=True)
        weights = ", ".join(f"{n}={w:+.3f}" for n, w in
                            zip(names, theta[:n_feat])) if args.hidden == 0 else ""
        if weights:
            print(f"          {weights}  bias={theta[-1]:+.3f}", flush=True)

    print(f"\nfinal scorer: {work / 'theta.npz'}")

    if args.holdout_perm_seeds:
        # THE RESULT. Every number above was measured on an ordering ES was
        # optimising against, so a rising theta_reward shows only that the
        # search worked. These runs use orderings the search never saw, and the
        # baselines are re-run on each of them so the comparison is within an
        # ordering rather than across one.
        print(f"\nmeta-test on held-out orders: "
              f"{', '.join(str(v) for v in args.holdout_perm_seeds)}")
        arms = ["nn1"] + ([] if args.skip_baselines else list(baselines))
        table = {}
        # One order at a time: every arm of a held-out order shares its task 0,
        # and run_warmed only avoids the duplicate fit within a single batch.
        for ps in args.holdout_perm_seeds:
            jobs = [{"selector": arm, "seed": args.seed, "perm_seed": int(ps),
                     "out_path": work / f"holdout_p{ps}_{arm}.json",
                     "scorer": work / "theta.npz" if arm == "nn1" else None}
                    for arm in arms]
            run_warmed(args, work, jobs, int(ps), warmed)
            # Re-read each result for BOTH scopes rather than keeping the
            # reward run_one returned: that one is whatever the search
            # optimised (validation, when carved), and the reported column has
            # to be test. Both are kept so the gap between them is visible
            # instead of having to be taken on trust.
            row = {}
            for arm, job in zip(arms, jobs):
                row[arm] = {"test": reward_of(job["out_path"], scope="seen")}
                try:
                    row[arm]["val"] = reward_of(job["out_path"], scope="val_seen")
                except RuntimeError:
                    pass                      # no validation split in this run
            table[str(ps)] = row
        has_val = any("val" in v for r in table.values() for v in r.values())
        orders = [int(v) for v in args.holdout_perm_seeds]
        # `none` is the reference arm: it performs no unlearning, so a paired
        # delta against it is the effect of the intervention rather than of one
        # selector over another.
        ref = "none" if "none" in arms else arms[-1]
        analysis = {}
        for scope in (["test", "val"] if has_val else ["test"]):
            label = ("TEST -- held-out orders, the reported result"
                     if scope == "test" else
                     "validation -- what the search optimised, for reference")
            print(f"\n  {label}")
            print(paired.render(table, arms, orders, scope, reference=ref))
            analysis[scope] = paired.summary(table, arms, orders, scope,
                                             reference=ref)
        # Rewritten now that the analysis exists, so the JSON carries the same
        # numbers the terminal showed rather than only their inputs.
        (work / "holdout.json").write_text(json.dumps(
            {"seed": args.seed,
             "perm_seeds": [int(v) for v in args.holdout_perm_seeds],
             "arms": arms, "reported_scope": "test", "reference_arm": ref,
             "search_scope": ("val_seen" if getattr(args, "hp_val_fraction", None)
                              else "seen"),
             "analysis": analysis, "results": table}, indent=2))
        print(f"\n  {work / 'holdout.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
