"""Per-arm run cost for one or more log groups.

    python run_times.py                        # the four stage-1 groups
    python run_times.py ember2018_stage1        # just one
    python run_times.py --root /other/Logs ...

Reports, per condition: number of runs, median wall time, and median
total_grad_steps.

Wall time on a SHARED machine is noisy -- another user's job inflates it and
nothing in the record says so. `total_grad_steps` is deterministic for a given
configuration, so it is the metric that reproduces. Both are shown because they
answer different questions: gradient steps is what the method costs, wall time
is what you waited.
"""
import json, sys, pathlib, statistics

DEFAULT = ["ember2018_stage1", "ember2024_stage1", "lamda_stage1", "tin_stage1"]

args = [a for a in sys.argv[1:]]
root = pathlib.Path("/mnt/delucio/iclr/Logs")
if "--root" in args:
    i = args.index("--root")
    root = pathlib.Path(args[i + 1])
    del args[i:i + 2]
groups = args or DEFAULT

def arm(cfg, hp):
    m, sel = cfg.get("method"), cfg.get("selector")
    name = f"{m} + {sel}" if sel else m
    bits = []
    if m == "madar_unlearn":
        bits.append(f"a={hp.get('alpha')}")
        bits.append(f"pool={hp.get('forget_pool') or 'all'}")
        sw = hp.get("scorer_weights")
        if sw:
            bits.append(pathlib.Path(sw).parent.parent.name)
    if cfg.get("task_setup") in ("100+10x10", "20+20x9"):
        bits.append(cfg["task_setup"])
    return name + (" [" + ", ".join(bits) + "]" if bits else "")

for g in groups:
    d = root / g / "jsons"
    if not d.is_dir():
        print(f"\n{g}: MISSING ({d})")
        continue
    rows = {}
    for f in sorted(d.glob("*.json")):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        s = r.get("summary") or {}
        t, gs = s.get("wall_time_s"), s.get("total_grad_steps")
        if not t:
            continue
        rows.setdefault(arm(r.get("config", {}), r.get("hyperparams") or {}),
                        []).append((t, gs))
    print(f"\n=== {g} ===")
    print(f"  {'arm':58} {'n':>3} {'median min':>11} {'median steps':>13}")
    for name, v in sorted(rows.items(), key=lambda kv: -statistics.median(
            [t for t, _ in kv[1]])):
        times = [t for t, _ in v]
        steps = [s for _, s in v if s]
        st = f"{int(statistics.median(steps)):,}" if steps else "--"
        print(f"  {name:58} {len(v):>3} {statistics.median(times)/60:>11.1f} {st:>13}")
