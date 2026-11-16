"""Held-out tables: raw per-arm means AND paired differences.

WHY BOTH, AND WHY PAIRED IS THE ONE THAT ANSWERS THE QUESTION.

Presentation order dominates everything else in this setting. Measured over
eight orderings on EMBER 2018, seen-class accuracy ranged from 65.9 to 79.4
with sd ~3.6, while the effect of unlearning at all was ~1.7. Ordering noise is
more than twice the treatment effect, so a table of raw means over a couple of
held-out orderings is mostly measuring which families happened to land in which
task. Two such tables, run on different machines, put `none` first and last.

Pairing removes that entirely: both arms see the SAME ordering, so the ordering
term cancels in the difference. The same eight orderings that give a raw sd of
3.6 give a paired sd of 1.36 for random-vs-none and 0.29 for
random-vs-gradconflict -- enough to separate arms differing by a third of a
point with five orderings, where the unpaired comparison would need hundreds.

The raw means are still reported. They are what a reader expects to see, they
carry the absolute level that the paired deltas discard, and a delta whose
components are never shown invites the suspicion that the components were
unflattering. Both, always.

NO p-VALUES HERE. This reports the mean difference, its paired standard
deviation, the standard error, a t statistic and the sign count. Turning t into
p needs a distributional assumption that five orderings cannot support, and the
sign count is the more honest summary at these sample sizes -- 8/8 positive
says more than a p-value computed from eight points.
"""

from __future__ import annotations

import math
import statistics as st


def paired_stats(a: dict, b: dict) -> dict | None:
    """Difference a - b over the orderings both arms share.

    `a` and `b` map ordering -> score. Orderings missing from either side are
    dropped rather than filled, because a pair is the unit of analysis: half a
    pair carries the ordering term it exists to cancel.
    """
    keys = sorted(set(a) & set(b))
    diffs = [a[k] - b[k] for k in keys if a[k] is not None and b[k] is not None]
    if not diffs:
        return None
    mean = st.mean(diffs)
    out = {"n": len(diffs), "mean": mean,
           "positive": sum(d > 0 for d in diffs),
           "per_order": {str(k): a[k] - b[k] for k in keys
                         if a[k] is not None and b[k] is not None}}
    if len(diffs) > 1:
        sd = st.stdev(diffs)
        se = sd / math.sqrt(len(diffs))
        out.update({"sd": sd, "se": se})
        # A t statistic from TWO points is not a statistic. Its sd has one
        # degree of freedom, so it is whatever the gap between the two numbers
        # happens to be, and it prints large or small for reasons that have
        # nothing to do with the effect: a real run produced t = 2.73 for one
        # arm and t = -2.60 for another from two orderings apiece, and both are
        # meaningless. Withholding it is the honest move -- the mean, the sd
        # and the sign count still say everything two points can say.
        if len(diffs) >= 3:
            out["t"] = (mean / se) if se else float("inf")
        # Orderings needed to reach |t| = 2 at this effect and spread. Reported
        # so an inconclusive comparison says how much more would settle it
        # instead of just failing to reach significance.
        if mean:
            out["n_for_t2"] = math.ceil((2 * sd / abs(mean)) ** 2)
    return out


def render(table: dict, arms: list, orders: list, scope: str,
           reference: str | None = None) -> str:
    """One scope's block: raw means, then paired deltas against `reference`.

    `table` maps str(order) -> arm -> {scope: value}.
    """
    lines = []
    width = max(max(len(a) for a in arms), 12)
    get = lambda o, a: (table.get(str(o), {}).get(a) or {}).get(scope)

    lines.append("  raw means")
    lines.append("  " + " " * 6 + "".join(f"{a:>{width + 2}s}" for a in arms))
    for o in orders:
        cells = "".join(
            (f"{get(o, a):>{width + 2}.2f}" if get(o, a) is not None
             else f"{'-':>{width + 2}s}") for a in arms)
        lines.append(f"  p{o:<5d}" + cells)
    if len(orders) > 1:
        means = {}
        for a in arms:
            vals = [get(o, a) for o in orders if get(o, a) is not None]
            means[a] = st.mean(vals) if vals else None
        lines.append("  " + "mean".ljust(6) + "".join(
            (f"{means[a]:>{width + 2}.2f}" if means[a] is not None
             else f"{'-':>{width + 2}s}") for a in arms))

    ref = reference if reference in arms else ("none" if "none" in arms
                                               else arms[-1])
    lines.append("")
    lines.append(f"  paired difference vs `{ref}` (same ordering, "
                 f"so the ordering term cancels)")
    lines.append(f"  {'arm':>{width}s} {'delta':>8} {'sd':>7} {'se':>7} "
                 f"{'t':>7} {'sign':>7} {'n for |t|=2':>12}")
    by_arm = {a: {o: get(o, a) for o in orders} for a in arms}
    for a in arms:
        if a == ref:
            continue
        s = paired_stats(by_arm[a], by_arm[ref])
        if s is None:
            lines.append(f"  {a:>{width}s} {'-':>8}")
            continue
        sd = f"{s['sd']:7.3f}" if "sd" in s else f"{'-':>7}"
        se = f"{s['se']:7.3f}" if "se" in s else f"{'-':>7}"
        t = f"{s['t']:7.2f}" if "t" in s else f"{'n/a':>7}"
        n2 = f"{s['n_for_t2']:>12}" if "n_for_t2" in s else f"{'-':>12}"
        lines.append(f"  {a:>{width}s} {s['mean']:+8.3f} {sd} {se} {t} "
                     f"{s['positive']:>3}/{s['n']:<3} {n2}")
    if len(orders) < 3:
        lines.append(f"  ! {len(orders)} ordering(s): t is withheld -- it would "
                     f"be computed from {len(orders)} point(s) and means "
                     f"nothing here.")
    if len(orders) < 5:
        lines.append(f"  ! only {len(orders)} ordering(s): ordering variance "
                     f"(sd ~3.6 on EMBER) swamps effects of this size, and the "
                     f"effect itself")
        lines.append(f"    varies by ordering (+0.12 to +3.99 for "
                     f"random-vs-none over eight). Five or more, and the "
                     f"deltas become interpretable.")
    return "\n".join(lines)


def summary(table: dict, arms: list, orders: list, scope: str,
            reference: str | None = None) -> dict:
    """The same numbers as `render`, for the JSON record."""
    ref = reference if reference in arms else ("none" if "none" in arms
                                               else arms[-1])
    get = lambda o, a: (table.get(str(o), {}).get(a) or {}).get(scope)
    by_arm = {a: {o: get(o, a) for o in orders} for a in arms}
    means = {}
    for a in arms:
        vals = [v for v in by_arm[a].values() if v is not None]
        means[a] = st.mean(vals) if vals else None
    return {"scope": scope, "reference": ref, "raw_mean": means,
            "paired_vs_reference": {a: paired_stats(by_arm[a], by_arm[ref])
                                    for a in arms if a != ref}}
