"""VENDORED VERBATIM from the user's lamda_data.py -- do not edit here.

This is the project's existing, tested LAMDA data layer, copied unchanged so the
runner's family selection is byte-identical to the one that produced the prior
LAMDA results. Reimplementing it would risk a silently different family set and
make old and new runs incomparable.

NOTE ON DUPLICATION: L2U_Code/ has its own copy for the legacy scripts. That is
two copies of one source of truth, which is the exact pattern this refactor
exists to remove -- collapse to this one when the legacy scripts are retired.

lamda_data.py -- shared LAMDA Class-IL data layer.

Single source of truth for family selection, task construction, and turning the
raw cache (lamda_class_il_cache.npz) into train-ready arrays. Imported by
build_lamda_cache.py, every clmd_lamda_classil_*.py condition, and the meta
pipeline, so the selection skeleton cannot drift between them.

The selection logic is an EXACT mirror of the EMBER top-N block:

    counts_dict = dict(zip(unique.tolist(), counts.tolist()))
    eligible = sorted([f for f, c in counts_dict.items() if c >= 200],
                      key=lambda f: counts_dict[f], reverse=True)[:100]
    id_map = {old_id: new_id for new_id, old_id in enumerate(eligible)}

i.e. descending TRAIN frequency, ties by ascending key, new ids 0..N-1, tasks as
contiguous prefix slices. The mask/eval machinery downstream depends on all of
that -- see PREFIX-MASK INVARIANT below.

PREFIX-MASK INVARIANT
---------------------
new ids run 0..N-1 in descending train frequency, and task_families tiles them as
contiguous prefix slices with no gap or overlap. Everything downstream rides on
this: mask[:active_count], eval slicing [:, :active_count], KD's
[:prev_active_count], and the forget term's [prev_active_count:active_count].
verify_prefix_invariant() asserts it; call it once after building arrays.
"""

import numpy as np

__all__ = ['select_families', 'make_task_families', 'build_arrays_lamda',
           'verify_prefix_invariant', 'DEFAULTS']

# settled protocol defaults (see LAMDA_PROTOCOL.md)
DEFAULTS = {
    'random':   dict(num_classes=80, task0_classes=30, step_classes=5,
                     min_family_samples=200, cl_iters=1000, mem_size=2500),
    'temporal': dict(num_classes=60, task0_classes=30, step_classes=5,
                     min_family_samples=200, temporal_cut=2020, min_test_samples=10),
}


def select_families(train_counts, min_samples=200, num_classes=80):
    """EMBER-identical selection. train_counts: {family_key: train_count}.

    Returns (id_map, ordered) with ordered sorted by descending count, ties by
    ascending key, and id_map[family] = new_id in 0..len(ordered)-1.
    """
    eligible = [f for f, c in train_counts.items() if c >= min_samples]
    ordered = sorted(eligible, key=lambda f: (-train_counts[f], f))[:num_classes]
    return {f: i for i, f in enumerate(ordered)}, ordered


def make_task_families(num_classes=80, task0_classes=30, step_classes=5):
    """Contiguous prefix slices: [0..task0), then steps of `step_classes`."""
    if (num_classes - task0_classes) % step_classes:
        raise ValueError(f"{num_classes} - {task0_classes} is not divisible by "
                         f"{step_classes}; the schedule would not tile evenly")
    tasks = [list(range(task0_classes))]
    for i in range((num_classes - task0_classes) // step_classes):
        lo = task0_classes + i * step_classes
        tasks.append(list(range(lo, lo + step_classes)))
    return tasks


def verify_prefix_invariant(task_families, num_classes):
    """Fail loudly if the prefix-mask assumption is broken."""
    flat = [c for t in task_families for c in t]
    if sorted(flat) != list(range(num_classes)):
        raise RuntimeError("task_families does not tile 0..N-1 exactly")
    if len(flat) != len(set(flat)):
        raise RuntimeError("task_families has overlapping classes")
    starts = [t[0] for t in task_families]
    if starts != sorted(starts):
        raise RuntimeError("task_families are not in ascending order")
    return True


def build_arrays_lamda(cache_path, split_mode='random', temporal_cut=None,
                       min_family_samples=200, num_classes=80,
                       task0_classes=30, step_classes=5, min_test_samples=0,
                       verbose=True):
    """LAMDA analogue of load_ember_data() + the EMBER top-N selection block.

    split_mode='random'   -> use the curation's own train/test split (PRIMARY).
    split_mode='temporal' -> train = year < temporal_cut, test = year >= cut
                             (SECONDARY drift-preserving run). Pass
                             min_test_samples to require adequate test support,
                             which the random split does not need.

    Returns dict with X_train, y_train, X_test, y_test, task_families, ordered,
    id_map, input_dim, and per-task train sizes.
    """
    d = np.load(cache_path, allow_pickle=True)
    X = d['X']
    fam = d['family'].astype(str)
    split = np.char.lower(d['split'].astype(str))
    year = d['year']

    if split_mode == 'random':
        train_sel = split == 'train'
    elif split_mode == 'temporal':
        if temporal_cut is None:
            raise ValueError("split_mode='temporal' requires temporal_cut")
        train_sel = year < temporal_cut
    else:
        raise ValueError(f"unknown split_mode {split_mode!r}")
    test_sel = ~train_sel

    # --- selection on the TRAIN side only, exactly as in EMBER ---
    keys, cnts = np.unique(fam[train_sel], return_counts=True)
    train_counts = dict(zip(keys.tolist(), cnts.tolist()))

    if min_test_samples > 0:
        tk, tc = np.unique(fam[test_sel], return_counts=True)
        test_counts = dict(zip(tk.tolist(), tc.tolist()))
        train_counts = {f: c for f, c in train_counts.items()
                        if test_counts.get(f, 0) >= min_test_samples}

    id_map, ordered = select_families(train_counts, min_family_samples, num_classes)
    if len(ordered) < num_classes:
        raise RuntimeError(
            f"only {len(ordered)} families clear >= {min_family_samples} train"
            + (f" and >= {min_test_samples} test" if min_test_samples else "")
            + f" under split_mode={split_mode}; need {num_classes}")

    task_families = make_task_families(num_classes, task0_classes, step_classes)
    verify_prefix_invariant(task_families, num_classes)

    # --- remap; index ONCE per split to avoid a second full-size copy ---
    sel_arr = np.array(ordered)
    in_sel = np.isin(fam, sel_arr)

    def take(split_mask):
        m = split_mask & in_sel
        if not m.any():
            return X[:0].copy(), np.zeros(0, dtype=np.int64)
        y = np.fromiter((id_map[f] for f in fam[m]), dtype=np.int64, count=int(m.sum()))
        return X[m], y

    X_train, y_train = take(train_sel)
    X_test, y_test = take(test_sel)

    task_sizes = [int((np.isin(y_train, t)).sum()) for t in task_families]

    if verbose:
        print(f"[lamda_data] split_mode={split_mode}"
              + (f" cut={temporal_cut}" if split_mode == 'temporal' else ""))
        print(f"[lamda_data] families={len(ordered)} "
              f"tasks={len(task_families)} input_dim={X.shape[1]}")
        print(f"[lamda_data] train={len(y_train):,} test={len(y_test):,}")
        print(f"[lamda_data] task sizes: t0={task_sizes[0]:,} "
              f"incr={task_sizes[1:]}")
        if len(task_sizes) > 1:
            mean_inc = sum(task_sizes[1:]) / len(task_sizes[1:])
            print(f"[lamda_data] mean incremental={mean_inc:,.1f} "
                  f"-> 121-epoch CL_ITERS ~ {round(121 * mean_inc / 256 / 50) * 50}")

    return dict(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                task_families=task_families, ordered=ordered, id_map=id_map,
                input_dim=int(X.shape[1]), task_sizes=task_sizes)