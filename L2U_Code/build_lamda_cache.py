"""
build_lamda_cache.py  --  LAMDA Class-IL feature cache builder (Option A join).

WHAT THIS DOES
--------------
The IQSeC-Lab/LAMDA-class-il curation defines WHICH samples/families are in the
Class-IL study but ships no feature vectors. The main IQSeC-Lab/LAMDA repo ships
the 4561-dim "Baseline" features per year but not the curation. This script joins
them by sha256 and writes ONE raw cache:

    lamda_class_il_cache.npz

The cache is a PURE FEATURE STORE: all curated families (145) and both splits,
with year retained. It applies NO family selection, NO remapping, NO scaling.
That is deliberate -- the primary experiment (random split, 80 families) and the
secondary temporal-split experiment (63 families) select DIFFERENT family sets
downstream, so selection must not be frozen into the cache. Use
build_arrays_lamda() (below) to turn the cache into train-ready arrays.

Usage
-----
    pip install huggingface_hub pyarrow pandas numpy
    python build_lamda_cache.py                 # build the cache
    python build_lamda_cache.py --preview-only  # no download; just show the
                                                #   80-family schedule from the CSV

NOTE ON TESTING: the parquet-reading path (read_parquet_features) requires pyarrow
and network and was NOT executable in the authoring sandbox. The join mechanics,
coverage accounting, and selection skeleton WERE tested against synthetic frames.
Verify feature_dim and coverage on the first real run before trusting the cache.
"""

import argparse
import os
import sys
import json
import numpy as np

# ------------------------------------------------------------------ config / args
p = argparse.ArgumentParser(description="Build the LAMDA Class-IL feature cache (Option A)")
p.add_argument('--classil-repo', default='IQSeC-Lab/LAMDA-class-il')
p.add_argument('--main-repo',    default='IQSeC-Lab/LAMDA')
p.add_argument('--classil-dir',  default='./Datasets/lamda-class-il')
p.add_argument('--main-dir',     default='./Datasets/lamda')
p.add_argument('--out',          default='./Datasets/lamda_class_il_cache.npz')
p.add_argument('--curation-csv', default='split_samples.csv',
               help='which curation CSV carries the split (confirmed: split_samples.csv)')
p.add_argument('--expected-dim', type=int, default=4561,
               help='asserted feature dim; build fails loudly if the parquet disagrees')
p.add_argument('--min-family-samples', type=int, default=200)
p.add_argument('--num-classes',        type=int, default=80)
p.add_argument('--task0-classes',      type=int, default=30)
p.add_argument('--step-classes',       type=int, default=5)
p.add_argument('--preview-only', action='store_true',
               help='skip all downloads; print the family selection + schedule from the CSV only')
args = p.parse_args()

import re
import pandas as pd

# shared selection logic -- single source of truth (see lamda_data.py)
from lamda_data import select_families, make_task_families

# Feature columns in the LAMDA Baseline parquet are named feat_0 .. feat_N.
# An ALLOWLIST on that pattern is safer than a blocklist of metadata names:
# the first build failed because the parquet calls its VT column `vt_count`
# while the curation CSV calls it `vt_detection`, so a blocklist miss silently
# promoted a metadata column to a feature (caught only by the dim assertion).
FEAT_RE = re.compile(r'^feat_(\d+)$')
# fallback only, if the feat_N naming is ever absent
META_COLS = {'hash', 'sha256', 'sha', 'md5', 'label', 'y', 'target', 'family',
             'class_id', 'year', 'year_month', 'split', 'vt_detection', 'vt_count',
             'added', 'first_appearance_year', 'global_family_count',
             'usable_family_count', 'feature_path'}
HASH_CANDIDATES = ('sha256', 'hash', 'sha', 'md5')


def select_feature_columns(names):
    """Return (feature_cols_in_index_order, how). Prefers feat_N; verifies the
    indices are contiguous 0..N-1 so a missing/extra column cannot shift the
    matrix silently."""
    feats = []
    for c in names:
        m = FEAT_RE.match(c)
        if m:
            feats.append((int(m.group(1)), c))
    if feats:
        feats.sort(key=lambda t: t[0])          # numeric, not lexicographic
        idx = [i for i, _ in feats]
        if idx != list(range(len(idx))):
            gap = next(k for k, v in enumerate(idx) if k != v)
            raise RuntimeError(f"feat_ indices not contiguous: expected {gap}, got {idx[gap]}")
        return [c for _, c in feats], 'feat_N allowlist'
    return [c for c in names if c.lower() not in META_COLS], 'metadata blocklist'


# ------------------------------------------------------------------ helpers
def norm_hash(series):
    """Normalise both sides of the join identically. Verified on real data that
    both sides are 64-char lowercase hex, so this is defensive, not corrective."""
    s = series.map(lambda v: v.decode('utf-8', 'ignore')
                   if isinstance(v, (bytes, bytearray)) else v)
    return s.astype(str).str.strip().str.strip('"\'').str.replace(
        r'^0[xX]', '', regex=True).str.lower()


def pick_hash_col(names):
    low = {c.lower(): c for c in names}
    for cand in HASH_CANDIDATES:
        if cand in low:
            return low[cand]
    return None


def preview_schedule(df, key='family'):
    """Print the selection + schedule from the curation CSV alone (no features)."""
    tr = df[df['split'].astype(str).str.lower().isin(['train', 'training', '1', 'true'])]
    counts = tr[key].value_counts().to_dict()
    id_map, ordered = select_families(counts, args.min_family_samples, args.num_classes)
    if len(ordered) < args.num_classes:
        sys.exit(f"FATAL: only {len(ordered)} families clear >= {args.min_family_samples} "
                 f"train; need {args.num_classes}. Re-run the inspector's threshold sweep.")
    tasks = make_task_families(args.num_classes, args.task0_classes, args.step_classes)
    inv = {v: k for k, v in id_map.items()}
    sizes = [sum(counts[inv[c]] for c in t) for t in tasks]
    print(f"\n=== schedule preview: {args.task0_classes} + "
          f"{args.step_classes} x {len(tasks) - 1} ({len(tasks)} tasks) ===")
    print(f"  task 0 ({len(tasks[0])} fams): {sizes[0]:,} train samples")
    for i, (t, s) in enumerate(zip(tasks[1:], sizes[1:]), 1):
        print(f"  task {i:>2} ({len(t)} fams): {s:>6,}")
    inc = sizes[1:]
    print(f"  incremental mean: {sum(inc) / len(inc):,.0f}  "
          f"(-> CL_ITERS ~ {round(121 * (sum(inc) / len(inc)) / 256 / 50) * 50})")
    print(f"  total selected:   {sum(sizes):,}")
    return id_map, ordered, tasks


# ------------------------------------------------------------------ parquet read (UNTESTED here)
def read_parquet_features(path, expected_hash_col=None):
    """Return (norm_hashes, feature_frame, feature_cols, hash_col, labels, how).
    `labels` is the parquet's `label` column if present (0 benign / 1 malware),
    used as a join-integrity check: the curation is malware_only, so every
    successfully joined row MUST be label==1. A scrambled join would show ~50%
    zeros, which is otherwise undetectable across 4561 anonymous binary columns."""
    import pyarrow.parquet as pq
    names = pq.ParquetFile(path).schema_arrow.names
    hcol = expected_hash_col or pick_hash_col(names)
    if hcol is None:
        raise RuntimeError(f"{os.path.basename(path)}: no hash-like column in {names[:6]}")
    feat_cols, how = select_feature_columns(names)
    lcol = next((c for c in names if c.lower() == 'label'), None)
    read_cols = [hcol] + ([lcol] if lcol else []) + feat_cols
    tbl = pq.read_table(path, columns=read_cols).to_pandas()
    labels = tbl[lcol] if lcol else None
    return norm_hash(tbl[hcol]), tbl[feat_cols], feat_cols, hcol, labels, how


# ------------------------------------------------------------------ 1. curation CSV
if not args.preview_only:
    from huggingface_hub import snapshot_download, list_repo_files, hf_hub_download
    print(f"=== curation repo {args.classil_repo} (csv/json only) ===")
    local = snapshot_download(args.classil_repo, repo_type='dataset',
                              local_dir=args.classil_dir, allow_patterns=['*.csv', '*.json'])
else:
    local = args.classil_dir

csv_path = os.path.join(local, args.curation_csv)
if not os.path.exists(csv_path):
    sys.exit(f"FATAL: {csv_path} not found. (Expected the split-bearing curation CSV.)")

usecols = ['sha256', 'family', 'class_id', 'year', 'split']
cur = pd.read_csv(csv_path, usecols=usecols, dtype={'sha256': str})
cur['sha256'] = norm_hash(cur['sha256'])
if cur['sha256'].duplicated().any():
    n = int(cur['sha256'].duplicated().sum())
    print(f"  !! {n} duplicate curated hashes; keeping first of each")
    cur = cur.drop_duplicates('sha256', keep='first').reset_index(drop=True)
print(f"curated samples: {len(cur):,}  "
      f"(train {int((cur.split == 'train').sum()):,} / "
      f"test {int((cur.split == 'test').sum()):,})")

# family selection uses the family STRING (matches inspector output)
id_map, ordered, tasks = preview_schedule(cur, key='family')

if args.preview_only:
    print("\n(preview-only: no features joined, no cache written)")
    sys.exit(0)

# ------------------------------------------------------------------ 2. enumerate parquets
print(f"\n=== main repo {args.main_repo}: enumerating Baseline parquets ===")
all_files = sorted(f for f in list_repo_files(args.main_repo, repo_type='dataset')
                   if f.endswith('.parquet') and 'var_thresh' not in f)
if not all_files:
    sys.exit("FATAL: no Baseline parquet files found in the main repo.")
print(f"  {len(all_files)} parquet files (expect 24 = 12 years x train/test)")

# ------------------------------------------------------------------ 3. preallocate + fill
# sha256 -> row index in the output, over ALL curated samples
index_of = {h: i for i, h in enumerate(cur['sha256'].tolist())}
n = len(cur)
X = None                     # allocated once we know feature_dim from file 0
filled = np.zeros(n, dtype=bool)
feature_cols = None
hash_col = None
total_benign_hits = 0

for k, rel in enumerate(all_files):
    fp = hf_hub_download(args.main_repo, rel, repo_type='dataset', local_dir=args.main_dir)
    hashes, feats, fcols, hcol, labels, how = read_parquet_features(
        fp, expected_hash_col=hash_col)

    if feature_cols is None:
        feature_cols, hash_col = fcols, hcol
        dim = len(feature_cols)
        if args.expected_dim and dim != args.expected_dim:
            sys.exit(f"FATAL: feature dim {dim} != expected {args.expected_dim}. "
                     f"Wrong parquet variant? (selected via {how}; "
                     f"first cols: {feature_cols[:4]})")
        X = np.zeros((n, dim), dtype=np.float32)
        print(f"  feature_dim = {dim}  (via {how}, from {os.path.basename(rel)})")
        if labels is None:
            print("  !! no `label` column in the parquet; skipping join-integrity check")
    else:
        if fcols != feature_cols:
            sys.exit(f"FATAL: {rel} feature columns differ from file 0 "
                     f"(order/identity mismatch would corrupt the matrix).")

    # rows of this parquet that are curated samples
    vals = feats.to_numpy(dtype=np.float32, copy=False)
    lab = labels.to_numpy() if labels is not None else None
    hit = redundant = benign_hits = 0
    for row_i, h in enumerate(hashes.tolist()):
        j = index_of.get(h)
        if j is None:
            continue
        if filled[j]:
            redundant += 1          # same hash in two parquets -> should not happen
            continue
        if lab is not None and lab[row_i] != 1:
            benign_hits += 1        # curation is malware_only -> must not happen
        X[j] = vals[row_i]
        filled[j] = True
        hit += 1
    msg = f"  [{k + 1:>2}/{len(all_files)}] {os.path.basename(rel):<22} "\
          f"rows={len(hashes):>7,} curated_hits={hit:>7,}"
    if redundant:
        msg += f"  !! {redundant} duplicate-hash rows skipped"
    if benign_hits:
        msg += f"  !!! {benign_hits} joined rows have label!=1"
    print(msg)
    total_benign_hits += benign_hits

# ------------------------------------------------------------------ 4. coverage report
cov = int(filled.sum())
print(f"\n=== coverage ===")
print(f"  {cov:,} / {n:,} curated samples got features ({cov / n:.2%})")
if total_benign_hits == 0:
    print(f"  join integrity: OK -- every joined row has label==1 (malware), "
          f"consistent with the curation's malware_only protocol")
else:
    print(f"  !!! JOIN INTEGRITY FAILURE: {total_benign_hits:,} joined rows have "
          f"label != 1, but the curation is malware_only. This indicates hash "
          f"misalignment -- DO NOT USE THIS CACHE.")
miss = cur.loc[~filled]
if len(miss):
    print(f"  !! {len(miss):,} curated samples had NO feature match")
    by_year = miss['year'].value_counts().sort_index()
    print("     missing by year:")
    for y, c in by_year.items():
        print(f"       {y}: {c:,}")
    # a selected-family miss is worse than a discarded-family miss
    sel_fams = set(ordered)
    sel_miss = int(miss['family'].isin(sel_fams).sum())
    print(f"     of these, {sel_miss:,} belong to the 80 SELECTED families "
          f"(these would silently shrink the schedule's tasks).")
    if cov / n < 0.99:
        print("     coverage < 99%: investigate before trusting the cache.")

# drop unmatched rows so the cache has no all-zero feature vectors
keep = filled
Xk = X[keep]
curk = cur.loc[keep].reset_index(drop=True)

# ------------------------------------------------------------------ 5. save
print(f"\n=== writing {args.out} ===")
np.savez_compressed(
    args.out,
    X=Xk,
    sha256=curk['sha256'].to_numpy(),
    family=curk['family'].astype(str).to_numpy(),
    class_id=curk['class_id'].to_numpy(dtype=np.int64),
    split=curk['split'].astype(str).to_numpy(),
    year=curk['year'].to_numpy(dtype=np.int64),
    feature_dim=np.int64(Xk.shape[1]),
    feature_columns=np.array(feature_cols, dtype=object),
    build_meta=np.array([json.dumps({
        'classil_repo': args.classil_repo, 'main_repo': args.main_repo,
        'n_curated': int(n), 'n_cached': int(len(curk)),
        'coverage': float(cov / n), 'feature_dim': int(Xk.shape[1]),
        'note': 'raw feature store; no selection/remap/scaling applied',
    })], dtype=object),
)
sz = os.path.getsize(args.out) / 1e6
print(f"  saved {len(curk):,} samples x {Xk.shape[1]} features  ({sz:,.0f} MB compressed)")
print(f"  raw cache: apply build_arrays_lamda() downstream for train-ready arrays.")


# ==================================================================
# build_arrays_lamda() now lives in lamda_data.py -- import it from there:
#     from lamda_data import build_arrays_lamda