"""Print the buffer and selection fingerprint of a run record.

    python3 bufcmp.py <record.json>

Run it on the SAME run id on two machines and diff the output. If the buffer
composition differs, the replay contents differ, and the accuracy gap is
downstream of sample SELECTION -- which on this corpus means the per-family
IsolationForest, i.e. scikit-learn. If the buffer matches and accuracy still
diverges, the cause is in the optimisation, i.e. torch/CUDA arithmetic.
"""
import hashlib, json, sys

if len(sys.argv) != 2:
    sys.exit(__doc__)
rec = json.load(open(sys.argv[1]))
env = rec.get("environment") or {}
print("run_id       ", rec["run_id"])
print("git_commit   ", env.get("git_commit"))
for k in ("torch", "sklearn", "numpy", "python", "device", "gpu"):
    if k in env:
        print(f"{k:13s}", env[k])
print()
print(f"{'task':>4} {'seen':>8} {'task0':>8} {'recent':>8} "
      f"{'buf size':>9} {'fams':>5} {'removed':>8} {'buf_after':>10}")
for t in rec["per_task"]:
    b = t.get("buffer") or {}
    s = t.get("selection") or {}
    m = t["metrics"]
    print(f"{t['task']:>4} {m['seen']['accuracy']:8.3f} "
          f"{m['task0']['accuracy']:8.3f} {m['recent']['accuracy']:8.3f} "
          f"{str(b.get('size','-')):>9} {str(b.get('n_families','-')):>5} "
          f"{str(s.get('n_forget','-')):>8} "
          f"{str(s.get('buffer_size_after','-')):>10}")

# A hash over the per-family occupancy: identical buffers hash alike, so a
# single line settles whether the two machines replayed the same thing.
comp = [(t["task"], (t.get("buffer") or {}).get("size"),
         (t.get("buffer") or {}).get("n_families")) for t in rec["per_task"]]
print("\nbuffer fingerprint",
      hashlib.sha256(json.dumps(comp, sort_keys=True).encode()).hexdigest()[:16])
