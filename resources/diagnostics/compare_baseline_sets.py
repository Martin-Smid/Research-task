"""Compare two baseline sets (e.g. Quadro P4000 vs Koios P100): relative L2 difference per array, in %.

Usage: python compare_baseline_sets.py <dir_a> <dir_b>   (dirs holding the .npz files, searched recursively)
"""
import sys
from pathlib import Path

import numpy as np

a_root, b_root = Path(sys.argv[1]), Path(sys.argv[2])
rows = []
for a_path in sorted(a_root.rglob("*.npz")):
    b_path = b_root / a_path.relative_to(a_root)
    if not b_path.exists():
        print(f"missing in B: {a_path.relative_to(a_root)}")
        continue
    with np.load(a_path, allow_pickle=False) as a, np.load(b_path, allow_pickle=False) as b:
        for key in sorted(set(a.files) & set(b.files)):
            x, y = a[key], b[key]
            if x.dtype.kind not in "fc" or x.shape != y.shape or x.size == 0:
                continue
            norm = np.linalg.norm(x)
            if norm == 0:
                continue
            rel = 100 * np.linalg.norm(x - y) / norm
            rows.append((str(a_path.relative_to(a_root)), key, rel))
rows.sort(key=lambda r: -r[2])
print(f"{len(rows)} float arrays compared; relative L2 difference |A-B|/|A| in %")
print(f"identical (0%): {sum(r[2] == 0 for r in rows)}, max: {rows[0][2]:.3e} %, median: {np.median([r[2] for r in rows]):.3e} %")
print("\nworst 25:")
for f, k, r in rows[:25]:
    print(f"  {r:11.3e} %  {f}  {k}")
per_file = {}
for f, k, r in rows:
    per_file[f] = max(per_file.get(f, 0), r)
print("\nmax per file:")
for f, r in sorted(per_file.items(), key=lambda t: -t[1]):
    print(f"  {r:11.3e} %  {f}")
