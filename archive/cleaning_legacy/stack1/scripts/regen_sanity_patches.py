"""Re-sample sanity patches uniformly across the full lab/wild/vocalmat pools.

After the wild top-up, the original sanity_patches/wild_*.png are biased
toward the 12 5970 recordings (the only wild patches present when prep
finished). This script:

1. Deletes existing sanity_patches/{vocalmat,lab,wild}_*.png
2. Draws 50 random per cohort from the FULL patch inventory:
   - vocalmat: 50 random rows from manifest_all.csv
   - lab:      50 random PNGs from patches/lab/
   - wild:     50 random PNGs from patches/wild/
3. Copies them with the prep's naming convention (cohort_NN_*.png)

Uses the prep's seed (default 1729) so the lab + vocalmat samples are
reproducible. Wild is re-sampled with the same seed so the choices are
deterministic.

Diagnostic-only, doesn't modify production code or manifests.
"""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path("data/lab_cnn_training")
SANITY_DIR = OUTPUT_DIR / "sanity_patches"
N_PER_COHORT = 50
SEED = 1729


def main() -> int:
    if not SANITY_DIR.is_dir():
        print(f"ERROR: {SANITY_DIR} missing", file=sys.stderr)
        return 1

    rng = np.random.default_rng(SEED)

    # Pool each cohort.
    pools = {}

    # vocalmat: pull from manifest_all.csv (path column).
    vm_paths = []
    with open(OUTPUT_DIR / "manifest_all.csv", newline="") as f:
        for row in csv.DictReader(f):
            vm_paths.append(row["path"])
    pools["vocalmat"] = vm_paths

    # lab + wild: glob the patch dirs.
    for cohort in ("lab", "wild"):
        cohort_dir = OUTPUT_DIR / "patches" / cohort
        if cohort_dir.is_dir():
            pools[cohort] = sorted(str(p) for p in cohort_dir.glob("*.png"))
        else:
            pools[cohort] = []

    for cohort, pool in pools.items():
        print(f"  {cohort:10} pool size: {len(pool)}")

    # Wipe existing sanity patches.
    n_deleted = 0
    for p in SANITY_DIR.glob("*.png"):
        p.unlink()
        n_deleted += 1
    print(f"  deleted {n_deleted} existing sanity patches")

    # Re-sample.
    for cohort, pool in pools.items():
        if not pool:
            print(f"  WARN: {cohort} pool empty, skipping")
            continue
        n_sample = min(N_PER_COHORT, len(pool))
        idx = rng.choice(len(pool), size=n_sample, replace=False)
        for i, j in enumerate(idx):
            src = Path(pool[j])
            dst = SANITY_DIR / f"{cohort}_{i:02d}_{src.name}"
            shutil.copy2(src, dst)
        print(f"  {cohort:10} wrote {n_sample} sanity patches")

    final = sorted(SANITY_DIR.glob("*.png"))
    print(f"\nDONE. sanity_patches/ has {len(final)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
