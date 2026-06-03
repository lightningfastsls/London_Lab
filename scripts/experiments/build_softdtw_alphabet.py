"""Build a full-corpus elastic (soft-DTW) shape alphabet — Track A of
PLAN_elastic_shape_clustering.md.

This is a PARALLEL artifact to the incumbent registration alphabet
(`models/shape_kmeans/k20.joblib`) — it does NOT overwrite it.

OOM-safe scale strategy (a full 67k×67k DTW matrix is ~36 GB — never built):
  1. Fit soft-DTW k-means (tslearn TimeSeriesKMeans, metric='softdtw') on a
     cohort-balanced stratified SUBSAMPLE (~8-10k) to learn barycenter centroids.
  2. Assign all 67,337 ridges to the nearest centroid by soft-DTW distance,
     STREAMED in batches (cdist_soft_dtw_normalized centroid-vs-batch).

Compute note (measured 2026-06-03 smoke): soft-DTW k-means is CPU-bound (not
memory-bound — ridges are 50-d), scaling super-linearly: ~15→50 s/iter at
N=300→1500. A full ~8k fit is ~1-3 h CPU. Per the plan's compute&safety, the
heavy fit belongs on the rig (gated launch); the box can do the smoke + the
streamed assignment. Use --smoke for a tiny end-to-end validation on the box.

This builder is NOT on the GATE-1 critical path: GATE 1 is decided by the
human-anchored eval on the 200×200 labeled soft-DTW matrix
(eval_shape_human_anchored.py). This alphabet feeds the SECONDARY NMI metric and
is the production preprocessor candidate for Phase 3.

Outputs (parallel, do NOT overwrite incumbent):
  - models/shape_kmeans/k20_softdtw.joblib
  - models/shape_kmeans/k20_softdtw_letters.parquet   (per-call alphabet letter)
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd


def stratified_cohort_subsample(cohort: np.ndarray, n_total: int, seed: int) -> np.ndarray:
    """Cohort-balanced subsample indices. Aims for n_total/n_cohorts per cohort;
    cohorts smaller than that quota contribute ALL their rows, and the freed
    quota is NOT redistributed (keeps it simple + reproducible). Prints the
    realized per-cohort counts (per feedback_analysis_print_params)."""
    rng = np.random.default_rng(seed)
    cohorts = sorted(set(cohort.tolist()))
    quota = max(1, n_total // len(cohorts))
    picks = []
    realized = {}
    for c in cohorts:
        idx = np.where(cohort == c)[0]
        take = min(quota, len(idx))
        chosen = rng.choice(idx, take, replace=False)
        picks.append(chosen)
        realized[c] = int(take)
    out = np.concatenate(picks)
    print(f"  [SUBSAMPLE] target≈{n_total} ({quota}/cohort) -> realized {realized} = {len(out)} rows")
    return np.sort(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", default="/home/shachar/.claude/jobs/9a954f32/tmp/shape_data/true_registered_ridges_meta.npz")
    ap.add_argument("--out-model", default="models/shape_kmeans/k20_softdtw.joblib")
    ap.add_argument("--out-letters", default="models/shape_kmeans/k20_softdtw_letters.parquet")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--subsample", type=int, default=8000)
    ap.add_argument("--max-iter", type=int, default=15)
    ap.add_argument("--n-init", type=int, default=2)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--assign-batch", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end run on the box: subsample=400, k=8, max_iter=3, "
                         "assign 1000 rows only; proves the pipeline + times it. Writes *_SMOKE outputs.")
    args = ap.parse_args()

    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.metrics import cdist_soft_dtw_normalized

    if args.smoke:
        args.subsample, args.k, args.max_iter, args.n_init = 400, 8, 3, 1
        args.out_model = args.out_model.replace(".joblib", "_SMOKE.joblib")
        args.out_letters = args.out_letters.replace(".parquet", "_SMOKE.parquet")

    print("=" * 92)
    print("BUILD SOFT-DTW SHAPE ALPHABET" + ("  [SMOKE]" if args.smoke else ""))
    print("=" * 92)
    print(f"  meta={args.meta}")
    print(f"  k={args.k}  subsample={args.subsample}  max_iter={args.max_iter}  n_init={args.n_init}"
          f"  gamma={args.gamma}  assign_batch={args.assign_batch}  seed={args.seed}")

    m = np.load(args.meta, allow_pickle=True)
    Sh = m["shapes"].astype(np.float64)
    coh = m["cohort"].astype(str)
    ws = m["wav_stem"].astype(str)
    cid = m["call_id"]
    print(f"  ridges={Sh.shape}  cohorts={dict(zip(*np.unique(coh, return_counts=True)))}")

    sub_idx = stratified_cohort_subsample(coh, args.subsample, args.seed)
    Xsub = Sh[sub_idx][:, :, None]

    print(f"\n  [FIT] TimeSeriesKMeans(softdtw, gamma={args.gamma}) on {len(sub_idx)} ridges ...")
    t0 = time.time()
    km = TimeSeriesKMeans(
        n_clusters=args.k, metric="softdtw", metric_params={"gamma": args.gamma},
        max_iter=args.max_iter, n_init=args.n_init, random_state=args.seed, n_jobs=-1, verbose=False,
    )
    km.fit(Xsub)
    print(f"  [FIT] done in {time.time() - t0:.1f}s  inertia={km.inertia_:.4f}")

    centroids = km.cluster_centers_  # (k, sz, 1)

    # ---- streamed assignment of ALL ridges to nearest centroid by soft-DTW ----
    n_assign = 1000 if args.smoke else len(Sh)
    print(f"\n  [ASSIGN] {n_assign} ridges -> nearest of {args.k} centroids, batch={args.assign_batch} ...")
    letters = np.full(n_assign, -1, dtype=np.int32)
    t0 = time.time()
    for s in range(0, n_assign, args.assign_batch):
        e = min(s + args.assign_batch, n_assign)
        batch = Sh[s:e][:, :, None]
        D = cdist_soft_dtw_normalized(batch, centroids, gamma=args.gamma)  # (b, k)
        letters[s:e] = D.argmin(axis=1).astype(np.int32)
        if s == 0 or (s // args.assign_batch) % 20 == 0:
            print(f"    .. {e}/{n_assign}  ({time.time() - t0:.0f}s)")
    print(f"  [ASSIGN] done in {time.time() - t0:.1f}s")
    print(f"  [ASSIGN] letter histogram = {dict(zip(*np.unique(letters[letters >= 0], return_counts=True)))}")

    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    import joblib
    joblib.dump({"kmeans": km, "k": args.k, "gamma": args.gamma, "metric": "softdtw",
                 "subsample_idx": sub_idx, "seed": args.seed}, args.out_model)
    letters_df = pd.DataFrame({
        "wav_stem": ws[:n_assign], "call_id": cid[:n_assign],
        "cohort": coh[:n_assign], "softdtw_letter": letters,
    })
    letters_df.to_parquet(args.out_letters)
    print(f"\n[OUT] {args.out_model}")
    print(f"[OUT] {args.out_letters}")
    if args.smoke:
        print("\n[SMOKE OK] pipeline validated end-to-end. For the full fit, run WITHOUT --smoke "
              "on the rig (gated) or as a long box background job.")


if __name__ == "__main__":
    main()
