# Shape-invariance benchmark — method plug-in contract

This package is the shared foundation for the USV shape-invariance bake-off
(handoff: `usv_shape_invariance_handoff.md`). Phase-1 method agents (M2/M3/M4)
read THIS file and plug in without touching the harness or other methods' files.

## The eval contract (identical for every method)

- **Labels.** 611 human-labeled calls from `data/manual_shape_labels.csv`, joined
  to the registered ridges via the SPEC `build_join(ws, cid, human_df, offset=-1)`
  with `shape_label=='unclear'` dropped. Families: `group_family` ->
  `{chevron:44, jump:204, flat:125, complex:67}` (PRIMARY) plus
  `{Noise:66, Down-FM:54, Up-FM:27, Short:24}` for context.
- **Decision metric.** Leave-one-out kNN retrieval purity vs human family,
  1000x bootstrap 95% CI. Primary `k=10`. Decide on **NON-overlapping CIs**.
- **Baselines on the same rows.** registration-Euclidean on `contour50`
  (= IDENTITY / incumbent) and soft-DTW
  (`tslearn.metrics.cdist_soft_dtw_normalized(X[:,:,None], gamma=1.0)`,
  distance-native) = THE BAR.

### DATA DEVIATION FROM THE HANDOFF (verified 2026-06-07)
The handoff says "ALL 611 labels are cohort lab_131204; wild UNLABELED ->
stratification is VACUOUS." **This is STALE.** The current CSV is the
Phase-2-expanded gold set: the 611 labeled rows span cohorts
`{lab_131204:182, 5970:204, 9252:140, 3452:85}`. Cross-cohort stratification is
therefore REAL. The harness uses **`cohort` (4 levels) as the within-stratum
field** (the cage axis the handoff actually wants). `pairing` (parsed from
wav_stem) is provided too but is degenerate for wild (singleton stems).

## How to plug a method in

A method is either:
1. an **embedding** `(N, d)` — preferred; or
2. a **distance matrix** `(N, N)` — distance-native (soft-DTW).

```python
import sys; sys.path.insert(0, "scripts/experiments")
from shape_invariance.loader import load_labeled
from shape_invariance.harness import benchmark
from shape_invariance.io import save_features
from shape_invariance.reversal import reversal_test

data = load_labeled()                      # dict aligned to the 611 rows
X = my_encode_batch(data["contour50"])     # (611, d)  -- or contour128
save_features("m3_persistence", X, params={...})

res = benchmark(
    X, kind="embedding",
    meta=data,                             # must carry 'family' and 'stratum'
    families=["chevron", "jump", "flat", "complex",
              "Noise", "Down-FM", "Up-FM", "Short"],
    k=10, ks=(1, 5, 15),
    side=data["side"],                     # (N,3) raw -> z-scored inside
)
rev = reversal_test(my_encode_single, data["contour50"])
```

### `benchmark(X_or_D, *, kind, meta, families, k=10, ks=(1,5,15), side=None, n_boot=1000, seed=42)`

Returns a nested dict:

```
{
  "pooled_invariant":        {family: [point, lo, hi]},
  "pooled_sidechannel":      {family: [point, lo, hi]} | {"_note": ...},
  "withinstratum_invariant": {family: [point, lo, hi]},
  "withinstratum_sidechannel:{family: [point, lo, hi]} | {"_note": ...},
  "k_sweep":                 {"1": {family: point}, "5": {...}, "15": {...}}  # pooled invariant
}
```

- **Settings** = `{pooled, within-stratum} x {invariant-only, invariant + z-scored side-channels}`.
- **Side-channels** (handoff rule 2): `(N,3)` raw `[duration_ms, freq_range, freq_std]`,
  z-scored inside the harness, then `hstack`ed onto the invariant matrix. The
  invariant matrix itself is **not** rescaled (per the handoff text). NOTE: if a
  method's features are on a very different magnitude than the unit-scale z-scored
  side-channels, the side-channels will be dominated (e.g. Hz-scale `contour50`).
  Report both settings regardless; for `O(1)`-scale features (e.g. the turning
  function in radians) the side-channels meaningfully participate.
- **within-stratum**: each query's neighbour pool is restricted to its own
  `cohort`; per-point purity is pooled across cohorts (N-weighted by
  construction) and bootstrapped with the IDENTICAL resample-target-points +
  2.5/97.5 percentile recipe the SPEC bootstrap uses. If a method only wins
  pooled, suspect cage leakage, not shape.
- For `kind='distance'`, side-channels are not applicable (no embedding to
  concatenate) -> the two side-channel settings carry `{"_note": ...}`.

## Cross-cutting rules enforced as code

1. **Reversal (rule 1).** `reversal.reversal_test(encode_fn, contours50)` returns
   `{passed, self_reverse_median, decile_threshold, note}`; PASS iff median
   self-reverse distance >= 90th percentile of pairwise distances. If a method
   FAILS, append a signed direction feature (net slope) and re-test; record both
   verdicts.
4. **scale_invariant: bool, default False** (keep modulation depth as signal).
   Test both where cheap (M4/M5).

## Output convention (reproducibility)

- Features: `features/shape_invariance/{method}__{paramhash}.npy` + sibling
  `.json` of params (`io.save_features`). `paramhash` = md5(sorted params)[:8].
- Per-method scorecard: `results/shape_invariance/{method}_result.json` with the
  structure in the handoff CONTEXT (method, status, feature_path, params, d,
  reversal{...}, purity{setting:{family:[p,lo,hi]}}, k_sweep, prediction_held,
  notes).
- Print all params / Ns / thresholds / row-counts (repo convention
  `feedback_analysis_print_params`).

## SPEC functions (REUSE, never reimplement)

In `scripts/experiments/eval_shape_human_anchored.py`:
`build_join`, `group_family`, `loo_knn_purity`, `knn_purity_from_distance`,
`bootstrap_purity_ci`, `bootstrap_purity_ci_from_distance`. They are
unit-tested-as-spec (`tests/experiments/test_eval_shape_human_anchored.py`).
The pooled settings call them directly; the within-stratum settings reuse their
exact bootstrap recipe on per-stratum-restricted per-point purities.
```
