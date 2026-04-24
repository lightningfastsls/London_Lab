# Module: sis_baselines (17.1)

## Purpose

Compute the **Syntax Information Score at depth 1** — `MI(X_n ; X_{n-1})` —
for an arbitrary sequence of USV call labels. At depth D=1, Hertz et al. 2020's
SIS definition `H(X_n) - H(X_n | X_{n-1}..X_{n-D})` reduces to the mutual
information between consecutive syllables. This is the *decision-gate baseline*
for Phase 17 of `ROADMAP_SIS_BENCHMARK.md`: before implementing any new labeling
method (iMSA / Oren-ridge / AMVOC / SIM), we need to know how well our existing
labelings already capture sequential structure.

## Public Interface

### `SISResult` (frozen dataclass)

| Field | Type | Semantics |
|---|---|---|
| `name` | `str` | User-supplied label (e.g. `"scattoni-7"`) |
| `n_calls` | `int` | Length of the input label sequence |
| `n_labels` | `int` | Alphabet size (unique labels present) |
| `mi_at_lag_1` | `float` | Mutual information in bits (log2) |
| `marginal_entropy` | `float` | `H(X)` in bits |
| `conditional_entropy` | `float` | `H(X | X_{prev}) = H(X) - MI` |
| `entropy_reduction_pct` | `float` | `(MI / H) * 100`, guarded to `0.0` when `H = 0` |

Immutable (`frozen=True`) — attribute assignment raises `FrozenInstanceError`.

### `compute_sis_depth_1(labels, name, sort_by_time=None) -> SISResult`

- `labels: np.ndarray` — accepts **both string and integer** arrays. Factorized
  internally via `pd.factorize(sort=True)`, so `['Flat', 'Down', 'Flat']` and
  `[0, 1, 0]` produce identical MI values.
- `name: str` — stored verbatim on the returned `SISResult`.
- `sort_by_time: np.ndarray | None` — if given, labels are reordered by a
  stable argsort on this key before MI computation. Use when your CSV is not
  yet chronologically sorted by recording time.

### `compute_sis_depth_1_bout_aware(labels, ici_gap_s, bout_threshold_s, name) -> tuple[SISResult, int, int]`

Bout-filtered variant added 2026-04-19 per
`docs/handoffs/sis-baselines-17.1-bout-filter-rerun.md`. Same 7-field
`SISResult` contract, but MI is computed only over within-bout pairs
(pairs whose silent gap `>= bout_threshold_s` are excluded from the
joint-count matrix). Delegates the math to the shared helper
`usv_language.analysis.sequence_analysis.mutual_information_within_bouts`,
which is also used by Phase A2 — any future refinement of bout filtering
happens in one place.

- `labels` must already be chronologically sorted; caller is responsible.
- `ici_gap_s` must have length `len(labels) - 1` and be in the same
  order as `labels`.
- Returns `(SISResult, n_within_pairs, n_excluded_pairs)`.

**Edge cases handled:**

| Input | Behavior |
|---|---|
| Empty array | Returns `SISResult` with all zeros; no crash |
| Single element | `n_calls=1`, `mi=0`; no crash |
| Single unique label | `K=1, mi=0, H=0, pct=0.0` (never NaN) |

## Usage

### As a library

```python
from usv_spectrogram.classification import compute_sis_depth_1
import pandas as pd

df = pd.read_csv("classified_detections_full.csv")
df = df.sort_values(["file", "begin_time_s"])
result = compute_sis_depth_1(
    df["syllable_type"].to_numpy(),
    name="scattoni-7",
)
print(f"MI = {result.mi_at_lag_1:.4f} bits over {result.n_calls} calls")
```

### As a CLI driver

```bash
.venv/bin/python scripts/run_sis_baselines.py \
    --classified-csv classified_detections_full.csv \
    --umap-csv results/recluster_umap_hdbscan/reclassified_detections.csv \
    --output-dir results/sis_baselines/ \
    --dataset 5970
```

Produces:

- `results/sis_baselines/baselines.csv` — one row per labeling with all 7
  `SISResult` fields.
- `results/sis_baselines/baselines.png` — bar chart with horizontal reference
  lines at Hertz 2020's 0.10 (iVoICE), 0.13 (iMUPET), 0.22 (iMSA) bits.
- `results/sis_baselines/parameters.json` — machine-readable sidecar with
  inputs, methodology (including `bout_detection.threshold_s` and
  `ici_gap_source`), and per-labeling pair counts
  (`n_within_bout_pairs`, `n_excluded_pairs`).

The driver:

1. Joins the two CSVs on `(file, begin_time_s)` (or `det_index` as fallback).
2. Sorts by filename-derived absolute datetime + `begin_time_s` when every
   filename follows the canonical `YYYY-MM-DD_HH-MM-SS_...` prefix, matching
   Phase A2's sort. Falls back to lexicographic `(file, begin_time_s)` when
   filenames don't parse (e.g. synthetic test CSVs).
3. Resolves the bout threshold from `--bout-threshold-s` (CLI) ▷
   `data/corpus_facts/<dataset>.json:bout_detection_a2.threshold_s` (auto) ▷
   `None` (raw mode). Pass a negative value to disable filtering explicitly.
4. Resolves the ICI-gap array from `--ici-gap-npy` (CLI) ▷
   `results/sequential_structure{_<dataset>}/ici_gap.npy` (auto-discovery,
   byte-aligned with Phase A2) ▷ inline computation from `end_time_s`.
5. Computes SIS via `compute_sis_depth_1_bout_aware` when a threshold and
   gap array are available; otherwise falls back to `compute_sis_depth_1`.

Missing labeling columns (`syllable_type`, `label`, `hdbscan_label`) are
skipped with a warning rather than crashing the run.

## Reproducibility Check

As of 2026-04-19, SIS 17.1 and Phase A2 use the **same** bout-filtered MI
primitive (`usv_language.analysis.sequence_analysis.mutual_information_within_bouts`)
on the same Scattoni-7 labels for dataset 5970 and reproduce the same value:

| Source                                    | Pairs              | Method                               | Scattoni-7 MI   |
|-------------------------------------------|--------------------|--------------------------------------|-----------------|
| Phase A2 (`results/sequential_structure/`) | 6,350 within-bout  | bout_detection_a2 (0.6 s threshold)  | **0.0921 bits** |
| SIS 17.1 (`results/sis_baselines/`)        | 6,350 within-bout  | bout_detection_a2 (0.6 s threshold)  | **0.0921 bits** |

The two rows **must** agree to within 1e-3 bits; `audit_corpus.py` flags
`scattoni_7_sis_17_1.deprecated = true` whenever they match (confirming SIS
17.1 is a redundant cross-check) and `deprecated = false` with an explicit
`deprecation_reason` describing the drift when they don't.

### Which value is canonical vs the Hertz 2020 benchmark?

**0.0921 bits is the canonical Scattoni-7 value** for comparisons against
the Hertz reference lines (iVoICE 0.10, iMUPET 0.13, iMSA 0.22, SIM 0.23
bits/symbol at depth 1). Verified from the Hertz 2020 Methods section:

> "we divided the labeled syllables into sequences, based on their ISI (with
> 160 ms as a threshold). … An ISI of more than 160 ms represented the end
> of the current sequence and the beginning of a new one."

Our 0.6 s threshold is data-derived (3× median IOI) — a different *value*
than Hertz's 160 ms, but the same *family* of methods (silence-gap
segmentation → per-sequence MI → aggregate across sequences), which is
what makes the numerical comparison meaningful.

### What this means for module 17.9's benchmark chart

`results/sis_baselines/baselines.csv` now contains bout-filtered MI values
for all three labelings. The `baselines.png` reference lines (Hertz's
0.10 / 0.13 / 0.22 / 0.23 bits) are methodology-aligned and can be plotted
without caveats in module 17.9.

### Historical note — pre-2026-04-19

Before the 17.1 rerun, the driver produced raw-consecutive MI values with
no bout filter (Scattoni-7 = 0.0758 bits). The registry flagged that entry
`canonical_for_downstream: false` with an explicit warning that it was not
comparable to Hertz's benchmark lines. The rerun handoff
(`docs/handoffs/sis-baselines-17.1-bout-filter-rerun.md`) drove the
resolution — SIS 17.1 now applies the same bout filter as Phase A2, and
the `scattoni_7_raw_consecutive` registry entry is gone. Its successor
`scattoni_7_sis_17_1` exists only as a drift-detection cross-check.

## Integration Points

- **Upstream:** `classified_detections_full.csv` (output of the DeepSqueak
  classification bridge) and `reclassified_detections.csv` (output of
  `scripts/recluster_umap_hdbscan.py`).
- **Downstream:**
  - Module 17.8 (SIM optimization) consumes the same integer-label sequences
    as initial conditions.
  - Module 17.9 (SIS benchmark report) aggregates `baselines.csv` with results
    from 17.4, 17.7, 17.8 to produce the final comparison table.

## Decision Gate Criterion

Per ROADMAP 17.1: if **all three baselines** are below 0.05 bits on the real
data, the sequential structure of our USV calls may be intrinsically weak and
feature engineering (17.5, 17.6) may not help. In that case, the user is
advised to discuss before proceeding to 17.2+.

## ADR References

None — this module operates on integer label sequences only, no signal
processing. Sample rate / STFT invariants (ADR-001, ADR-002) do not apply.

## Source Files

- `src/usv_spectrogram/classification/sis_baselines.py` — module
- `scripts/run_sis_baselines.py` — CLI driver
- `tests/test_sis_baselines.py` — 17 test cases (pre-implementation spec from
  `test-architect`, 8 ROADMAP + 9 additional gap-pattern)
