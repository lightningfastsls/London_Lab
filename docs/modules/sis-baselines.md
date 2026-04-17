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
    --output-dir results/sis_baselines/
```

Produces:

- `results/sis_baselines/baselines.csv` — one row per labeling with all 7
  `SISResult` fields.
- `results/sis_baselines/baselines.png` — bar chart with horizontal reference
  lines at Hertz 2020's 0.10 (iVoICE), 0.13 (iMUPET), 0.22 (iMSA) bits.

The driver joins the two CSVs on `det_index`, sorts by `(file, begin_time_s)`,
and computes SIS for three labeling columns: `syllable_type` (Scattoni-7),
`label` (DeepSqueak-27), `hdbscan_label` (HDBSCAN-3). Missing columns are
skipped with a warning rather than crashing the run.

## Reproducibility Check

Phase A2 previously computed MI at lag 1 on Scattoni-7 labels for the 5970
dataset and obtained **0.093 bits**. Running this module over the same dataset
should reproduce that value exactly — it reuses the same underlying estimator
(`usv_language.analysis.sequence_analysis.mutual_information_at_lag`).

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
