# Repertoire Statistical Analysis

**Phase:** 14.3 (Syllable Repertoire Statistical Analysis)
**ADRs:** None (statistical analysis, no DSP or ML architecture decisions)
**Tests:** `tests/test_classification/test_repertoire_stats.py` (33 tests)
**Review:** `docs/reviews/repertoire-stats-review.md`

## Purpose

Compares syllable repertoires between mouse populations (wild vs lab) using DeepSqueak classification labels from Phase 14.2. This is the scientific payoff of the classification pipeline: quantifying whether wild mice vocalize differently from lab mice.

```
Classified CSV (Phase 14.2) -> THIS MODULE -> statistical tests + plots + report
```

Methods: per-animal proportions, Shannon entropy (diversity), transition matrices (sequential structure), PERMANOVA (multivariate composition), Jensen-Shannon divergence (distribution distance), chi-squared (frequency differences), Frobenius norm (transition structure).

## Public Interface

### `RepertoireConfig`

```python
@dataclass(frozen=True)
class RepertoireConfig:
    classified_data_path: Path          # CSV from Phase 14.2
    population_column: str = "population"
    animal_id_column: str = "animal_id"
    syllable_column: str = "label"
    time_column: str = "begin_time_s"
    n_permutations: int = 1000
    random_seed: int = 42
    output_dir: Path = Path("analysis/repertoire")
```

Validates: `n_permutations > 0`, auto-converts string paths to `Path`.

### Core Functions

| Function | Signature | Purpose |
|----------|-----------|---------|
| `syllable_proportions` | `(df, group_col, syllable_col) -> DataFrame` | Per-group syllable type proportions (sum to 1.0) |
| `syllable_diversity` | `(df, group_col, syllable_col) -> DataFrame` | Shannon entropy H per group (bits) |
| `transition_matrix` | `(df, animal_id, ..., all_labels) -> (ndarray, list)` | Row-stochastic K x K transition matrix for one animal |
| `compare_repertoires` | `(df, method, ...) -> dict` | Statistical comparison dispatcher (3 methods) |
| `compare_transition_matrices` | `(df, ...) -> dict` | Frobenius norm + permutation test on transition structure |
| `plot_repertoire_comparison` | `(df, ..., output_dir) -> list[Path]` | 4 publication-ready figures |
| `generate_report` | `(results, output_dir) -> Path` | Markdown report with plain-language interpretation |
| `analyze_repertoire` | `(RepertoireConfig) -> dict` | Full pipeline orchestrator |

### Comparison Methods

| Method | Statistic | Unit | Populations | Notes |
|--------|-----------|------|-------------|-------|
| PERMANOVA | pseudo-F | dimensionless | 2+ | Primary test. Operates on per-animal Bray-Curtis distances. R² effect size. |
| JSD | Jensen-Shannon divergence | [0, 1] (bits, base-2) | exactly 2 | Symmetric, bounded. `scipy.jensenshannon()²` to get divergence not distance. |
| Chi-squared | chi² | dimensionless | 2+ | Cramer's V effect size. **Caution:** pools calls across animals (pseudoreplication). |
| Transition | Frobenius norm | dimensionless | exactly 2 | Compares mean transition matrices. Reports top 10 differential transitions. |

### Key Limitations

1. **JSD and transition comparison require exactly 2 populations.** Passing 3+ populations raises `ValueError`. Use PERMANOVA or chi-squared for multi-population comparisons.
2. **Chi-squared treats each call as independent** (pseudoreplication). With N=5 animals and 100 calls each, chi-squared sees 500 "observations" per group, inflating significance. Prefer PERMANOVA for primary inference; use chi-squared as a supplementary check.
3. **PERMANOVA with 1 animal per population** produces R²=1.0 and pseudo-F=0.0. Results are not interpretable — need at least 2 animals per group.

## Output Structure

```
analysis/repertoire/
+-- syllable_proportions.csv         # Per-animal proportions
+-- diversity_comparison.csv         # Shannon entropy per animal
+-- transition_matrices/
|   +-- wild_transition_matrix.csv   # Mean transition matrix
|   +-- lab_transition_matrix.csv
+-- figures/
|   +-- syllable_proportions.png     # Stacked bar chart
|   +-- diversity_boxplot.png        # Shannon entropy box + strip plot
|   +-- transition_heatmaps.png      # Side-by-side heatmaps
|   +-- animal_scatter.png           # PCA of proportion vectors
+-- statistical_tests.json           # All test results
+-- repertoire_report.md             # Plain-language summary
```

## CLI Usage

```bash
# Standard analysis (data already has population + animal_id columns)
python scripts/analyze_repertoire.py \
    --classified-data classified_detections.csv \
    --output-dir analysis/repertoire

# Join metadata first (e.g. to add population labels)
python scripts/analyze_repertoire.py \
    --classified-data classified_detections.csv \
    --metadata animal_metadata.csv \
    --output-dir analysis/repertoire

# Custom column names, fewer permutations
python scripts/analyze_repertoire.py \
    --classified-data classified_detections.csv \
    --population-column group \
    --syllable-column type \
    --n-permutations 500

# Dry run
python scripts/analyze_repertoire.py \
    --classified-data classified_detections.csv \
    --dry-run -v
```

## Key Decisions

- **No skbio dependency:** PERMANOVA implemented from scratch using `scipy.spatial.distance.pdist(metric="braycurtis")`. The `skbio` package has C extension dependencies that fail on Windows. The algorithm (Anderson 2001) is straightforward: pseudo-F from squared Bray-Curtis distances + permutation test.
- **JSD = `jensenshannon()²`:** scipy returns the Jensen-Shannon *distance* (square root of divergence). We square it to get the actual divergence bounded in [0, 1] with base-2 logarithm.
- **`all_labels` parameter on `transition_matrix`:** Ensures consistent K x K matrix dimensions across all animals, even when some animals don't use all syllable types. Essential for meaningful matrix averaging.
- **Metadata join in CLI, not library:** Library functions expect columns to exist in the DataFrame. The CLI script handles the optional metadata CSV join, keeping library code pure and testable.
- **Permutation p-value formula:** `(count_ge + 1) / (n_perm + 1)` following standard convention — the +1 accounts for the observed statistic itself, preventing p=0.

## Integration Points

- **Reads from:** Classified CSV from Phase 14.2 (`import_deepsqueak_results`)
- **Dependencies:** `numpy`, `pandas`, `scipy` (distance, stats), `sklearn` (PCA), `matplotlib` (Agg backend)
- **No new packages** — all dependencies already in the environment
