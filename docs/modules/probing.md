# Probing Framework

**Phase:** Hidden state interpretability (extends Phase 8.4 analysis suite)
**ADRs:** None (operates on pre-extracted hidden states and labels)
**Tests:** `usv_language/tests/test_probing.py` -- 16 tests (14 functional + 2 visualization smoke tests)
**Dependencies:** NumPy, scikit-learn, matplotlib; seaborn optional (fallback to matplotlib imshow)
**Review:** `docs/reviews/probing-review.md` (Tier 2)

## Purpose

Determine **which transformer layer encodes which acoustic property** using cross-validated linear and MLP probes. The resulting layers x properties heatmap directly answers which layer the VQ-VAE should operate on -- the layer with richest acoustic encoding is the best candidate.

Technique reference: Belinkov (2022), "Probing Classifiers: Promises, Shortcomings, and Advances."

## Data Flow

```
SpectrogramTransformer  -->  hidden_states[layer]  (N, d_model)
acoustic_properties.py  -->  labels[property]      (N,)
ProbingExperiment       -->  cross-validated score
ProbingAnalysisPipeline -->  (layers x properties) heatmap
```

## Public Interface

### ProbingConfig (frozen dataclass)

| Field | Default | Description |
|-------|---------|-------------|
| `probe_types` | `("linear", "mlp")` | Probe architectures to evaluate |
| `mlp_hidden_size` | 64 | MLP hidden layer width |
| `n_folds` | 5 | Cross-validation folds (>= 2) |
| `max_samples` | 0 | Subsample cap (0 = use all) |
| `random_seed` | 42 | Seed for reproducibility |
| `regression_properties` | 5 properties | peak_frequency, spectral_centroid, energy, bout_position, time_since_last_usv |
| `classification_properties` | 2 properties | is_voiced, frequency_direction |
| `ridge_alpha` | 1.0 | Ridge regularization (> 0) |
| `logistic_max_iter` | 1000 | LogisticRegression max iterations |
| `mlp_max_iter` | 500 | MLP probe max iterations |

### ProbingResult (frozen dataclass)

Per-experiment result: `property_name`, `layer`, `probe_type`, `task_type`, `score` (mean CV), `score_std`, `n_samples`, `fold_scores`.

### ProbingAnalysisResult (mutable dataclass)

Aggregated results. Computed properties:

| Method/Property | Returns | Description |
|----------------|---------|-------------|
| `heatmap_data(probe_type)` | `ndarray (n_layers, n_properties)` | Score matrix, clamped to [0, 1] |
| `best_layer_by_property` | `dict[str, (int, float)]` | Best layer per property |
| `best_overall_layer` | `int` | Layer with highest mean score |
| `summary` | `str` | Human-readable summary |

### ProbingExperiment

Single-experiment runner. `run(X, y, property_name, layer, probe_type, task_type)` returns `ProbingResult`.

### ProbingAnalysisPipeline

Orchestrator. `run(hidden_states, properties)` returns `ProbingAnalysisResult`.

### Visualization Functions

| Function | Description |
|----------|-------------|
| `plot_probing_heatmap(analysis, probe_type, ...)` | Layers x properties heatmap (seaborn or matplotlib fallback) |
| `plot_layer_comparison(analysis, ...)` | Line plot of mean score per layer with error bars |

## Key Design Decisions

1. **StandardScaler inside Pipeline** -- prevents data leakage. The scaler fits only on training folds, never on test data.

2. **Frame-level probing** -- no temporal pooling. Each spectrogram frame is an independent data point. This gives maximum resolution but means the probe fits N = T * (number of bouts) data points.

3. **Sentinel filter for time_since_last_usv** -- values of -1.0 (no preceding onset) are filtered before regression. No other acoustic property can produce -1.0.

4. **R^2 clamping in heatmap** -- negative R^2 (probe worse than mean prediction) is clamped to 0 for display. Raw scores preserved in ProbingResult.

5. **ConvergenceWarning suppression** -- MLP probes may not converge within max_iter on small datasets. Warnings are locally suppressed during cross_val_score.

## CLI Script

`usv_language/scripts/run_probing.py` -- loads transformer checkpoint, extracts hidden states from all layers, extracts acoustic properties, runs probing analysis. Outputs: `results.json`, `probing_report.md`, `probing_heatmap_{type}.png`, `layer_comparison.png`.

Spectrogram orientation is determined by matching one dimension to `model.config.n_freq` (not by fragile size heuristic).
