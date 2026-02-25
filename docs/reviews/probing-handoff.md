# Implementation Handoff: Probing Framework

**Module:** `usv_language/analysis/probing.py`
**Review Tier:** 2 (new module with ML correctness concerns — data leakage prevention, proper CV)
**Date:** 2026-02-24
**Branch:** main (uncommitted)

## What Changed

- Built a cross-validated probing framework that tests whether acoustic properties are linearly accessible from each transformer layer's hidden states
- Linear probes (Ridge/LogisticRegression) and MLP probes with StandardScaler inside sklearn Pipeline to prevent data leakage
- Frame-level probing (no temporal pooling) — each spectrogram frame is an independent data point
- Produces a layers x properties heatmap that directly answers which layer the VQ-VAE should operate on
- CLI script for end-to-end analysis from transformer checkpoint to output plots

## Files Changed

| File | Action | Lines | Description |
|------|--------|-------|-------------|
| `usv_language/analysis/probing.py` | NEW | ~470 | Core module: ProbingConfig, ProbingResult, ProbingAnalysisResult, ProbingExperiment, ProbingAnalysisPipeline, plot_probing_heatmap, plot_layer_comparison |
| `usv_language/tests/test_probing.py` | NEW | ~500 | 18 tests (16 functional + 2 visualization smoke tests) |
| `usv_language/scripts/run_probing.py` | NEW | ~250 | CLI entry point following run_null_model_analysis.py pattern |
| `usv_language/analysis/__init__.py` | MODIFIED | +10 | Added 7 exports (5 classes + 2 viz functions) and docstring line |
| `docs/modules/probing.md` | NEW | ~80 | Module documentation |
| `IMPLEMENTATION_PROGRESS.md` | MODIFIED | +16 | New "Probing Framework" entry |

## Architecture

### Data Structures (3 dataclasses)

| Class | Type | Purpose |
|-------|------|---------|
| `ProbingConfig` | frozen | 11 fields: probe types, CV folds, alpha, max_iter, property lists |
| `ProbingResult` | frozen | Single experiment: score, std, fold_scores, n_samples |
| `ProbingAnalysisResult` | mutable | Aggregated results with computed properties (heatmap_data, best_layer, summary) |

### Core Classes (2)

| Class | Purpose |
|-------|---------|
| `ProbingExperiment` | Single experiment runner: filter labels, encode, build sklearn Pipeline, cross-validate |
| `ProbingAnalysisPipeline` | Orchestration: iterate (layer x property x probe_type), collect ProbingResults |

### Visualization Functions (2)

| Function | Output |
|----------|--------|
| `plot_probing_heatmap(analysis, probe_type)` | Annotated heatmap (seaborn or matplotlib imshow fallback) |
| `plot_layer_comparison(analysis)` | Line plot with error bars, mean score per layer |

## Key Decisions Made

1. **StandardScaler inside sklearn Pipeline** (probing.py:436-439) — The scaler is wrapped in a Pipeline with the probe estimator so that `cross_val_score` only fits the scaler on training folds. This is the #1 correctness requirement. If someone refactors the scaler outside the pipeline, scores would be inflated by data leakage.

2. **Frame-level probing, no temporal pooling** (probing.py:463-471 docstring) — Each spectrogram frame is an independent data point. Maximizes statistical power but assumes temporal independence. For screening which layer to use, this is fine. For publishing, bout-level pooling would be needed.

3. **Sentinel filter for time_since_last_usv == -1.0** (probing.py:378-390) — Values of -1.0 mean "no preceding onset" and must be excluded from regression. The filter also removes NaN and inf. For classification labels (string/bool dtype), the filter is bypassed.

4. **R^2 clamping in heatmap display** (probing.py:185) — `max(0.0, r.score)` clamps negative R^2 to 0 for the heatmap. Raw negative scores are preserved in ProbingResult.score.

5. **Seaborn optional** (probing.py:549-565) — `try/except ImportError` with matplotlib `imshow` fallback. Seaborn is not currently installed in the venv.

6. **Spectrogram orientation via model.config.n_freq** (run_probing.py:175-187) — The CLI script matches one spectrogram dimension to the transformer's known n_freq config rather than using a fragile shape heuristic. Correctly handles short bouts where T < n_freq.

## What I'm Unsure About

1. **Probe selectivity not implemented** — Raw accuracy is reported for classification, not accuracy-minus-majority-baseline. For imbalanced data (e.g., is_voiced with 85/15 split), a probe predicting the majority class gets high accuracy without encoding anything. Deferred to first use on real recordings.

2. ~~**ConvergenceWarning suppression**~~ — **RESOLVED.** Removed the dead message-based filter (`".*ConvergenceWarning.*"`). Only the class-based filter remains.

3. **ProbingAnalysisResult is mutable** (probing.py:154) — Unlike ProbingResult (frozen), the analysis result container is not frozen because computed properties depend on the full result set. This matches the precedent of `FullAnalysisResult` in `statistical_tests.py`.

4. **Empty-after-filtering edge case** (probing.py:300-310) — When all labels are filtered out or fewer valid samples than n_folds remain, the experiment returns score=0.0 with empty fold_scores. Covered by `test_too_few_samples_graceful_return`.

5. ~~**No MLP probe end-to-end test**~~ — **RESOLVED.** Added `test_mlp_probe_regression`: MLP on perfectly linear data achieves R² > 0.8.

6. ~~**Three-class classification untested**~~ — **RESOLVED.** Added `test_three_class_classification`: StratifiedKFold with 3 imbalanced classes (45/45/10 split) works correctly, accuracy > 0.8.

## Test Results

```
pytest usv_language/tests/test_probing.py -v
18 passed in 5.00s

pytest usv_language/tests/ -v
272 passed, 1 skipped in ~16s
```

The 1 skipped test is pre-existing (HMM requires hmmlearn).

### Test Coverage Details

| # | Test | Assertion |
|---|------|-----------|
| 1 | Perfect linear encoding | R^2 > 0.95 |
| 2 | Random noise encoding | R^2 < 0.1 |
| 3 | Deeper layers = stronger signal | Scores monotonically increase (d=8, n=500, noise [2.0, 1.0, 0.5, 0.1]) |
| 4 | Perfectly separable classification | accuracy > 0.95 |
| 5 | Random classification labels | accuracy in [0.35, 0.65] |
| 6 | CV fold count | len(fold_scores) == n_folds |
| 7 | Heatmap shape | (n_layers, n_properties) |
| 8 | best_overall_layer correctness | Selects highest-scoring layer |
| 9 | Config: invalid probe_type | ValueError |
| 10 | Config: n_folds < 2 | ValueError |
| 11 | Pipeline with missing properties | Warns, runs on available subset |
| 12 | Subsampling respects max_samples | result.n_samples <= max_samples |
| 13 | Sentinel filter | -1.0 values excluded before regression (n_samples verified) |
| 14 | Too few samples | Graceful zero return when n_samples < n_folds |
| 15 | MLP probe regression | MLP on perfect linear data: R² > 0.8 |
| 16 | Three-class classification | StratifiedKFold with 3 imbalanced classes: accuracy > 0.8 |
| + | plot_probing_heatmap smoke test | Figure file created without crash |
| + | plot_layer_comparison smoke test | Figure file created without crash |

## ROADMAP Exit Criteria Status

This module extends Phase 8.4's analysis suite (not a named ROADMAP phase). Applying Phase 8.4's exit criteria:

- [x] All visualizations generate without errors on synthetic data
- [x] All tests pass
- [x] py_compile passes on all new files
- [x] Module documentation created
- [x] Implementation progress updated

## Docs Written/Updated

- `docs/modules/probing.md` — created (module doc)
- `docs/reviews/probing-handoff.md` — this file
- `IMPLEMENTATION_PROGRESS.md` — updated with probing framework entry
- `DECISIONS.md` — no new ADRs needed (no DSP parameters, no training splits)
- `docs/architecture/patterns.md` — no new patterns introduced

## Dependencies

All already in `usv_language/requirements.txt`:
- numpy, scikit-learn (Ridge, LogisticRegression, MLPRegressor, MLPClassifier, cross_val_score, Pipeline, StandardScaler, KFold, StratifiedKFold)
- matplotlib (visualization)
- seaborn (optional, graceful fallback)

## Upstream Dependencies

- `usv_language/analysis/acoustic_properties.py` — `extract_all_properties()` provides probing labels
- `usv_language/models/transformer.py` — `SpectrogramTransformer.forward(return_hidden_states=True)` for CLI

## Risks

- **Data leakage if Pipeline is refactored** — If someone moves StandardScaler outside the Pipeline, scores silently inflate. Documented in module doc.
- **Frame-level independence assumption** — Adjacent frames are correlated. Probing scores may be optimistic vs bout-level analysis. Documented in docstring and module doc.
- **No DSP parameter changes** — This module consumes hidden states, doesn't modify STFT/FFT parameters
