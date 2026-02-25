# Phase 14.3: Syllable Repertoire Statistical Analysis -- Handoff

**Date:** 2026-02-25
**Review Tier:** 2 (new module + script + tests, no DSP/ML changes)
**Status:** Implementation complete, review warnings fixed, ready for independent review

## What Was Built

Statistical machinery for answering the core research question: do wild mice vocalize differently than lab mice? Takes classified detection CSV (from Phase 14.2) and runs per-animal syllable proportions, Shannon entropy, transition matrices, three population comparison tests (PERMANOVA, JSD, chi-squared), transition structure comparison, four publication-ready figures, and a plain-language report.

## Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `src/usv_spectrogram/classification/repertoire_stats.py` | ~700 | Core module: config, proportions, entropy, transitions, PERMANOVA, JSD, chi-squared, Frobenius norm, plots, report, orchestrator |
| `scripts/analyze_repertoire.py` | ~185 | CLI entry point with `--metadata` join, `--dry-run`, `-v` |
| `tests/test_classification/test_repertoire_stats.py` | ~560 | 33 tests across 13 test classes |
| `docs/modules/repertoire-stats.md` | ~130 | Module documentation |
| `docs/reviews/repertoire-stats-handoff.md` | -- | This file |

## Files Modified

| File | Change |
|------|--------|
| `src/usv_spectrogram/classification/__init__.py` | Added `RepertoireConfig` and 8 public functions to imports and `__all__` |
| `IMPLEMENTATION_PROGRESS.md` | Added dated Phase 14.3 entry |

## Architecture Decisions

### No skbio — PERMANOVA from scratch with scipy
The `skbio` package provides PERMANOVA but has heavy C dependencies that fail on Windows. The Anderson 2001 algorithm is straightforward: pseudo-F from squared Bray-Curtis distances + permutation test. Implemented in ~40 lines using `scipy.spatial.distance.pdist(metric="braycurtis")`.

### JSD = jensenshannon()² (divergence, not distance)
`scipy.spatial.distance.jensenshannon()` returns the Jensen-Shannon *distance* (square root of divergence). We square the result to get the actual divergence, bounded in [0, 1] with base-2 logarithm. This matches the information-theoretic definition.

### `all_labels` parameter on `transition_matrix`
Ensures consistent K x K matrix dimensions across all animals, even when some animals don't use all syllable types. Without this, matrix averaging would fail on mismatched dimensions.

### Metadata join in CLI, not library
Library functions expect columns (`population`, `animal_id`, `label`) to exist in the DataFrame. The CLI script handles the optional metadata CSV join via `--metadata` and `--metadata-key`. This keeps library code pure and testable with synthetic data.

### Explicit 2-population validation (post-review fix)
JSD and transition comparison require exactly 2 populations (they compare a pair of distributions). Instead of silently ignoring extra populations, both functions raise `ValueError` if `len(unique_populations) != 2`. PERMANOVA and chi-squared handle N>=2 populations correctly.

### Permutation p-value convention
`p = (count_ge + 1) / (n_perm + 1)` — the +1 accounts for the observed statistic itself, preventing p=0 and following standard permutation test convention.

## Public API

```python
from usv_spectrogram.classification import (
    RepertoireConfig,              # frozen dataclass: paths + column names + permutation settings
    syllable_proportions,          # per-group syllable proportions (sum to 1.0)
    syllable_diversity,            # Shannon entropy H per group (bits)
    transition_matrix,             # row-stochastic K x K for one animal
    compare_repertoires,           # statistical comparison dispatcher (3 methods)
    compare_transition_matrices,   # Frobenius norm + permutation test
    plot_repertoire_comparison,    # 4 publication-ready figures
    generate_report,               # markdown report with interpretation
    analyze_repertoire,            # full pipeline orchestrator
)
```

## What I'm Unsure About

- **Chi-squared pseudoreplication**: The chi-squared test pools all calls across animals, treating each call as independent. With 5 animals x 100 calls, it sees N=500 per group when the true independent sampling unit is the animal (N=5). This inflates significance. PERMANOVA is the correct primary test; chi-squared is supplementary. The limitation is documented but not programmatically enforced.
- **PERMANOVA with N=1 per group**: Produces R²=1.0 and pseudo-F=0.0 — technically not an error but the results are meaningless. A logging warning could be added but isn't currently.
- **Column name defaults**: The defaults (`population`, `animal_id`, `label`, `begin_time_s`) match the Phase 14.2 output schema. If DeepSqueak column names differ in practice, the CLI `--syllable-column` flag handles remapping.

## ROADMAP Exit Criteria Status

- [x] All statistical methods run without error on synthetic classified data
- [x] Visualizations are publication-quality (labeled axes, legend, appropriate colors)
- [x] `repertoire_report.md` provides plain-language interpretation
- [x] On synthetic data with known differences: methods correctly detect them (p < 0.05)
- [x] On synthetic data with no differences: methods correctly fail to reject (p > 0.05)
- [x] All tests pass (33/33)
- [x] py_compile passes on all new files (3/3)

## Test Coverage

| Category | Count | What's tested |
|----------|-------|---------------|
| Config validation | 4 | defaults, string->Path, zero permutations, negative permutations |
| Column validation | 2 | missing columns raises, all present passes |
| Syllable proportions | 2 | sum to 1.0 per animal, proportions match counts |
| Shannon entropy | 3 | H=0 for single type, H=log2(K) for uniform, monotonicity |
| Transition matrix | 4 | row-stochastic, zero rows for unseen, all_labels dims, deterministic cycle |
| PERMANOVA | 2 | identical -> p > 0.05, different -> p < 0.05 |
| JSD | 2 | near zero for identical, positive for different |
| Chi-squared | 2 | detects differences, not significant for identical |
| Transition comparison | 2 | cyclic vs random detected, identical not significant |
| Edge cases | 2 | single animal per pop, single syllable type |
| Input validation | 5 | missing population col, missing syllable col, unknown method, 3-pop JSD, 3-pop transitions |
| Plots | 1 | all 4 figure files created with nonzero size |
| Report | 1 | markdown generated with expected content |
| Integration | 1 | end-to-end pipeline on synthetic CSV |

## Dependencies

- `numpy`, `pandas` (already in project) — data manipulation
- `scipy.spatial.distance` (already in project) — Bray-Curtis, Jensen-Shannon
- `scipy.stats` (already in project) — chi2_contingency, mannwhitneyu
- `sklearn.decomposition.PCA` (already in project) — scatter plot dimensionality reduction
- `matplotlib` (already in project) — visualization (Agg backend)
- **No new packages required**

## Statistical Method Reference

| Method | Reference | Implementation detail |
|--------|-----------|----------------------|
| PERMANOVA | Anderson 2001 | pseudo-F on squared Bray-Curtis, permutation p-value |
| Shannon entropy | Shannon 1948 | H = -sum(p_i * log2(p_i)), units = bits |
| JSD | Lin 1991 | = 0.5*KL(P\|\|M) + 0.5*KL(Q\|\|M), M=(P+Q)/2 |
| Cramer's V | Cramer 1946 | sqrt(chi2 / (n * min(r-1, c-1))) |
| Frobenius norm | matrix analysis | ||M_wild - M_lab||_F with permutation test |

## Docs Written/Updated

- `docs/modules/repertoire-stats.md` — created
- `docs/reviews/repertoire-stats-handoff.md` — created (this file)
- `src/usv_spectrogram/classification/__init__.py` — updated
- `IMPLEMENTATION_PROGRESS.md` — updated

## What's Next

This module enables:
1. Run the full analysis on real classified data once DeepSqueak clustering is complete (Phase 14.2 MATLAB step)
2. Phase 14 Gate: all sub-phases (14.1 Raven export, 14.2 DeepSqueak import, 14.3 repertoire stats) are now complete
3. Integrate with LMT behavioral data for USV-behavior correlation analysis
