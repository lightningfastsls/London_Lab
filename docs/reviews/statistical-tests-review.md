# Statistical Comparison Framework -- Review

**Reviewed by:** Master Reviewer (claude-sonnet-4-6)
**Date:** 2026-02-24
**Module:** `usv_language/analysis/statistical_tests.py` + `usv_language/scripts/run_null_model_analysis.py`
**Tier:** 2 (Standard -- cross-domain statistics + information theory)
**Verdict:** APPROVED (all blockers and warnings addressed)

## Test Run

```
usv_language/tests/test_statistical_tests.py: 14 passed in 17.05s
Full usv_language suite: 232 passed, 1 skipped, 0 failures (0 regressions)
py_compile: all 4 files pass
```

## Findings and Fixes

### Blockers (both fixed)

| ID | Issue | Fix |
|----|-------|-----|
| B1 | Missing module doc and handoff | Created `docs/modules/statistical-tests.md` and `docs/reviews/statistical-tests-handoff.md` |
| B2 | IMPLEMENTATION_PROGRESS.md not updated | Added dated entry |

### Warnings (all fixed)

| ID | Issue | Fix |
|----|-------|-----|
| W1 | mi_decay_half_life degenerate at large K | Added diagnostic note in `_build_summary` when all comparisons have null_std=0 for this metric |
| W2 | CLI histograms use Normal approximation without caveat | Added `ax.text()` annotation on each histogram: "Approx: Normal(mean, std) / (raw null values not stored)" |
| W3 | `__init__.py` docstring missing `statistical_tests` | Added line to module listing |
| W4 | No warning when n_surrogates=0 in comparison | Added `warnings.warn()` in `full_analysis()` when `comparison.n_surrogates == 0` |

### Suggestions applied

| ID | Issue | Fix |
|----|-------|-----|
| S1 | Alphabetical null model ordering | Added `_NULL_MODEL_ORDER` hierarchy list, used as sort key |
| S2 | "Cohen's d" misnomer | Updated docstring to "Standardized distance from null (analogous to Cohen's d)" |
| S3 | CLI missing sys.path guard + exit code | Added `if _REPO_ROOT not in sys.path` guard, changed `main() -> int`, added `sys.exit(main())` |

### Not applied (documented)

| ID | Issue | Rationale |
|----|-------|-----------|
| S4 | Test 3 uses soft z-score criterion | Acceptable: i.i.d. sequences can occasionally produce z-scores above 2 due to random fluctuation. The `abs(z) < 4.0` threshold is a pragmatic guard against flaky test failures while still verifying the metric is in a reasonable range. |

## What Passed (from original review)

- Statistical correctness (Bessel's correction, z-score, rank p-value)
- NaN propagation and edge cases
- Integration with information_theory.py and null_models.py function signatures
- to_markdown() and to_dataframe() output format
- CLI structure follows run_analysis.py pattern
- Frozen dataclass pattern
- Test anti-greenwashing compliance
- Reproducibility via seeded RNG
