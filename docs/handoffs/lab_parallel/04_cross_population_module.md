# Stream 4 — Build Cross-Population Comparison Module

**Status:** Ready to run
**Estimated time:** 3–4 hours (code + tests)
**Compute:** None (module + unit tests only)

## Goal

Build `src/usv_spectrogram/classification/cross_population.py` — a reusable module that takes two classified-detection CSVs (with cohort labels) and outputs a structured comparison report. This module IS the comparison machinery for the wild-vs-lab analysis. It must be ready before lab detection finishes so the comparison is one function call away.

## Why this is its own module (not a script)

Existing scripts (`analyze_acoustic_features.py`, `analyze_temporal_dynamics.py`) operate on a single dataset. Cross-population comparison is a new responsibility with reusable methods (chi-squared on type proportions, JSD between distributions, transition-matrix differences) that the lab analysis, between-individual analyses (5970 vs 3452 vs 9252), and any future cohort comparison will all use. Putting it in `src/` makes it importable and testable.

## API

```python
from usv_spectrogram.classification.cross_population import (
    CrossPopulationComparison,
    ComparisonReport,
)

comparison = CrossPopulationComparison(
    pop_a_csv="results/traditional_taxonomy/classified_traditional.csv",
    pop_a_label="wild_5970",
    pop_b_csv="results/traditional_taxonomy_lab_131204/classified_traditional.csv",
    pop_b_label="lab_131204",
    strata_note="wild-vs-lab-strain",       # required (schema 1.1+); see
                                            # feedback_cross_animal_population_strata.md
    bout_threshold_s=0.6,                   # canonical, from corpus_facts
    type_column="syllable_type",
    confidence_column="classification_confidence",
)

report = comparison.run_all()
report.write_json("results/cross_population/wild_5970_vs_lab_131204.json")
report.write_markdown("results/cross_population/wild_5970_vs_lab_131204.md")
report.write_figures("results/cross_population/wild_5970_vs_lab_131204/")
```

## Required metrics

Each implemented as a method returning a typed dataclass result. All include a p-value where applicable, and a bootstrap CI where applicable.

| Metric | Method | Notes |
|--------|--------|-------|
| Type proportion difference | `compare_type_proportions()` | Chi-squared + per-type Cohen's h effect size |
| Repertoire JSD | `compare_repertoires_jsd()` | Use `repertoire_stats.py` if it has JSD; otherwise add |
| Shannon entropy | `compare_entropy()` | Per-population entropy + bootstrap 95% CI + permutation p-value |
| Transition matrix difference | `compare_transitions()` | Bout-aware (uses bout_threshold_s); Frobenius distance + per-row JSD |
| MI at lag 1 | `compare_mi_lag1()` | Bout-aware; matches the canonical `scattoni_7_bout_aware` from corpus_facts |
| Zipf exponent | `compare_zipf()` | MLE from `information_theory.py` |
| Burstiness CV | `compare_burstiness()` | CV of inter-call intervals |
| IOI distribution | `compare_ioi_distributions()` | KS test + median IOI + IQR |
| Acoustic feature distributions | `compare_features()` | Per-feature: KS test, Cohen's d, plot violins side-by-side |
| Joint UMAP | `compare_umap_overlap()` | Fit UMAP on combined population, compute per-population density, output overlap coefficient |

## File layout

```
src/usv_spectrogram/classification/cross_population.py    <-- main module
tests/test_cross_population.py                            <-- unit tests (synthetic data)
```

## Implementation rules

1. **Reuse existing functions.** `repertoire_stats.py` has Shannon entropy, JSD, PERMANOVA. `information_theory.py` has Zipf MLE, MI at lag, burstiness CV, idiom detection. `corpus.py` has the canonical sample rate / freq band. Import; do not reimplement.
2. **Bout-aware methods only.** Any sequential metric (transitions, MI, burstiness within bouts) must use the bout segmentation from `analyze_sequential_structure.py` or the helper used by SIS 17.1. Never compute over file-spanning sequences (would inflate apparent randomness).
3. **Bootstrap CIs**: 1,000 resamples, percentile method. Make this a parameter (`n_bootstrap=1000`).
4. **Random seed**: accept `random_state` and pass through to all sklearn / UMAP / numpy random ops.
5. **No global state.** Module functions take CSVs / DataFrames as input, return dataclasses or write to explicit paths.
6. **Type hints throughout.** Use `dataclasses.dataclass` for results.
7. **Print parameters and row counts** when `run_all()` is called.

## Testing

`tests/test_cross_population.py` must include:

- **Synthetic-identical test**: generate two fake populations with identical type distributions; expect chi-squared p > 0.05, JSD ≈ 0, entropy difference ≈ 0
- **Synthetic-different test**: generate two populations with deliberately different type proportions; expect chi-squared p < 0.001
- **Empty / single-type edge case**: one population has only one type — methods should not crash, should return clear "single-type" sentinel
- **Bootstrap reproducibility**: same `random_state` → same CI bounds
- **Schema test**: JSON output validates against documented schema

Use `pytest tests/test_cross_population.py -v` to verify. All tests green before commit.

## Constraints

1. **No corpus constants redeclared.** Sample rate, freq band — import from `corpus.py`.
2. **Bout threshold parameterized**, default 0.6 s. Document in docstring that this matches `scattoni_7_bout_aware` in corpus_facts.
3. **Test pre-existing files (if any) are spec.** Don't modify tests written by `test-architect` if you find them.
4. **No emojis in code.**
5. **Default to no comments** — only when the why is non-obvious.

## Validation

Done when:
- [ ] `src/usv_spectrogram/classification/cross_population.py` exists with all 10 metric methods
- [ ] `tests/test_cross_population.py` exists, all tests pass
- [ ] `python -m py_compile src/usv_spectrogram/classification/cross_population.py` succeeds
- [ ] Smoke-test: instantiate with `(5970 traditional, 3452 traditional)` and run `run_all()` — produces non-empty report
- [ ] Module doc: 1-paragraph overview + API example committed (in module docstring, NOT a separate .md)
- [ ] Commit SHA recorded

## Decision-needed signals

- If `repertoire_stats.py` JSD differs from textbook formula — surface back, do not silently adopt
- If existing bout-detection helper is hard to reuse cleanly — propose extraction into `bout_detection.py` rather than duplicating
- If joint UMAP on (5970 + 3452 + 9252) takes >5 min — propose a subsample strategy

## Smoke test acceptance

```bash
.venv/bin/python -c "
from usv_spectrogram.classification.cross_population import CrossPopulationComparison
cmp = CrossPopulationComparison(
    pop_a_csv='results/traditional_taxonomy/classified_traditional.csv',
    pop_a_label='wild_5970',
    pop_b_csv='results/traditional_taxonomy_3452/classified_traditional.csv',
    pop_b_label='wild_3452',
    strata_note='wild-vs-wild between-couple',
    bout_threshold_s=0.6,
)
report = cmp.run_all()
print(report.summary())
"
```

This must run end-to-end without error and print non-trivial numbers.

## Result section

### Population strata — read before any number below

Both `wild_5970` and `wild_3452` are **wild-mouse couples** (male + female
pairs of wild-caught animals; the male is the vocalizer). N = **1 couple**
per cohort, NOT N = 1 animal. Same species in both cohorts. This smoke
test is therefore a **wild-vs-wild between-couple** comparison.

The actual research-goal axis is **wild-vs-lab-strain**, blocked pending
lab-strain data (see `docs/handoffs/HANDOFF_05_LAB_DATA_PIPELINE.md`).
Every divergence number reported below (JSD = 0.1377 bits, MI 0.092 vs
0.197 bits, max|h| = 0.657 on `Short`, etc.) is therefore the **noise floor**
that a future wild-vs-lab signal must exceed to count as a strain effect
rather than between-couple variability within the wild stratum. These are
baseline-estimation numbers, not headline findings.

Terminology choice: this memo uses "couple" (matching `project_wild_mice.md`
and Stream 1's `01_RESULTS_3452_vs_5970.md`). Project-wide vocabulary
("couple" vs "cohort" vs "dyad") is still open — the user has deferred
standardization. See `feedback_cross_animal_population_strata.md`.

---

- **Status:** DONE (2026-04-24)
- **Commit SHA:** `375d4bdc` — note: a parallel Stream 2 chat bulk-staged five streams' untracked files at 2026-04-24T23:48 and committed them under the misleading title `feat(9252-analysis): merge CSV + rate-anomaly investigation`. Stream 4 contributions (cross_population.py + tests + __init__.py + this handoff + JSON/MD outputs) are inside that commit. The four smoke-test PNGs are intentionally not versioned (`*.png` is gitignored project-wide, line 9) — they are regenerable from the committed JSON via `report.write_figures()`.
- **Tests passing:** 16/16 in 3.72s (`pytest tests/test_cross_population.py -v`)
- **Smoke test output (5970 vs 3452, skip umap_overlap, bootstrap=1000):**
  - N: A=7,864 calls / 1,338 files, B=401 calls / 110 files, K=7 types
  - Type proportions: chi²=381.90, p=2.17e-79, Cramer's V=0.215, max|h|=0.657 on `Short`
  - JSD = 0.1377 bits [95% CI 0.1121, 0.1714]
  - Shannon H: A=2.548, B=2.310, diff=+0.238, permutation p≈0.001
  - **MI lag-1 (bout-aware):** A=0.0916 bits, B=0.1966 bits (3452 has ~2× the sequential structure)
  - Transitions: Frobenius=0.9461 on bout-aware matrices (A n_within=6,305, B n_within=252)
  - Zipf: insufficient unique types (K=7 < 10 threshold) — sentinel 0.0 with caveat
  - Burstiness CV within bouts: A=0.526, B=0.551
  - IOI median (within-bout): A=159.4 ms, B=187.3 ms, KS p=3.82e-08
  - Features max|Cohen's d| = 0.862 on `mean_power_db`
  - Artifacts: `results/cross_population/wild_5970_vs_wild_3452.json`, `.md`, 4 PNGs
- **MI canary against `corpus_facts/5970.json`:** observed 0.0916 bits vs canonical `scattoni_7_bout_aware = 0.0921 bits` — difference 5e-4 bits, 6,305 vs 6,350 within-bout pairs. Methodologically aligned but not byte-identical due to per-file vs sorted-global bout segmentation (see Decision D1).
- **Decisions surfaced:**
  - **D1 (bout helper extraction):** chosen option (c) — reused `sequence_analysis.segment_into_bouts` directly, added a private `_bout_pairs_per_file` helper that groups by WAV file. Per-file grouping is stricter than the canonical sorted-global segmentation (explains the 45-pair / 5e-4 bit MI gap) and is arguably more correct for cross-population work where recording protocols may differ. Stream 5 can formalize the bout module once its threshold sensitivity conclusions land.
  - **D2 (per-animal partitioning not available):** chosen option (c) — each wild cohort is one couple (male + female; the male vocalizes), not one individual. We can't reliably split calls per animal from the audio, so animal-level PERMANOVA from `repertoire_stats` doesn't apply; the module operates on pooled per-cohort proportions. The lab cohort `131204` (6 couples × 2 timepoints → 12 sessions) is where between-couple inference becomes statistically possible. Wild side will always be limited until more couples are recorded.
  - **D3 (Cohen's h):** chosen as-written — Cohen's h = 2·(arcsin√p_a − arcsin√p_b) computed per type, returned as a dict plus `max_abs_cohens_h` summary. Largest effect on wild cohorts was `Short` with |h|=0.657 (medium-large effect).
- **Non-obvious findings worth a note:**
  1. 3452 shows substantially higher sequential structure than 5970 (MI 0.197 vs 0.092 bits, >2× difference) despite smaller sample. If real, this is a cross-animal signal worth adding to `data/corpus_facts/3452.json` once audit_corpus runs.
  2. The smoke test surfaces a methodology gap — per-file bouts vs sorted-global bouts give slightly different within-bout pair counts. Both are defensible; the question is which the project wants canonical. Flagged to Stream 5.
- **How to reproduce:** `PYTHONPATH=src .venv/bin/python -c "..."` (the handoff's smoke-test block needs `PYTHONPATH=src` prepended — the script pattern used by `scripts/analyze_sequential_structure.py` isn't in effect for ad-hoc `-c` invocations).
