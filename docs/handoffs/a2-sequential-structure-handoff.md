# Handoff: Phase A2 — Sequential Structure Analysis

**Date:** 2026-04-05
**Prerequisite:** A1 (Temporal Dynamics) complete. Results in `results/temporal_dynamics/`.

---

## Context

We're analyzing USV (ultrasonic vocalization) data from a single mouse (cage 5970, usv_lmt_034). Detection pipeline and classification are complete — 7,864 classified calls across a ~32-hour continuous recording (Sep 30 11:18 → Oct 1 19:42, 2024).

### Key finding from A1
- **USV1–USV5 folders are download batches, NOT temporal sessions.** Ignore folder structure for temporal analysis. The real timeline comes from filename timestamps (`YYYY-MM-DD_HH-MM-SS_XXXXXXX`) + `begin_time_s` offset.
- Calling is **bursty** — 1,579 bouts detected (median 4 calls/bout, 0.6s threshold), bimodal ICI distribution.
- **Down calls are overrepresented as bout starters** (~28% bout-initial vs ~17% overall), suggesting sequential structure exists.
- Type composition is stable during active periods — the repertoire doesn't shift over hours.

### Data file
**`results/traditional_taxonomy/classified_traditional.csv`** — 7,921 rows (57 have NaN `file`, drop those → 7,864 usable).

Key columns:
- `file` / `wav_stem`: filename with timestamp (e.g., `2024-09-30_11-18-17_0000001`)
- `begin_time_s`: call onset within the WAV file (float seconds)
- `syllable_type`: one of `Flat`, `Down`, `Chevron`, `Short`, `Complex`, `Frequency_Jump`, `Up`
- `classification_confidence`: `high`, `medium`, `low`

To get absolute time: parse datetime from filename + add `begin_time_s`.

---

## Task: A2 Sequential Structure

**Question:** Are call sequences random, or do certain types follow each other more than chance predicts?

### Analyses to run

| Analysis | Function (already implemented) | Location |
|----------|-------------------------------|----------|
| Transition matrix P(B\|A) | `transition_matrix()` | `src/usv_spectrogram/classification/repertoire_stats.py:197` |
| Compare transitions (permutation test) | `compare_transition_matrices()` | `repertoire_stats.py:594` |
| Entropy rate (bigram/trigram) | `entropy_rate()` | `usv_language/analysis/sequence_analysis.py:152` |
| Conditional entropy H(C_{t+1}\|C_t) | `conditional_entropy()` | `sequence_analysis.py:391` |
| Mutual information at lag | `mutual_information_at_lag()` | `sequence_analysis.py:105` |
| Idiom detection (shuffle surrogates) | `detect_idioms()` | `usv_language/analysis/information_theory.py` |
| Zipf distribution fit | `zipf_mle()` | `information_theory.py` |
| Burstiness by context | `burstiness_by_context()` | `information_theory.py:1051` |

### Expected outputs (save to `results/sequential_structure/`)

1. **Transition matrix heatmap** — 7×7 P(type_B | type_A), row-stochastic
2. **Entropy convergence plot** — entropy rate vs n-gram order (does it plateau?)
3. **Mutual information at lag** — MI(T, T+k) for k=1..10 (how far does "memory" extend?)
4. **Zipf plot** — rank-frequency of syllable types with MLE fit
5. **Idiom report** — recurring n-grams above chance (shuffle surrogate p-values)
6. **Summary CSV** — key metrics (entropy rate, Zipf exponent, top idioms, etc.)

### Notes

- The existing functions in `repertoire_stats.py` and `information_theory.py` / `sequence_analysis.py` do the heavy lifting. The task is mostly **wiring them together** with the classified CSV and generating visualizations.
- `repertoire_stats.py` is 1,142 lines — read in chunks (Large File Protocol).
- `information_theory.py` is 1,075 lines — read in chunks.
- Sort calls by absolute time before extracting sequences. Within-bout vs across-bout sequences may show different structure — consider analyzing both.
- The A1 script (`scripts/analyze_temporal_dynamics.py`) shows how to parse timestamps and compute absolute time — reuse that pattern.

### Roadmap reference
Full analysis plan: `docs/analysis-roadmap.md` §2 (Sequential Structure).
