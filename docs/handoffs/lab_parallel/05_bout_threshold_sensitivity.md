# Stream 5 — Bout Threshold Sensitivity + File-Aware Logic Prototype

**Status:** Ready to run
**Estimated time:** 2–3 hours
**Compute:** Light

## Goal

Resolve the bout threshold question (Q1 in `docs/questions-for-mickey.md`) at least quantitatively, so when lab data lands we have a defensible canonical value AND know how sensitive every downstream number is to that choice. Without this, every wild-vs-lab sequential structure number is challengeable on "but what if you'd used a different threshold?"

## Background (from `docs/questions-for-mickey.md` Q1)

- Current canonical: **0.6 s** (`bout_detection_a2.threshold_s` in `data/corpus_facts/5970.json`), giving MI lag 1 = 0.0921 bits.
- Mixture model on within-file ICIs only: dominant peak at 74 ms (78% weight), tail at 184 ms (22%), crossover at **0.143 s**.
- Original 0.6 s came from `3 × median(onset-to-onset ICI)` over ALL ICIs — wrong derivation, but file boundaries silently did most of the work.
- **Open question for Mickey**: should file boundaries always be bout breaks? (Recorder stops after ~2 s silence — 90 cross-file gaps under 1 s, others much longer.)

## Steps

### 1. Sensitivity sweep — current logic (no file-awareness)

Recompute on 5970 across thresholds: `[0.10, 0.143, 0.20, 0.25, 0.40, 0.60, 0.80, 1.00, 2.00]` seconds.

For each threshold, output:

| metric | source |
|--------|--------|
| n_bouts | `analyze_sequential_structure.py` bout detection |
| n_within_bout_pairs | same |
| n_cross_bout_pairs | same |
| MI_lag1_bits (bout-aware) | `mutual_information_within_bouts` |
| Marginal entropy | same |
| Conditional entropy | same |
| Mean bout duration | computed |
| Mean calls per bout | computed |

Output: `results/bout_threshold_sensitivity/sweep_no_file_aware.csv` + `sweep_no_file_aware.png` (MI vs threshold + n_bouts vs threshold).

### 2. Sensitivity sweep — file-aware logic

Implement the proposed two-layer logic:

```
Bout boundary if:  (different file)  OR  (same file AND gap > threshold)
```

Same threshold sweep, plus an additional row for `threshold = ∞` (file = bout, no within-file splitting).

Output: `results/bout_threshold_sensitivity/sweep_file_aware.csv` + `sweep_file_aware.png`.

### 3. Direct comparison

Plot both sweeps overlaid: MI vs threshold under each logic. Annotate the canonical 0.6 s point and the 0.143 s mixture-crossover point.

Output: `results/bout_threshold_sensitivity/comparison.png`.

Compute the difference: at the same threshold, how much do `n_bouts`, `MI_lag1`, etc. change between logics? This is the "magnitude of the file-boundary decision."

### 4. Within-file gap distribution diagnostic

Reproduce the within-file ICI mixture model diagnostic:

- Histogram of within-file gaps with the 2-component Gaussian mixture overlaid
- Histogram of cross-file gaps
- KS test: are within-file and cross-file gap distributions distinguishable?

Output: `results/bout_threshold_sensitivity/gap_distributions.png`.

### 5. Cross-dataset application

Repeat steps 1–2 for 3452 (using `results/traditional_taxonomy_3452/classified_traditional.csv`). Add 9252 if Stream 2 has finished by then.

This answers: "is the optimal threshold dataset-dependent?" If yes, the canonical needs per-dataset configuration; if no, one value works.

Outputs: `sweep_no_file_aware_3452.csv`, `sweep_file_aware_3452.csv`, etc. Mirror naming.

### 6. Recommendation memo

Write `docs/handoffs/lab_parallel/05_RESULTS_bout_threshold.md` with:

- The sweep findings (key plot embedded as path reference)
- File-aware vs non: which is more theoretically defensible (file = recorder-imposed silence ≥ 2 s)?
- Recommended canonical threshold + recommended logic, with rationale
- Sensitivity bounds: "MI lag 1 ranges from X to Y across reasonable thresholds — wild-vs-lab differences must exceed this band to be meaningful"
- A "questions-for-Mickey resolution" section: based on the data, which of Q1's three definition options (a/b/c) is the data most consistent with?

## Constraints

1. **DO NOT change the canonical** in `data/corpus_facts/5970.json` in this handoff. This is sensitivity analysis only. The canonical is changed via a separate decision after Mickey signs off.
2. **Use existing code where possible** — `analyze_sequential_structure.py`, `information_theory.py` `mutual_information_within_bouts`, the bout detection helper. Do not reimplement.
3. **Print parameters and counts.**
4. **Sample rate / freq band** — import from `corpus.py`.
5. **Bootstrap CIs** for MI estimates (1,000 resamples) — small N for some bouts.

## Validation

Done when:
- [ ] `results/bout_threshold_sensitivity/` populated with all 6 plot/CSV outputs
- [ ] Recommendation memo exists with explicit threshold + logic recommendation
- [ ] Memo includes per-dataset comparison (5970, 3452, optionally 9252)
- [ ] Sensitivity bound ("MI varies by ±X bits across reasonable choices") quantified
- [ ] Commit SHA recorded
- [ ] No changes to canonical files (corpus_facts unchanged, source modules unchanged)

## Decision-needed signals

- If file-aware vs non-file-aware logic gives qualitatively different MI rankings (e.g., one logic says lag-1 MI is significant, the other says not) — surface immediately
- If 3452 and 5970 prefer different optimal thresholds — implies the canonical needs to be per-dataset, not global; surface for design discussion
- If the within-file gap distribution turns out NOT to be bimodal (mixture model fits one component) — the bout concept itself is questionable

## Result section

- Commit SHA:
- Recommended threshold:
- Recommended logic (file-aware Y/N):
- MI sensitivity range across reasonable thresholds:
- Per-dataset agreement:
- Open Q for Mickey (refined version):
