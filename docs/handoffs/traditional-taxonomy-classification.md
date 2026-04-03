# Handoff: Rule-Based Traditional Taxonomy Classification
Date: 2026-04-03
From: Claude Code
To: Claude Code (plan mode)

## Task

Classify the 7,518 USV calls into traditional syllable types (Holy & Guo 2005 / Scattoni taxonomy) using deterministic rules on the acoustic features already in `classified_detections_full.csv`. This provides literature-comparable labels without any training data or ML — pure feature thresholds.

**Input:** `classified_detections_full.csv` — 7,518 matched USV calls with 18 acoustic features from DeepSqueak.

**Acceptance criteria:**
1. A script that reads the classified CSV, applies rule-based classification, and outputs a new CSV with traditional syllable type labels
2. Rules documented with citations to Holy & Guo (2005) and Scattoni et al. (2008) definitions
3. Output CSV preserves all original columns, adds: `syllable_type` (categorical label), `classification_confidence` (rule match quality)
4. Summary statistics: count and proportion per syllable type, with bar chart
5. Per-type acoustic feature summary (mean ± std for duration, frequency, slope, sinuosity)
6. Gallery PNGs organized by syllable type (reuse `scripts/generate_cluster_gallery.py` pattern)
7. Cross-tabulation: DeepSqueak k-means cluster vs traditional type (to see how the 27 clusters map)

## Files to Modify

- **NEW** `scripts/classify_traditional_taxonomy.py` — Rule-based classification script
- **NEW** `results/traditional_taxonomy/` — Output directory for CSV, figures, summary

## Relevant Constraints (from vault)

1. **Classifiers trained on lab mice generalize poorly to wild mice.** BootSnap showed F1 drops significantly across populations. Rule-based classification avoids this problem entirely — rules are population-agnostic.
   Source: `classifiers trained on lab mice generalize poorly to wild mice requiring population-specific training data`
   Verified: 2026-04-03

2. **Traditional taxonomy may obscure continuous variation.** Holy & Guo types impose discrete categories on a continuum. This approach is for literature comparability, not definitive classification.
   Source: `forcing USVs into discrete categories may obscure the continuous variation that distinguishes populations`
   Verified: 2026-04-03

3. **BootSnap found simple categories (up, down, flat, short) overlap between wild and lab mice, while complex types (inverted-U, complex) differ.** This predicts which types will be most interesting when lab data arrives.
   Source: `BootSnap snapshot ensemble CNN on gammatone spectrograms outperformed DeepSqueak classification with F1 67 percent on wild mice`
   Verified: 2026-04-03

4. **DeepSqueak frequency values appear to be in kHz (range 47-91).** Rules must use consistent units. The column names say `_hz` but values are kHz-scale.
   Source: Observed in `classified_detections_full.csv` this session
   Verified: 2026-04-03

5. **Hertz et al. 2020 Syntax Information Score can evaluate whether these categories capture meaningful sequential structure.** After classification, computing SIS would validate whether the traditional types predict next-syllable better than random.
   Source: `Hertz et al 2020 Syntax Information Score ranks classification schemes by how well syllable labels predict next syllable`
   Verified: 2026-04-03

## Context

### Traditional USV syllable types

Based on Holy & Guo (2005), Scattoni et al. (2008), and Grimsley et al. (2011), the standard types are:

| Type | Key features | Typical rules |
|------|-------------|---------------|
| **Short** | < 10ms, any shape | `call_length_s < 0.010` |
| **Flat** | Low slope, low sinuosity | `abs(slope) < threshold AND sinuosity < threshold` |
| **Up** | Positive slope, low sinuosity | `slope > threshold AND sinuosity < threshold` |
| **Down** | Negative slope, low sinuosity | `slope < -threshold AND sinuosity < threshold` |
| **Chevron** (inverted-U) | High freq in middle, moderate sinuosity | `sinuosity in mid-range AND bandwidth > threshold` |
| **Complex** | High sinuosity, multiple direction changes | `sinuosity > high_threshold` |
| **Step up** | Abrupt frequency jump upward | Requires contour analysis (may not be classifiable from summary stats alone) |
| **Step down** | Abrupt frequency jump downward | Same limitation as step up |
| **Two-component** | Two distinct segments | Same limitation — needs contour, not just summary |
| **Frequency jump** | Large bandwidth, low sinuosity | `bandwidth > high_threshold AND sinuosity < threshold` |

**Important:** Some types (step, two-component, frequency jump) require the frequency *contour* to classify properly. We only have summary statistics. The plan should address this limitation — likely collapsing to 6-7 classifiable types: short, flat, up, down, chevron, complex, and possibly frequency-jump.

### Feature ranges in our data (for threshold calibration)
From the cluster feature summary computed this session:
- `call_length_s`: 0.031 – 0.405 (median ~0.08)
- `slope`: -793 to +704 (wide range)
- `sinuosity`: 1.2 to 8.1 (most calls 1.5-3.5)
- `bandwidth_hz`: 18 to 91 (in kHz despite column name)
- `tonality`: 0.18 to 0.34

### Threshold determination strategy
Thresholds should be determined by:
1. Start from published definitions where quantitative thresholds exist
2. Examine feature distributions (histograms) to find natural break points
3. Use the UMAP/HDBSCAN results (from the companion handoff) to validate — do traditional types map cleanly to density-based clusters?

### Dependencies
- `matplotlib`, `pandas`, `numpy`, `seaborn` — all available in venv
- No additional packages needed — this is pure rule-based logic

### Pattern to follow
See `scripts/generate_cluster_gallery.py` for script structure.

## Validation

1. `python -m py_compile scripts/classify_traditional_taxonomy.py`
2. Script runs end-to-end and produces output CSV + figures
3. Every row gets a `syllable_type` label (no NaN in output)
4. Distribution is not degenerate — at least 4 types have > 5% of calls
5. Cross-tabulation with k-means clusters shows interpretable mapping (e.g., "Cluster_15 is mostly flat")
6. Visual spot-check: gallery PNGs for each type should look plausibly correct

## Open Questions / Known Risks

1. **Threshold sensitivity:** Small changes in slope/sinuosity thresholds can dramatically shift proportions. The plan should include sensitivity analysis or at least document chosen thresholds and rationale.
2. **Step and multi-component types are unclassifiable from summary statistics.** These require the frequency contour (time series), which is in the DeepSqueak .mat files but not in our CSV. Calls that would be "step" or "two-component" will likely be classified as "complex" or "chevron" — document this limitation.
3. **No ground truth for wild mice.** We can't compute accuracy — only face validity from visual inspection and literature comparability. This is inherent to the approach.
4. **Unit ambiguity in frequency columns.** The CSV columns say `_hz` but values look like kHz. Verify before setting thresholds — getting this wrong would make all frequency-based rules fail.
