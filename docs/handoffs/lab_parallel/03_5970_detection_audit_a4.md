# Stream 3 — 5970 Phase A4 Detection Quality Audit

**Status:** Ready to run
**Estimated time:** 2–3 hours
**Compute:** Light, plus some manual spectrogram review

## Goal

Phase A4 from the analysis roadmap: assess CNN detection quality on 5970 (the most-analyzed dataset) before we extend trust to lab data.

The four questions to answer:

1. **Does CNN confidence correlate with syllable type?** — are some types harder to detect?
2. **Match-quality cross-tab vs syllable_type** — do "loose" types have worse match quality?
3. **What are the 37 HDBSCAN noise points?** — real rare USVs or artifacts?
4. **What is the 98-call ultra-short HDBSCAN cluster 0?** — real USVs or sub-syllable noise?

## Inputs

| File | Purpose |
|------|---------|
| `results/traditional_taxonomy/classified_traditional.csv` | 7,864 rows with syllable_type + features + CNN metadata |
| `results/recluster_umap_hdbscan/reclassified_detections.csv` | Same calls + hdbscan_label + umap_x/umap_y |
| `results/batch_5970/manual_review_all_detections.csv` | Raw detection metadata (confidence, match quality, file) |
| `data/corpus_facts/5970.json` | Reference numbers — read first |

## Steps

### 1. CNN confidence by syllable type

Output: `results/detection_quality_audit/confidence_by_type.png` (box+strip plot).
Plus: `results/detection_quality_audit/confidence_stats_by_type.csv` (median, IQR, n).

Compute Kruskal-Wallis across types. If significant, run pairwise Mann-Whitney with Holm correction. Document which types have lowest confidence — these are candidates for "model is unsure" categories that may bias the lab comparison.

### 2. Match quality cross-tab

`match_quality` column (categorical: e.g., good/fair/poor) crossed with `syllable_type`. Output:

- `cross_tab_match_quality.csv` (counts + chi-squared p-value)
- `cross_tab_match_quality_heatmap.png` (proportions per row, normalized)

Flag any type with >30% poor-match — those calls are unreliable.

### 3. HDBSCAN noise points (label = -1, N=37)

For all 37 calls with `hdbscan_label == -1`:

- Generate spectrograms (use existing spectrogram generation — see `src/usv_spectrogram/spectrogram.py` or `scripts/generate_spectrograms.py`)
- Save to `results/detection_quality_audit/hdbscan_noise_spectrograms/`
- Manually annotate each: USV / artifact / boundary / unsure (write CSV with annotations)
- Tabulate: how many per syllable_type? Per CNN confidence bin? Per file?

Decision criterion: if >70% are real USVs, the HDBSCAN noise points are rare-call detections (keep them in analyses with a "rare" tag). If <30%, they're artifacts (consider filtering).

### 4. Ultra-short HDBSCAN cluster 0 (N=98)

These are calls in `hdbscan_label == 0`. The orient-context note says "ultra-short cluster 0 (N=98)". Verify by computing duration distribution for cluster 0 vs other clusters.

- `results/detection_quality_audit/cluster_0_duration_dist.png`
- Spectrograms of 30 random samples from cluster 0
- Manual annotation: real USV / sub-syllable artifact / spectral noise

This connects to roadmap Q5: "Short calls — real USVs or detection artifacts?" The 14% Short syllables (1,113 calls) and HDBSCAN cluster 0 (98 calls) overlap. Cross-tab `syllable_type == 'Short'` against `hdbscan_label == 0` to see overlap.

### 5. CNN confidence calibration

Use `user_action` (from manual review) as ground truth for the calls that were reviewed. Plot:

- Reliability diagram: predicted prob (binned) vs fraction confirmed-real
- Brier score
- ECE (Expected Calibration Error)

Output: `results/detection_quality_audit/calibration.png` + `calibration_metrics.json`.

If miscalibration is large, document — it affects how we threshold lab detections.

### 6. Summary report

Write `docs/handoffs/lab_parallel/03_RESULTS_5970_detection_audit.md`:

- Per-type confidence ranking (worst to best)
- Match-quality red flags
- HDBSCAN noise points: real or artifact (with annotated counts)
- Ultra-short cluster 0: real or artifact
- Calibration ECE / Brier
- Implications for trusting CNN on lab data

## Constraints

1. **Do NOT modify the underlying detection CSVs.** Read-only.
2. **Use canonical sample rate / freq band** from `corpus.py` for spectrogram generation.
3. **Manual annotations are subjective** — note that single-annotator decisions are not ground truth, just flags for follow-up.
4. **Print parameters and row counts** at script start.

## Validation

Done when:
- [ ] All plots in `results/detection_quality_audit/` populated
- [ ] All annotation CSVs written
- [ ] Calibration metrics JSON exists
- [ ] `03_RESULTS_5970_detection_audit.md` written with explicit recommendations for lab-data trust
- [ ] Commit SHA recorded

## Decision-needed signals

- If calibration ECE > 0.1, surface back — affects all confidence-thresholded analyses
- If >50% of HDBSCAN noise are real USVs, the noise label is misleading — propose renaming
- If a syllable type has median confidence < 0.6, flag as "model doesn't really know this type"

## Result section

- Commit SHA:
- Headline confidence finding:
- HDBSCAN noise verdict (real / artifact / mixed):
- Cluster 0 verdict:
- Calibration ECE:
- Lab-data trust recommendation:
