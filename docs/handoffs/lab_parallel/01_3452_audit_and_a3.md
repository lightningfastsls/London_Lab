# Stream 1 — 3452 Audit + A3 Acoustic Deep-Dive

**Status:** Ready to run
**Estimated time:** 1–2 hours
**Compute:** Light (pandas + sklearn + UMAP)

## Goal

Bring the 3452 wild dataset (animal `usv_lmt_035`) to the same analysis state as 5970:
1. Verify the existing 3452 classification CSVs are correct and current
2. Generate `data/corpus_facts/3452.json` (mirror 5970.json schema exactly)
3. Run Phase A3 acoustic feature deep-dive on 3452
4. Compare A3 findings to 5970's A3

## What already exists (do NOT regenerate)

```
results/traditional_taxonomy_3452/
  classified_traditional.csv         <-- main classification output
  cluster_vs_type_crosstab.csv
  cluster_vs_type_heatmap.png
  feature_summary.csv
  feature_summary.png
  type_distribution.png

results/recluster_umap_hdbscan_3452/
  reclassified_detections.csv        <-- UMAP+HDBSCAN output
  cluster_summary.csv
  contingency_matrix.png
  umap_hdbscan_scatter.png
  umap_kmeans_scatter.png
```

These are presumed complete but unverified.

## Steps

### 1. Verify classification freshness

```bash
.venv/bin/python -c "
import pandas as pd
df = pd.read_csv('results/traditional_taxonomy_3452/classified_traditional.csv')
print('rows:', len(df))
print('columns:', df.columns.tolist())
print('type distribution:'); print(df['syllable_type'].value_counts())
print('confidence distribution:'); print(df['classification_confidence'].value_counts())
print('NaN counts per col:'); print(df.isna().sum()[df.isna().sum()>0])
"
```

Compare against the source detection CSVs in `results/batch_3452_reviewed/` and `results/batch_3452_sample/`. If row count mismatches batch detections, classification is stale — re-run `scripts/classify_traditional_taxonomy.py` against the current batch outputs.

### 2. Generate corpus_facts

```bash
.venv/bin/python scripts/audit_corpus.py --dataset 3452
```

This writes `data/corpus_facts/3452.json`. Schema MUST match `data/corpus_facts/5970.json` (same keys: `counts`, `timing`, `bout_detection_a2`, `sequential_structure_mi`, `labeling_distributions`, `references`).

If `audit_corpus.py` doesn't accept `--dataset 3452` yet, read the script and add the path mapping (input CSVs are in `results/traditional_taxonomy_3452/`, `results/recluster_umap_hdbscan_3452/`, etc.). Mirror the 5970 logic exactly.

### 3. Run A3 acoustic feature deep-dive

```bash
.venv/bin/python scripts/analyze_acoustic_features.py \
    --input results/traditional_taxonomy_3452/classified_traditional.csv \
    --output-dir results/acoustic_feature_analysis_3452/
```

This produces correlation matrix, PCA biplot, UMAP overlays, within-type violins, boundary case analysis, and `summary.txt`. The script is from the A3 work on 5970 — it has `--input` / `--output-dir` flags exactly for this re-run case.

### 4. Compare A3 findings to 5970

Read `docs/handoffs/a3-acoustic-feature-deep-dive.md` (5970 findings) and write a brief comparison memo at `docs/handoffs/lab_parallel/01_RESULTS_3452_vs_5970.md`. Cover:

- Type distribution differences (chi-squared on type proportions)
- Are the same feature correlations present (mean_power↔tonality r=0.94 in 5970 — does it hold in 3452?)
- PCA structure — same first-two-PC variance percentages? same loading patterns?
- UMAP shape — same "single continuum" finding or genuine clusters?
- Boundary case rate — is Chevron still the leakiest?
- Per-type CV — is Short still the loosest type?

## Constraints

1. **Canonical sample rate / STFT / freq band** — import from `src/usv_spectrogram/corpus.py`. Never redeclare.
2. **Bout threshold = 0.6 s** for any sequential numbers in corpus_facts (matches 5970 canonical). Q1 may change this later.
3. **Print parameters and row counts** at the start of every script run (`feedback_analysis_print_params`).
4. **3452 = wild mouse** (`project_wild_mice` memory) — variability is expected to be higher than lab.
5. **CNN was trained on wild data** — no retraining or threshold-shifting in this handoff.

## Validation

Done when:
- [ ] `data/corpus_facts/3452.json` exists, schema-matches 5970.json, all keys populated
- [ ] `results/acoustic_feature_analysis_3452/` populated with all plots + summary.txt + UMAP coordinates CSV
- [ ] `docs/handoffs/lab_parallel/01_RESULTS_3452_vs_5970.md` written
- [ ] Commit with SHA recorded back in this handoff

## Decision-needed signals — surface back, do NOT auto-decide

- If type distributions are radically different (e.g., one type missing entirely)
- If PCA structure differs qualitatively (PC1 < 20% variance, or new dominant axis)
- If UMAP shows discrete clusters in 3452 but continuum in 5970
- If row counts disagree with batch detections by more than 1%

## Result section (fill in when done)

- Commit SHA:
- Total 3452 calls:
- Type distribution:
- A3 headline finding vs 5970:
- Any decision-needed flags raised:
