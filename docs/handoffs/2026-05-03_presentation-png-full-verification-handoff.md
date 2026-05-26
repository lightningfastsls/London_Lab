# Handoff: Presentation PNG Full Verification
Date: 2026-05-03

## Task

Continue the presentation PNG provenance audit by verifying every PNG in `presentation/figures/`, one image at a time or one tightly related image family at a time.

Primary scope:

- `presentation/figures/` - 196 PNGs

Secondary scope:

- `presentation/web_upload/` - 17 PNGs. Treat these as upload copies first. Verify by SHA-256 against `presentation/figures/`; only audit separately if a web-upload PNG has no matching primary figure.

Do not rely on visual inspection alone. A PNG is verified only when the provenance fields below are recorded.

## Required Verification Fields Per PNG

For each PNG, record:

- what the image shows
- source data file(s)
- generating script/notebook/function
- key parameters and assumptions
- whether it can be reproduced now
- mismatches, missing provenance, stale claims, or follow-up work

Use these reproducibility labels:

- reproducible now
- reproducible after small script fix
- data found but generator missing
- generator found but data missing
- copied artifact verified by hash
- unresolved

## Start Here

Read first:

1. `AGENTS.md`
2. `docs/handoffs/2026-05-03_presentation-png-provenance-audit.md`
3. `presentation/HANDOFF_FROM_CLAUDE_CODE.md`
4. `presentation/FIGURE_GUIDE_FOR_WEB_SESSION.md`

The previous audit solved four representative examples:

- `presentation/figures/00_raw_waveform.png`
- `presentation/figures/03_training_data/training_set_composition.png`
- `presentation/figures/09_umap/umap_by_type.png`
- `presentation/figures/08_classification/gallery/Flat/01_2024-09-30_11-20-14_0000022_5.478s.png`

Use those solved examples as templates, but do not assume their conclusions apply to every family.

## Recommended Order

1. Hash-match `presentation/web_upload/*.png` against `presentation/figures/**/*.png`. Mark matching copies as `copied artifact verified by hash`.
2. Finish quick scripted families:
   - `06_deepsqueak_validation/`
   - `08_classification/` summary PNGs
   - `09_umap/` non-gallery PNGs
3. Verify gallery families with representative sampling plus source-row checks:
   - `08_classification/gallery/`
   - `09_umap/hdbscan_gallery/`
   - `09_umap/kmeans_gallery/`
4. Verify analysis output families:
   - `10_temporal_dynamics/`
   - `10_sequential_structure/`
   - `11_sis_baselines/`
   - `12_cross_population/`
5. Investigate weaker-provenance families:
   - `00_raw_waveform*.png`
   - `01_raw_signal_pipeline/`
   - `04_cnn_architecture/`
   - `05_signal_detection/`

## Known Counts

As of 2026-05-03:

- `presentation/`: 213 PNGs total
- `presentation/figures/`: 196 PNGs
- `presentation/web_upload/`: 17 PNGs

`presentation/figures/` breakdown:

| Cluster | PNG count |
| --- | ---: |
| `00_raw_waveform.png` | 1 |
| `00_raw_waveform_zoomed.png` | 1 |
| `01_raw_signal_pipeline/` | 6 |
| `03_training_data/` | 6 |
| `04_cnn_architecture/` | 1 |
| `05_signal_detection/` | 3 |
| `06_deepsqueak_validation/` | 2 |
| `08_classification/` | 38 |
| `09_umap/` | 122 |
| `10_sequential_structure/` | 6 |
| `10_temporal_dynamics/` | 5 |
| `11_sis_baselines/` | 1 |
| `12_cross_population/` | 4 |

## Likely Provenance Map

Use this as a starting hypothesis only; verify each claim.

| Figure family | Likely generator | Likely source data |
| --- | --- | --- |
| `03_training_data/training_set_composition.png` | `presentation/figures/_generate.py::make_training_composition()` | `spectrograms_training/{train,val,test}.csv` |
| `06_deepsqueak_validation/*.png` | `presentation/figures/_generate.py` | `results/validation_new_filter/*`, `results/validation_old_filter/*` |
| `08_classification/*.png` | `scripts/classify_traditional_taxonomy.py` | `classified_detections_full.csv`, `results/traditional_taxonomy/classified_traditional.csv` |
| `08_classification/gallery/**/*.png` | `scripts/classify_traditional_taxonomy.py`, `scripts/generate_cluster_gallery.py::render_call_spectrogram()` | `results/traditional_taxonomy/classified_traditional.csv`, source WAVs under `5970 USV/` or `5970_reviewed/` |
| `09_umap/umap_by_type.png`, `umap_by_feature.png`, `boundary_cases.png` | `scripts/analyze_acoustic_features.py` | `results/traditional_taxonomy/classified_traditional.csv`, `results/acoustic_feature_analysis/*` |
| `09_umap/umap_hdbscan_scatter.png`, `umap_kmeans_scatter.png`, `contingency_matrix.png`, galleries | `scripts/recluster_umap_hdbscan.py`, `scripts/generate_cluster_gallery.py` | `classified_detections_full.csv`, `results/recluster_umap_hdbscan/*`, source WAVs |
| `10_temporal_dynamics/*.png` | `scripts/analyze_temporal_dynamics.py` | likely `results/traditional_taxonomy/classified_traditional.csv`, `results/temporal_dynamics/*` |
| `10_sequential_structure/*.png` | `scripts/analyze_sequential_structure.py`, `scripts/bout_threshold_sensitivity.py` | likely `results/traditional_taxonomy/classified_traditional.csv`, `results/sequential_structure/*`, `results/bout_threshold_sensitivity/*` |
| `11_sis_baselines/baselines.png` | `scripts/run_sis_baselines.py` | `data/corpus_facts/5970.json`, SIS result files |
| `12_cross_population/*.png` | `src/usv_spectrogram/classification/cross_population.py` or caller | `results/cross_population/wild_5970_vs_wild_3452.json`, `.md` |
| `00_raw_waveform*.png` | generator not found in prior audit | WAV and JSONs named in `presentation/HANDOFF_FROM_CLAUDE_CODE.md` |
| `01_raw_signal_pipeline/*.png` | generator not found in prior audit | likely `results/pipeline_comparison/{1_raw_cnn,2_hysteresis_only,3_full_pipeline}/` plus source WAVs |
| `04_cnn_architecture/training_curves.png` | likely `scripts/plot_training_curves.py` | model training history/checkpoint artifacts |
| `05_signal_detection/*.png` | likely `scripts/evaluate_model.py` or `scripts/generate_roc_curve.py` | `models/hard_neg_retrain/evaluation/test_metrics.json`, test split data |

## Output Format

Update or create a durable audit document under `docs/handoffs/`, for example:

`docs/handoffs/YYYY-MM-DD_presentation-png-full-verification-results.md`

For a large audit, use a table plus details for exceptions:

| PNG | Shows | Source data | Generator | Key params | Repro status | Notes |
| --- | --- | --- | --- | --- | --- | --- |

For gallery families, do not write 100 repeated paragraphs if the same generator and parameters apply. Instead:

- verify the family generator and shared parameters once
- verify each PNG's source row exists
- verify source WAV exists
- record a compact per-PNG row with source stem, begin/end time, label/cluster, and status

## Validation Required Before Final Answer

For docs-only verification:

- Re-read the created or updated audit file.
- Verify every referenced repo file/path exists.
- Explicitly list any external paths not verified.
- If claiming `reproducible now`, run the generator or a tightly scoped reproduction probe.
- If claiming `copied artifact verified by hash`, record the matching source path and hash result.
- Do not claim byte-identical reproduction unless SHA-256 was checked.

## Important Findings Already Known

- `presentation/figures/03_training_data/training_set_composition.png` plots `spectrograms_training`, not production `data/training/matched_windows_v2`. This is a real slide-meaning mismatch.
- `presentation/figures/00_raw_waveform.png` has source data but no committed generator found in the prior audit.
- `presentation/figures/09_umap/umap_by_type.png` is byte-identical to `results/acoustic_feature_analysis/umap_by_type.png`.
- `presentation/figures/08_classification/gallery/Flat/01_2024-09-30_11-20-14_0000022_5.478s.png` is byte-identical to its `results/traditional_taxonomy/gallery/` counterpart.
- Some deck prose says USV band 30-110 kHz or 20-120 kHz, while `scripts/generate_cluster_gallery.py` renders galleries with 20-125 kHz. Record the actual script parameter per figure family.

## Open Questions / Known Risks

- Some presentation PNGs may be copied from ignored result artifacts. If a copied result PNG is absent in a fresh clone, script-level reproduction is still needed.
- UMAP and Matplotlib output can vary by package version. Content reproduction and byte-identical reproduction are different standards.
- Git may report dubious ownership from PowerShell on this WSL UNC checkout. That is not a blocker for docs-only audit work.

## Worth Remembering For Claude

The goal is not to admire or caption every image. The goal is to make each image accountable: data, generator, parameters, reproducibility status, and mismatches. Prefer compact family-level verification where the same script generated many gallery PNGs, but still verify every PNG has a source row and source WAV where applicable.
