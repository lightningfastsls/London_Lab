# Handoff: Presentation PNG Full Verification Results
Date: 2026-05-03

## Task

Continue the PNG provenance audit for `presentation/figures/` and `presentation/web_upload/`.

Delivered:

- Verified the inventory: 196 PNGs under `presentation/figures/` and 17 PNGs under `presentation/web_upload/`.
- Hash-matched all 17 `web_upload` PNGs to primary figures.
- Hash-matched 182 of 196 primary figures to already-rendered PNG artifacts under `results/`.
- Hash-matched 9 additional primary figures to `noise_samples/` or `models/hard_neg_retrain/`.
- Reproduced the three committed-generator figures from `presentation/figures/_generate.py` into a temporary directory and confirmed SHA-256 identity.
- Verified every gallery PNG has a source CSV row and source WAV.
- Left only the two raw waveform figures in the weaker `data found but generator missing` category.

## Files Changed

- `docs/handoffs/2026-05-03_presentation-png-full-verification-results.md` - this audit result.

## Inventory Summary

| Scope | PNG count | Verification result |
| --- | ---: | --- |
| `presentation/figures/` | 196 | All accounted for. 194 have either hash-copy or generator provenance; 2 raw waveform PNGs have source data but no committed generator. |
| `presentation/web_upload/` | 17 | All byte-identical to primary figures in `presentation/figures/`. |

Primary figure status by family:

| Figure family | PNGs | Status | Source data | Generator / copy source | Key params / assumptions | Notes |
| --- | ---: | --- | --- | --- | --- | --- |
| `00_raw_waveform.png` | 1 | data found but generator missing | `5970/USV3/usv_lmt_034/2024-10-01_18-51-28_0006209.wav`; `results/pipeline_comparison/{1_raw_cnn,3_full_pipeline}/2024-10-01_18-51-28_0006209.json` | No committed generator found; provenance comes from `presentation/HANDOFF_FROM_CLAUDE_CODE.md` | `sr=300000`; full 0.0-2.47456 s clip; Matplotlib 200 dpi, dark slate trace | Source WAV and JSONs exist; cannot reproduce from committed code. |
| `00_raw_waveform_zoomed.png` | 1 | data found but generator missing | Same WAV and JSONs as above | No committed generator found; provenance comes from `presentation/HANDOFF_FROM_CLAUDE_CODE.md` | `sr=300000`; 1.013-1.113 s zoom around first CNN detection at 1.0333866667-1.09312 s | Source data exists; cannot reproduce from committed code. |
| `01_raw_signal_pipeline/` | 6 | copied artifact verified by hash | `results/pipeline_comparison/{1_raw_cnn,2_hysteresis_only,3_full_pipeline}/*.json`; source WAVs for stems `2024-10-01_18-51-28_0006209` and `2024-10-01_17-46-45_0005839` | Byte-identical to matching `results/pipeline_comparison/*/*.png`; committed renderer not found | `sr=300000`; stage folders encode raw CNN, hysteresis, full pipeline. Keeps-all example has 4 events in all stages; rejects example has 4 raw/hysteresis events and 1 full-pipeline event | Strong artifact provenance, weaker script provenance. |
| `03_training_data/training_set_composition.png` | 1 | reproducible now | `spectrograms_training/{train,val,test}.csv`; `models/hard_neg_retrain/evaluation/test_metrics.json` | `presentation/figures/_generate.py::make_training_composition()` | Reads `label == "USV"` as positive; plots 376/476 train, 155/210 val, 65/97 test; 8 x 4.8 in, 200 dpi | SHA-identical temp reproduction. Still a slide-meaning mismatch if used to claim production `matched_windows_v2` size. |
| `03_training_data/noise_examples/` | 5 | copied artifact verified by hash | `noise_samples/*.png` | Byte-identical copies from `noise_samples/` | Hard-negative spectrogram examples; per-file source encoded in filename | Generator for these trimmed PNGs was not separately traced. |
| `04_cnn_architecture/training_curves.png` | 1 | copied artifact verified by hash | `models/hard_neg_retrain/training_history.json` | Byte-identical to `models/hard_neg_retrain/training_curves.png`; generator `scripts/plot_training_curves.py` via `plot_training_history()` | Production `mid` CNN training curves; saved by model training/plot helper | Data and plot artifact found. |
| `05_signal_detection/*.png` | 3 | copied artifact verified by hash | `models/hard_neg_retrain/evaluation/test_metrics.json`; held-out test split used by the evaluation run | Byte-identical to `models/hard_neg_retrain/evaluation/{confusion_matrix,precision_recall_curve,roc_curve}.png`; generator `scripts/evaluate_model.py` / `src/usv_spectrogram/models/evaluate.py` | Production checkpoint evaluation; metrics file records TP=479, FP=50, TN=1238, FN=62 | Did not rerun full model evaluation in this docs audit. |
| `06_deepsqueak_validation/*.png` | 2 | reproducible now | `results/validation_new_filter/comparison_report.csv`; `results/validation_new_filter/comparison_summary.json`; `results/validation_old_filter/comparison_summary.json` | `presentation/figures/_generate.py::{make_deepsqueak_agreement,make_fp_filter_v1_vs_v2}` | 197 files; v2 agreement 252, CNN-only 10, DS-only 25; v1 agreement 241, CNN-only 10, DS-only 36 | Both figures reproduced SHA-identically in temp output. |
| `08_classification/` summary PNGs | 3 | copied artifact verified by hash | `classified_detections_full.csv`; `results/traditional_taxonomy/classified_traditional.csv`; `results/traditional_taxonomy/{feature_summary,cluster_vs_type_crosstab}.csv` | Byte-identical to `results/traditional_taxonomy/{type_distribution,feature_summary,cluster_vs_type_heatmap}.png`; generator `scripts/classify_traditional_taxonomy.py` | Traditional Scattoni-style rule labels; generator also writes `classified_traditional.csv` | Strong provenance. |
| `08_classification/gallery/` | 35 | copied artifact verified by hash | `results/traditional_taxonomy/classified_traditional.csv`; source WAVs under `5970_reviewed` or `5970 USV` | Byte-identical to `results/traditional_taxonomy/gallery/**/*.png`; generator `scripts/classify_traditional_taxonomy.py::generate_gallery()` using `scripts/generate_cluster_gallery.py::render_call_spectrogram()` | `sr=300000`; 20-125 kHz render band; NFFT=1024; hop=128; 50 ms padding; seed=42; 5 examples per type | All 35 PNGs have matching source rows and WAVs. |
| `09_umap/` non-gallery PNGs | 6 | copied artifact verified by hash | `results/traditional_taxonomy/classified_traditional.csv`; `results/acoustic_feature_analysis/*`; `results/recluster_umap_hdbscan/*` | Byte-identical to `results/acoustic_feature_analysis/{umap_by_type,umap_by_feature,boundary_cases}.png` and `results/recluster_umap_hdbscan/{umap_hdbscan_scatter,umap_kmeans_scatter,contingency_matrix}.png` | UMAP on 10 standardized acoustic features; `n_neighbors=15`, `min_dist=0.1`, `random_state=42`; HDBSCAN/k-means reclustering from `scripts/recluster_umap_hdbscan.py` | Strong artifact provenance; byte reproduction would require same package versions. |
| `09_umap/hdbscan_gallery/` | 10 | copied artifact verified by hash | `results/recluster_umap_hdbscan/reclassified_detections.csv`; source WAVs under `5970_reviewed` or `5970 USV` | Byte-identical to `results/recluster_umap_hdbscan/gallery/HDBSCAN_*/*.png`; generator `scripts/recluster_umap_hdbscan.py::generate_gallery()` using `render_call_spectrogram()` | `sr=300000`; 20-125 kHz render band; NFFT=1024; hop=128; 50 ms padding; seed=42 | All 10 PNGs have matching `hdbscan_label`, source rows, and WAVs. |
| `09_umap/kmeans_gallery/` | 106 | copied artifact verified by hash | `classified_detections_full.csv`; source WAVs under `5970_reviewed` or `5970 USV` | Byte-identical to `results/cluster_gallery/Cluster_*/*.png`; generator `scripts/generate_cluster_gallery.py` | `sr=300000`; 20-125 kHz render band; NFFT=1024; hop=128; 50 ms padding; seed=42 | All 106 PNGs have matching `label` source rows and WAVs. |
| `10_temporal_dynamics/` | 5 | copied artifact verified by hash | `results/traditional_taxonomy/classified_traditional.csv`; `results/temporal_dynamics/temporal_summary.csv` | Byte-identical to `results/temporal_dynamics/*.png`; generator `scripts/analyze_temporal_dynamics.py` | Filename timestamps parsed into absolute timeline; bout threshold derived from ICI distribution; output dpi=150 | Strong artifact provenance. |
| `10_sequential_structure/` | 6 | copied artifact verified by hash | `results/traditional_taxonomy/classified_traditional.csv`; `results/sequential_structure/{sequential_structure_summary.csv,ici_gap.npy,ici_onset.npy,idiom_report.csv}` | Byte-identical to `results/sequential_structure/*.png`; generators `scripts/analyze_sequential_structure.py` and `scripts/bout_threshold_sensitivity.py` | Within-bout sequential analysis; A2 uses bout-aware pairs; sensitivity plots use file-aware/no-file-aware comparisons | Strong artifact provenance. |
| `11_sis_baselines/baselines.png` | 1 | copied artifact verified by hash | `data/corpus_facts/5970.json`; `results/sis_baselines/{baselines.csv,parameters.json}` | Byte-identical to `results/sis_baselines/baselines.png`; generator `scripts/run_sis_baselines.py` | Raw-consecutive SIS depth-1 MI by labeling; references Hertz 2020 lines | Strong artifact provenance. |
| `12_cross_population/` | 4 | copied artifact verified by hash | `results/cross_population/wild_5970_vs_wild_3452.{json,md}` | Byte-identical to `results/cross_population/wild_5970_vs_wild_3452/{type_proportions,features,transitions,ioi_medians}.png`; generator `src/usv_spectrogram/classification/cross_population.py::write_figures()` | Wild 5970 vs wild 3452 comparison; this is wild-vs-wild between-couple variability, not lab-vs-wild | Strong artifact provenance. |

## Web Upload Hash Matches

All `presentation/web_upload/*.png` are upload copies, not independent primary figures.

| Web upload PNG | Matching primary figure |
| --- | --- |
| `presentation/web_upload/01_raw_pipeline_keeps_stage1.png` | `presentation/figures/01_raw_signal_pipeline/example_filter_keeps_all/stage1_raw_cnn.png` |
| `presentation/web_upload/02_raw_pipeline_rejects_stage3.png` | `presentation/figures/01_raw_signal_pipeline/example_filter_rejects/stage3_full_pipeline.png` |
| `presentation/web_upload/03_training_set_composition.png` | `presentation/figures/03_training_data/training_set_composition.png` |
| `presentation/web_upload/04_training_curves.png` | `presentation/figures/04_cnn_architecture/training_curves.png` |
| `presentation/web_upload/05_confusion_matrix.png` | `presentation/figures/05_signal_detection/confusion_matrix.png` |
| `presentation/web_upload/07_deepsqueak_agreement.png` | `presentation/figures/06_deepsqueak_validation/deepsqueak_agreement.png` |
| `presentation/web_upload/08_type_distribution.png` | `presentation/figures/08_classification/type_distribution.png` |
| `presentation/web_upload/09_feature_summary.png` | `presentation/figures/08_classification/feature_summary.png` |
| `presentation/web_upload/10_umap_by_type.png` | `presentation/figures/09_umap/umap_by_type.png` |
| `presentation/web_upload/11_umap_by_feature.png` | `presentation/figures/09_umap/umap_by_feature.png` |
| `presentation/web_upload/12_umap_hdbscan_scatter.png` | `presentation/figures/09_umap/umap_hdbscan_scatter.png` |
| `presentation/web_upload/13_call_rate_hourly.png` | `presentation/figures/10_temporal_dynamics/call_rate_hourly.png` |
| `presentation/web_upload/14_ici_distribution.png` | `presentation/figures/10_temporal_dynamics/ici_distribution.png` |
| `presentation/web_upload/15_transition_matrix.png` | `presentation/figures/10_sequential_structure/transition_matrix.png` |
| `presentation/web_upload/16_mutual_information_lag.png` | `presentation/figures/10_sequential_structure/mutual_information_lag.png` |
| `presentation/web_upload/17_sis_baselines.png` | `presentation/figures/11_sis_baselines/baselines.png` |
| `presentation/web_upload/18_cross_population_type_proportions.png` | `presentation/figures/12_cross_population/type_proportions.png` |

## Gallery Source-Row Verification

Gallery PNGs share a renderer, so the useful per-PNG verification is whether each filename resolves to the expected source row and WAV. This was checked by parsing filenames of the form `NN_<wav_stem>_<begin>.png`, then matching source CSV rows with a +/-0.0006 s tolerance.

| Gallery family | PNGs checked | Source row criterion | WAV criterion | Result |
| --- | ---: | --- | --- | --- |
| `presentation/figures/08_classification/gallery/` | 35 | `wav_stem`, rounded `begin_time_s`, and `syllable_type` in `results/traditional_taxonomy/classified_traditional.csv` | `wav_stem` found in `5970_reviewed` or `5970 USV` | 35/35 matched rows and WAVs |
| `presentation/figures/09_umap/hdbscan_gallery/` | 10 | `wav_stem`, rounded `begin_time_s`, and `hdbscan_label` in `results/recluster_umap_hdbscan/reclassified_detections.csv` | `wav_stem` found in `5970_reviewed` or `5970 USV` | 10/10 matched rows and WAVs |
| `presentation/figures/09_umap/kmeans_gallery/` | 106 | `wav_stem`, rounded `begin_time_s`, and `label` in `classified_detections_full.csv` | `wav_stem` found in `5970_reviewed` or `5970 USV` | 106/106 matched rows and WAVs |

Per-PNG source details are encoded directly in each gallery filename:

- parent folder = expected type or cluster
- filename stem = source WAV stem
- final numeric field = rounded `begin_time_s`
- exact `end_time_s`, confidence, frequency, and feature values live in the matched CSV row

## Reasoning

This audit treats byte identity, source data, and generator provenance as separate claims.

Many presentation PNGs are not regenerated directly in `presentation/figures/`; they are copied from `results/`, `models/`, or `noise_samples/`. For those, `copied artifact verified by hash` is the strongest accurate status for the presentation copy, while the family notes still identify the upstream generator and source data when found.

Only three unmatched figures have a committed presentation generator: `training_set_composition.png`, `deepsqueak_agreement.png`, and `fp_filter_v1_vs_v2.png`. I reproduced those into a temporary folder and verified SHA-256 identity.

The raw waveform figures remain weaker because the source WAV and detection JSONs are present, but no committed script creates the exact PNGs. `presentation/HANDOFF_FROM_CLAUDE_CODE.md` is detailed enough to recreate the content, but that is not the same as generator provenance.

## Validation

Docs-only validation performed:

- Re-read this handoff after writing.
- Counted `presentation/figures/**/*.png`: 196.
- Counted `presentation/web_upload/*.png`: 17.
- SHA-256 matched all 17 `web_upload` PNGs to `presentation/figures/`.
- SHA-256 matched 182/196 primary figure PNGs to `results/**/*.png`.
- SHA-256 matched the 5 training noise examples to `noise_samples/*.png`.
- SHA-256 matched `04_cnn_architecture/training_curves.png` to `models/hard_neg_retrain/training_curves.png`.
- SHA-256 matched the 3 signal detection figures to `models/hard_neg_retrain/evaluation/*.png`.
- Reproduced these figures into a temporary directory and confirmed SHA-256 equality with the presentation copies:
  - `presentation/figures/03_training_data/training_set_composition.png`
  - `presentation/figures/06_deepsqueak_validation/deepsqueak_agreement.png`
  - `presentation/figures/06_deepsqueak_validation/fp_filter_v1_vs_v2.png`
- Verified all gallery source rows and WAVs:
  - `08_classification/gallery/`: 35/35
  - `09_umap/hdbscan_gallery/`: 10/10
  - `09_umap/kmeans_gallery/`: 106/106
- Verified raw waveform source facts:
  - `5970/USV3/usv_lmt_034/2024-10-01_18-51-28_0006209.wav` exists, `sr=300000`, 742368 samples, 2.47456 s.
  - `results/pipeline_comparison/1_raw_cnn/2024-10-01_18-51-28_0006209.json` exists and has 4 events.
  - `results/pipeline_comparison/3_full_pipeline/2024-10-01_18-51-28_0006209.json` exists and has 4 events.
- Verified raw pipeline source facts:
  - `2024-10-01_18-51-28_0006209.wav` exists with `sr=300000`, 2.47456 s; stage JSONs have 4/4/4 events.
  - `2024-10-01_17-46-45_0005839.wav` exists with `sr=300000`, 2.14688 s; stage JSONs have 4/4/1 events.
- Searched for committed raw waveform and raw pipeline renderers with `rg`; no renderer was found beyond handoff documentation and copied result artifacts.

External paths not verified:

- None. All source paths recorded here were checked inside the current repo checkout.

## Open Questions / Known Risks

- `00_raw_waveform.png` and `00_raw_waveform_zoomed.png` should get a small committed generator if the presentation needs durable rebuilds.
- `01_raw_signal_pipeline/*.png` are byte-identical to `results/pipeline_comparison/*.png`, but the script that rendered those stage-overlay PNGs is not in the repo. If these are important for future rebuilds, recover or recreate that renderer.
- `03_training_data/training_set_composition.png` is technically reproducible, but it plots `spectrograms_training/`, not the production `data/training/matched_windows_v2/` corpus. Use it only if the slide intends the smaller labeling-app working corpus.
- Gallery figures render 20-125 kHz. Some presentation prose still says 30-110 kHz or 20-120 kHz; cite the actual renderer band for gallery PNGs unless the script changes.
- Byte-identical reproduction for UMAP and Matplotlib analysis figures may depend on package versions. This audit only claims byte identity where SHA-256 was actually checked against existing artifacts.

## Worth Remembering For Claude

- The presentation image set is now mostly accountable by hash: `presentation/figures/` is primarily a copied presentation layer over `results/`, `models/`, and `noise_samples/`.
- Do not treat `web_upload/` as independent provenance; every PNG there is a byte-identical upload copy.
- The two real unresolved provenance gaps are raw waveform rendering and raw pipeline overlay rendering.
- If the deck narrative says production training set size, replace or relabel `03_training_data/training_set_composition.png`.
