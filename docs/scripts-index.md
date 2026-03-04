# Scripts Index

Categorized index of all entry-point scripts in `scripts/`. For module-level docs, see `docs/modules/`.

---

## Core Pipeline (spectrogram, detection, extraction)

| Script | Purpose |
|--------|---------|
| `make_spectrogram.py` | Generate spectrogram PNG from a WAV file |
| `run_detection.py` | Run energy-based USV detection pipeline |
| `extract_spectrograms.py` | Extract spectrogram images for detected candidates |
| `extract_split_spectrograms.py` | Extract spectrograms with train/val/test split awareness |
| `extract_comparison_spectrograms.py` | Extract side-by-side comparison spectrograms |
| `extract_visual_samples.py` | Extract visual sample images for review |

## Training Data Preparation

| Script | Purpose |
|--------|---------|
| `assemble_training_data.py` | Assemble full training dataset (Phase 9.1 DatasetAssembler) |
| `prepare_dataset.py` | Prepare training dataset from detections |
| `prepare_labeled_dataset.py` | Prepare dataset from labeled examples |
| `create_train_val_test_splits.py` | Create stratified train/val/test splits |
| `create_experiment_dataset.py` | Create experiment-specific dataset subset |
| `create_full_training_dataset.py` | Create full training dataset from all sources |
| `check_splits.py` | Validate split integrity (no leakage) |
| `check_overlap.py` | Check for overlap between dataset splits |
| `diagnose_dataset.py` | Diagnose dataset issues (class balance, corrupted files) |
| `analyze_training_data_composition.py` | Analyze training data class/source composition |

## Negative Sample Generation

| Script | Purpose |
|--------|---------|
| `generate_random_negatives.py` | Generate random negative examples from background |
| `generate_comprehensive_negatives.py` | Generate comprehensive negative set (multiple strategies) |
| `extract_noise_samples.py` | Extract noise-only spectrogram samples |
| `regenerate_noise_spectrograms.py` | Regenerate noise spectrograms from updated config |
| `build_noise_dataset.py` | Build structured noise dataset |

## CNN Training & Evaluation

| Script | Purpose |
|--------|---------|
| `train_cnn.py` | Train CNN classifier |
| `predict.py` | Run inference with trained model |
| `evaluate_model.py` | Evaluate model performance (precision, recall, F1) |
| `plot_training_curves.py` | Plot loss/accuracy training curves |
| `collect_baseline_metrics.py` | Collect baseline performance metrics |

## Active Learning Cycle

| Script | Purpose |
|--------|---------|
| `run_training_cycle.py` | Run full active learning cycle (detect → train → evaluate) |
| `mine_hard_negatives.py` | Mine hard negatives from false positives |
| `compare_detection_results.py` | Compare detection results across runs |

## Threshold & Parameter Tuning

| Script | Purpose |
|--------|---------|
| `threshold_sweep.py` | Sweep detection threshold parameter |
| `recalibrate_threshold.py` | Recalibrate detection threshold from data |
| `optimize_threshold.py` | Optimize detection threshold for F1 |
| `test_continuity_tuning.py` | Test continuity parameter settings |
| `test_merge_gap_settings.py` | Test merge gap configuration |
| `test_middle_ground_config.py` | Test middle-ground detection config |

## Analysis & Comparison

| Script | Purpose |
|--------|---------|
| `analyze_predictions.py` | Analyze prediction outputs and error patterns |
| `analyze_recording_performance.py` | Analyze per-recording model performance |
| `compare_probability_distributions.py` | Compare probability distribution shapes |
| `compare_preprocessing.py` | Compare preprocessing method effects |
| `evaluate_experiment.py` | Evaluate experiment results end-to-end |
| `find_label_outliers.py` | Find outlier labels in dataset |

## Desktop App Launchers

| Script | Purpose |
|--------|---------|
| `run_app.py` | Launch PyQt6 detection desktop app |
| `run_app_safe.py` | Safe app launcher (catches import errors) |

## Streamlit Apps

| Script | Purpose |
|--------|---------|
| `usv_parameter_lab.py` | Launch Streamlit parameter exploration lab |
| `usv_labeling_tool.py` | Launch Streamlit USV labeling tool |
| `noise_review_tool.py` | Launch Streamlit noise sample review tool |

## Clustering (4-script pipeline)

| Script | Purpose |
|--------|---------|
| `batch_detect_for_clustering.py` | Run batch detection to feed clustering |
| `clustering_extract_features.py` | Extract features for clustering |
| `clustering_cluster.py` | Run clustering algorithm |
| `clustering_analyze.py` | Analyze clustering results |
| `clustering_visualize.py` | Visualize cluster assignments |

## Classification Bridge (DeepSqueak / Raven)

| Script | Purpose |
|--------|---------|
| `export_raven_tables.py` | Export detections as Raven selection tables |
| `import_deepsqueak_results.py` | Import DeepSqueak classification results |
| `analyze_repertoire.py` | Analyze USV repertoire statistics |

## LMT Integration

| Script | Purpose |
|--------|---------|
| `run_event_triggered_analysis.py` | Run event-triggered USV analysis with LMT behavioral data |

## Data Utilities

| Script | Purpose |
|--------|---------|
| `pick_random_wavs_9252.py` | Pick random WAV files from recording set |
| `sample_wav_subset.py` | Sample a WAV subset for quick experiments |
| `detect_9252.py` | Run detection on project 9252 recordings |

## Debugging & Diagnostics

| Script | Purpose |
|--------|---------|
| `debug_merge.py` | Debug candidate merge operations |
| `debug_outputs.py` | Debug output file generation |
| `debug_prediction.py` | Debug prediction logic |
| `debug_sliding_inference.py` | Debug sliding window inference |
| `diagnose_cnn_batch_detection.py` | Diagnose CNN batch detection issues |
| `test_boundary_adjustment.py` | Test boundary adjustment logic |
| `test_detection_backend.py` | Test detection backend integration |
| `test_gui_import.py` | Test GUI import chain |
| `test_ui_integration.py` | Test UI integration |

## Reporting & Notion

| Script | Purpose |
|--------|---------|
| `generate_project_timeline.py` | Generate project timeline visualization |
| `generate_timeline.py` | Generate timeline (alternate format) |
| `upload_report_to_notion.py` | Upload reports to Notion page |
| `check_kb_projects_prop.py` | Check Notion KB projects property |
| `batch_move_project.py` | Batch move Notion project pages |
| `explore_project_notes.py` | Explore project notes in Notion |
| `tag_specific_pages.py` | Tag specific Notion pages |
| `find_exact_pages.py` | Find exact Notion pages by title |
| `find_course_55008.py` | Find course 55008 in Notion |

## MATLAB Interop (legacy)

| Script | Purpose |
|--------|---------|
| `create_deepsqueak_mats.m` | Create DeepSqueak MAT files from detections |
| `diagnose_deepsqueak.m` | Diagnose DeepSqueak import issues |
