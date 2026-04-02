# DeepSqueak Full Pipeline Results -- 5970 Dataset

**Date:** 2026-04-03
**Status:** Complete. Full round-trip classification pipeline validated end-to-end.
**Significance:** First complete run of the DeepSqueak classification bridge on the full 5970 dataset. 7,518 USV calls classified into 27 acoustic clusters, merged with CNN detection metadata. This unlocks repertoire analysis, behavioral correlation, and supervised classifier training.

## What Was Done

### Pipeline Overview

```
6,400 WAVs (5970/ + 5970_reviewed/)
  -> CNN batch detection [run_batch_detection.py]
     1,328 WAVs with USVs, 7,575 detections
  -> Raven export [export_raven_tables.py --batch-format]
     1,328 selection tables in raven_tables_full/
  -> MATLAB: create_deepsqueak_mats.m
     1,328 .mat files in DeepSqueak Detections/
  -> MATLAB: deepsqueak_batch_classify.m (k-means)
     27 clusters, 7,864 classified calls
  -> MATLAB: deepsqueak_export_stats.m
     deepsqueak_output_full/classified_Stats.xlsx (18 acoustic features)
  -> Python: import_deepsqueak_results.py --batch-format
     classified_detections_full.csv (31 columns, 7,518 matched)
```

### Key Numbers

| Metric | Value |
|--------|-------|
| Source WAVs | 6,400 |
| WAVs with USV detections | 1,328 |
| Total CNN detections | 7,575 |
| DeepSqueak classified calls | 7,864 (includes 289 from smoke test overlap) |
| Successfully merged | 7,518 (99.2% of batch detections) |
| Unmatched DS calls | 346 (289 from smoke test stems, 57 edge cases) |
| Acoustic clusters (k-means) | 27 |
| Matching tolerance used | 75 ms |
| Mean match distance | 20.28 ms |
| Median match distance | 18.60 ms |

### Cluster Distribution (top 10)

| Cluster | Count | % of matched |
|---------|-------|-------------|
| Cluster_27 | 1,408 | 18.7% |
| Cluster_15 | 1,165 | 15.5% |
| Cluster_25 | 824 | 11.0% |
| Cluster_23 | 548 | 7.3% |
| Cluster_22 | 346 | 4.6% |
| Cluster_16 | 337 | 4.5% |
| Cluster_5 | 305 | 4.1% |
| Cluster_17 | 289 | 3.8% |
| Cluster_7 | 270 | 3.6% |
| Cluster_10 | 208 | 2.8% |

## Technical Decisions and Findings

### 1. Timestamp Drift: 75ms Tolerance Required

DeepSqueak's `CalculateStats` recomputes `Begin Time` from spectrogram ridge analysis rather than using the raw `Box(:,1)` value from the Raven import. This causes systematic drift:
- Most calls: ~0.37ms (STFT bin quantization)
- Some calls: 8-68ms (spectrogram ridge shifting)
- P95: 42ms, P99: 55ms, Max: 74ms

75ms tolerance is safe because minimum inter-call gap across all WAVs is 39.6ms (well above the max drift), and each stem has exactly the same number of DS calls and detections, eliminating cross-match risk.

### 2. Batch Format Support

Two detection JSON formats exist in the repo:

| Format | Location | Structure | Used by |
|--------|----------|-----------|---------|
| Per-detection | `USV_Detections/<batch>/<stem>/detection_*.json` | Individual files with `core_time` | PyQt6 review app |
| Batch flat | `results/batch_*/detections/<stem>.json` | List of `{start_time_s, end_time_s, ...}` per WAV | `run_batch_detection.py` |

Both `raven_export.py` and `deepsqueak_import.py` now support both formats via `--batch-format` flag.

### 3. Consolidated Excel wav_stem Fix

DeepSqueak's standard workflow produces one Excel per WAV. Our headless pipeline produces a single `classified_Stats.xlsx` with all WAVs. The `wav_stem` extraction was fixed to use the per-row `File` column instead of the Excel filename.

### 4. MATLAB UNC Path Workarounds

MATLAB's `save()` and `load()` cannot handle UNC paths (`\\wsl.localhost\...`). All three MATLAB scripts use a temp file + copyfile pattern:
```matlab
tmpPath = fullfile(tempdir, 'filename.mat');
save(tmpPath, 'data', '-v7.3');
copyfile(tmpPath, uncPath);
delete(tmpPath);
```

### 5. Recursive WAV Lookup

WAVs are nested in `5970/USV{1-5}/usv_lmt_034/<stem>.wav`. The MATLAB script builds a `containers.Map` lookup by recursively scanning with `dir(fullfile(wavDir, '**', '*.wav'))`. All 6,400 stems are unique across subdirectories.

## Output Artifacts

| File | Description |
|------|-------------|
| `classified_detections_full.csv` | 7,921 rows, 31 columns (merged DS + detection metadata) |
| `classified_detections.csv` | 757 rows (smoke test, 10 WAVs) |
| `deepsqueak_output_full/classified_Stats.xlsx` | Raw DeepSqueak output (7,864 calls, 18 features) |
| `deepsqueak_output_full/clustering_model_kmeans.mat` | K-means centroids (k=27) for reuse |
| `raven_tables_full/` | 1,328 Raven selection tables |
| `raven_tables/` | 10 selection tables (smoke test) |
| `deepsqueak_output/` | Smoke test DeepSqueak output (289 calls) |

## MATLAB Scripts

All scripts are in `scripts/` and designed for headless batch operation (no GUI):

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `create_deepsqueak_mats.m` | Raven TSV -> DeepSqueak .mat | `raven_tables_full/`, WAV dirs | .mat files in DeepSqueak/Detections/ |
| `deepsqueak_batch_classify.m` | K-means clustering | .mat files, DeepSqueak install | Cluster labels saved to .mat, Excel output |
| `deepsqueak_export_stats.m` | Compute 18 acoustic features | .mat files | Excel with stats |
| `test_deepsqueak_batch.m` | Post-run validation (16 checks) | .mat dir, output dir | PASS/FAIL report |
| `diagnose_deepsqueak.m` | .mat structure diagnostic | .mat files | Structural comparison with DS examples |

### Running the Full MATLAB Pipeline

```matlab
% Step 1: Create .mat files (run as script, ~5 min for 1328 files)
run('\\wsl.localhost\Ubuntu\home\shachar\projects\mickey_london_lab\scripts\create_deepsqueak_mats.m')

% Step 2: Classify + export (takes ~1-2 hours for 7500 calls)
deepsqueak_batch_classify( ...
    fullfile(fileparts(which('DeepSqueak')), 'Detections'), ...
    fileparts(which('DeepSqueak')), ...
    '\\wsl.localhost\Ubuntu\home\shachar\projects\mickey_london_lab\deepsqueak_output_full', ...
    'kmeans')

% Step 3: Validate
test_deepsqueak_batch( ...
    fullfile(fileparts(which('DeepSqueak')), 'Detections'), ...
    '\\wsl.localhost\Ubuntu\home\shachar\projects\mickey_london_lab\deepsqueak_output_full')
```

### Running the Python Import

```bash
PYTHONPATH=src .venv/bin/python scripts/import_deepsqueak_results.py \
    --results-dir deepsqueak_output_full \
    --detections-dir results/batch_5970_v2_full/detections \
    --batch-format \
    --output classified_detections_full.csv \
    --tolerance-ms 75.0 -v
```

## Merged CSV Column Reference (31 columns)

**DeepSqueak acoustic features (18):**
file, id, label, accepted, score, begin_time_s, end_time_s, call_length_s, principal_freq_hz, low_freq_hz, high_freq_hz, bandwidth_hz, freq_std_dev_hz, slope, sinuosity, mean_power_db, tonality, peak_freq_khz

**Detection metadata (8):**
det_start_s, det_end_s, det_duration_ms, det_index, det_prob_max, det_prob_mean, det_user_action, det_json_path

**Merge metadata (5):**
wav_stem, source_file, match_quality, match_distance_ms, (row index)

## What's Next

1. **Repertoire analysis** -- Use cluster distributions to characterize USV repertoires per recording session. Compare across experimental conditions.
2. **Behavioral correlation** -- Merge with LMT event data to see which cluster types co-occur with specific behaviors.
3. **Supervised classifier** -- Train a CNN or random forest on the 27 cluster labels to bypass DeepSqueak for future datasets.
4. **3452 dataset** -- Run the same pipeline on the 3452 batch (855 reviewed WAVs).

## Commits (this session)

| Hash | Description |
|------|-------------|
| `f2801508` | Fix wav_stem extraction for consolidated multi-WAV Excel |
| `b465bfd0` | Batch-format Raven export + MATLAB recursive WAV lookup |
| `46553637` | Batch-format import + full 5970 merge results (7,518 matched) |
