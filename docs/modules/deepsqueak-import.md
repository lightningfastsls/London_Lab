# DeepSqueak Results Import

**Phase:** 14.2 (DeepSqueak Results Ingestion)
**ADRs:** None (format conversion + timestamp matching, no DSP parameters involved)
**Tests:** `tests/test_classification/test_deepsqueak_import.py`

## Purpose

Reads DeepSqueak Excel classification outputs (.xlsx), normalizes column names to snake_case, and re-merges them with our detection JSONs using timestamp proximity matching. This completes the round-trip classification pipeline:

```
Our detections -> Raven tables -> DeepSqueak (MATLAB) -> Excel -> THIS MODULE -> merged CSV
```

The merged CSV contains both DeepSqueak's acoustic features/labels and our detection metadata, enabling downstream repertoire statistics (Phase 14.3).

## Public Interface

### `DeepSqueakImportConfig`

```python
@dataclass(frozen=True)
class DeepSqueakImportConfig:
    results_dir: Path             # Directory with .xlsx files
    detections_dir: Path          # USV_Detections/ root
    output_path: Path             # Where to write merged CSV
    tolerance_ms: float = 5.0    # Max time difference for matching (ms)
```

Validates: tolerance > 0, auto-converts string paths to `Path`.

### `ImportSummary`

```python
@dataclass
class ImportSummary:
    total_ds_calls: int           # Total rows from DeepSqueak Excel files
    total_detections: int         # Total detection JSONs loaded
    matched: int                  # Successfully matched pairs
    unmatched_ds: int             # DS calls with no detection match
    unmatched_det: int            # Detections with no DS call match
    files_processed: int          # Number of Excel files loaded
    per_file_counts: dict[str, int]  # {filename: n_rows}
```

### Core Functions

| Function | Signature | Purpose |
|----------|-----------|---------|
| `load_deepsqueak_excel` | `(Path) -> pd.DataFrame` | Load one Excel, normalize columns, add wav_stem |
| `load_all_deepsqueak_results` | `(Path) -> pd.DataFrame` | Batch load all .xlsx from directory |
| `load_detections_for_merge` | `(Path) -> dict[str, list[dict]]` | Load detection JSONs grouped by WAV stem |
| `merge_with_detections` | `(DataFrame, dict, float) -> (DataFrame, ImportSummary)` | Timestamp proximity matching |
| `export_classified_detections` | `(DataFrame, Path) -> Path` | Write merged CSV |
| `import_deepsqueak_results` | `(DeepSqueakImportConfig) -> ImportSummary` | Full pipeline orchestrator |

### DeepSqueak Excel Columns (16 fields, normalized to snake_case)

```
ID -> id                           Begin Time (s) -> begin_time_s
Label/Type -> label                End Time (s) -> end_time_s
Call Length (s) -> call_length_s   Principal Frequency (kHz) -> principal_freq_hz
Low Freq (kHz) -> low_freq_hz     High Freq (kHz) -> high_freq_hz
Delta Freq (kHz) -> bandwidth_hz  Frequency Standard Deviation (kHz) -> freq_std_dev_hz
Slope (kHz/s) -> slope            Sinuosity -> sinuosity
Mean Power (dB/Hz) -> mean_power_db  Tonality -> tonality
Peak Frequency (kHz) -> peak_freq_hz
```

### Merged CSV Output Schema

| Source | Columns |
|--------|---------|
| DeepSqueak | All 15 normalized columns (id through peak_freq_hz) |
| Detection JSON | det_start_s, det_end_s, det_duration_ms, det_index, det_prob_max, det_prob_mean, det_user_action, det_json_path |
| Merge metadata | wav_stem, source_file, match_quality (exact/fuzzy/unmatched_ds/unmatched_det), match_distance_ms |

## Matching Algorithm

Greedy 1:1 nearest-neighbor matching by WAV stem:

1. Group DS results and detections by WAV stem
2. For each DS call, find detection with minimum `|ds.begin_time_s - det.start_s|`
3. Accept if distance <= `tolerance_ms / 1000`; exact if distance == 0.0, fuzzy otherwise
4. Once matched, remove detection from candidate pool (greedy 1:1)
5. Report unmatched from both sides

**Why greedy?** DeepSqueak reads our Raven tables, so there should be a 1:1 correspondence. Greedy matching prevents double-assignment.

**Default tolerance: 5.0 ms.** Conservative — timestamps should be near-identical since DS reads the Raven tables we exported.

## CLI Usage

```bash
# Standard import
python scripts/import_deepsqueak_results.py \
    --results-dir deepsqueak_output \
    --detections-dir USV_Detections \
    --output classified_detections.csv

# Tighter tolerance
python scripts/import_deepsqueak_results.py \
    --results-dir deepsqueak_output \
    --detections-dir USV_Detections \
    --output classified_detections.csv \
    --tolerance 2.0

# Dry run (match without writing files)
python scripts/import_deepsqueak_results.py \
    --results-dir deepsqueak_output \
    --detections-dir USV_Detections \
    --output classified_detections.csv \
    --dry-run -v
```

## Key Decisions

- **Separate `_load_detection_json_extended`**: Extracts additional fields (probabilities, index, user_action) beyond what `load_detection_json` returns, without modifying the existing function.
- **Snake_case normalization**: Consistent with Python conventions; `_COLUMN_MAP` handles known DS columns, fallback handles unknown ones.
- **`match_quality` column**: Enables downstream filtering — keep only "exact" and "fuzzy" for analysis, investigate "unmatched_*" rows separately.
- **WAV stem extraction with suffix stripping**: Handles DeepSqueak's convention of appending `_Detections`, `_calls`, etc. to filenames.

## Integration Points

- **Reads from:** DeepSqueak Excel output (.xlsx), `DetectionExporter` output (detection JSONs)
- **Reuses:** `_is_detection_json` from `raven_export.py` for filtering non-detection files
- **Feeds into:** Repertoire statistics (Phase 14.3), population comparison analysis
- **Dependencies:** `pandas`, `openpyxl` (Excel reading), standard library
