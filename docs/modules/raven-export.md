# Raven Selection Table Export

**Phase:** 14.1 (Raven Selection Table Export Adapter)
**ADRs:** None (pure format conversion, no DSP parameters involved)
**Tests:** `tests/test_classification/test_raven_export.py` -- 33 tests across 9 test classes

## Purpose

Converts individual detection JSON files (produced by the USV detection app's `DetectionExporter`) into Raven Pro selection table format -- the standard bioacoustics annotation interchange format. Output files can be opened directly in Raven Pro, imported into DeepSqueak for batch classification, or used with Audacity's label import.

The module bridges the gap between our custom detection pipeline and established bioacoustics tools, enabling the syllable classification workflow.

## Public Interface

### `RavenExportConfig`

```python
@dataclass(frozen=True)
class RavenExportConfig:
    detections_dir: Path          # USV_Detections/ root
    wav_dir: Path                 # Source WAV file directory
    output_dir: Path              # Where to write .txt files
    low_freq_hz: float = 25_000  # Mouse USV band lower bound (Hz)
    high_freq_hz: float = 125_000 # Mouse USV band upper bound (Hz)
```

Validates: low < high, non-negative frequencies, auto-converts string paths to `Path`.

### `ExportSummary`

```python
@dataclass
class ExportSummary:
    total_wav_files: int          # WAVs that had detections
    total_detections: int         # Total detection count across all WAVs
    total_tables_written: int     # Number of .txt files created
    unmapped_dirs: list[str]      # Detection dirs with no matching WAV
    empty_detection_dirs: list[str]  # Matched WAV but zero detection JSONs
    per_wav_counts: dict[str, int]   # {wav_stem: detection_count}
```

### Core Functions

| Function | Signature | Purpose |
|----------|-----------|---------|
| `load_detection_json` | `(Path) -> dict` | Load one detection JSON, extract `core_time` fields |
| `discover_wav_detection_mapping` | `(Path, Path) -> dict[str, list[Path]]` | Map detection subdirectories to WAV stems |
| `detections_to_raven_table` | `(list[dict], float, float) -> pd.DataFrame` | Convert detections to Raven-format DataFrame |
| `export_raven_tables` | `(RavenExportConfig) -> list[Path]` | Full pipeline: discover, convert, write TSV + summary |

### Detection JSON Format (input)

Only `core_time` fields are read (not `saved_region` which includes context padding):

```json
{
  "core_time": {
    "start_s": 0.14165,
    "end_s": 0.20139,
    "duration_ms": 59.73
  }
}
```

### Raven Selection Table Format (output)

Tab-delimited `.txt` with 7 columns:

```
Selection	View	Channel	Begin Time (s)	End Time (s)	Low Freq (Hz)	High Freq (Hz)
1	Spectrogram 1	1	0.1417	0.2014	25000	125000
```

Output naming: `{wav_stem}.Table.1.selections.txt` (Raven convention).

## CLI Usage

```bash
# Standard export
python scripts/export_raven_tables.py \
    --detections-dir USV_Detections \
    --wav-dir "5970 USV" \
    --output-dir raven_tables

# Dry run (mapping + counts only)
python scripts/export_raven_tables.py \
    --detections-dir USV_Detections \
    --wav-dir "5970 USV" \
    --dry-run -v
```

## Key Decisions

- **`core_time` over `saved_region`**: Raven needs actual USV boundaries, not the padded extraction window.
- **Fixed frequency bounds**: Per-syllable frequency extraction isn't available; the full 25-125 kHz band is written for every row.
- **One table per WAV**: Follows Raven's convention for associating selection tables with sound files.
- **Directory name = WAV stem**: Supports both exact match and prefix match (longest stem wins).

## Integration Points

- **Reads from:** `DetectionExporter` output (`detection_exporter.py:202-242`)
- **Feeds into:** DeepSqueak batch classification, Raven Pro visualization
- **Dependencies:** `pandas` (DataFrame + TSV writing), standard library only otherwise
