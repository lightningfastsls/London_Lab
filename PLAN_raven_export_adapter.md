# PLAN: Raven Selection Table Export Adapter for DeepSqueak Classification

## Goal
Build a Python module that converts our existing CNN detection JSONs into Raven selection table format (.txt), enabling batch syllable classification in DeepSqueak without re-running detection. Then build a results ingestion module that reads DeepSqueak's Excel output back into our pipeline for statistical analysis.

## Context
- We have ~840 positive USV detections across ~200-300 WAV files for the 5970 mouse group
- Detections are stored as individual JSON files (one per detection) with `core_time.start_s`, `core_time.end_s`, and `core_time.duration_ms`
- JSONs do NOT contain: source WAV filename, frequency bounds
- WAV files are recorded at 300 kHz sample rate
- The file/folder structure linking JSONs to WAVs needs to be discovered by exploring the project directory

## Phase 1: Discovery — Understand the file structure

1. Explore the project directory to find:
   - Where WAV files live (likely under a `5970 USV/` or similar folder)
   - Where detection JSONs live relative to their source WAVs
   - The naming convention that links JSONs to WAVs (could be: subfolder per WAV, filename prefix, a manifest file, or the `session_id` field)
   - Any existing index/manifest that maps detections to recordings
2. Look at the existing codebase for how detections are currently loaded — there's likely already a function that resolves JSON→WAV mappings
3. Check if there's a labels CSV or similar file that aggregates detection info
4. Document findings before proceeding

## Phase 2: Build the Raven Export Module

### Location: `src/usv_spectrogram/classification/raven_export.py`

### Raven Selection Table Format
Tab-separated `.txt` file with these columns:
```
Selection	View	Channel	Begin Time (s)	End Time (s)	Low Freq (Hz)	High Freq (Hz)
1	Spectrogram 1	1	1.7006	1.7420	25000	125000
2	Spectrogram 1	1	3.5020	3.5580	25000	125000
```

### Key decisions:
- **Frequency bounds**: Use fixed range 25000–125000 Hz (standard mouse USV band). DeepSqueak regenerates its own spectrograms from audio, so the bounding box just needs to be a reasonable region of interest. We can refine later if needed.
- **Time values**: Use `core_time.start_s` and `core_time.end_s` from each JSON (NOT `saved_region` which includes context padding)
- **One Raven file per WAV**: Each `.txt` file corresponds to one WAV file and contains all detections from that recording
- **Naming**: `{wav_filename_without_extension}.Table.1.selections.txt` (Raven convention)

### Functions to implement:
```python
def load_detection_json(json_path: Path) -> dict:
    """Load a single detection JSON and extract core fields."""

def discover_wav_detection_mapping(data_dir: Path) -> dict[Path, list[Path]]:
    """Walk the directory tree and map each WAV to its detection JSONs.
    Return {wav_path: [json_path1, json_path2, ...]}
    Must handle whatever directory structure exists."""

def detections_to_raven_table(
    detections: list[dict],
    low_freq_hz: float = 25000,
    high_freq_hz: float = 125000,
) -> pd.DataFrame:
    """Convert a list of detection dicts to Raven selection table DataFrame."""

def export_raven_tables(
    wav_detection_map: dict[Path, list[Path]],
    output_dir: Path,
    low_freq_hz: float = 25000,
    high_freq_hz: float = 125000,
) -> list[Path]:
    """Export one Raven .txt per WAV file. Return list of created files."""

def export_all(data_dir: Path, output_dir: Path) -> None:
    """Main entry point: discover mappings, export all Raven tables."""
```

### CLI entry point:
```bash
python -m usv_spectrogram.classification.raven_export --data-dir ./5970_USV --output-dir ./raven_tables
```

## Phase 3: Build the DeepSqueak Results Ingestion Module

### Location: `src/usv_spectrogram/classification/deepsqueak_import.py`

After running DeepSqueak classification (done manually in MATLAB), the user will have Excel files with per-call classification labels. This module reads them back.

### DeepSqueak Excel output columns (16 fields):
- ID, Label/Type, Begin Time, End Time, Call Length, Principal Frequency
- Low Freq, High Freq, Bandwidth, Freq Std Dev, Slope, Sinuosity
- Mean Power, Tonality, Peak Frequency

### Functions to implement:
```python
def load_deepsqueak_excel(excel_path: Path) -> pd.DataFrame:
    """Load a DeepSqueak Excel export into a standardized DataFrame."""

def load_all_deepsqueak_results(results_dir: Path) -> pd.DataFrame:
    """Load all Excel files from a directory, adding source_file column."""

def merge_with_detections(
    ds_results: pd.DataFrame,
    wav_detection_map: dict,
    tolerance_ms: float = 5.0,
) -> pd.DataFrame:
    """Match DeepSqueak classifications back to original detection JSONs
    using timestamp proximity matching (within tolerance_ms)."""

def syllable_repertoire_summary(
    classified_df: pd.DataFrame,
    group_col: str = "population",  # 'wild' vs 'lab'
) -> pd.DataFrame:
    """Compute per-animal and per-group syllable type proportions,
    counts, Shannon entropy, and transition matrices."""
```

## Phase 4: Statistical Analysis Module

### Location: `src/usv_spectrogram/classification/repertoire_stats.py`

```python
def syllable_proportions(classified_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Per-animal syllable type proportions."""

def compare_repertoires(
    wild_df: pd.DataFrame,
    lab_df: pd.DataFrame,
    method: str = "permanova",  # or "chi_square", "jsd"
) -> dict:
    """Compare syllable repertoire distributions between populations."""

def transition_matrix(classified_df: pd.DataFrame, animal_id: str) -> np.ndarray:
    """Compute syllable-to-syllable transition probability matrix for one animal."""

def compare_transition_matrices(wild_matrices: list, lab_matrices: list) -> dict:
    """Compare transition structure between populations using permutation test."""

def plot_repertoire_comparison(wild_df, lab_df, output_path: Path) -> None:
    """Generate publication-ready figures: stacked bar charts of syllable proportions,
    heatmaps of transition matrices, etc."""
```

## Phase 5: Tests

### Location: `tests/test_classification/`

- `test_raven_export.py`:
  - Test Raven table format compliance (correct columns, tab-separated, proper header)
  - Test with synthetic detection JSONs
  - Test frequency bounds are in Hz (not kHz)
  - Test one-file-per-WAV output
  - Test CLI entry point

- `test_deepsqueak_import.py`:
  - Test Excel loading with mock DeepSqueak output
  - Test timestamp matching with tolerance
  - Test handling of unmatched detections

- `test_repertoire_stats.py`:
  - Test syllable proportion computation
  - Test PERMANOVA with known-different distributions
  - Test transition matrix is row-stochastic

## Implementation Order
1. Phase 1 (Discovery) — MUST happen first, blocks everything
2. Phase 2 (Raven export) — the critical deliverable
3. Phase 5 tests for Phase 2
4. Phase 3 (DeepSqueak import) — can be stubbed with mock data
5. Phase 4 (Stats) — can proceed with mock classified data
6. Phase 5 remaining tests

## Important Notes
- Follow existing project conventions: check pyproject.toml, existing test structure, import patterns
- Use pathlib throughout, no os.path
- All functions should have type hints and docstrings
- The Raven export is the URGENT deliverable — the import and stats modules can wait
- Don't assume the directory structure; discover it programmatically in Phase 1
- If you can't find the WAV↔JSON mapping, STOP and ask — don't guess
