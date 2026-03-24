# Dataset Assembler

**Phase:** 9.1 (Training Data Assembly Pipeline)
**ADRs:** ADR-001 (Sample Rate), ADR-002 (STFT Parameters), ADR-004 (Recording-Based Splits), ADR-008 (3-Source Negatives), ADR-010 (LabelStorage JSON Format)
**Tests:** `tests/test_dataset_assembler.py` -- 10 tests across 10 test classes

## Purpose

The DatasetAssembler unifies the multi-script training data preparation workflow into a single reproducible pipeline. It reads label JSON files from the desktop detection app, generates positive candidates with jitter augmentation, creates negatives from 3 sources (ADR-008), extracts training spectrograms, splits by recording (ADR-004), validates quality, and writes train/val/test CSVs ready for `train_cnn.py`.

Replaces the manual workflow: `generate_comprehensive_negatives.py` -> `create_full_training_dataset.py` -> manual combination.

## Public Interface

### `AssemblyConfig`

```python
@dataclass(frozen=True)
class AssemblyConfig:
    labels_dir: Path            # Dir with LabelStorage JSON files
    wav_dir: Path               # WAV file directory
    jitter_n_samples: int = 5   # Jittered versions per positive
    jitter_window_ms: float = 40.0
    jitter_context_padding_ms: float = 20.0
    jitter_min_overlap: float = 0.5
    neg_random_frac: float = 0.5
    neg_inter_usv_frac: float = 0.3
    neg_low_energy_frac: float = 0.2
    neg_ratio: float = 1.0     # Negatives per positive (soft target)
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    output_dir: Path = Path("data/training/assembled")
```

Validates: split ratios sum to 1.0, negative fractions sum to 1.0, all parameters positive.

### `AssemblyReport`

```python
@dataclass
class AssemblyReport:
    total_positives: int    # Includes originals + jittered
    total_negatives: int    # From all 3 sources
    train_count: int
    val_count: int
    test_count: int
    n_recordings: int
    warnings: list[str]     # Failed quality checks
    output_dir: Path
```

### `DatasetAssembler`

```python
class DatasetAssembler:
    def __init__(self, config: AssemblyConfig): ...

    def assemble(self, dry_run: bool = False) -> AssemblyReport:
        """Run full pipeline: collect -> jitter -> negate -> extract -> split -> validate -> write.
        With dry_run=True, computes statistics without writing files."""
```

### CLI Entry Point

```
python scripts/assemble_training_data.py \
    --labels-dir USV_Detections \
    --wav-dir "5970 USV" \
    --output-dir data/training/milestone_1 \
    --jitter-samples 5 \
    --neg-ratio 1.0 \
    --dry-run
```

## Key Decisions

1. **Direct JSON parsing (no LabelStorage import).** Avoids coupling to PyQt6 app dependencies. Reads `metadata.wav_file` and `detections[]` with `json.load()` directly. Filters out `user_action == "deleted_by_user"` detections.

2. **Hamilton's method for negative allocation.** Proportional distribution across 3 negative types uses largest-remainder allocation instead of `round()` to prevent starving the smallest category at low counts.

3. **Frame-level detection buffer masking.** Low-energy negatives mask out detection buffer zones (50ms) at the STFT frame level *before* grouping frames into contiguous regions. This avoids rejecting entire regions that merely border a detection.

4. **Jitter before split.** Augmented positives are created before recording-based splitting, ensuring all variants of a USV stay in the same split (no leakage).

5. **Jitter threshold formula.** Jitter is impossible when `usv_duration >= jitter_window_ms / (2 * jitter_min_overlap)`. With defaults (40ms window, 0.5 overlap), USVs >= 40ms silently receive no jitter augmentation.

6. **neg_ratio is a soft target.** The `max(1, ...)` floor per recording ensures every recording gets at least one negative, which can cause the actual negative count to overshoot `neg_ratio * n_positives` by up to ~40% with many low-detection recordings. This is logged when overshoot exceeds 20%.

7. **Sample rate validation.** Low-energy negatives skip recordings with `sr != 300000` (returns early) rather than computing STFT with wrong parameters that would produce mis-timed candidates.

## Output Structure

```
data/training/assembled/
  spectrograms/          # All spectrogram PNGs (positives + negatives)
  train.csv              # candidate_id, source_file, label, spectrogram_path
  val.csv
  test.csv
  assembly_report.json   # Statistics, config, timestamp
```

## DSP Parameters

All ADR-002 compliant: `sr=300000`, `n_fft=512`, `hop_length=128`, Hann window. Low-energy analysis uses STFT in 20-120 kHz band with 20th percentile threshold.
