# Implementation Handoff: Training Data Assembly Pipeline (Phase 9.1)

**Module:** Unified Dataset Assembly
**Review Tier:** 2
**Date:** 2026-02-21
**Branch:** main

## What Changed

- Built a unified training data assembly pipeline (`DatasetAssembler`) that replaces the manual multi-script workflow (`generate_comprehensive_negatives.py` -> `create_full_training_dataset.py` -> manual combination)
- Collects labels from LabelStorage JSON files (ADR-010), creates positive candidates with jitter augmentation, generates negatives from 3 sources (ADR-008: random, inter-USV gap, low-energy), extracts training spectrograms, splits by recording (ADR-004), validates quality, and writes train/val/test CSVs
- Created CLI entry point following Pattern 4 (script with argparse, `--dry-run` support)
- All DSP parameters match ADR-001 (sr=300000) and ADR-002 (n_fft=512, hop=128, Hann window)
- 8 tests covering assembly, leakage prevention, negative sources, jitter counts, dry-run, report accuracy, error handling, and spectrogram file existence

## Files Changed

- `src/usv_spectrogram/dataset/assembler.py` (NEW) -- Core assembly logic: `AssemblyConfig`, `AssemblyReport`, `DatasetAssembler` (~480 lines)
- `scripts/assemble_training_data.py` (NEW) -- CLI entry point with argparse (~85 lines)
- `tests/test_dataset_assembler.py` (NEW) -- 8 test cases with synthetic WAV/JSON fixtures (~270 lines)
- `src/usv_spectrogram/dataset/__init__.py` (MODIFIED) -- Added exports for `AssemblyConfig`, `AssemblyReport`, `DatasetAssembler`

## Key Decisions Made

1. **JSON parsed directly, no LabelStorage import.** The `LabelStorage` class lives in `app/core/` which depends on PyQt6. Instead of importing it, we `json.load()` the files directly and extract `metadata.wav_file` and `detections[]`. This avoids coupling the data pipeline to the desktop app's GUI dependencies.

2. **Largest-remainder allocation for negative distribution.** Using `round()` for proportional allocation across 3 negative sources starved the smallest category at small counts (e.g., 3 negatives: round(1.5)=2 random + round(0.9)=1 gap + 0 low-energy). Switched to Hamilton's method (floor + distribute remainder by largest fractional part) which guarantees fair allocation.

3. **Frame-level detection buffer masking for low-energy negatives.** Originally rejected entire contiguous low-energy regions if they overlapped any detection's 50ms buffer. This was too aggressive — silent regions naturally border detections, so almost every region was rejected. Fixed by masking out detection buffer zones at the STFT frame level *before* grouping frames into regions, so eligible regions are safe by construction.

4. **Source file stored with .wav extension.** The `Candidate.source_file` field stores the full WAV filename (e.g., `rec_001.wav`) matching the existing pipeline convention. The `SpectrogramExtractor.extract_single()` looks up `wav_dir / candidate.source_file.name`, so the extension must be present. Candidate IDs use the stem (no extension) for readability.

5. **Jitter before split.** Augmented (jittered) positives are created before splitting by recording. This ensures jittered versions stay with their parent recording, preventing subtle data leakage where the same USV appears in different augmented forms across train/test splits.

## What I'm Unsure About

- **Jitter min_overlap constraint with long USVs.** Jitter is impossible when `usv_duration >= jitter_window_ms / (2 * jitter_min_overlap)`. With defaults (40ms window, 0.5 overlap), USVs >= 40ms get no augmentation. This is handled gracefully (returns empty list) and documented in `_jitter_candidate()`. May need a larger default window or adaptive window sizing at scale.
- **Low-energy 20th percentile threshold.** This threshold works for synthetic test data but may need tuning for real recordings with varying noise floors. The threshold is hardcoded; could be made configurable.
- **`soundfile.info()` called per-detection in `_create_positive_candidates`.** For recordings with many detections, this reads WAV metadata repeatedly for the same file. Could be optimized with a cache, but kept simple for now since it's just metadata (fast I/O).

## Test Results

```
.\.venv\Scripts\python.exe -m pytest tests/test_dataset_assembler.py -v
8 passed in 6.13s

.\.venv\Scripts\python.exe -m pytest tests/ -v
432 passed, 0 failed in 17.72s
```

## ROADMAP Exit Criteria Status

- [ ] `assemble_training_data.py --dry-run` runs without error on real label data (needs real data)
- [x] Full assembly produces train.csv, val.csv, test.csv in correct format
- [x] Quality checks pass: no leakage, acceptable class balance, all files exist
- [ ] Output can be fed directly to `train_cnn.py` and training starts successfully (needs real data)
- [x] All tests pass (432/432)
- [x] py_compile passes on all new files

## Docs Written/Updated

- `docs/reviews/dataset-assembler-handoff.md` -- this file
- `IMPLEMENTATION_PROGRESS.md` -- not yet updated (pending review)
- `DECISIONS.md` -- no new ADRs needed (uses existing ADR-001, 002, 004, 008, 010)
- `docs/architecture/patterns.md` -- no new patterns needed (follows existing Pattern 4)
