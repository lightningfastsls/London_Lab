# Phase 8.1: USV Bout Data Pipeline — Handoff

**Date:** 2026-02-18
**Tier:** 2 — Standard (new module, multi-file)

## Summary

Implemented the v2 data preparation pipeline that converts CNN detection results into PyTorch-ready datasets for the autoregressive transformer (Phase 8.2). This is the data foundation for the v2 architecture (ADR-007).

## Files Created (11 total)

### Source (6 files)
| File | Lines | Purpose |
|------|-------|---------|
| `usv_language/data/__init__.py` | 42 | Package init, public API exports |
| `usv_language/data/bout_extractor.py` | 248 | Bout extraction from detection results |
| `usv_language/data/spectrogram.py` | 136 | STFT spectrogram via _stft_core.py |
| `usv_language/data/normalization.py` | 126 | Per-frequency normalization (Welford's) |
| `usv_language/data/dataset.py` | 331 | PyTorch dataset, bucketed sampler, augmentation |
| `usv_language/data/prepare_data.py` | 199 | CLI end-to-end pipeline |

### Config (1 file)
| File | Purpose |
|------|---------|
| `usv_language/configs/default_config.yaml` | Master YAML config reference |

### Tests (4 files, 56 tests)
| File | Tests | Coverage |
|------|-------|----------|
| `test_bout_extractor.py` | 15 | Config, grouping, splitting, CSV/JSON parsing |
| `test_bout_spectrogram.py` | 12 | Shape, freq bins, short audio, dB range, dtype |
| `test_bout_normalization.py` | 8 | Stats, save/load, mean~0/std~1, edge cases |
| `test_bout_dataset.py` | 21 | Chunking, masking, next-col, batching, splits |

## Architectural Decisions

1. **Reuses `_stft_core.py`** — exact same STFT as CNN detection pipeline for consistency
2. **Recording-based splits** (ADR-004) — prevents data leakage between train/val/test
3. **Welford's online algorithm** — numerically stable incremental mean/variance
4. **Bucketed batch sampler** — 6 length buckets (64–512) minimize padding waste
5. **Lazy audio loading** — bout extraction returns metadata, audio loaded only during spectrogram computation
6. **Next-column prediction** — input=frames[:-1], target=frames[1:], matching the autoregressive transformer design

## Test Results

```
56 passed in 12.30s (module tests)
351 passed in 20.97s (full suite, 0 regressions)
```

## Known Limitations / Future Work

1. **No HDF5 output** — saves individual .npy files + JSON manifest (sufficient for v2 Phase 8.2)
2. **No multi-GPU DataLoader** — num_workers configurable but DistributedSampler not yet added
3. **Augmentation in dB domain** — gain is additive in dB which is correct; Gaussian noise is approximate
4. **prepare_data.py untested end-to-end** — would need real WAV files; individual components all tested

## Review Focus Areas

1. DSP correctness of `spectrogram.py` (reuses _stft_core — should be safe)
2. Bout splitting algorithm for edge cases (recursive split on oversized bouts)
3. Chunking overlap behavior near bout boundaries
