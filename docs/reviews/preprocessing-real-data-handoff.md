# Implementation Handoff: Phase 11.1 — Bout Extraction & Preprocessing on Real Data

**Module:** Phase 11.1 — Real Data Preprocessing
**Review Tier:** 1 (Housekeeping)
**Date:** 2026-02-22
**Branch:** main

## What Changed

- Fixed `_parse_tracking_json()` and `extract_from_tracking_json()` in `bout_extractor.py` to filter out records where `user_action == "deleted_by_user"`, preventing rejected detections from leaking into training data
- Added recursive WAV path resolution with lazy-built index (`_build_wav_index` + `_resolve_wav_path` fast/slow path) for future nested WAV directory support
- Created `validate_preprocessing.py` validation script with 4 checks: directory structure, sample shapes/dtypes, normalization stats, optional DataLoader test and plotting
- Added 9 new tests: 5 for tracking JSON filtering, 4 for recursive WAV lookup
- Updated `.gitignore` to exclude `usv_language/prepared_data/`
- Ran full pipeline on real `5970 USV/` data with `USV_Detections/5970/` detection results

## Files Changed

- `usv_language/data/bout_extractor.py` (MODIFIED) — filtering + recursive WAV resolution
- `usv_language/scripts/validate_preprocessing.py` (NEW) — validation CLI
- `usv_language/tests/test_bout_extractor.py` (MODIFIED) — 9 new tests
- `.gitignore` (MODIFIED) — added `usv_language/prepared_data/`

## Key Decisions Made

- Filtering uses `r.get("user_action") != "deleted_by_user"` — records without this field (the common case) pass through. Only explicitly deleted records are excluded.
- Recursive index is lazy-built on first cache miss — for flat directories (current `5970 USV/`), `rglob` is never called.
- Validation script follows existing `validate_shapes.py` pattern with `sys.exit(0/1)` convention.

## What I'm Unsure About

- Nothing significant. The code changes are minimal and well-tested.

## Test Results

```
184 passed, 0 failed (24 bout_extractor tests, 9 new)
```

## Pipeline Run Results

```
Step 1: 124 bouts extracted (2 deleted_by_user records correctly filtered)
Step 2: 27 valid spectrograms (97 skipped — WAV files not present on disk)
Step 3: 8 train / 1 val / 1 test recordings (19/4/4 spectrograms)
Step 4: 103,089 frames for normalization stats
Step 5: Saved to usv_language/prepared_data/

Validation: ALL CHECKS PASSED
- Shape (170, T), float32, all finite
- Normalization stats: mean/std (170,), count=103,089, all std > 0
- DataLoader: batch (4, 255, 170) from 813 dataset items
- 10 sample plots saved
```

## ROADMAP Exit Criteria Status

- [x] Bug fix: deleted_by_user filtering
- [x] Feature: recursive WAV path resolution
- [x] Validation script created and passing
- [x] Tests written and passing (184 total)
- [x] Pipeline run on real data with valid output
- [x] .gitignore updated

## Docs Written/Updated

- `docs/reviews/preprocessing-real-data-handoff.md` — this file
- `.gitignore` — updated
- No new ADRs needed (all DSP parameters unchanged)
