# Handoff: Recording-Level Triage and Batch Output (ROADMAP 15.7)

**Date:** 2026-03-28
**Review Tier:** 2 (standard)
**ROADMAP Section:** 15.7

## Files Created

| File | Purpose |
|------|---------|
| `src/usv_spectrogram/postprocessing/triage.py` | TriageConfig, RecordingResult, triage_recording() |
| `src/usv_spectrogram/postprocessing/batch_output.py` | write_batch_results() — parquet + JSON output |
| `scripts/run_batch_detection.py` | CLI for full batch pipeline |
| `docs/modules/recording-triage.md` | Module documentation |

## Files Modified

| File | Change |
|------|--------|
| `src/usv_spectrogram/postprocessing/__init__.py` | Added exports for triage + batch_output |
| `requirements.txt` | Added pyarrow |

## Implementation Decisions

1. **Tier precedence**: auto_reject first (max(probs) <= 0.10), then auto_accept (all peaks >= 0.90), then manual_review. This prevents edge cases where a noisy recording with a few high peaks gets auto-accepted.

2. **noise_floor_p90_threshold**: Added to TriageConfig (not in original ROADMAP spec) with default 0.4. Required for the noise floor QC flag logic that the pre-existing tests expect.

3. **mean_event_confidence**: Computed as mean of per-event `peak_probability` (not `mean_probability`), confirmed by test assertion at line 488.

4. **confidence_score**: Set to `max_confidence` as simplest meaningful metric. The ROADMAP notes this will need recalibration after the first batch run.

5. **Two-pass triage in batch script**: First pass computes per-recording metrics, second pass adds batch-level outlier flagging with computed mean/std.

## Test Results

- **Pre-existing tests from test-architect:** 19 (all pass)
- **Test modifications:** None (zero changes to test expectations)
- **Full suite:** 811 passed, 5 failed (pre-existing deepsqueak_import failures), 5 skipped

## Dependencies

- **Upstream:** USVEvent (hysteresis.py), probabilities (SlidingInference)
- **Downstream:** Desktop app (reads per-recording JSON), batch analysis (reads parquet)
- **New dependency:** pyarrow (for pandas parquet I/O)

## Known Limitations

- Triage thresholds (0.90/0.10/0.4) are initial defaults; recalibration expected after first real batch
- Batch script re-triage uses `np.array([r.noise_floor_p90])` as placeholder probabilities for the second pass (QC metrics are preserved from first pass)
- No parallelism in batch script yet (sequential processing)

## Fixes Applied (Post-Review)

1. **BLOCKER 1**: Replaced re-triage second pass with direct `qc_flags` mutation (`run_batch_detection.py:215-232`). Also switched to `ddof=1` sample std.
2. **BLOCKER 2**: Added `start_col`/`end_col` to `_event_to_adr010_dict()` (`batch_output.py:31-45`). Computed from `event.start_window * hop_px`.
3. **BLOCKER 3**: Changed `confidence_score = mean_event_confidence` (`triage.py:187`). Per ROADMAP Resolved Ambiguity #5.
4. **WARNING 1**: Added `n_events == 0 → auto_reject` guard (`triage.py:176-178`). Per ROADMAP Resolved Ambiguity #6.

## Test Counts

- Pre-existing tests from test-architect: 19
- Hardener tests: 44 (in `tests/test_triage_hardened.py`)
- Total: 63 (all pass)
- Bugs found by hardener: 0

## Risks

- pyarrow adds ~50MB to the environment; could use fastparquet as lighter alternative if needed
- Latent gap: `triage_recording()` with empty probabilities array would raise on `np.percentile`. Architecturally abnormal (SlidingInference always produces ≥1 window).
