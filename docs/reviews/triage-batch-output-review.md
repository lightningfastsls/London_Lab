# Recording-Level Triage and Batch Output Module Review

**Date:** 2026-03-28
**Reviewer:** master-reviewer
**Review Tier:** 2 (standard)
**ROADMAP Section:** 15.7
**Handoff:** `docs/reviews/triage-batch-output-handoff.md`

## Verdict: CHANGES NEEDED → FIXES APPLIED

Three blockers were identified and resolved:

### BLOCKER 1 (FIXED): Second-Pass Triage Corrupted Tier Assignments

**Problem:** `run_batch_detection.py` re-called `triage_recording()` with `np.array([r.noise_floor_p90])` as probabilities, causing quiet recordings with valid events to be re-triaged as `auto_reject`.

**Fix:** Replaced re-triage with direct mutation of `qc_flags` on existing results. Also switched to `ddof=1` (sample std) per SUGGESTION 1. File: `scripts/run_batch_detection.py:215-232`.

### BLOCKER 2 (FIXED): Per-Recording JSONs Missing `start_col`/`end_col`

**Problem:** Desktop app's `label_storage.py:150` accesses `detection_dict["start_col"]` with no fallback — would crash on batch output JSONs.

**Fix:** Added `start_col` and `end_col` to `_event_to_adr010_dict()`, computed from `event.start_window * hop_px`. File: `batch_output.py:31-45`.

### BLOCKER 3 (FIXED): `confidence_score` Contradicted ROADMAP Spec

**Problem:** Implementation used `max_confidence`; ROADMAP Resolved Ambiguity #5 mandates `mean_event_confidence`.

**Fix:** Changed `confidence_score = mean_event_confidence` in `triage.py:187`.

### WARNING 1 (FIXED): Empty Events Not Routed to `auto_reject`

**Problem:** ROADMAP Resolved Ambiguity #6 says empty events → `auto_reject`. Code only reached `auto_reject` if `max(probs) <= 0.10`.

**Fix:** Added explicit `n_events == 0` check before probability-based tier logic. File: `triage.py:176-186`.

## Test Results After Fixes

- 19/19 pre-existing tests pass
- No test expectations modified
- Full suite: 811 passed, 5 failed (pre-existing), 5 skipped

## Re-Review Required

Blockers required re-review per protocol. See below.
