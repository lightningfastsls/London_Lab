# Task Brief: Investigate pq6 app missing detection lines

## Goal
Create a clear, reproducible diagnosis plan and evidence trail to explain why the pq6 app shows **zero detections** (and therefore no vertical detection lines), especially after the 2026-02-02 updates. The output should identify the most likely root cause(s) and where in the pipeline detections are being lost.

## Scope
- Reproduce the "no detections" behavior using the pq6 app on any WAV file.
- Confirm whether detections are produced in non-UI pipelines (CLI or direct inference) using the same model and thresholds.
- Trace pq6 app configuration: model path, threshold values, and inference preprocessing (padding, normalization).
- Trace the UI drawing path to confirm whether detections are arriving but not rendered vs. not produced.
- Collect minimal logs/prints where needed to prove the break point.
- Summarize findings and propose next steps or minimal fixes.

## Non-Scope
- Retraining models or changing training data.
- Major UI refactors or feature additions.
- Changing public APIs or adding dependencies.
- Performance optimizations unrelated to the missing detections.

## Constraints
- Follow AGENTS.md rules (no scope creep, small diffs, no fabricated results).
- Ask before changing public APIs or adding dependencies.
- Use the real input data location via `USV_WAV_DIR` or default `<repo_root>/5970 USV`.
- If adding temporary logging, keep it minimal and scoped; remove before final if not requested to keep.

## Assumptions
- pq6 app uses the CNN inference pipeline and the 2026-02-02 model/threshold updates.
- The issue is new or became noticeable after the 2026-02-02 changes (model path + thresholds + padding fix).
- Missing detections imply either inference returns zero hits or UI does not render them.

## Acceptance Criteria
- A written diagnosis that pinpoints where detections drop to zero (model load, preprocessing, inference, thresholding, or UI render).
- Evidence for the diagnosis (logs, outputs, or code references).
- A minimal next-step recommendation (e.g., config change, threshold update, preprocessing fix, or UI toggle).

## Files to Inspect / Potentially Touch
- `src/usv_spectrogram/app/main_window.py` (pq6 app config, thresholds, UI flow)
- `src/usv_spectrogram/app/core/sliding_inference.py` (padding, windowing, preprocessing)
- `src/usv_spectrogram/models/cnn_classifier.py` (model loading, default threshold)
- `scripts/test_detection_backend.py` (sanity check pipeline)
- `scripts/diagnose_cnn_batch_detection.py` (probability distribution checks)

## Plan (Small Diffs)
### Stage 1 — Reproduce + Baseline Checks
1. Run pq6 app on any WAV and confirm zero detections.
2. Run CLI sanity checks (e.g., `scripts/test_detection_backend.py`) on the same WAV to see if detections are produced outside UI.
3. Record outputs (detection counts and basic probability ranges if available).

### Stage 2 — Configuration & Model Load Verification
1. Confirm the pq6 app is loading `models/production/best_model.pt`.
2. Confirm thresholds in pq6 app (expect low=0.05, high=0.10 per 2026-02-02).
3. Confirm the model file exists and loads without errors.

### Stage 3 — Inference Preprocessing Trace
1. Verify inference windows are padded to 512px (per 2026-02-02 fix).
2. Confirm normalization/mode matches training (`render_mode="training"`, consistent preprocessing).
3. If needed, add minimal logging around preprocessing and inference to confirm tensor shapes and score ranges.

### Stage 4 — UI Rendering Path
1. Trace how detections are passed to the UI layer and plotted.
2. Confirm the "show detections" toggle or equivalent is enabled.
3. Verify time-to-pixel mapping is valid (sample rate, time axis, spectrogram width).

### Stage 5 — Findings & Recommendation
1. Summarize the exact break point with evidence.
2. Recommend a minimal fix or next step (config, threshold, preprocessing, or UI).
3. If no clear root cause, list remaining hypotheses and targeted follow-ups.

## Handoff Note
Implementer: read this task brief and document findings in `10_impl_notes.md`. Only proceed to code changes if the brief explicitly calls for it.
