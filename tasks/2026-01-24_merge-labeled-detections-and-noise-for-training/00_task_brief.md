# Task Brief

Goal:
Update the training inputs by merging completed noise review results with the labeled detections so training can proceed using the expected dataset files.

Context assumptions (explicit; label each assumption):
- Assumption: Training consumes `labels.csv` and `noise_samples/noise_samples_final.csv` via `scripts/prepare_dataset.py` (no need to merge into a single file).
- Assumption: Noise review statuses map as follows:
  - Clean -> keep as Not USV
  - Trimmed -> keep as Not USV, but use the trimmed candidate boundaries and `_trimmed` candidate_id
  - Contains USV -> exclude from noise set (not auto-promoted to USV)
  - Skip -> exclude
- Assumption: If a Trimmed sample is missing its `_trimmed.png`, we should skip it and report it.
- Assumption: Existing `labels.csv` does not need modification.

Scope:
- Create or update a script to build `noise_samples/noise_samples_final.csv` from:
  - `noise_samples/noise_samples.csv`
  - `noise_samples/noise_reviews.csv`
- Apply trim math for Trimmed entries and update `candidate_id`, `start_ms`, `end_ms`, `duration_ms`, `context_start_ms`, `context_end_ms`, and `spectrogram_path` accordingly.
- Write output to `noise_samples/noise_samples_final.csv` in the expected format.
- Provide a short console summary of counts per status and any skipped items.

Non-scope:
- Re-running detection or labeling tools.
- Regenerating spectrogram images (unless required for missing trimmed files).
- Training the model or recreating splits.
- Changing formats of `labels.csv` or `noise_samples.csv`.

Constraints (deps, API stability, style):
- No new dependencies.
- Keep diffs small.
- Do not change public APIs or existing file formats without approval.

Risks / unknowns:
- Unclear desired handling for "Contains USV" noise reviews; assumption is to exclude.
- Trimmed PNGs may be missing; decide whether to skip or regenerate.

Acceptance criteria (verifiable; include commands where possible):
- `noise_samples/noise_samples_final.csv` is regenerated with only Clean + Trimmed samples.
- Trimmed rows use `_trimmed` candidate_id and adjusted time bounds consistent with the review tool:
  - if trim_ms > 0: new_start = start_ms + trim_ms, new_end = end_ms
  - if trim_ms < 0: new_start = start_ms, new_end = end_ms + trim_ms
  - context_before = start_ms - context_start_ms; context_after = context_end_ms - end_ms
  - new_context_start = max(0, new_start - context_before); new_context_end = new_end + context_after
- Spectrogram path for trimmed rows points to `noise_samples/<candidate_id>_trimmed.png`.
- Script prints a summary of counts for Clean/Trimmed/Contains USV/Skip and any missing files.

Proposed file touch list:
- `scripts/build_noise_samples_final.py` (new) or update a nearby existing script
- `noise_samples/noise_samples_final.csv` (regenerated)
- `tasks/2026-01-24_merge-labeled-detections-and-noise-for-training/10_impl_notes.md`

Step-by-step plan in small diffs:
- Stage 1: Implement a small script to merge `noise_samples.csv` + `noise_reviews.csv` into `noise_samples_final.csv` with the mapping rules above.
- Stage 2: Run the script once to regenerate `noise_samples_final.csv` and record results in `10_impl_notes.md`.
- Stage 3: (Optional) If trimmed PNGs are missing, decide whether to regenerate via `scripts/regenerate_noise_spectrograms.py` or skip.
