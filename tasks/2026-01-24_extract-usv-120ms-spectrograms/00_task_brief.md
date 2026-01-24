# Task Brief

Title: Extract USV >120ms spectrograms for review
Date: 2026-01-24

## Goal
Identify USV candidates longer than 120 ms and copy their existing spectrogram PNGs into a new folder for quick visual inspection.

## Context
Assumptions:
- Default candidates source is `candidates_extracted.csv` (has `duration_ms` + `spectrogram_path` pointing to `spectrograms_review/`).
- “USVs” means labeled `USV` in `labels.csv`, matched on `candidate_id`.
- Output folder default is `spectrograms_usv_over_120ms/` at repo root, with a manifest CSV of what was copied.
Uncertainties:
- Which candidates CSV / spectrogram set should be targeted if not `candidates_extracted.csv`?
- Should unlabeled candidates be included, or only those labeled `USV`?

## Scope
In scope:
- Read chosen candidates CSV and (optionally) join with `labels.csv` to keep `label == USV`.
- Filter to `duration_ms > 120`.
- Copy corresponding `spectrogram_path` PNGs to a new output folder.
- Write a manifest CSV of copied files (candidate_id, duration_ms, source path, dest path).
Out of scope:
- Re-running detection or regenerating spectrograms.
- Changing model training data or labeling workflows.

## Constraints
Dependencies:
- No new dependencies without approval; use stdlib or existing pandas usage if already available.
Performance:
- Operate as a one-off batch copy; avoid loading audio.
File ownership:
- New script under `scripts/` (or small helper in `tools/` if preferable).
API stability:
- Do not change existing script APIs.
Style:
- Follow existing script patterns; keep diff small.

## Acceptance criteria
- A runnable script exists that copies spectrogram PNGs for `duration_ms > 120` (and `label == USV` if applicable).
- Output folder is created and populated with the filtered spectrograms.
- A manifest CSV is written listing copied items and any missing files are reported.

## File touch list
New files:
- `scripts/extract_long_usv_spectrograms.py` (name can vary but must be documented in the brief notes)
Modified files:
- (Optional) `README.md` if a user-facing command is added.

## Plan (small diffs)
Stage 1) Confirm inputs: candidates CSV, whether to filter on `labels.csv`, and output folder name.
Stage 2) Implement a small script to filter, copy, and write a manifest.
Stage 3) Run sanity check (`python -m py_compile`) and execute the script on the confirmed inputs.

## Implementer instructions
Do:
- Read this brief and capture decisions in `10_impl_notes.md`.
- Prefer `pathlib` + `shutil.copy2` and ensure missing spectrograms are logged and skipped.
- Keep output deterministic (sorted by candidate_id or duration).
Do not:
- Do not regenerate spectrograms or re-run detection.
- Do not add dependencies without approval.

## Verifier checklist
- Read `00_task_brief.md` and `10_impl_notes.md`.
- Run checks per `AGENTS.md` (or sanity run protocol).
- Record commands and results in `20_verification.md`.
