# Task Brief

Title: Ensure skip-noise PNGs are found during split creation
Date: 2026-01-22

## Goal
Ensure dataset split creation can locate spectrogram PNGs for noise-derived samples that are labeled as USV (e.g., "Skip" from noise review), so they are not skipped.

## Context
Assumptions:
- Noise review "Skip" samples are moved into the USV set and their PNGs may live in `spectrograms_review/`.
- `labels.csv` can include candidate_ids with `_noise_` labeled as `USV`.
- `load_labeled_samples()` currently assumes `_noise_` candidate_ids always live under `noise_samples/`.
Uncertainties:
- Whether the user wants skip-noise PNGs to stay in `spectrograms_review/` permanently or mirror into `noise_samples/` as well.

## Scope
In scope:
- Update `load_labeled_samples()` path resolution to find `_noise_` samples labeled as `USV` in `spectrograms_review/`.
- Add a fallback to the alternate folder if the expected PNG is missing.
Out of scope:
- Changing label schemas or candidate CSV formats.
- Reworking the noise review app UI/logic.

## Constraints
Dependencies:
- No new Python dependencies.
Performance:
- Minimal impact (simple path checks).
File ownership:
- Implementer edits dataset split logic only.
API stability:
- No breaking changes to CLI entrypoints.
Style:
- Small, focused diff; add docstrings only for changed public functions if needed.

## Acceptance criteria
- `prepare_dataset.py --create-splits` no longer skips skip-noise samples when their PNGs are present in `spectrograms_review/`.
- `_noise_` samples labeled `Not USV` continue to load from `noise_samples/`.
- If a PNG is missing in the expected folder, the loader tries the alternate folder before skipping.

## File touch list
New files:
- None.
Modified files:
- src/usv_spectrogram/dataset/splits.py

## Plan (small diffs)
1) Update path resolution in `load_labeled_samples()` to check label + fallback folder.

## Implementer instructions
Do:
- Prefer `spectrograms_review/` for `_noise_` samples labeled `USV`.
- Prefer `noise_samples/` for `_noise_` samples labeled `Not USV`.
Do not:
- Do not change CSV schemas or split ratios.
- Do not add dependencies.

## Verifier checklist
- Read `00_task_brief.md` and `10_impl_notes.md`.
- Run checks per `AGENTS.md` (or sanity run protocol).
- Record commands and results in `20_verification.md`.
