# Task Brief

Title: Expand detection control in USV labeling app (opposite of trim)
Date: 2026-01-24

## Goal
Add an “expand detection” control to the USV labeling app that mirrors the noise review trim flow but expands the candidate boundaries so the detection appears larger for review, with preview and save.

## Context
Assumptions:
- “Opposite of trim” means: positive value expands the start earlier (start_ms decreases), negative value expands the end later (end_ms increases).
- Expansion is a per-candidate adjustment saved alongside the label without breaking existing consumers (extra columns are ok).
- Expanded spectrograms can be generated on demand from WAVs using the same `SpectrogramExtractor` as noise review.
- WAV inputs are located via `USV_WAV_DIR` env var (fallback to `<repo>/5970 USV`).
Uncertainties:
- Whether expansions should be stored in a separate CSV instead of `labels.csv`.
- Whether expanded PNGs should overwrite or be stored alongside originals (assume new filename/suffix).

## Scope
In scope:
- Add expand controls (input + preview + save) to `labeling_app.py` similar to `noise_review_app.py`.
- Implement expansion helper to adjust candidate start/end and re-extract spectrogram.
- Persist expansion metadata (e.g., `expand_ms`) with labels and surface current value in UI.
- Save expanded PNGs with a suffix (e.g., `_expanded`) in a predictable output folder.
- Update labeling README to describe the expand feature and CSV format changes (if any).
Out of scope:
- Changing detection pipeline or candidates CSV generation.
- Batch reprocessing or auto-expanding all candidates.
- Major UI redesign.

## Constraints
Dependencies:
- No new third-party dependencies.
Performance:
- Only expand on preview/save (single candidate); avoid re-rendering on every label.
File ownership:
- Labeling app + labeling README only unless expansion requires shared utilities.
API stability:
- Avoid breaking existing `labels.csv` readers; if adding columns, keep required fields intact.
Style:
- Follow existing Streamlit patterns and noise review trim UX.

## Acceptance criteria
- Labeling app includes an “Expand detection” section with the same interaction pattern as noise review trim.
- User can enter a signed ms value with clear guidance (positive expands start, negative expands end).
- Preview button renders an expanded spectrogram and displays it in-app.
- Save button persists expansion data and generates a saved expanded PNG.
- Expansion is bounded (no negative start_ms; handles overrun past recording end gracefully).
- If WAVs are missing or expansion invalid, the UI shows a clear error and does not crash.
- Labels and expansion data remain readable by existing scripts (candidate_id/label/labeled_at preserved).
- README mentions the expand feature and any CSV field changes.

## File touch list
New files:
- (Optional) `labels_expansions.csv` if we decide not to extend `labels.csv`.
Modified files:
- `src/usv_spectrogram/labeling/labeling_app.py`
- `src/usv_spectrogram/labeling/README.md`

## Plan (small diffs)
1) Add expansion helper + session state to `labeling_app.py` (mirror trim logic; wire WAV dir).
2) Add expand UI (input, preview, save) and persist expansion value to CSV or sidecar file.
3) Update README + in-app guide text; verify sanity run for touched script(s).

## Implementer instructions
Do:
- Mirror `noise_review_app.py` trim UX to keep behavior consistent.
- Keep expansion fields optional; default to 0 when absent.
- Use `USV_WAV_DIR` env var with fallback to repo `5970 USV`.
Do not:
- Overwrite original spectrogram PNGs.
- Change label options or the labeling flow unless required for expand controls.

## Verifier checklist
- Read `00_task_brief.md` and `10_impl_notes.md`.
- Run checks per `AGENTS.md` (or sanity run protocol).
- Record commands and results in `20_verification.md`.
