# Task Brief

Title: Document real input path and verify on real data
Date: 2026-01-08

## Goal
Document the real input data location for all agents and run the pipeline on a real WAV to produce observable outputs.

## Context
Assumptions:
- Real input WAVs live under `D:\mickey_london_lab\5970 USV`.
Uncertainties:
- Which specific WAV file should be used for the verification run if multiple exist.

## Scope
In scope:
- Update agent/task documentation to include the real input path.
- Run the spectrogram script on a real WAV and record output paths.
Out of scope:
- Any functional code changes to the spectrogram pipeline.

## Constraints
Dependencies:
- Do not add new dependencies.
Performance:
- Prefer a smaller real WAV if multiple are available to keep verification fast.
File ownership:
- Only update documentation and task files for this task.
API stability:
- No API changes.
Style:
- ASCII-only edits; keep diffs small.

## Acceptance criteria
- `AGENTS.md` includes the real input data location.
- `TASK_TEMPLATE.md` prompts for the real input path when available.
- A real WAV is processed and output locations are recorded in verification notes.

## File touch list
New files:
- None.
Modified files:
- `AGENTS.md`
- `TASK_TEMPLATE.md`
- `tasks/2026-01-08_document-real-input-path-and-verify-on-real-data/00_task_brief.md`
- `tasks/2026-01-08_document-real-input-path-and-verify-on-real-data/10_impl_notes.md`
- `tasks/2026-01-08_document-real-input-path-and-verify-on-real-data/20_verification.md`

## Plan (small diffs)
1) Update `AGENTS.md` and `TASK_TEMPLATE.md` with the real input path.
2) Run the spectrogram script on a real WAV and record outputs.

## Implementer instructions
Do:
- Keep changes documentation-only.
- Record output files created during verification.
Do not:
- Modify pipeline code.

## Verifier checklist
- Read `00_task_brief.md` and `10_impl_notes.md`.
- Run checks per `AGENTS.md` (or sanity run protocol).
- Record commands and results in `20_verification.md`.
