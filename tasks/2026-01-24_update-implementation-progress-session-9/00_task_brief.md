# Task Brief

Title: Update IMPLEMENTATION_PROGRESS Session 9
Date: 2026-01-24

## Goal
Document Session 9 CNN test set diagnostic findings in `IMPLEMENTATION_PROGRESS.md`.

## Context
Assumptions:
- Session 9 findings listed in `codex_tasks/task1_update_implementation_progress.md` are accurate and complete.
- The new section should be appended at the end of `IMPLEMENTATION_PROGRESS.md` and match its tone.
Uncertainties:
- Exact wording and subheading structure used in nearby sections (confirm before writing).

## Scope
In scope:
- Add a new section titled `## Session 9: CNN Test Set Performance Diagnostic`.
- Include the specified problem, diagnostics, fixes, key findings, and files created.
Out of scope:
- Any edits to earlier sections or other files.
- Reformatting unrelated content.

## Constraints
Dependencies: None.
Performance: Not applicable.
File ownership: Modify only `IMPLEMENTATION_PROGRESS.md`.
API stability: Not applicable.
Style: Match existing document format; concise bullets and short paragraphs.

## Acceptance criteria
- New section exists at the end with the exact title.
- Section includes all items: problem identified, diagnostics completed, fix implemented, key findings, and files created.
- Performance metrics are stated clearly (F1 0.43 -> 0.76, Recall 0.30 -> 0.92).
- No other content changes in `IMPLEMENTATION_PROGRESS.md`.

## File touch list
New files: None.
Modified files:
- `IMPLEMENTATION_PROGRESS.md`

## Plan (small diffs)
1) Review the current end of `IMPLEMENTATION_PROGRESS.md` to mirror its structure.
2) Append the new Session 9 section with concise bullets.
3) Re-read for tone/consistency and save.

## Implementer instructions
Do:
- Keep bullets compact and list file paths where relevant.
- Use plain ASCII symbols (use `->` for arrows).
Do not:
- Change any existing sections or headings.
- Add new files or dependencies.

## Verifier checklist
- Read `00_task_brief.md` and `10_impl_notes.md`.
- Confirm the new section title and content are present and accurate.
- No verification commands needed for a doc-only change.
- Record verification in `20_verification.md`.
