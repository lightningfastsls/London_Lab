# Tasks workflow

This repo uses file-based task handoffs under `tasks/` to avoid copy/paste between agents.

## How to start a task
1) Create a new task folder from templates:
```
python tools/new_task.py "My Task Title"
```
2) Spec Refiner fills `tasks/<date>_<slug>/00_task_brief.md`.
3) Implementer reads the brief and writes `tasks/<date>_<slug>/10_impl_notes.md`.
4) Verifier runs checks and writes `tasks/<date>_<slug>/20_verification.md`.

## Roles and responsibilities
Spec Refiner:
- Reads the user request and `AGENTS.md`.
- Writes the Task Brief in `00_task_brief.md`.
- Records assumptions, scope, constraints, and acceptance criteria.

Implementer:
- Reads `00_task_brief.md` and follows the plan.
- Updates `10_impl_notes.md` with decisions, commands, and file changes.

Verifier:
- Reads `00_task_brief.md` and `10_impl_notes.md`.
- Runs checks per `AGENTS.md` (or the sanity run protocol).
- Writes a full transcript to `20_verification.md`.

Optional roles:
- Refactorer: only after verification is green, improves readability without behavior changes.
- Docs: updates README/docs or docstrings if user-facing usage changed.
- Reviewer: scans diffs for regressions and missing tests.

## Safety rules
- No copy/paste between agents: always read/write the task files.
- No two sessions edit the same file at the same time.

## Usage examples
Create a task:
```
python tools/new_task.py "Add task handoff workflow"
```

What gets created:
```
tasks/YYYY-MM-DD_add-task-handoff-workflow/
  00_task_brief.md
  10_impl_notes.md
  20_verification.md
```
