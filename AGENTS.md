# AGENTS.md

This repo appears to be a small Python data analysis and plotting project with standalone scripts and CSV outputs.

## Real input data location
- WAV inputs live under `C:\Users\shach\PycharmProjects\mickey_london_lab\5970 USV` when available.

## Working agreements
- Plan first, then implement.
- Keep diffs small; verify continuously.
- Ask before adding new dependencies or changing public APIs.
- Always run the smallest relevant tests after edits; run the full suite before a PR.

## How to use this workflow
- Run multiple Codex sessions (tabs) with distinct roles to parallelize.
- Use skills when appropriate by mentioning their names (e.g., $verify-app, $code-simplifier).
- Use custom prompts via `/prompts:<name>` (e.g., `/prompts:verify`).

## Task intake (Spec Refiner)
Before any code changes:
- Start a new task folder: `python tools/new_task.py "<Task Title>"`.
- Fill `tasks/<date>_<slug>/00_task_brief.md` with Goal, Scope/Non-scope, Constraints, Acceptance criteria, File touch list, and a plan in small diffs.
- Only then proceed to implementation.

## Handoff protocol (no copy/paste)
All roles communicate via files in a task folder under `tasks/`:
- Spec Refiner writes: `tasks/<date>_<slug>/00_task_brief.md`.
- Implementer reads the brief and writes: `tasks/<date>_<slug>/10_impl_notes.md`.
- Verifier reads both and writes: `tasks/<date>_<slug>/20_verification.md`.
Do not copy/paste task content between agents; always read/write the shared files.

## Commands

Packaging detection:
- None detected: no `pyproject.toml`, `requirements.txt`, `setup.cfg`, or `setup.py` found.

Setup/install (Windows PowerShell; run from repo root):
- None configured. If you add dependencies, add exact setup/install commands here.

Lint/format:
- None configured. Use the sanity run protocol below.

Unit tests (fast):
- None configured (no tests/ or pytest config detected).

Full test suite:
- None configured.

Sanity run protocol (when no tests/lint):
- `python -m py_compile <script>.py` for any touched script files.
- `python <entrypoint_script>.py [args]` if the script has documented arguments.
- Validate any expected output files exist and are non-empty.

Run the app / typical workflow:
- Primary script run (PowerShell): `python <entrypoint_script>.py [args]`
- If unclear, choose an entrypoint from: `analysis.py`, `data_processing.py`, `loader_mice_files.py`, `plot_training.py`, `water_formater.py`, `wav_format.py`, `playground.py`.
- WSL/macOS note: use `python3 <entrypoint_script>.py [args]`.

## Definition of done
Verification checklist:
- Lint/format pass (if configured).
- Unit tests pass.
- Full test suite pass (if configured).
- Build/package steps pass (if applicable).
- Documentation updated:
  - Docstrings added/updated for NEW or behavior-changed public code.
  - README/docs updated if user-facing usage changed.
- Paste a verification transcript with commands run and results in `tasks/<date>_<slug>/20_verification.md`.

## Code documentation standards
- For any NEW public function/class/module (or any function/class whose behavior you change):
  - Add/maintain a short docstring that states:
    - Purpose (1-2 lines)
    - Parameters + types (if not obvious)
    - Return value (and type) or side effects
    - Errors/edge cases only if non-obvious
- Do NOT add verbose comments everywhere.
  - Prefer docstrings for "what/why," and inline comments only for tricky logic, non-obvious math, or critical assumptions.
- If a change affects expected inputs/outputs of a script, update:
  - The script/module docstring (top-of-file) OR a short section in README (whichever is already used in this repo).
- Keep docs aligned with the "small diffs" rule: document as you go, not in a later sweep.

## Parallel sessions suggestion
- Spec Refiner (Prompt Engineer): creates the Task Brief in `00_task_brief.md` and captures assumptions and acceptance criteria.
- Implementer: makes code changes according to the Task Brief and records decisions in `10_impl_notes.md`.
- Verifier: runs checks per Commands and writes a verification transcript to `20_verification.md`.
- Refactorer: simplifies/cleans up after functionality is stable (no behavior changes), then re-runs verification.
- Docs: updates docstrings for new/changed public code and updates README/docs if user-facing usage changed.
- Reviewer: scans diffs for regressions, API changes, missing tests, and mismatches vs acceptance criteria.

## Parallel work safety
- Never allow two Codex sessions to edit the same file at the same time.
- Assign file ownership per role (e.g., Implementer: scripts, Verifier: tasks/20_verification.md, Docs: README/docs).
- Coordinate edits through the task folder and avoid overlapping file changes.
