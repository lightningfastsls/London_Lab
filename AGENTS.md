# AGENTS.md

## Relationship to CLAUDE.md

**CLAUDE.md** contains detailed technical guidance for Claude Code (architecture, algorithms, specialized agents, planning workflows).

**THIS FILE (AGENTS.md)** describes the manual Codex handoff protocol for background/long-running tasks.

When to use which:
- **Claude Code (CLAUDE.md workflow):** Real-time implementation, architecture, debugging (default)
- **Codex handoff (THIS file):** User triggers via `/codex-task` for background work

This file focuses on the **file-based coordination protocol** (tasks/ folder structure) for manual handoffs.

---

This repo appears to be a small Python data analysis and plotting project with standalone scripts and CSV outputs.

## Real input data location
- WAV inputs live under the path specified by the `USV_WAV_DIR` environment variable.
- Default fallback: `<repo_root>/5970 USV` if the env var is not set.
- To set on Windows: `$env:USV_WAV_DIR = "C:\path\to\your\5970 USV"` (PowerShell)

## Working agreements
- Plan first, then implement.
- Keep diffs small; verify continuously.
- Ask before adding new dependencies or changing public APIs.
- Always run the smallest relevant tests after edits; run the full suite before a PR.
- Write to `notes/claude_responses.md` only when the user says they cannot see the full response; exception: if the response is a long dense block (~10+ lines with no spacing), write it there proactively.

## How to use this workflow
- Run multiple Claude Code sessions (tabs/terminals) with distinct roles to parallelize.
- Use custom commands via `/<command-name>` (e.g., `/verify-app`, `/spec-refiner`).
- Commands are defined in `.claude/commands/` as markdown files.

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

Dependencies:
- Managed via `requirements.txt` (exists in repo root)
- To install: `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`

Unit tests:
- Test suite: `tests/` directory with 12+ test files
- Run tests: `.\.venv\Scripts\python.exe -m pytest tests/ -v`
- Fast subset: `.\.venv\Scripts\python.exe -m pytest tests/test_energy_detector.py -v`

Sanity checks:
- Syntax check: `.\.venv\Scripts\python.exe -m py_compile <file.py>`
- After any edit, run py_compile on modified files

Run the app / typical workflow:
- **USV Detection pipeline:** `python scripts/run_detection.py --help`
- **Spectrogram extraction:** `python scripts/extract_spectrograms.py --help`
- **Labeling tool:** `.\.venv\Scripts\streamlit.exe run scripts/usv_labeling_tool.py`
- **Parameter lab:** `python src/usv_spectrogram/param_lab/app.py`
- **CNN training:** `python scripts/train_cnn.py --help`
- **Model evaluation:** `python scripts/evaluate_model.py --help`

Legacy scripts (in mice_learning_files/): analysis.py, data_processing.py, plot_training.py
(These are older utilities; primary workflow is USV pipeline above)

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

## Parallel Sessions Coordination

When user runs multiple Codex sessions via this workflow:

**File ownership (prevents conflicts):**
- Never edit same file simultaneously in multiple sessions
- Each session owns specific files in `tasks/<date>_<slug>/`
- Session 1 (Spec Refiner): writes `00_task_brief.md`
- Session 2 (Implementer): writes `10_impl_notes.md`, code files
- Session 3 (Verifier): writes `20_verification.md`

**Coordination:**
- Sessions communicate via task folder files (no copy/paste)
- Each session reads previous outputs from files
- Update IMPLEMENTATION_PROGRESS.md when completing work

**Note:** For real-time Claude Code work, use specialized agents in CLAUDE.md instead of parallel sessions.
