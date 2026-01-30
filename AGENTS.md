# AGENTS.md

## Relationship to CLAUDE.md

**CLAUDE.md** contains detailed technical guidance for Claude Code (architecture, algorithms, specialized agents, planning workflows).

**THIS FILE (AGENTS.md)** describes the manual Codex handoff protocol for background/long-running tasks.

When to use which:
- **Claude Code (CLAUDE.md workflow):** Real-time implementation, architecture, debugging (default)
- **Codex handoff (THIS file):** User triggers via `/codex-task` for background work

---

This repo is a Python USV (ultrasonic vocalization) analysis project with spectrogram generation, detection pipeline, and labeling tools.

## ⛔ Integrity Rules (Non-Negotiable)

These rules apply regardless of task:

1. **No test corruption**: Never modify test expected values to make tests pass. Fix the code or flag for discussion.
2. **No fabrication**: Don't claim a file contains something without reading it. Don't claim tests pass without running them.
3. **No silent scope creep**: Do exactly what the task brief specifies. If you discover something that needs different work, note it in impl_notes and STOP.
4. **Surface blockers**: If stuck after 2-3 attempts, write a BLOCKED section in your notes file instead of continuing to try random approaches.

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

## Struggle Protocol

If you hit a blocker (can't figure out approach, tests keep failing unexpectedly, requirements unclear), write this in your notes file:

```markdown
## 🚨 BLOCKED

**What I understand**: [specific understanding of the task]
**What I tried**:
1. [Attempt 1 - outcome]
2. [Attempt 2 - outcome]
**Where I'm stuck**: [specific blocker]
**What would help**: [specific request - clarification, different approach, human review]
```

Then STOP. Do not continue trying random approaches. Surfacing blockers is correct behavior, not failure.

## Commands

Setup/install (Windows PowerShell; run from repo root):
```powershell
# If .venv exists, activate it
.\.venv\Scripts\Activate.ps1
# Or use the Python directly
.\.venv\Scripts\python.exe <script>
```

Dependencies:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Lint/format:
- Run `py_compile` on any touched files: `.\.venv\Scripts\python.exe -m py_compile <file.py>`

Unit tests:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
# Fast subset for detection work:
.\.venv\Scripts\python.exe -m pytest tests/test_energy_detector.py -v
```

Run the app / typical workflow:
```powershell
# USV Detection pipeline
python scripts/run_detection.py --help

# Spectrogram extraction
python scripts/extract_spectrograms.py --help

# Labeling tool (Streamlit)
.\.venv\Scripts\streamlit.exe run scripts/usv_labeling_tool.py

# Parameter lab (Streamlit)
python src/usv_spectrogram/param_lab/app.py

# CNN training
python scripts/train_cnn.py --help

# Model evaluation
python scripts/evaluate_model.py --help
```

Sanity run protocol:
- `python -m py_compile <script>.py` for any touched script files.
- `python <entrypoint_script>.py [args]` if the script has documented arguments.
- Validate any expected output files exist and are non-empty.

## Definition of Done

Verification checklist (write results to `20_verification.md`):
- [ ] py_compile passes on all touched files
- [ ] Unit tests pass (paste output)
- [ ] Changes match task brief scope (no scope creep)
- [ ] Docstrings added/updated for new or behavior-changed public code
- [ ] If user-facing usage changed, README/docs updated
- [ ] IMPLEMENTATION_PROGRESS.md updated with what changed

## Code documentation standards
- For any NEW public function/class/module (or any function/class whose behavior you change):
  - Add/maintain a short docstring: Purpose, Parameters + types, Return value, Errors/edge cases if non-obvious
- Do NOT add verbose comments everywhere.
- Keep docs aligned with the "small diffs" rule: document as you go.

## Parallel sessions suggestion
- **Spec Refiner**: Creates Task Brief in `00_task_brief.md`, captures assumptions and acceptance criteria.
- **Implementer**: Makes code changes per Task Brief, records decisions in `10_impl_notes.md`.
- **Verifier**: Runs checks per Commands, writes verification transcript to `20_verification.md`.
- **Refactorer**: Simplifies/cleans up after functionality stable (no behavior changes), re-runs verification.
- **Docs**: Updates docstrings for new/changed public code, updates README if user-facing usage changed.
- **Reviewer**: Scans diffs for regressions, API changes, missing tests, mismatches vs acceptance criteria.

## Parallel work safety
- Never allow two sessions to edit the same file at the same time.
- Assign file ownership per role.
- Coordinate edits through the task folder.

## Signal Processing Reminders (USV-Specific)

- Sample rate is 250,000 Hz - always specify, never use librosa defaults
- n_fft: 512, hop_length: 128
- Frequency range: 25-110 kHz
- Don't change STFT parameters without noting frequency resolution impact
- Detection threshold changes need baseline comparison

## What Claude Code Handles vs What You Handle

**Claude Code** is for reasoning-heavy tasks:
- Architecture decisions
- Complex debugging
- Algorithm design
- Code review

**You (Codex)** are better for:
- Writing tests for existing functions
- Adding docstrings and type hints
- Boilerplate and scaffolding
- Repetitive mechanical edits

If a task feels like it needs deep reasoning about the approach, note it in impl_notes and suggest human route it to Claude Code.
