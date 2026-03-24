# AGENTS.md

## Codex Role In This Repo

Codex is the focused implementation agent for `mickey_london_lab`.

Default responsibilities:
- edit code, tests, scripts, and targeted docs
- debug failures and inspect generated artifacts
- run validation for the changes it makes
- write explicit handoffs for anything worth carrying forward

Codex does not own orchestration, long-term memory, or knowledge-graph maintenance in this repo.

## Read First

At the start of a session, read:
1. `AGENTS.md`
2. `ops/goals.md`
3. `ops/reminders.md`
4. `docs/codex_index.md`
5. task-specific docs referenced from the index

If the task touches DSP, detection thresholds, STFT behavior, or architectural decisions, read the relevant module/reference docs before editing.

## Ownership Boundaries

Default read-only areas for Codex unless the user explicitly asks otherwise:
- `.claude/`
- `ops/`
- `notes/`
- `methodology/`
- `reference/`
- `templates/`
- `inbox/`

Default writable areas for Codex:
- `src/`
- `tests/`
- `scripts/`
- `usv_language/`
- `docs/handoffs/`
- targeted files under `docs/`

Treat Claude Code plus arscontexta as the owner of memory, workflow orchestration, and vault structure. If something should persist, write a handoff instead of editing Claude-owned systems by default.

## Core Safety Rules

- Always specify `sr=300000` explicitly when touching WAV loading, spectrogram generation, or DSP-related paths.
- Do not change test expectations just to make tests pass. Fix the code or raise the mismatch.
- Do not claim completion without validation.
- Do not make casual changes to STFT parameters, dB scaling, detection thresholds, or `energy_detector.py` without first reading the relevant docs and explaining the impact.
- Prefer code truth over stale docs. If docs and code disagree, inspect the codebase before deciding.
- Never use bulk staging or destructive git commands casually. Review what changed and stage specific files.

## Git

- Stage specific files by name. Never use `git add -A` or `git add .`.
- Commit on feature branches (e.g., `codex/feature-name`), not directly on `main`.
- Push to your feature branch with `-u` flag.
- Do NOT merge to `main` or push to `main`. That is Claude Code's or the user's responsibility.
- Do NOT force-push, amend published commits, or use destructive git commands.
- Include the handoff file in the same commit or as a follow-up commit on the same branch.

## Working Style

- Keep changes scoped to the user request.
- Explain reasoning when making non-obvious choices.
- Make assumptions visible.
- For durable implementation context, use `docs/handoffs/` rather than implicit memory.
- For bug-hunt sessions, keep going autonomously across multiple likely targets instead of stopping after the first fix. Only stop to report when blocked, when the user explicitly asks for a check-in, or when the current bug-hunt pass is genuinely exhausted.

## Validation

For code changes:
1. Run `py_compile` on every changed Python file.
2. Run relevant `pytest` coverage for the affected behavior.
3. Report validation results accurately.

For docs-only changes:
1. Re-read every created or edited file.
2. Verify references point to real files or directories.
3. Confirm ownership boundaries and instructions are internally consistent.

Do not say work is done until the appropriate validation has been run.

## Architecture Rules To Preserve

Follow established repo patterns:
- frozen config dataclasses with validated defaults and unit-suffixed fields
- script bootstrap that adds `src/` to `sys.path`
- PyQt separation between `app/core/` and `app/widgets/`
- synthetic test fixtures instead of real recordings
- shared STFT logic in `_stft_core.py`

See `docs/codex_index.md` and `docs/architecture/patterns.md` for routing and details.

## Handoffs

Write handoffs to `docs/handoffs/` for non-trivial work, architectural reasoning, unresolved issues, or anything Claude may want to ingest later.

Use handoffs to record:
- what changed
- why it changed
- how it was validated
- assumptions, risks, and open questions
- durable implementation knowledge worth preserving

**Orientation reads:** Only read top-level `docs/handoffs/*.md`. Skip `archive/`.
**Archival:** Move resolved handoffs to `docs/handoffs/archive/` when all action items are done.
**Review handoffs from Claude Code** follow `From/To/Re` header convention — check for any addressed to you.

See `docs/handoffs/README.md` for the required format.
