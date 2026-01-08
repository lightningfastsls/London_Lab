---
name: verify-app
description: Runs checks per AGENTS.md and writes the verification transcript to tasks/20_verification.md.
---

# verify-app

## When to use
Use this skill whenever:
- Code changed and you need to confirm it is correct.
- Tests or lint are failing and you need to debug/fix them.
- You are about to claim "done" and need a verification transcript.

## Rules
- If the task uses staged delivery, explicitly note whether the verified stage is complete in `20_verification.md` (and add a brief note to `10_impl_notes.md` when requested).
- Read `AGENTS.md` Commands for canonical checks.
- Read the task brief and implementation notes in the active task folder.
- If commands are missing, follow the sanity run protocol in `AGENTS.md` and note gaps.
- Record a full transcript in `tasks/<date>_<slug>/20_verification.md`.

## Method
1. Read `tasks/<date>_<slug>/00_task_brief.md` and `10_impl_notes.md`.
2. Run the smallest relevant checks first, then full checks if configured.
3. Fix failures minimally and re-run failed checks.
4. Write the exact commands and results to `20_verification.md`.

## Output requirements
- `20_verification.md` includes environment, commands, results, and any rerun transcript.
