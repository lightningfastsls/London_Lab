# Task Brief

Title: Implementor Stage Gate Skill
Date: 2026-01-09

## Goal
Add an implementor-focused skill that enforces stage-by-stage execution with explicit user confirmation between stages unless instructed otherwise.

## Context
Assumptions:
- Skills live under `.codex/skills/` and are triggered by name/description.
- This is repo-local skill content similar to `verify-app`.
Uncertainties:
- Whether a packaging step is desired; default to un-packaged repo skill.

## Scope
In scope:
- New skill folder with `SKILL.md` describing stage-gated implementor workflow.
Out of scope:
- Changes to existing code, tests, or workflow scripts.

## Constraints
Dependencies:
- None.
Performance:
- N/A.
File ownership:
- Implementer owns the new skill folder.
API stability:
- No API changes.
Style:
- Keep instructions concise and actionable; ASCII only.

## Acceptance criteria
- New skill exists under `.codex/skills/` with clear trigger description.
- Skill instructs implementors to execute a single stage at a time and request confirmation before proceeding.

## File touch list
New files:
- `.codex/skills/implementor-stage-gate/SKILL.md`
Modified files:
- `tasks/2026-01-09_implementor-stage-guardrails/00_task_brief.md`
- `tasks/2026-01-09_implementor-stage-guardrails/10_impl_notes.md`

## Plan (small diffs)
1) Define the skill metadata and workflow in `SKILL.md`.
2) Record implementer notes.

## Implementer instructions
Do:
- Follow skill-creator guidance for concise, high-signal content.
Do not:
- Add extra resource folders unless needed.

## Verifier checklist
- Read `00_task_brief.md` and `10_impl_notes.md`.
- Run checks per `AGENTS.md` (or sanity run protocol).
- Record commands and results in `20_verification.md`.
