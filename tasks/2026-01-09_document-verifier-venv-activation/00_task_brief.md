# Task Brief

Title: Document Verifier Venv Activation
Date: 2026-01-09

## Goal
Clarify the verifier workflow so test runs consistently use the intended Python environment (e.g., activating `.venv` or explicitly using its interpreter).

## Context
Assumptions:
- Verification guidance should live in repo docs (AGENTS.md) and/or the verifier skill.
- The project uses a local `.venv` when present.
Uncertainties:
- Whether the team prefers activation steps or explicit interpreter paths in guidance.

## Scope
In scope:
- Update verifier documentation to mention activating `.venv` (or equivalent) before running tests.
- Make the guidance concise and aligned with existing workflow rules.
Out of scope:
- Changing any code or tests.
- Adding dependencies or tooling automation.

## Constraints
Dependencies:
- None.
Performance:
- N/A.
File ownership:
- Spec Refiner owns task brief; Implementer owns docs updates.
API stability:
- No API changes.
Style:
- Keep instructions concise; ASCII only.

## Acceptance criteria
- Verifier workflow explicitly instructs how to ensure the correct Python environment is used for tests.
- Guidance is placed in the most appropriate doc location(s) without expanding scope.

## File touch list
Modified files:
- AGENTS.md
- Optional: .codex/skills/verify-app/SKILL.md

## Plan (small diffs)
Stage 1: Add a short verifier note about activating `.venv` (or using its interpreter) in AGENTS.md.
Stage 2: If needed, mirror the guidance in verify-app skill for consistency.

## Implementer instructions
Do:
- Keep the note short and actionable.
- Align wording with existing verification steps.
Do not:
- Add new dependencies or automation steps.

## Verifier checklist
- Confirm docs update is present and readable.
- No tests required for doc-only change.
