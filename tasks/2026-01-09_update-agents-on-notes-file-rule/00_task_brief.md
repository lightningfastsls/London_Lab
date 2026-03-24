# Task Brief

Title: Update Agents on Notes File Rule
Date: 2026-01-09

## Goal
Document the updated rule for writing to `notes/codex_responses.md` so all agents follow it.

## Context
Assumptions:
- The shared agent instructions live in `AGENTS.md` and are the source of truth for repo-level workflows.
- The notes file is `notes/codex_responses.md`.
Uncertainties:
- Whether to also mirror the rule in `.codex/skills/*` docs; default to AGENTS.md only unless requested.

## Scope
In scope:
- Add a short rule describing when agents should write to `notes/codex_responses.md`.
Out of scope:
- Any code changes.
- Changes to existing skills or tooling unless explicitly requested.

## Constraints
Dependencies:
- None.
Performance:
- N/A.
File ownership:
- Implementer owns AGENTS.md update.
API stability:
- No API changes.
Style:
- Keep instructions concise; ASCII only.

## Acceptance criteria
- AGENTS.md includes the new rule:
  - Write to `notes/codex_responses.md` only when the user says they can?t see the full response.
  - Exception: if the response is a long dense block (~10+ lines with no spacing), write it there proactively.
- Rule is easy to find and unambiguous.

## File touch list
Modified files:
- AGENTS.md

## Plan (small diffs)
Stage 1: Add the notes-file rule in AGENTS.md under working agreements or presentation guidance.
Stage 2: Confirm wording is concise and ASCII-only.

## Implementer instructions
Do:
- Keep the rule in one short paragraph or two bullets.
Do not:
- Add new tooling or dependencies.

## Verifier checklist
- Confirm AGENTS.md contains the rule.
- No tests required for doc-only change.
