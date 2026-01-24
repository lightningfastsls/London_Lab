# Task Brief

Title: Notion commands note in London Lab
Date: 2026-01-24

## Goal
Create a Notion note inside the "London Lab" area that documents the commonly used commands (detection + extraction, USV labelling app, noise labelling app) so the user can reference them later.

## Context
Assumptions:
- There is a Notion page or database named "London Lab" accessible to the Notion integration.
- The note should be a new subpage (or database entry) under "London Lab".
- Command details can be sourced from existing repo docs/scripts if not explicitly provided.
Uncertainties:
- Exact Notion location (page vs database) and preferred note title.
- Exact command lines and arguments for detection/extraction and the labeling apps.
- Whether to include environment setup steps (e.g., USV_WAV_DIR) in the note.

## Scope
In scope:
- Identify the target Notion location and create/append a commands note under "London Lab".
- Document the known command lines for detection/extraction and the two labeling apps.
- Format with clear headings and code blocks.
Out of scope:
- Changing repo code or adding new dependencies.
- Running the apps or verifying runtime behavior beyond documenting commands.

## Constraints
Dependencies:
- Use existing Notion API tools only; do not add dependencies.
Performance:
- Not applicable (single note creation).
File ownership:
- Only write task files and Notion content.
API stability:
- Do not change any public APIs or script interfaces.
Style:
- Keep note concise, practical, and organized by task.

## Acceptance criteria
- A Notion note exists under "London Lab" containing sections for:
  - Detection + extraction command(s)
  - USV labelling app command(s)
  - Noise labelling app command(s)
- Commands are in code blocks and include any required environment setup or arguments.
- The note title is confirmed or matches an agreed default.

## File touch list
New files:
- None
Modified files:
- tasks/2026-01-24_notion-commands-note/00_task_brief.md
- tasks/2026-01-24_notion-commands-note/10_impl_notes.md (Implementer)
- tasks/2026-01-24_notion-commands-note/20_verification.md (Verifier)

## Plan (small diffs)
Stage 1: Confirm Notion target location/title and exact command lines (or agree to infer from repo).
Stage 2: Use Notion API to create the note under "London Lab" with formatted command sections.
Stage 3: Record implementation notes and verification transcript in the task folder.

## Implementer instructions
Do:
- Search Notion for the "London Lab" page/database and confirm the destination.
- Use headings and code blocks for readability.
- Note any assumptions about command arguments.
Do not:
- Add dependencies or modify scripts.
- Write to notes/claude_responses.md unless user can’t see the response.

## Verifier checklist
- Read `00_task_brief.md` and `10_impl_notes.md`.
- Run checks per `AGENTS.md` (or sanity run protocol).
- Record commands and results in `20_verification.md`.
