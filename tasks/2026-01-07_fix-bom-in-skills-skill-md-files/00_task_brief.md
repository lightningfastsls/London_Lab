# Task Brief

Title: Fix BOM and frontmatter in local skills SKILL.md files
Date: 2026-01-07

## Goal
Remove BOMs and ensure valid YAML frontmatter in the three reported SKILL.md files so skills load normally.

## Context
Assumptions: The failure is due to BOM/encoding or missing frontmatter delimiters.
Uncertainties: Actual current content of each SKILL.md.

## Scope
In scope:
- .codex/skills/code-simplifier/SKILL.md
- .codex/skills/spec-refiner/SKILL.md
- .codex/skills/verify-app/SKILL.md
Out of scope:
- Changing skill semantics beyond minimal frontmatter fixes
- Updating other skill files

## Constraints
Dependencies: None
Performance: N/A
File ownership: Single agent edits only
API stability: Preserve existing skill behavior
Style: ASCII-only where possible

## Acceptance criteria
- Each SKILL.md begins with valid YAML frontmatter delimited by ---
- No UTF-8 BOM remains in those files
- Skills load without the reported warnings

## File touch list
New files: None
Modified files:
- tasks/2026-01-07_fix-bom-in-skills-skill-md-files/00_task_brief.md
- .codex/skills/code-simplifier/SKILL.md
- .codex/skills/spec-refiner/SKILL.md
- .codex/skills/verify-app/SKILL.md
- tasks/2026-01-07_fix-bom-in-skills-skill-md-files/10_impl_notes.md
- tasks/2026-01-07_fix-bom-in-skills-skill-md-files/20_verification.md

## Plan (small diffs)
1) Inspect the three SKILL.md files for BOM/frontmatter issues.
2) Fix frontmatter delimiters and remove BOMs with minimal edits.
3) Record notes and verification results.

## Implementer instructions
Do:
- Keep diffs minimal
- Preserve existing content after frontmatter
Do not:
- Introduce new dependencies
- Change skill behavior

## Verifier checklist
- Read `00_task_brief.md` and `10_impl_notes.md`.
- Run checks per `AGENTS.md` (or sanity run protocol).
- Record commands and results in `20_verification.md`.