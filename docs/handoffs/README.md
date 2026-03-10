# Agent Handoffs

`docs/handoffs/` is the bridge between Codex and Claude Code. Both agents write handoffs here. Codex writes implementation handoffs; Claude Code writes review handoffs.

## When To Write A Handoff

Write a handoff after non-trivial work. Strongly prefer a handoff when the task includes:
- architectural reasoning
- multiple file changes
- unresolved risks or open questions
- user-visible workflow changes
- implementation knowledge that should survive the session

Handoffs complement code comments, tests, and module docs. They do not replace them.

## File Naming

Use:

`YYYY-MM-DD_short-kebab-task-name.md`

Example:

`2026-03-06_codex-workflow-bootstrap.md`

## Required Sections

Every handoff should include:

```markdown
# Handoff: Short Task Title
Date: YYYY-MM-DD

## Task

## Files Changed

## Reasoning

## Validation

## Open Questions / Known Risks

## Worth Remembering For Claude
```

## Section Guidance

`Task`
- state what was requested and what was delivered

`Files Changed`
- list the files changed and what each change was for

`Reasoning`
- capture the main implementation decisions and trade-offs

`Validation`
- record what checks were run and their outcome
- for docs-only work, say that the files were re-read and references were verified

`Open Questions / Known Risks`
- include unresolved issues, assumptions, or follow-up points
- if there are none, state that explicitly

`Worth Remembering For Claude`
- include only durable implementation knowledge, decisions, constraints, gotchas, or recommended follow-up
- do not copy transient debugging chatter into this section

## Archival

Resolved handoffs go to `docs/handoffs/archive/` to keep the active folder small for agent orientation reads.

**When to archive:**
- The handoff's action items are all resolved (reviewed, merged, or superseded)
- Either agent can archive by moving the file to `archive/`

**Never archive:**
- Rolling files: `current_bug_hunt.md`, `README.md`, `codex_context_briefing.md`
- Handoffs with unresolved open questions or pending action items

**Reading convention:**
- To orient: read only top-level `docs/handoffs/*.md` (not `archive/`)
- To research past work: grep `docs/handoffs/archive/` by keyword

## Boundary Rule

If Codex discovers something that should persist, prefer writing it here instead of editing `ops/`, `notes/`, or other Claude-managed systems by default.

## Vault-Enriched Handoffs

When Claude Code writes a Codex task spec, it must search the vault for relevant constraints first (see `CLAUDE.md` § "Codex Handoff Vault Search"). Constraints are flattened into plain text in the handoff so Codex can respect invariants it has no way to discover on its own. Use `templates/codex-handoff.md` for the handoff structure. Cap at 5 constraints per handoff.