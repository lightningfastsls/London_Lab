# Handoff: Codex Workflow Bootstrap
Date: 2026-03-06

## Task

Create a Codex-facing workflow bootstrap for this repo by replacing the root `AGENTS.md`, adding a compact navigation index for Codex, defining handoff conventions, and recording the boundary decisions in a handoff artifact.

## Files Changed

- `AGENTS.md` - replaced the previous Claude-style contract with a Codex-specific operating contract for this repo
- `docs/codex_index.md` - added a compact routing document for future Codex sessions
- `docs/handoffs/README.md` - defined the handoff format and usage rules
- `docs/handoffs/2026-03-06_codex-workflow-bootstrap.md` - recorded this bootstrap work and the main design decisions

## Reasoning

The existing `AGENTS.md` was effectively a duplicate of `CLAUDE.md`, which blurred the ownership boundary between Codex and Claude Code. The bootstrap replaces that with a smaller contract that keeps Claude as the owner of orchestration, memory, and arscontexta-managed structures while keeping Codex focused on implementation, debugging, validation, and explicit handoffs.

The new `docs/codex_index.md` is intentionally compact. Its job is to reduce startup scanning cost for future Codex chats without duplicating module docs or the full script index.

The handoff README formalizes `docs/handoffs/` as the durable bridge for implementation context that should survive the session without writing directly into Claude-managed memory systems.

## Validation

- Re-read `AGENTS.md`, `docs/codex_index.md`, `docs/handoffs/README.md`, and this handoff after writing them
- Verified that the referenced repo paths and documents exist
- Confirmed the instructions are consistent about default read-only Claude-owned areas and default writable Codex areas
- No Python files were changed, so `py_compile` and `pytest` were not run for this docs-only task

## Open Questions / Known Risks

No open questions were required for the initial boundary definition.

Known risk: future repo workflow changes may require keeping `AGENTS.md`, `docs/codex_index.md`, and `docs/handoffs/README.md` synchronized so the Codex boundary does not drift back toward Claude-specific behavior.

## Worth Remembering For Claude

- `AGENTS.md` was intentionally replaced rather than preserved because the prior file was not Codex-specific
- Durable Codex context should go to `docs/handoffs/` by default, not directly into `ops/`, `notes/`, or other Claude-owned memory structures
- This bootstrap did not modify `.claude/`, `ops/`, `notes/`, `methodology/`, `reference/`, `templates/`, `inbox/`, or `IMPLEMENTATION_PROGRESS.md`