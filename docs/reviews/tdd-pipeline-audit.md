# TDD Pipeline Audit: test-architect Never Invoked

**Date:** 2026-04-05
**Severity:** Design gap (not a regression)
**Status:** FIXED

## Root Cause

The `test-architect` agent was fully defined but never wired into the pipeline. Three independent gaps allowed this:

1. **`roadmap-from-plan` Step 7.5** was a passive reminder ("consider running test-architect") rather than an active invocation. Users naturally skip optional suggestions during the momentum of plan-to-code transitions.

2. **`implement` Phase 2, Step 3.5** had a passive check: "if pre-existing tests exist, use them." Since nothing ever created them, the condition was always false — a no-op masquerading as a safety check.

3. **`CLAUDE.md`** had no mention of pre-implementation test verification in its workflow rules. The Implementation Completion Sequence started at "write code" with no pre-test gate.

## Why This Matters

The test-architect agent was designed to catch spec ambiguities *before* implementation begins. When tests are written after code, they tend to test what the code does rather than what the spec requires — a subtle but important distinction. Pre-implementation tests:

- Force the spec to be precise enough to generate assertions
- Catch ambiguous `/implement` blocks before a full session is invested
- Provide a clear "done" signal (all tests green) rather than subjective judgment
- Prevent the anti-greenwashing failure mode where tests are unconsciously shaped to match buggy code

## Fix: Defense in Depth

Two independent enforcement points were added:

### Proactive Generator: `roadmap-from-plan` Step 6.5

After the ROADMAP file is written and approved, test-architect is spawned for each module. Tests are committed with `test(<module>): pre-implementation test spec from ROADMAP`. This happens as part of the normal roadmap workflow — no extra user action needed.

### Active Gate: `implement` Phase 0

Before any planning or code writing, `/implement` checks for pre-existing test files. If missing, it warns the user explicitly and offers to spawn test-architect or get an explicit opt-out. **Never silently skips.**

### Authority: `CLAUDE.md` Implementation Completion Sequence

Step 0 of the sequence now requires test verification before any code is written. This is the behavioral contract that governs all implementation work.

## Design Principle

Either enforcement point alone could be bypassed:
- If someone runs `/implement` without `/roadmap-from-plan`, Phase 0 catches the gap
- If Step 6.5 fails or is skipped, Phase 0 catches the gap
- If someone ignores Phase 0, the CLAUDE.md behavioral contract makes it a visible violation

The old design relied on a single passive check that was structurally guaranteed to be false. The new design has two active checks at different pipeline stages.

## Files Changed

| File | Change |
|------|--------|
| `.claude/commands/roadmap-from-plan.md` | Replaced passive Step 7.5 with active Step 6.5 |
| `.claude/commands/implement.md` | Added Phase 0 gate, removed passive Step 3.5, updated task list |
| `CLAUDE.md` | Added Implementation Completion Sequence with Step 0 |
| `docs/reviews/tdd-pipeline-audit.md` | This file |
