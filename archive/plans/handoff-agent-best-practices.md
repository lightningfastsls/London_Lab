# Handoff: Apply Agent Best Practices to Our Projects

## Context for Claude Code

This handoff comes from a research session on Web Claude analyzing the Claude Code source leak (March 31, 2026) and extracting actionable improvements for our workflow. The findings are organized into two phases:

1. **Phase 1 — Discovery**: You explore the repos and report back what you find. Do NOT make changes yet. Just answer the questions.
2. **Phase 2 — Implementation**: Based on your Phase 1 answers, we decide together what to apply and where.

We have three codebases to consider:
- **mickey-lab** (USV detection pipeline — Python)
- **Cloudy Claude** (MHS company website — TypeScript/Next.js)
- **arscontexta** (knowledge vault)

---

## Phase 1 — Discovery

Answer each question below. Be specific — file paths, line counts, tool versions. Don't assume anything, go look.

### A. Current CLAUDE.md State

1. Find and read the CLAUDE.md (or equivalent) in each repo. How many lines is each? Does any of them have a section about what to preserve during compaction? Does any of them have a verification protocol (run tests/typecheck after edits)?

2. Does any repo have a `.claude/` directory? If so, what's in it — any `rules/` subdirectory, any `MEMORY.md`, any topic files?

3. Is there a `CLAUDE.local.md` anywhere? What's in it?

### B. Verification Infrastructure

4. In **mickey-lab**: What's the test runner? Is there a `mypy` config (mypy.ini, pyproject.toml section, setup.cfg)? What's the command to run type checking? What's the command to run the test suite? Are there any existing pre-commit hooks?

5. In **Cloudy Claude**: Is TypeScript strict mode enabled in `tsconfig.json`? What's the lint command (if ESLint is configured)? What's the test command (if any test framework is set up)? What's the typecheck command?

6. For each repo — what's the single fastest command that would catch "the agent wrote broken code"? Something that runs in under 10 seconds ideally.

### C. Large Files and Dead Code

7. In each repo, list all files over 500 lines of code. For any over 1000 LOC, flag them specifically — these are compaction hazards and chunked-read risks.

8. In each repo, are there obvious dead code patterns? Unused imports, commented-out blocks, orphaned exports? Don't fix anything — just give me a rough sense of the hygiene level.

### D. Memory Architecture (arscontexta)

9. In **arscontexta**: How is the vault currently structured? How many topic maps exist? What format are they in? How many total notes? What's the current index/routing mechanism?

10. Is there a `MEMORY.md` or equivalent index file? If so, how many lines is it? Is it under or over 200 lines?

11. Is there any existing consolidation or maintenance process — a script, a cron job, a skill that cleans up stale entries?

12. Find and read any existing plan files related to vault health, retrieval redesign, or topic map routing. Summarize what was already planned vs. what's been implemented.

### E. Existing Workflow Patterns

13. In each repo, is there an `AGENTS.md` or any agent configuration files? What agents/subagents are defined?

14. Are there any existing skills (`.claude/skills/` or similar)? List them.

15. What's the git branching strategy in each repo? Any branch protection, CI/CD pipelines, or automated checks?

---

## Phase 2 — Implementation Plan

Once Phase 1 is answered, we'll implement the following. **Do not start these until Phase 1 is complete and reviewed.**

### 2.1 — CLAUDE.md Verification Protocol

**What we learned**: Claude Code's internal definition of "task complete" is "did bytes hit disk?" — not "does it compile." The leaked source confirms a 29-30% false claims rate. Post-edit verification was gated behind internal flags.

**What to add** (adapt based on Phase 1 answers about available tooling):

For Python repos:
```markdown
## Verification Protocol
After EVERY code modification, before reporting success:
1. Run `<syntax check command from Phase 1>` for syntax
2. Run `<typecheck command from Phase 1>` for types
3. Run `<test command from Phase 1> -x -q` for relevant tests
4. If ANY fail, fix before reporting. Never say "Done" with failing checks.
If no tests exist for modified code, state that explicitly.
```

For TypeScript repos:
```markdown
## Verification Protocol
After EVERY code modification, before reporting success:
1. Run `<typecheck command from Phase 1>`
2. Run `<lint command from Phase 1>` if configured
3. Run relevant tests if they exist
4. If ANY fail, fix before reporting.
```

### 2.2 — Compaction Preservation Section

**What we learned**: When auto-compaction fires (~95% of 200K context), it compresses everything into a summary and re-injects only the 5 most recently accessed files (capped at 5K tokens each). Your CLAUDE.md survives intact because it's reloaded from disk. You can guide what the summary preserves by adding explicit instructions.

**What to add** to each CLAUDE.md:

```markdown
## Compaction Preservation
When compacting or summarizing this conversation, you MUST preserve:
- All file paths that were modified, with the specific changes made
- The current task and its completion status
- Any failing test output or error messages still being debugged
- Architectural decisions made during this session
- The current phase number if working through a phased plan
Do NOT discard: debugging hypotheses, variable/function names under discussion,
or the specific line numbers of active work.
```

### 2.3 — Pre-Refactor Dead Code Cleanup ("Step 0")

**What we learned**: Dead code accelerates context compaction. Every unused import, orphaned export, and debug log consumes tokens that contribute nothing to the task but push the context window toward the compaction threshold faster. Compaction is lossy — it's not compression, it's amputation.

**What to add**:

```markdown
## Pre-Refactor Protocol
Before any structural refactor on a file >300 LOC:
1. FIRST PASS (cleanup only): Remove dead imports, unused exports,
   orphaned variables, leftover debug logs, commented-out code.
   Commit separately: "chore: dead code cleanup before refactor"
2. SECOND PASS (actual refactor): Only after cleanup is committed.
Each phase should touch no more than 5 files to avoid triggering
mid-task compaction.
```

### 2.4 — Context Decay and Re-Read Rule

**What we learned**: Auto-compaction silently replaces your detailed memory of files with a compressed summary. The agent doesn't know it happened. After ~10 messages it may confidently edit files based on stale/hallucinated memory of their contents.

**What to add**:

```markdown
## Context Decay Rule
After 10+ messages in a conversation:
- Re-read ANY file before editing it. Do not trust cached memory.
- After any compaction event, treat ALL file knowledge as stale.
- The cost of a redundant read is trivial; the cost of editing against
  stale context is a broken codebase.
```

### 2.5 — Rename/Refactor Search Checklist

**What we learned**: Claude Code uses grep, not AST analysis. A single grep for a renamed symbol will miss dynamic imports, string references, re-exports, barrel files, type references, and test mocks.

**What to add**:

```markdown
## Rename/Signature Change Protocol
When renaming any function, type, class, or export, search separately for:
1. Direct calls and references
2. Type-level references (interfaces, generics, annotations)
3. String literals containing the name (routes, logs, test descriptions)
4. Dynamic imports and require() calls
5. Re-exports and barrel file entries (index.ts / __init__.py)
6. Test files: mocks, fixtures, monkeypatches
7. Config files (tsconfig paths, webpack/vite, pyproject.toml)
Do not assume a single grep caught everything.
```

### 2.6 — Large File and Search Truncation Rules

**What we learned**: File reads are capped at ~2,000 lines / 25K tokens. Content beyond that is silently truncated — the agent doesn't know what it didn't see. Similarly, search results over ~50K characters are truncated to a ~2K preview.

**What to add**:

```markdown
## Large File Protocol
For files over 500 LOC: read in sequential chunks (lines 1-500, 501-1000, etc.).
For files over 1500 LOC: consider whether the file should be split before editing.
State the total line count after your first read.

## Search Truncation Awareness
If any search returns suspiciously few results, assume truncation occurred.
Re-run with narrower scope (single directory, stricter glob).
State when you suspect truncation: "Results may be truncated — narrowing scope."
```

### 2.7 — arscontexta Consolidation Skill

**What we learned**: Anthropic's AutoDream system performs a four-phase memory consolidation cycle that maps directly onto arscontexta's architecture. MEMORY.md = topic map index. Topic files = individual notes. The consolidation process merges duplicates, converts relative dates to absolute, deletes contradicted facts, and keeps the index under 200 lines.

**What to implement** (as a Claude Code skill or a standalone script — decide after Phase 1):

A periodic consolidation process with four phases:

```
Phase 1 — Orient:
  Read the current index/routing layer.
  List all topic maps and their status.
  Identify what exists before making changes.

Phase 2 — Gather recent signal:
  Find notes that were added or modified since last consolidation.
  Identify topic maps that have drifted from their actual content.
  Find contradictions between the index and the notes it points to.

Phase 3 — Consolidate:
  Merge near-duplicate notes into single entries.
  Convert relative dates to absolute dates.
  Delete or correct contradicted facts at the source.
  Update topic maps to reflect current state of their notes.

Phase 4 — Prune and index:
  Keep the main index under 200 lines / 25KB.
  Remove pointers to notes that no longer exist.
  Add pointers to newly important notes.
  Resolve contradictions between topic maps.
```

**Write discipline rule**: Only update the index AFTER a successful write to the topic file. Never update the index speculatively.

### 2.8 — Coordinator Mode (Experimental)

**What we learned**: Claude Code has a multi-agent coordinator mode (`CLAUDE_CODE_COORDINATOR_MODE=1`) that spawns parallel workers with independent context windows. Each worker gets its own ~200K token budget.

**Discovery question for Phase 1**: Before enabling this, we need to understand which tasks in our repos are genuinely independent (no shared file edits). After Phase 1 answers about repo structure, we'll identify specific use cases — likely independent Next.js pages or tRPC routes in Cloudy Claude, or independent test files in mickey-lab.

---

## How To Use This Document

1. Start Claude Code in any of the three repos
2. Give it this file: "Read this handoff and begin Phase 1"
3. Let it explore and answer all Phase 1 questions
4. Share the Phase 1 answers with Web Claude (me) if you want to discuss before proceeding
5. Or proceed directly to Phase 2 if the answers are straightforward
6. Implement Phase 2 items one at a time, committing each separately

The Phase 2 items are ordered by impact: verification protocol first (highest ROI), compaction preservation second, then the rest. arscontexta consolidation (2.7) is the most complex and should probably be its own session.

---

## Source

This knowledge was extracted from:
- The Claude Code source leak (March 31, 2026) — 512K lines of TypeScript exposed via npm sourcemap in version 2.1.88
- Analysis of the three-layer compaction system (MicroCompact → AutoCompact → Full Compact)
- The AutoDream memory consolidation system prompt (services/autoDream/consolidationPrompt.ts)
- The MEMORY.md architecture (index-of-pointers pattern, 200-line cap, on-demand topic file loading)
- Multiple community analyses from VentureBeat, The Register, Hacker News, and GitHub mirrors
