# Phase 2 Implementation — Tailored to Phase 1 Findings

## Before You Start

Both CLAUDE.md files are already long (mickey-lab: 347 lines, Cloudy Claude: 465 lines).
Adherence degrades with length. Before adding new sections:

1. Read the current CLAUDE.md
2. Identify any sections that are **path-specific** (only relevant when working in certain directories) and could be moved to `.claude/rules/<topic>.md` with a `paths:` frontmatter
3. Identify any sections that duplicate information available in README, AGENTS.md, or other docs
4. Propose trims BEFORE adding new content. Show me what you'd move/cut and where it would go
5. Target: keep each CLAUDE.md under 300 lines after adding the new sections

## Items to Add (in order)

### 1. Compaction Preservation Section (both repos)

Add this section near the top of each CLAUDE.md, right after project description.
This is the highest-ROI addition — it costs almost nothing but protects against the biggest failure mode in long sessions.

```markdown
## Compaction Preservation
When compacting or summarizing this conversation, ALWAYS preserve:
- All file paths modified and what changed in each
- Current task, its phase, and completion status
- Failing test output or error messages still being debugged
- Architectural decisions made this session
- Active debugging hypotheses
Do NOT discard line numbers, variable names, or function signatures under active discussion.
```

### 2. Context Decay Rule (both repos)

Add immediately after the compaction section.

```markdown
## Context Decay
After 10+ messages: re-read any file before editing it.
After any compaction event: treat ALL file memory as stale.
Never edit from memory alone in a long session.
```

### 3. Verification Protocol — mickey-lab

The existing verification in CLAUDE.md mentions py_compile + pytest. Strengthen it:

Find the current verification section and ADD the following:

```markdown
- After every code change, before reporting success:
  1. `python -m py_compile <modified_file>` (syntax)
  2. `pytest <relevant_test> -x -q` if tests exist for modified code
  3. If no tests exist, state: "No test coverage for this change"
- Never say "Done" or "Complete" with failing checks
- NOTE: mypy is not configured. Do not claim type-safety without it.
```

**Separate action**: Consider whether mypy should be configured. This is a real gap for a scientific pipeline — type errors in numpy/scipy array operations are a common source of silent bugs. But this is a project decision, not a CLAUDE.md change. Flag it for Shachar to decide.

### 4. Verification Protocol — Cloudy Claude

Find the current verification approach (anti-greenwashing test table) and ADD:

```markdown
- After every code change, before reporting success:
  1. `pnpm type-check` (TypeScript strict across monorepo)
  2. `pnpm lint` if touching packages with ESLint
  3. Run relevant vitest suite if tests exist (api/, madaf-sync/)
  4. If no tests exist for the package, state: "No test coverage for this change"
- Never say "Done" or "Complete" with failing checks
```

### 5. Large File Protocol — mickey-lab only

Five files over 1000 LOC need this. Add to mickey-lab CLAUDE.md:

```markdown
## Large File Protocol
These files exceed 1000 LOC and MUST be read in chunks:
- main_window.py (1,819 lines) — read in 500-line segments
- assembler.py (1,490 lines)
- repertoire_stats.py (1,142 lines)
- test_fp_filter.py (1,119 lines)
- information_theory.py (1,075 lines)
For ANY file over 500 LOC: use offset/limit to read in chunks.
Never assume a single read captured the full file.
State the total line count after your first read.
```

### 6. Rename/Refactor Checklist (both repos, as a .claude/rules/ file)

This is long enough that it should NOT go in the main CLAUDE.md.
Create `.claude/rules/rename-safety.md` in each repo:

```markdown
---
paths:
  - "**/*.py"
  - "**/*.ts"
  - "**/*.tsx"
---
## Rename/Signature Change Protocol
When renaming any function, type, class, or export, search separately for:
1. Direct calls and references
2. Type-level references (interfaces, generics, type annotations)
3. String literals containing the name (routes, logs, test descriptions)
4. Dynamic imports / require() calls
5. Re-exports and barrel files (index.ts / __init__.py)
6. Test files: mocks, fixtures, monkeypatches
7. Config files (tsconfig, pyproject.toml, webpack/vite)
If results from any search seem suspiciously few, assume truncation and re-run with narrower scope.
```

### 7. Search Truncation Awareness (both repos, append to rename-safety.md or add to CLAUDE.md — your call)

```markdown
## Search Truncation
If any search returns suspiciously few results (<5 when you'd expect >10), assume truncation.
Re-run directory-by-directory. State when you suspect truncation occurred.
```

## Items to Skip

### Pre-Refactor Dead Code Cleanup (2.3)
Both repos are already clean. Add as a single-line preventive note if you want, but don't create a full section:
```
Before refactoring files >300 LOC, first commit a cleanup pass removing dead imports/exports.
```

### arscontexta Consolidation Skill (2.7)
Do NOT build a new skill. The existing infrastructure (/health, /rethink, condition triggers, vault canary) already implements the AutoDream pattern. The gap is operational, not architectural:
- Run `/rethink` NOW to clear 11 pending observations and 6 pending tensions
- After clearing, check whether the condition-based triggers need threshold tuning
- Consider adding a periodic reminder (weekly?) to check thresholds, since the current system depends on someone noticing they're exceeded

The one idea worth borrowing from AutoDream that your system may not have: **write discipline** — only update the topic map index AFTER a successful write to the note it points to. Check if your existing /reflect and /rethink skills already enforce this ordering. If not, it's a small but important addition.

### Coordinator Mode (2.8)
Defer. Identify specific use cases first. Possible candidates after reviewing repo structure:
- Cloudy Claude: parallel work on independent Next.js pages or tRPC routes
- mickey-lab: parallel test file updates that don't share imports
But don't enable it blindly.

## Execution Order

1. Clear arscontexta backlog: `/rethink` (do this first, no code changes needed)
2. Trim CLAUDE.md files — move path-specific content to `.claude/rules/`
3. Add Compaction Preservation (item 1) to both repos
4. Add Context Decay (item 2) to both repos
5. Strengthen verification protocols (items 3-4)
6. Add Large File Protocol to mickey-lab (item 5)
7. Create rename-safety.md rules file in both repos (item 6)

Items 2-7 can be a single commit per repo: "chore: add agent safety protocols from leak analysis"
