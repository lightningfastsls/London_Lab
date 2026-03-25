---
name: refresh-human-docs
description: Regenerate human-readable docs from knowledge graph and ops state. Reads notes/, ops/goals.md, ROADMAP files, and topic maps to synthesize up-to-date PROJECTS.md and DECISIONS.md in docs/human/. Triggers on "/refresh-human-docs", "refresh human docs", "regenerate project dashboard".
version: "1.0"
user-invocable: true
allowed-tools: Read, Write, Grep, Glob, Bash
context: fork
---

## Purpose

Regenerate human-readable documentation in `docs/human/` from the knowledge graph
and operational state. These files are for HUMANS — they should be comprehensive,
well-formatted, and include context that agents already have access to through
`ops/goals.md` and `notes/`.

## Step 1: Gather State

Read these files to understand current project state:

1. **`ops/goals.md`** — active threads, completed items, blockers
2. **`notes/index.md`** — knowledge graph entry point, list of topic maps
3. **All topic maps** listed in index.md (e.g., `notes/detection.md`, `notes/classification.md`, etc.)
4. **`ROADMAP.md`** — main pipeline phases and status
5. **All `ROADMAP_*.md` files** in the repo root — additional project roadmaps
6. **`IMPLEMENTATION_PROGRESS.md`** — latest session log entries (read first 100 lines for recent activity)

## Step 2: Summarize to User

Present a brief summary:

```
Knowledge Graph: N notes across M topic maps
Active Threads: [list from ops/goals.md]
ROADMAP files found: [list]
Decision notes found: [count from rg "^type: decision" notes/ -l]
```

## Step 3: Ask What to Refresh

Use AskUserQuestion to ask:

**"Which human-readable docs should I regenerate?"**

Options:
- (a) PROJECTS.md — project dashboard with status, phases, blockers, next actions
- (b) DECISIONS.md — architecture decision records reconstructed from decision notes
- (c) Both

## Step 4: Generate PROJECTS.md (if selected)

Synthesize `docs/human/PROJECTS.md` from:
- `ops/goals.md` active threads → project list with status
- `ROADMAP*.md` files → phase tables with completion status
- Topic maps → domain descriptions
- Recent `IMPLEMENTATION_PROGRESS.md` entries → latest activity

**Format to follow:** Match the structure of the existing `docs/human/PROJECTS.md`:
- Quick Status Dashboard table at top
- Per-project sections with: What It Is, Phase Status table, Blockers, Next Action
- External Dependencies & Blockers table
- Suggested Priority Order

Add a generation timestamp at the top:
```
> **Generated:** YYYY-MM-DD by `/refresh-human-docs`
> **Source:** ops/goals.md, notes/, ROADMAP*.md
```

## Step 5: Generate DECISIONS.md (if selected)

Synthesize `docs/human/DECISIONS.md` from:
- All notes with `type: decision` in frontmatter (search: `rg "^type: decision" notes/ -l`)
- Read each decision note to extract: the claim (title), description, confidence, conditions
- Reconstruct ADR format: ADR-NNN: Title, Status, Context, Decision, Consequences

**Ordering:** Group by topic map membership. Within each group, order by confidence
(proven → likely → experimental → speculative).

**For ADRs that have both a decision note AND an entry in the original DECISIONS.md:**
Use the note as the authoritative source (it may be more up-to-date), but preserve
the ADR numbering from the original file.

Add a generation timestamp at the top:
```
> **Generated:** YYYY-MM-DD by `/refresh-human-docs`
> **Source:** notes/ (type: decision), docs/human/DECISIONS.md (ADR numbering)
```

## Step 6: Report

After writing, report:
- Which files were regenerated
- Summary of changes (new sections added, removed, updated)
- Note count and topic map count used as source
- Any decision notes that couldn't be mapped to an ADR number (new decisions without legacy numbering)

## Important Rules

- **Never modify `ops/goals.md` or `notes/`** — this skill READS the KG, it doesn't write to it
- **Always overwrite** `docs/human/PROJECTS.md` and/or `docs/human/DECISIONS.md` completely — these are generated artifacts, not manually curated
- **Include the generation timestamp** — so readers know when the doc was last refreshed
- **Preserve ADR numbering** — ADR-001 through ADR-014 have established numbers. New decisions without ADR numbers should be listed in a separate "Additional Decisions" section
