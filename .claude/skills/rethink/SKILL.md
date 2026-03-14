---
name: rethink
description: Challenge system assumptions against accumulated evidence. Triages observations and tensions, detects patterns, generates proposals. The scientific method applied to knowledge systems. Triggers on "/rethink", "review observations", "challenge assumptions", "what have I learned".
user-invocable: true
allowed-tools: Read, Write, Edit, Grep, Glob, Bash, AskUserQuestion
context: fork
---

[Full rethink SKILL.md content - Evidence-driven system evolution. See original rethink/SKILL.md for complete documentation.]

## KG Conventions (embedded from CLAUDE.md)

### Operational Learning Loop
- **Observations** (ops/observations/) — friction, surprises, process gaps. Category: friction|surprise|process-gap|methodology.
- **Tensions** (ops/tensions/) — contradictions between notes or implementation vs methodology. Status: pending|resolved|dissolved.
- Triggers: 10+ observations -> /rethink. 5+ tensions -> /rethink.

### Maintenance Thresholds (relevant to rethink)
- Pending observations >= 10 -> Run /rethink
- Open tensions >= 5 -> Run /rethink
- Orphan notes persistent > 7 days -> Run /reflect
- Stale notes > 30 days old + < 2 incoming links -> Run /reweave
- Inbox items >= 3 -> Run /reduce or /pipeline

### Tension Resolution
- When two notes contradict, capture in ops/tensions/ with status: pending|resolved|dissolved
- Resolution updates both notes with the outcome
- Dissolved tensions mean the contradiction was apparent, not real

Full reference: `docs/workflow/knowledge-graph-reference.md`
