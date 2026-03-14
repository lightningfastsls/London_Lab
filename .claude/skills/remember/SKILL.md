---
name: remember
description: Capture friction as methodology notes. Three modes — explicit description, contextual (review recent corrections), session mining (scan transcripts for patterns). Triggers on "/remember", "/remember [description]".
version: "1.0"
generated_from: "arscontexta-v1.6"
user-invocable: true
context: fork
model: sonnet
allowed-tools: Read, Write, Edit, Grep, Glob, Bash
---

[Full remember SKILL.md content - Friction capture and methodology learning. See original remember/SKILL.md for complete documentation.]

## KG Conventions (embedded from CLAUDE.md)

### Operational Learning Loop
- **Observations** (ops/observations/) — friction, surprises, process gaps. Category: friction|surprise|process-gap|methodology.
- **Tensions** (ops/tensions/) — contradictions between notes or implementation vs methodology. Status: pending|resolved|dissolved.
- Triggers: 10+ observations -> /rethink. 5+ tensions -> /rethink.

### Content Routing
- Friction signals, patterns -> ops/observations/
- Methodology self-knowledge -> ops/methodology/
- Knowledge claims, insights -> notes/ (via /reduce, never directly)
- Time-bound commitments -> ops/reminders.md

Full reference: `docs/workflow/knowledge-graph-reference.md`
