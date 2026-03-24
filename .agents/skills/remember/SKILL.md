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

## EXECUTE NOW

**Target: $ARGUMENTS**

Parse immediately:
- If target contains a description: capture that as a methodology note
- If target is empty: review recent session for friction patterns

**Three modes:**

1. **Explicit** — "/remember [description]": Create a methodology note in ops/methodology/ capturing the described friction, insight, or process correction.

2. **Contextual** — "/remember": Review recent tool calls and corrections in this session. Identify patterns: what went wrong, what was corrected, what should be different next time.

3. **Session mining** — "/remember session": Scan session transcripts for recurring friction patterns across multiple sessions.

**Output:** Create atomic methodology note with:
- What was observed (the friction or insight)
- Why it matters (impact on workflow)
- What should change (proposed correction)

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
