---
name: verify
description: Combined verification — recite (description quality via cold-read prediction) + validate (schema compliance) + review (health checks). Use as a quality gate after creating notes or as periodic maintenance. Triggers on "/verify", "/verify [note]", "verify note quality", "check note health".
user-invocable: true
allowed-tools: Read, Write, Edit, Grep, Glob, mcp__qmd__vector_search
context: fork
---

[Full verify SKILL.md content - kept concise due to token limits. This performs three checks: recite (cold-read test), validate (schema), review (health). See original verify/SKILL.md for complete documentation.]

## KG Conventions (embedded from CLAUDE.md)

### Schema (required fields)
- `description` (max 200 chars, adds context beyond title), `topics` (wiki links to topic maps)
- Domain: `type` (finding|decision|method|hypothesis|baseline|open-question|pattern), `confidence` (proven|likely|experimental|speculative), `conditions`, `meta_state` (current|outdated|superseded)
- Templates in `templates/` are the single source of truth for schema

### Discovery-First Checklist
Every note must be findable by a future agent who doesn't know it exists. Verify:
1. **Title as claim** — reads naturally when linked: `since [[title]]`
2. **Description quality** — adds info beyond the title
3. **Topic map membership** — linked from at least one topic map
4. **Composability** — linkable without dragging irrelevant context

### Link Health
- Link density target: 3+ outgoing links per note
- Dangling links = demand signals — create the missing note or fix the link
- Orphan notes (no incoming links) need connections — run /reflect

Full reference: `docs/workflow/knowledge-graph-reference.md`
