---
name: reweave
description: Update old notes with new connections. The backward pass that /reflect doesn't do. Revisit existing notes that predate newer related content, add connections, sharpen claims, consider splits. Triggers on "/reweave", "/reweave [note]", "update old notes", "backward connections", "revisit notes".
user-invocable: true
allowed-tools: Read, Write, Edit, Grep, Glob, Bash, mcp__qmd__search, mcp__qmd__vector_search, mcp__qmd__deep_search, mcp__qmd__status
context: fork
---

[Full reweave SKILL.md content - kept concise due to token limits. This is the backward pass phase that updates older notes with new connections found after they were created. See original reweave/SKILL.md for complete documentation.]

## KG Conventions (embedded from CLAUDE.md)

### Wiki Links
- `[[title]]` basic | `since [[claim]]` as prose (preferred) | `contradicts [[finding]]` typed
- Link density target: 3+ outgoing links per note
- Never rename manually — use `ops/scripts/rename-note.sh "old" "new"`
- Dangling links = demand signals. Orphan notes = need /reflect.

### Topic Maps
- Three-tier: `index.md -> topic maps -> notes`. Context phrases required on all links.
- Split at ~50 notes (warning 40, critical 60). **Consult arscontexta-expert before any split/merge/creation.**
- Bad: `- [[note title]]` | Good: `- [[note title]] -- explains why this link matters`

### Maintenance Thresholds (relevant to reweave)
- Stale notes: > 30 days old + < 2 incoming links -> Run /reweave
- Orphan notes: persistent > 7 days -> Run /reflect
- Topic map oversized: > 40 notes -> Split into sub-topic-maps

### Schema (required fields)
- `description` (max 200 chars, adds context beyond title), `topics` (wiki links to topic maps)
- Domain: `type` (finding|decision|method|hypothesis|baseline|open-question|pattern), `confidence`, `conditions`, `meta_state`

### Description Quality Gate (when editing descriptions)
If you modify a note's `description` field, verify before saving:
- **No restatement**: description says something the title doesn't
- **No subject echo**: description starts with a different noun/subject than the title
- **Answers "so what?"**: a cold reader would know WHY to open this note from the description alone

Full reference: `docs/workflow/knowledge-graph-reference.md`
