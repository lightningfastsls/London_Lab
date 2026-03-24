---
name: seed
description: Add a source file to the processing queue. Checks for duplicates, creates archive folder, moves source from inbox, creates extract task, and updates queue. Triggers on "/seed", "/seed [file]", "queue this for processing".
version: "1.0"
generated_from: "arscontexta-v1.6"
user-invocable: true
context: fork
model: sonnet
allowed-tools: Read, Write, Edit, Grep, Glob, Bash
argument-hint: "[file] — path to source file to seed for processing"
---

[Full seed SKILL.md content - Entry point for processing pipeline. See original seed/SKILL.md for complete documentation.]

## KG Conventions (embedded from CLAUDE.md)

### Processing Pipeline
- **NEVER write directly to notes/.** Route: inbox/ -> /reduce -> notes/. Direct writes skip quality gates.
- Phases: /seed (research) -> /reduce (extract) -> /reflect (connect) -> /reweave (backward pass) -> /verify (quality check)
- Processing depth configured in ops/config.yaml (deep | standard | quick)

### Research Provenance
- Preserve the chain: `source query -> inbox file (metadata preserved) -> /reduce -> notes/`
- Every claim must be traceable to its origin
- Source files in inbox/ must preserve metadata (research_tool, research_query, date_accessed)

Full reference: `docs/workflow/knowledge-graph-reference.md`
