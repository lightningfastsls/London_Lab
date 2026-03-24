---
name: refactor
description: Plan vault restructuring from config changes. Compares config.yaml against derivation.md, identifies dimension shifts, shows restructuring plan, executes on approval. Triggers on "/refactor", "restructure vault".
version: "1.0"
generated_from: "arscontexta-v1.6"
user-invocable: true
context: fork
allowed-tools: Read, Write, Edit, Grep, Glob, Bash
argument-hint: "[dimension|--dry-run] — focus on specific dimension or preview without approval prompt"
---

[Full refactor SKILL.md content - Configuration-driven restructuring. See original refactor/SKILL.md for complete documentation.]

## KG Conventions (embedded from CLAUDE.md)

### Wiki Links — Rename Safety
- **Never rename manually** — use `ops/scripts/rename-note.sh "old" "new"` (renames with git mv, updates ALL wiki links)
- Scripts: `orphan-notes.sh`, `dangling-links.sh`, `backlinks.sh`, `link-density.sh`, `validate-schema.sh`

### Topic Maps — Split/Merge Rules
- Three-tier: `index.md -> topic maps -> notes`. Context phrases required on all links.
- Split at ~50 notes (warning 40, critical 60). **Consult arscontexta-expert before any split/merge/creation.**
- Expert validates split boundaries, prevents premature subordinate-level splits (< 15 notes)

### Graph Analysis
- Vault = graph database: nodes (markdown), edges (wiki links), properties (YAML frontmatter)
- Key operations: triangle detection, orphan detection, bridge detection, link density (target: 3+)
- Use /graph for interactive analysis

Full reference: `docs/workflow/knowledge-graph-reference.md`
