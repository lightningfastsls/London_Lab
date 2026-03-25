# Retrieval Overhaul — Phase 0 Audit Summary
<!-- Generated: 2026-03-25 -->

## Coverage: 98.2% — PASS (>95% threshold → proceed to Phase 1)

**518 / 527** non-MOC notes are referenced in at least one topic map.
26 MOCs confirmed. All `topics:` frontmatter references are valid (no broken links).

## Uncovered Notes (9)

| Note | Likely Type | Action |
|------|------------|--------|
| Clauset et al 2009 MLE produces the gold standard power law fit... | method | Could go in representation-learning |
| Crutchfield and Feldman 2003 block entropy extrapolation... | method | Could go in representation-learning |
| Miller-Madow correction compensates for finite sample bias... | method | Could go in representation-learning |
| conditional entropy by lag probes single-token influence... | method | Could go in representation-learning |
| entropy-based Zipf estimation cross-validates MLE... | method | Could go in representation-learning |
| n-gram idiom detection identifies compositional phrases... | method | Could go in representation-learning |
| null models are essential for interpreting information-theoretic... | method | Could go in representation-learning |
| pre-code questions for LMT integration... | planning | Could go in behavioral-integration |
| molt-cornelius-what-no-single-note-contains-agentic-notetaking... | meta/bridge | Meta artifact, OK to leave uncovered |

All 9 are information-theoretic methods or meta-notes — none are active knowledge claims or decisions. Ripgrep co-primary catches them by content. No backfill needed before Phase 1.

## Topic Map Quality

### Split Candidates (>40 notes)
- **agent-memory**: 62 note-links (highest — strong split candidate)
- **context-management**: 46 note-links
- **representation-learning**: 43 note-links
- **agent-governance**: 41 note-links

### Sparse (<5 notes)
- **experimental-methods**: 0 direct note-links (hub-only MOC linking to sub-maps)

### Section Headers
All 26 MOCs have `## Section` headers (none flat).

### Context Phrases
Most MOCs have context phrases on links. Two exceptions noted:
- **agent-external-cognition**: reported 0 context phrases (likely formatting variation)
- **graph-structure**: reported 0 context phrases

## qmd Reference Catalog

| Category | Count | Examples |
|----------|-------|---------|
| **Skills (active)** | 12 | kcheck, ask, architect, recommend, reflect, reweave, learn, reduce, health, verify, refresh-human-docs, sync |
| **Hook** | 1 | session-orient.sh (lines 211-277) |
| **Notes** | 6 | agent-memory MOC, knowledge-related notes |
| **Methodology (READ-ONLY)** | 16 | Various research claims |
| **Reference (READ-ONLY)** | 11 | kernel.yaml, components.md, etc. |
| **Documentation** | 7 | Knowledge-graph-reference, investigations |
| **Investigations** | 7 | 4.1-4.4 series, handoffs |
| **Operational** | 7 | goals.md, derivation, health reports |
| **Root docs/plans** | ~12 | Various plan/audit files |

## session-orient.sh Knowledge Activation (lines 211-277)

**Current behavior:**
1. Checks if `qmd` command exists and goal threads are non-empty
2. For each goal thread (max 5), extracts title and description via regex
3. Runs `qmd search "$clean_title" --limit 3 --json` (keyword/BM25)
4. Runs `qmd vsearch "$query" --limit 3 --json` (vector search)
5. Parses results via inline python3 JSON extraction
6. Writes `session-relevance.md` with results or "No strong matches"
7. Falls back to "qmd unavailable" message if qmd not found

**Dependencies:** `qmd` binary, `python3` (for JSON parsing), bash regex for thread extraction.
**Non-obvious:** Uses `sed -n '/\[/,$p'` to strip non-JSON prefix from vsearch output. Thread regex handles both `**Bold**` and plain title formats.

## Gate Decision

**PROCEED TO PHASE 1.** Coverage is 98.2%, well above the 95% threshold. All 9 uncovered notes are methods/meta-artifacts that ripgrep will catch by content.
