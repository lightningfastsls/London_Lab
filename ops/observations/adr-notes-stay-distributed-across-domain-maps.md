---
category: methodology
date: 2026-02-27
status: archived
archived: 2026-03-02
archived_by: rethink-2026-03-02
---

# ADR-derived notes stay distributed across domain topic maps

## Decision
ADR-derived notes (type: decision) remain distributed across domain topic maps
(signal-processing, detection, classification, etc.) rather than getting their
own dedicated `architectural-decisions.md` topic map.

## Rationale (arscontexta-expert analysis)
Seven methodology claims converge on this recommendation:

1. **Faceted classification** -- `type: decision` in YAML already provides the
   queryable axis (`rg "^type: decision" notes/ -l`). A dedicated MOC would
   duplicate this retrieval path.

2. **MOCs are attention management devices** -- domain maps orient agents to
   coherent work domains. An "architectural decisions" MOC organizes by
   content-kind (format), not domain (substance).

3. **Basic level categorization** -- "architectural decisions" sits at the
   superordinate level (covers everything, orients toward nothing).

4. **Cross-MOC membership** -- decision notes naturally bridge domains (e.g.,
   FFT params bridge signal-processing AND detection). A dedicated ADR map
   would sever these cross-domain connections.

5. **No topological community** -- decision notes link to domain neighbors, not
   to each other. A decision about JSON labels doesn't cluster with FFT params.

## For audit use cases
Run `rg "^type: decision" notes/ -l` or use qmd semantic search. If the pattern
recurs, consider a generated index in ops/ (temporal) rather than a curated MOC
in notes/ (durable).
