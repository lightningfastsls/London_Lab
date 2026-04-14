---
description: The vault's self-knowledge -- derivation rationale, configuration state, and operational evolution history
type: moc
---
# methodology

This folder records what the system knows about its own operation -- why it was configured this way, what the current state is, and how it has evolved. Meta-skills (/rethink, /arscontexta:architect) read from and write to this folder. /remember captures operational corrections here.

## Derivation Rationale
- [derivation-rationale](derivation-rationale.md) -- Why each configuration dimension was set the way it was

## Processing Strategy
- [bulk-source-processing-strategy](bulk-source-processing-strategy.md) -- Phase-batched cluster-grouped processing for ingesting multiple research topics
- [reflect-timing-within-cluster-processing](reflect-timing-within-cluster-processing.md) -- Defer /reflect until all cluster reductions complete; cross-source reflection beats single-source

## Configuration State
(Populated by /rethink, /arscontexta:architect)

## Evolution History
- [rethink-2026-03-02](rethink-2026-03-02.md) -- Triaged 9 observations + 9 tensions, identified 3 patterns (automation lifecycle gaps, self-review bias, topic map sprawl), generated 5 proposals. Archived 4 observations + 4 tensions. Key actions: doc freshness rule in CLAUDE.md, /reduce description self-check, inbox archival after /reduce, topic map splits scheduled.
- [rethink-2026-04-06](rethink-2026-04-06.md) -- Phase A1+A2 analysis complete, Tension 5 resolved (3 classification approaches operational), 9252 dataset emerged, 2 new patterns (data assumptions surface during analysis, tool-gap-to-abundance trajectory). 4 proposals.

## How to Use This Folder

Browse notes: `ls ops/methodology/`
Query by category: `rg '^category:' ops/methodology/`
Find active directives: `rg '^status: active' ops/methodology/`
Ask the research graph: `/arscontexta:ask [question about your system]`

Meta-skills (/rethink, /arscontexta:architect) read from and write to this folder.
/remember captures operational corrections here.
