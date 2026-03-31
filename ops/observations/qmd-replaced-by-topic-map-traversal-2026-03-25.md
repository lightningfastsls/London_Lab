---
description: qmd retrieval replaced by topic-map traversal + ripgrep after investigation 4.3 showed 0% recall on goal-thread prose
type: observation
status: archived
archived: 2026-03-29
archived_by: rethink-2026-03-29
created: 2026-03-25
---

# qmd replaced by topic-map traversal + ripgrep

## What happened
Knowledge activation was completely broken — every session started with "No strong matches above relevance threshold" for all goal threads. Investigation 4.3 showed qmd BM25 gets 0% on goal-thread prose and vector search had 4 independent bugs.

## What changed
- Built `ops/scripts/topic-map-index.mjs` (parses 26 MOCs + 527 notes into static JSON index, 209ms)
- Built `ops/scripts/vault-search.mjs` (3-layer search: topic map routing → ripgrep → merge/rank, 139ms)
- Modified `session-orient.sh` to use vault-search.mjs (qmd demoted to fallback)
- Updated 11 skills to remove `mcp__qmd__*` from allowed-tools and replace search instructions
- Updated CLAUDE.md with retrieval hierarchy

## Results
- 13/13 validation tests pass (7 known-answer, 3/4 ground truth, 4 negative, 1 performance)
- Full hook execution: 862ms (was 5-8s with qmd, and qmd returned nothing useful)
- Session relevance now shows real, relevant notes with descriptions and context phrases

## Why it worked
The vault's 26 topic maps already contain curated, structured indexes of 98.2% of notes. Structural traversal (which section matches the query?) provides precision that BM25 keyword matching cannot, while ripgrep provides recall for cross-vocabulary matches.
