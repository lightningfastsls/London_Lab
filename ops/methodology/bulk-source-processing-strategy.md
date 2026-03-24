---
description: Phase-batched cluster-grouped processing is optimal for ingesting multiple related research topics into the vault
category: processing
status: active
source: arscontexta methodology analysis (12 converging research claims)
created: 2026-03-01
---
# Bulk source processing strategy

When ingesting multiple research topics (5-10+), use **phase-batched, cluster-grouped processing** rather than pure breadth-first or pure depth-first.

## Why not breadth-first (all /learn, then all /reduce, then /reflect)

Triggers the Collector's Fallacy (PKM failure Stage 1). Inbox items lose context freshness per Ebbinghaus decay curves — beyond 72 hours is "critical" priority. Unprocessed inbox files are "organized debris" per [[throughput matters more than accumulation]].

## Why not depth-first (learn→reduce→reflect per topic in one session)

Chaining phases in a single session runs later phases on degraded context. Capture, process, connect, and verify are "genuinely different cognitive operations that interfere when mixed" per [[every knowledge domain shares a four-phase processing skeleton that diverges only in the process step]].

## The hybrid: cluster-grouped, phase-batched

1. **Group topics into clusters of 2-3 by context similarity** — processing context-similar items consecutively means loading context once and applying it multiple times, per [[batching by context similarity reduces switching costs in agent processing]].
2. **Run /learn for each topic in the cluster** (separate sessions).
3. **Run /reduce for each inbox file** (one per session, /clear between).
4. **Run /reflect across all new notes from the cluster** — gives reflect enough material (5-15 notes) to find cross-note patterns.
5. **Repeat for next cluster.**
6. **Final pass: comprehensive /reflect across ALL new notes** for cross-cluster connections.

## Key constraints

- **One /reduce per session.** Per [[fresh context per task preserves quality better than chaining phases]].
- **End every session with a handoff.** Write `ops/last-session.md` with: what was done, what's next (exact commands), current vault count, and any unresolved issues. The next session starts by reading this — if it's missing or stale, continuity breaks.
- **Process existing inbox backlog first.** Per [[WIP limits force processing over accumulation]], inbox should not exceed ~20 items.
- **Do not capture faster than you process.** Each capture batch should be processed before the next begins.
- **3-5 items per session.** Per [[continuous small-batch processing eliminates review dread]].

## Empirical grounding

The "one /reduce per session" and "/clear between phases" constraints are empirically validated by multi-turn degradation research: since [[concatenating all requirements into a single prompt restores approximately 95 percent of single-turn performance]], fresh context per task is the strongest mitigation against both multi-turn and context window degradation. The "separate sessions" design also avoids the compounding where [[context window and multi-turn degradation have distinct root causes but compound when both occur in long multi-turn sessions]].

## Risks and mitigations

- **Cross-cluster connections missed** → final comprehensive /reflect pass catches these
- **Cluster grouping imperfect** → note dangling links as demand signals for later clusters
- **Large note volume (100-250 from 10 topics)** → per-cluster /reflect prevents orphan accumulation
