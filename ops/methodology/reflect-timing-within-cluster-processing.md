---
description: "Defer /reflect until all reductions in a cluster are complete — cross-source reflection is more valuable than single-source, and processed notes don't decay"
category: processing
status: active
source: arscontexta methodology analysis (8 converging research claims)
created: 2026-03-01
---
# Reflect timing within cluster processing

When processing a cluster of 2-3 related topics through the /learn → /reduce → /reflect pipeline, defer /reflect until ALL reductions in the cluster are complete. Do not interrupt the capture-reduce cycle for intermediate reflection.

## Why defer /reflect

1. **Cross-source /reflect is more valuable than single-source.** Per [[incremental reading enables cross-source connection finding]], juxtaposing notes from two related sources produces the highest-value connections. Single-source /reflect finds within-source patterns that are often already captured during /reduce.

2. **Processed notes don't decay.** Per [[temporal processing priority creates age-based inbox urgency]], temporal urgency applies to inbox items (raw captures), not to notes already in notes/. The Ebbinghaus decay concern is about capture-to-processing gaps, not processing-to-reflection gaps. Notes in notes/ are stable and will wait.

3. **Early /reflect creates redundant work.** You would need to /reflect AGAIN after the remaining topics are processed, making the first pass partially redundant. The [[bulk-source-processing-strategy]] was designed to avoid this duplication.

4. **The collector's fallacy doesn't apply when inbox is empty.** /learn adds to inbox, but the WIP limit constraint (inbox < 20) only triggers when the inbox is already populated. Capture is safe when processing is current.

## When to reflect early (exceptions)

- If cluster processing will span more than 5 days, intermediate /reflect prevents orphan drift
- If a reduction surfaces a surprising tension with existing notes, capture the tension immediately (in ops/tensions/) but defer full /reflect
- If the session is long and context is degraded, end the session — but don't run /reflect in degraded context

## The sequence

Per [[bulk-source-processing-strategy]]:
1. /learn all topics in cluster (separate sessions)
2. /reduce each inbox file (one per session, /clear between)
3. /reflect across ALL new notes from the cluster
4. Repeat for next cluster
