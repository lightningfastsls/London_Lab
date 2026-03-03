---
description: "Web Claude implementation plans contain both task specs and domain knowledge — the /roadmap-from-plan skill originally only captured the task specs, losing theoretical content"
category: process-gap
trigger: "user pointed out that vacation master plan contained ~28 extractable domain claims that would have been lost without manual /reduce"
status: archived
archived: 2026-03-02
archived_by: rethink-2026-03-02
---

# Web Claude plans are dual-purpose documents requiring both ROADMAP conversion and KG extraction

While converting the vacation-master-plan-v2 into ROADMAP format, the user asked whether the theoretical knowledge in the plan would have been proactively extracted to the knowledge graph. The honest answer was no — the `/roadmap-from-plan` skill focused exclusively on generating `/implement` blocks, and the agent's task framing ("convert this to ROADMAP format") suppressed the knowledge extraction instinct.

The plan contained 28 extractable domain claims (information theory methods, null model hierarchies, probing methodology, LMT integration design) plus 6 enrichments to existing notes. All of this would have evaporated after the session ended if the user hadn't flagged it.

The root cause is a mode-switching failure: "engineer mode" (convert format) doesn't automatically trigger "researcher mode" (extract knowledge). Web Claude plans are uniquely likely to contain both because the user develops theoretical reasoning alongside implementation steps in those conversations.

## Proposed Action

**Already implemented** — Added Step 6 ("Extract Theoretical Knowledge to KG") to `/roadmap-from-plan` skill. The step scans the source plan for theoretical content, asks the user whether to run `/reduce`, and routes through the standard `/seed` → `/reduce` pipeline. This makes KG extraction a non-optional part of the plan conversion workflow.
