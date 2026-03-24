---
description: First topic map split revealed that arscontexta-expert consultation should precede structural KG changes, not follow them
category: process-gap
date: 2026-03-03
status: archived
archived: 2026-03-03
archived_by: rethink-2026-03-03
resolution: "Added arscontexta-expert consultation requirement to CLAUDE.md Topic Maps section (Proposal 6). Split threshold also raised from 35 to 50 (Proposal 7)."
---

# topic map split process requires arscontexta consultation before execution

## What Happened

During the classification topic map split (87 notes → 3 maps), the agent started writing child maps before consulting the arscontexta-expert. The expert was consulted mid-process after the user flagged the gap.

## What the Expert Contributed

The arscontexta-expert provided grounded guidance from 6 methodology claims that materially shaped the split:

1. **Rosch's basic-level categorization** — prevented a premature sub-split of few-shot learning (13 notes) into its own map. 13 notes is "subordinate level" — too narrow to justify maintenance overhead.
2. **Boundary objects** — literature notes appearing on multiple maps are structurally valuable cross-references, not misclassified nodes. The expert recommended keeping them on the parent hub rather than distributing to children.
3. **Context phrase clarity** — parent-to-child links need rich context phrases that enable "confident branch commitment" (Larson & Czerwinski 1998). Without them, deeper hierarchies degrade navigation.
4. **Stale navigation risk** — duplicating "preview" notes on the parent creates dual maintenance burden. The expert warned against this pattern.
5. **MOC health range** — 10-40 is healthy, >50 is warning. The planned 37-note child was validated as acceptable.
6. **`parent_map` convention** — confirmed as vault-local optional field, suggested adding to schema validation.

## Lesson

**For any structural KG change (splits, merges, new topic maps), consult arscontexta-expert FIRST.** The agent table in CLAUDE.md already says this ("Topic map strategy → arscontexta-expert") but the operational habit wasn't established. The expert's methodology grounding prevents over-engineering (premature splits) and under-engineering (missing context phrases).

## Follow-up

~70 notes that moved from the parent to child maps need their `topics` footer updated from `[[classification]]` to `[[classification-tools]]` or `[[classification-methodology]]`. This is a mechanical task that can be batched.
