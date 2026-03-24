---
description: "The Concat condition proves degradation comes from temporal distribution not information loss — validates the Fresh Context Pattern and plan-file-first workflows"
type: finding
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[multi-turn-degradation]]"
---

# Concatenating all requirements into a single prompt restores approximately 95 percent of single-turn performance

Laban et al. tested a "Concat" condition where all shards (the distributed information pieces) are concatenated as bullet points in a single turn rather than distributed across multiple turns. This condition preserves approximately 95.1% of full single-turn performance on average — a dramatic recovery from the 39% degradation seen in the Sharded condition.

This finding serves dual purposes. Methodologically, it is a causal verification: it proves the degradation is not caused by information loss during the sharding/rephrasing process. The same information, reformatted as bullet points in one turn, performs nearly as well as the original fully-specified instruction. The degradation comes specifically from temporal distribution — spreading information across conversation turns.

Practically, this validates the "Fresh Context Pattern" widely used in AI agent workflows: gather all requirements into a single specification, then submit to a fresh model instance. This is what plan files, spec documents, and consolidated prompts achieve. Smaller models (under 13B parameters) show more pronounced Concat drops (86-92%), indicating greater sensitivity to paraphrasing, but the principle holds across all model sizes.

The implication for this vault's workflow: the orient→work→persist rhythm, /clear between phases, and plan-file-first approach are not just operational preferences — they are empirically validated mitigations for a measured multi-turn degradation phenomenon.

## Dual-Mechanism Validation

The Fresh Context Pattern is even more general than the multi-turn finding alone suggests. Since [[context window and multi-turn degradation have distinct root causes but compound when both occur in long multi-turn sessions]], Fresh Context mitigates BOTH degradation types simultaneously: it prevents multi-turn commitment errors (by avoiding conversation turns) AND prevents attention dilution (by keeping the context window small and focused). Anthropic's context engineering guidance describes this as "discover the smallest collection of high-signal information that maximizes desired outcomes" — a principle that addresses both root causes. This dual-mechanism validation explains why Fresh Context dominates practical recommendations across both the multi-turn and context window research literatures.

---

Source: multi-turn-conversation-degradation-llms-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[LLMs lose approximately 25 percentage points average performance when tasks are distributed across conversational turns]] -- the degradation this mitigates
- [[even two conversational turns trigger multi-turn degradation regardless of task complexity]] -- why single-turn is worth pursuing
- [[snowball turn-by-turn accumulation mitigates 15-20 percent of the full-to-sharded performance deterioration]] -- the realistic alternative when single-turn is not possible

Topics:
- [[agent-cognition]]
