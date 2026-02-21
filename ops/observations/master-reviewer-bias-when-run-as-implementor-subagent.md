---
description: "Master-reviewer subagent catches real bugs but implementor interprets findings and marks own fixes — bias risk for high-tier reviews"
category: methodology
trigger: "Phase 8.4 review — user questioned whether same-chat subagent review is independent enough"
status: pending
---

# Master-reviewer run as implementor subagent creates bias in fix interpretation

During Phase 8.4 implementation (2026-02-21), the master-reviewer was spawned as a Task subagent within the implementor's chat session. The reviewer ran in isolated context (no access to implementation reasoning) and caught 4 genuine blockers — including a causal attention cross-contamination bug that would have produced scientifically invalid codebook profiles.

However, the implementor (same chat):
1. **Chose when to spawn** the reviewer (after implementation, not during)
2. **Interpreted the findings** — deciding severity and whether fixes were adequate
3. **Marked own homework** — wrote "APPROVED (after fixes applied)" about own fixes
4. **No re-review of fixes** — the reviewer never verified the applied fixes

The subagent's context isolation is real (it only sees files on disk, not implementation intent), so it does function as a genuine quality gate for catching bugs. The bias risk is in the *interpretation layer*, not the *detection layer*.

## Proposed Action

For **Tier 3 reviews** (critical modules like VQ-VAE, transformer, detection): require a separate chat session for the master-reviewer, where the reviewer reads the handoff cold and controls the verdict independently. The implementor should not be in the loop between "findings" and "verdict."

For **Tier 2 reviews** (analysis tools, scripts, utilities): current subagent approach is adequate — the detection quality is high and the iteration speed benefit is worth the interpretation bias risk.

Update `/implement` skill to encode this distinction.
