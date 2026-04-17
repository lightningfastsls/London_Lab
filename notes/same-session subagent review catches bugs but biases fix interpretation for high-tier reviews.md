---
description: "Context isolation enables genuine bug detection, but the implementor controlling spawn timing, finding interpretation, and verdict creates an interpretation bias layer that undermines independence for critical modules"
type: claim
confidence: likely
source: ops/observations/master-reviewer-bias-when-run-as-implementor-subagent.md
topics: "[[code-review-governance]]"
---

# same-session subagent review catches bugs but biases fix interpretation for high-tier reviews

When a master-reviewer is spawned as a Task subagent within the implementor's chat session, the reviewer operates in isolated context (only sees files on disk, not implementation reasoning). This produces genuine bug detection — in Phase 8.4, it caught 4 blockers including a causal attention cross-contamination bug that would have produced scientifically invalid codebook profiles.

However, the implementor controls the entire interpretation layer:
1. **Spawn timing** — reviewer is called after implementation, not during
2. **Finding interpretation** — implementor decides severity and whether fixes are adequate
3. **Verdict authority** — implementor writes "APPROVED (after fixes applied)" about own fixes
4. **No re-review** — reviewer never verifies the applied fixes

The bias is not in detection (which works) but in interpretation (which is compromised). Since [[fresh context swap between generation and review eliminates conversation drift and confirmation bias]], the detection half of the subagent pattern is sound. The gap is that the verdict half remains under implementor control.

**Practical resolution:** For Tier 3 reviews (critical modules like VQ-VAE, transformer, detection), require a separate chat session where the reviewer reads the handoff cold and controls the verdict independently. For Tier 2 reviews (analysis tools, scripts, utilities), the current subagent approach is adequate — detection quality is high and iteration speed benefit outweighs interpretation bias risk. Since [[layered review depth tiering mirrors human review practice by matching investment to complexity]], this tier distinction is methodologically supported.

See also: [[adversarial builder-critic separation catches silent performance risks that pass all tests]], [[same-model generation and review creates confirmation bias producing 8x duplicated code and 72 percent Java security failures]].
