---
description: "Standard tiered approach: cheap models for style and lint at ~$0.00006 per 300 tokens, mid-tier for logic, premium for security and architecture — the routing question is what is the minimal model for this task"
type: method
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[code-review-governance]]"
---

# Model cascading routes 70-90 percent of review to cheap models achieving 60-87 percent cost reduction

Model cascading is the dominant cost optimization strategy for AI code review, achieving 60-87% cost reduction by routing the majority of review tasks to cheap models. The standard three-tier approach: 90% of queries start with cheap models (Mistral 7B, Claude Haiku at ~$0.00006/300 tokens), escalate to mid-tier (GPT-4o mini, Claude Sonnet) when the cheap model signals uncertainty, and reserve premium models (GPT-4o, Claude Opus at $5-15/M input tokens) for the 10% requiring advanced reasoning.

The key insight is the routing question: "What is the minimal model that can confidently handle this query?" This reframes cost optimization from "how to make expensive models cheaper" to "how to avoid using expensive models." Simple style checks, format validation, and obvious lint violations do not need reasoning capabilities. Security analysis, architectural assessment, and cross-file dependency tracking do. Matching model capability to task complexity is the core optimization.

The empirical finding that using a cheaper model for 70% of routine tasks and an expensive model for 30% yields better ROI than all-in on the top model has a counterintuitive implication: spending less per token can improve overall quality. This is because the cheap model handles high-confidence easy cases quickly, freeing budget for deeper analysis on genuinely complex changes. Attempting to use a premium model for everything either exhausts the budget prematurely or creates rate-limiting bottlenecks.

Model cascading intersects with review depth tiering but addresses a different dimension — model cascading selects WHICH model, while since [[layered review depth tiering mirrors human review practice by matching investment to complexity]] determines HOW THOROUGHLY to review. Both can operate simultaneously. The tiering principle itself parallels how since [[tiered behavioral contracts must scale with project complexity because instruction-following degrades with instruction count]] -- matching governance investment to task complexity is a general design pattern, whether applied to contracts, review depth, or model selection.

---

Source: ai-code-review-optimization-multi-agent-architectures-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[layered review depth tiering mirrors human review practice by matching investment to complexity]] -- the complementary depth dimension
- [[3-5 actor-critic review rounds eliminate over 90 percent of issues at under 2 dollars per feature]] -- concrete cost data for multi-round review
- [[deterministic tools embedded inside agentic loops enforce constraints more reliably than prompt-based style guidance]] -- Tier 1 checks can be deterministic rather than LLM-based
- [[tiered behavioral contracts must scale with project complexity because instruction-following degrades with instruction count]] -- the same match-investment-to-complexity principle applied to governance

Topics:
- [[agent-governance]]
