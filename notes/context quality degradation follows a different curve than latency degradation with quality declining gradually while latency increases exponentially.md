---
description: "Context Discipline 2026 showed Llama-3.1-70B maintained 98 percent quality but suffered 1017 percent latency increase at 15K words — costs are computational before intellectual"
type: finding
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-cognition]]"
  - "[[context-management]]"
---

# Context quality degradation follows a different curve than latency degradation with quality declining gradually while latency increases exponentially

The "Context Discipline" paper (2026) demonstrated a critical but often overlooked distinction: quality and efficiency degrade on completely different curves as context grows. Testing Llama-3.1-70B, quality remained at 98% across 15K words of context — remarkably stable. But latency increased by 1,017% over the same range.

This means the costs of long context are computational before they are intellectual. A model may still give correct answers while taking 10x longer and consuming 10x more resources. For agent architectures that iterate many times (running tool calls, processing batches, exploring codebases), this latency multiplier compounds rapidly. Ten iterations at 10x latency is a 100x total time increase even though each individual answer is still correct.

The quality curve follows a gradually accelerating decline — for simple retrieval (NIAH-style), models maintain near-perfect performance to a task-dependent threshold, then decline. For reasoning tasks requiring latent inference, degradation begins almost immediately. Since [[simple retrieval tasks tolerate far more context than reasoning tasks requiring latent inference]], the quality curve is task-specific while the latency curve is more universal.

The practical implication: even when quality metrics suggest a context length is "safe," the latency cost may already be prohibitive. This reinforces why [[practical guidance converges on 60-70 percent of maximum context window as effective usable capacity]] — the 60-70% rule accounts for both quality and efficiency, not just quality alone.

---

Source: context-window-degradation-llms-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[simple retrieval tasks tolerate far more context than reasoning tasks requiring latent inference]] -- quality curves differ by task type
- [[practical guidance converges on 60-70 percent of maximum context window as effective usable capacity]] -- accounts for both quality and latency
- [[Claude Sonnet exhibits a qualitative performance cliff at 147K-152K tokens which is 73-76 percent of its 200K window]] -- a model-specific quality cliff

Topics:
- [[agent-cognition]]
