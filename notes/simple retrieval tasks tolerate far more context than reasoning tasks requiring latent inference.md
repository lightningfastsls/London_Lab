---
description: "NIAH stays near-perfect at 128K while NoLiMa degrades at 4K — task type is the primary determinant of effective context length, not architecture alone"
type: finding
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-cognition]]"
  - "[[context-management]]"
---

# Simple retrieval tasks tolerate far more context than reasoning tasks requiring latent inference

The most consistent finding across long-context benchmarks is that effective context length depends heavily on task type. Simple retrieval (matching a literal string or keyword) tolerates much more context than tasks requiring latent reasoning (connecting concepts, multi-hop inference, aggregation).

The evidence is stark. On standard NIAH (literal matching), models like GPT-4o maintain 99%+ accuracy even at 128K tokens. But on NoLiMa (latent reasoning, no lexical shortcuts), the same models drop below 50% at 32K. Since [[NoLiMa found 11 of 12 models dropped below 50 percent baseline at 32K when lexical shortcuts were removed]], and since [[RULER benchmark showed only half of long-context models maintained performance at 32K despite claiming 32K-plus support]] even with lexical cues available on harder task variants, the gap between retrieval and reasoning is not marginal — it can be an order of magnitude in effective context length.

The mechanism relates to how attention processes different task types. Simple retrieval can succeed with sparse attention — a few high-attention tokens are sufficient. Reasoning tasks require integrating information across multiple positions, which means the model needs sustained, distributed attention rather than focused retrieval. Since [[LLMs exhibit a U-shaped attention curve where information in the middle of context degrades by more than 30 percent]], reasoning tasks that require middle-positioned information suffer disproportionately.

For agent architecture, this means context budgets should be task-adaptive: file search and code navigation can use larger contexts; debugging, architectural reasoning, and synthesis should use smaller, more focused contexts. This is why [[subagent architecture isolates context degradation by giving each task a focused window and compressing summaries upward]] — each subagent can be scoped to the task's complexity.

---

Source: context-window-degradation-llms-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[NoLiMa found 11 of 12 models dropped below 50 percent baseline at 32K when lexical shortcuts were removed]] -- the strongest evidence
- [[maximum effective context window can differ from claimed context by as much as 99 percent and shifts by problem type]] -- the MECW confirmation
- [[subagent architecture isolates context degradation by giving each task a focused window and compressing summaries upward]] -- the architectural response

Topics:
- [[agent-cognition]]
