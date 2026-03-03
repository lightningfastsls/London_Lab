---
description: "Attention sinks (StreamingLLM ICLR 2024) — initial tokens are visible to all subsequent tokens during training, making them natural attractors for surplus attention even when semantically irrelevant"
type: finding
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-cognition]]"
  - "[[context-management]]"
---

# Initial tokens attract disproportionate attention regardless of semantic relevance due to autoregressive causal masking during training

The "attention sinks" phenomenon (StreamingLLM, ICLR 2024; "When Attention Sink Emerges," ICLR 2025) explains a specific component of the [[LLMs exhibit a U-shaped attention curve where information in the middle of context degrades by more than 30 percent]]: why initial tokens receive disproportionate attention weight regardless of their content.

The mechanism traces to training dynamics. Because of autoregressive causal masking, initial tokens are visible to all subsequent tokens during pretraining. This makes them natural attractors for surplus attention — when a query is not semantically aligned with most of its context, the surplus attention probability mass has to go somewhere, and initial tokens are the universal landing pad. This serves a functional purpose: attention sinks help deep transformers avoid "representational collapse" and over-mixing across layers.

The practical consequence is twofold. First, keeping the KV cache of initial tokens largely recovers the performance of window attention, enabling streaming inference without fine-tuning (the core StreamingLLM contribution). Second, the phenomenon has implications for KV cache quantization and prompt engineering — the first few tokens in any context carry outsized influence on the model's attention distribution, regardless of their semantic content.

This is distinct from the general recency bias because it is specifically about the FIRST tokens, not recent ones. The U-shaped curve has two separate causes: attention sinks explain the primacy peak, while recency bias (recent tokens being most contextually accessible) explains the recency peak. The middle suffers because it has neither advantage.

---

Source: context-window-degradation-llms-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[LLMs exhibit a U-shaped attention curve where information in the middle of context degrades by more than 30 percent]] -- the broader phenomenon this partially explains
- [[left-skewed position frequency distributions during pretraining cause effective context to rarely exceed half of training context length]] -- a related training-time root cause

Topics:
- [[agent-cognition]]
