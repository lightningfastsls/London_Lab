---
description: "Learns Discrete-Time Markov Chains from execution traces, performs probabilistic model checking at runtime — 93.6% safety on unsafe tasks, 100% prediction of traffic violations 38.66 seconds in advance"
type: method
confidence: likely
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-governance]]"
---

# Pro2Guard proactive enforcement predicts unsafe states via learned Markov chains before they occur

Pro2Guard (2025) addresses the reactive limitation of runtime enforcement systems like AgentSpec by learning Discrete-Time Markov Chains (DTMCs) from execution traces and performing probabilistic model checking at runtime. When the estimated risk of reaching an unsafe state exceeds a configurable threshold, the system intervenes preemptively — before the violation occurs.

Results demonstrate enforcement of safety on 93.6% of unsafe tasks using low thresholds, 100% prediction of traffic law violations up to 38.66 seconds in advance, and PAC (Probably Approximately Correct) guarantees on the learned models. The proactive approach fundamentally changes the cost profile: reactive enforcement detects violations and requires recovery, while proactive enforcement prevents violations entirely, eliminating recovery costs.

The prediction capability connects to the behavioral drift modeling in ABC: since [[recovery mechanisms convert exponential compliance decay to linear decay through structured intervention]], recovery is always possible, but prevention is strictly better. Pro2Guard's Markov chain approach essentially models the state transition probabilities that lead to drift, then intervenes when the predicted trajectory approaches unsafe regions.

The limitation is the learning requirement — Pro2Guard needs execution traces to build its Markov models, which means it cannot protect against novel failure modes not represented in training data. This creates a bootstrap problem similar to active learning: the system becomes more protective as it observes more violations, but is least protective when first deployed.

---

Source: behavioral-contracts-ai-coding-agents-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[AgentSpec DSL uses trigger-predicate-enforcement triples for lightweight runtime safety with negligible overhead]] -- the reactive approach Pro2Guard improves upon
- [[recovery mechanisms convert exponential compliance decay to linear decay through structured intervention]] -- recovery as the fallback when prediction fails
- [[active enforcement is necessary because passive monitoring cannot prevent all behavioral drift in self-evolving agents]] -- the theoretical justification

Topics:
- [[agent-governance]]
