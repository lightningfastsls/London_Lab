---
description: "ABC evaluation found that simply making contracts visible to agents improved compliance rates — enforcement catches violations, but visibility prevents them"
type: finding
confidence: likely
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-governance]]"
  - "[[agent-cognition]]"
---

# Contract visibility improves natural compliance even before enforcement the transparency effect

A key finding from the ABC framework's evaluation across 1,980 sessions: contract visibility alone improves natural compliance, independent of enforcement mechanisms. When agents can see the behavioral contract — even without runtime monitoring that catches violations — they comply more reliably than agents operating without any contract specification.

This is the "transparency effect" and it has a clear mechanism: LLMs trained via RLHF are highly responsive to contextual framing. A visible contract provides a frame of reference that shapes generation. The effect is analogous to how since [[externalized reasoning at approval gates forces agents to improve their plans before executing them]], making expectations explicit changes behavior even before enforcement applies.

The practical implication is that behavioral contracts serve two functions: (1) passive compliance improvement through visibility, and (2) active violation detection through enforcement. The transparency effect means that even imperfectly enforced contracts provide value — the contract itself is a behavioral intervention, not just a monitoring specification. This partly explains why practitioner-level contracts like Vass's CLAUDE.md approach work despite having no formal enforcement: the visibility effect provides baseline compliance improvement.

However, the transparency effect has limits. Since [[RLHF training rewards premature helpfulness causing LLMs to make early assumptions that anchor subsequent responses]], the same training that makes agents responsive to visible contracts also makes them susceptible to drifting when the contract becomes stale in context. The effect likely weakens in long sessions as the contract text moves further from the model's attention window -- and since [[behavioral contract effectiveness degrades beyond approximately 150-200 instructions requiring progressive disclosure]], even the visibility effect erodes when instruction density overwhelms the agent's attention.

---

Source: behavioral-contracts-ai-coding-agents-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[externalized reasoning at approval gates forces agents to improve their plans before executing them]] -- parallel mechanism of making expectations explicit
- [[RLHF training rewards premature helpfulness causing LLMs to make early assumptions that anchor subsequent responses]] -- the training that enables the effect
- [[ABC framework defines probabilistic compliance where hard constraints hold with high probability and soft violations recover within bounded steps]] -- the framework that discovered this effect
- [[behavioral contract effectiveness degrades beyond approximately 150-200 instructions requiring progressive disclosure]] -- the instruction density ceiling that limits the transparency effect
- [[active enforcement is necessary because passive monitoring cannot prevent all behavioral drift in self-evolving agents]] -- the theoretical proof that visibility (passive) is insufficient alone

Topics:
- [[agent-governance]]
