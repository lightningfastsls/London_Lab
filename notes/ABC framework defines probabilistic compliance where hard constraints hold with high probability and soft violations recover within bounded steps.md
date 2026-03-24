---
description: "Bhardwaj 2026 formalizes (p,delta,k)-Satisfaction — hard invariants never violated, soft constraints allow transient drift if recovery occurs within k steps, grounded in Ornstein-Uhlenbeck SDE"
type: method
confidence: likely
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-governance]]"
---

# ABC framework defines probabilistic compliance where hard constraints hold with high probability and soft violations recover within bounded steps

The Agent Behavioral Contracts (ABC) framework (Bhardwaj, Accenture, February 2026) is the most comprehensive academic formalization of behavioral contracts for AI agents. It adapts Design-by-Contract (Meyer, 1992) from individual function calls to multi-turn agent sessions, defining a contract as C = (P, I, G, R): Preconditions, Invariants (hard and soft), Governance (hard and soft constraints), and Recovery (corrective action mappings).

The key theoretical contribution is (p,delta,k)-Satisfaction: a probabilistic compliance framework where hard guarantees hold with probability >= p, and soft compliance drops recover within k steps. This acknowledges that stochastic agents cannot provide deterministic guarantees — the question is not "does the agent always comply?" but "how often, and how quickly does it recover from violations?" The framework models behavioral drift via an Ornstein-Uhlenbeck stochastic differential equation, proving that when recovery rate gamma exceeds natural drift rate alpha, drift converges to a bounded value D* = alpha/gamma.

The practical evaluation is compelling: 1,980 sessions across 7 models from 6 vendors showed contracted agents detect 5.2-6.8 soft violations per session that baselines miss entirely. Hard constraint compliance ranges 88-100%. Reliability index exceeds 0.90 uniformly. Since [[recovery mechanisms convert exponential compliance decay to linear decay through structured intervention]], the recovery component is what makes long sessions viable.

---

Source: behavioral-contracts-ai-coding-agents-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[recovery mechanisms convert exponential compliance decay to linear decay through structured intervention]] -- the key recovery result
- [[contract visibility improves natural compliance even before enforcement the transparency effect]] -- the surprising finding from ABC evaluation
- [[training-time alignment and runtime contracts are complementary because neither alone prevents behavioral drift in long sessions]] -- ABC's positioning

Topics:
- [[agent-governance]]
