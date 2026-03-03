---
description: "ICSE 2026 — three-tuple rules (trigger, predicates, enforcement) achieve 90%+ prevention of unsafe code executions with millisecond overhead; OpenAI o1 auto-generates rules at 95.56% precision"
type: method
confidence: likely
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-governance]]"
---

# AgentSpec DSL uses trigger-predicate-enforcement triples for lightweight runtime safety with negligible overhead

AgentSpec (Wang, Poskitt, and Sun, ICSE 2026) provides a lightweight domain-specific language for runtime enforcement of agent behavior. Rules are defined as three-tuples: r = (trigger, predicates, enforcement). Triggers include state_change, before_action, agent_finish, plus domain-specific events. Predicates are Boolean evaluation functions (is_destructive_cmd, is_fragile_object, obstacle_distance_leq). Enforcement types are stop, user_inspection, invoke_action, or LLM_self_examination.

Results demonstrate 90%+ prevention of unsafe code executions, 100% compliance in autonomous vehicle scenarios, and negligible millisecond overhead. OpenAI o1 can auto-generate AgentSpec rules from natural language descriptions with 95.56% precision, suggesting the specification burden can be partially automated.

The key limitation is that AgentSpec is purely reactive — it intervenes only when unsafe behavior is imminent. This contrasts with since [[Pro2Guard proactive enforcement predicts unsafe states via learned Markov chains before they occur]], which addresses the prediction gap. The reactive vs proactive distinction matters: reactive enforcement catches violations as they happen, but proactive enforcement prevents them from reaching the violation boundary.

The practical value of AgentSpec is its simplicity. Where ABC provides a comprehensive theoretical framework, AgentSpec provides an immediately deployable DSL that can be incrementally adopted. The auto-generation capability is particularly significant — it suggests behavioral contracts could be derived from existing documentation rather than manually authored. AgentSpec is a concrete instantiation of the principle that since [[boundary-level enforcement via type systems and linters is more robust than prompt-level behavioral instructions]] -- the DSL compiles behavioral expectations into deterministic runtime checks rather than relying on prompt compliance.

---

Source: behavioral-contracts-ai-coding-agents-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[Pro2Guard proactive enforcement predicts unsafe states via learned Markov chains before they occur]] -- the proactive complement to AgentSpec's reactive approach
- [[ABC framework defines probabilistic compliance where hard constraints hold with high probability and soft violations recover within bounded steps]] -- the comprehensive framework AgentSpec simplifies
- [[deterministic tools embedded inside agentic loops enforce constraints more reliably than prompt-based style guidance]] -- the enforcement principle AgentSpec implements
- [[boundary-level enforcement via type systems and linters is more robust than prompt-level behavioral instructions]] -- AgentSpec instantiates boundary-level enforcement as a DSL
- [[active enforcement is necessary because passive monitoring cannot prevent all behavioral drift in self-evolving agents]] -- the theoretical result AgentSpec implements as reactive enforcement

Topics:
- [[agent-governance]]
