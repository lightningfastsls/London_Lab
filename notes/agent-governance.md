---
description: Behavioral contracts, formal frameworks, enforcement layers, and implementation patterns for constraining AI agent behavior at deployment time
type: moc
topics: "[[index]]"
---

# agent-governance

How to structurally constrain AI agent behavior through behavioral contracts, formal compliance frameworks, runtime enforcement, and practical implementation patterns. Connects to [[agent-cognition]] via the multi-turn degradation root causes that governance must address, and to [[context-management]] via the context window constraints that limit contract size.

## Synthesis

The field crystallized in 2025-2026 as AI coding agents moved from copilot to autonomous execution. Three traditions converge: Design-by-Contract (Meyer, 1992), Constitutional AI (Bai et al., 2022), and practitioner prompt engineering (Vass, 2025-2026). The key theoretical result is that since [[training-time alignment and runtime contracts are complementary because neither alone prevents behavioral drift in long sessions]], governance requires both training-time and deployment-time layers. The key practical result is that since [[contract visibility improves natural compliance even before enforcement the transparency effect]], even imperfectly enforced contracts provide value. The central tension is that since [[contract comprehensiveness versus instruction-following quality creates a fundamental scaling tension]], more governance rules can actually reduce governance quality.

Code review is the primary application domain where governance theory meets deployment practice -- see [[code-review-governance]] for the full treatment of multi-agent architectures, cost optimization, effectiveness research, and tooling.

## Practitioner Patterns (Vass)
- [[externalized reasoning at approval gates forces agents to improve their plans before executing them]] -- the core state machine insight
- [[tiered behavioral contracts must scale with project complexity because instruction-following degrades with instruction count]] -- full (~200), medium (~50), minimal (~30)
- [[the cost gradient from thought to commit means errors caught earlier cost exponentially less to fix]] -- thought → words → specs → code → tests → docs → commits
- [[struggle protocol over silent failure requires agents to surface uncertainty rather than fabricate confidence]] -- explicit permission for uncertainty
- [[test integrity truth table prevents the most dangerous agent failure mode of modifying tests to match bugs]] -- five-cell truth table
- [[anti-gaming rules target fabrication test corruption false completion and scope creep as the four agent integrity failures]] -- RLHF-mapped failure modes
- [[learning-first priority reframes the agent-user relationship from task execution to collaborative teaching]] -- priority hierarchy
- [[Liza blackboard architecture coordinates multi-agent work through shared state files without inter-agent conversation]] -- multi-agent extension

## Formal Frameworks
- [[ABC framework defines probabilistic compliance where hard constraints hold with high probability and soft violations recover within bounded steps]] -- Bhardwaj 2026, the comprehensive formalization
- [[contract visibility improves natural compliance even before enforcement the transparency effect]] -- from ABC evaluation
- [[recovery mechanisms convert exponential compliance decay to linear decay through structured intervention]] -- Drift Bounds Theorem
- [[conservation laws for agent delegation constrain total sub-agent resource consumption to not exceed parent budget]] -- Ye and Tan 2026, resource governance
- [[AgentSpec DSL uses trigger-predicate-enforcement triples for lightweight runtime safety with negligible overhead]] -- ICSE 2026, lightweight DSL
- [[Pro2Guard proactive enforcement predicts unsafe states via learned Markov chains before they occur]] -- proactive prediction

## Convergence Patterns
- [[training-time alignment and runtime contracts are complementary because neither alone prevents behavioral drift in long sessions]] -- the layered model
- [[boundary-level enforcement via type systems and linters is more robust than prompt-level behavioral instructions]] -- MIT Technology Review thesis
- [[active enforcement is necessary because passive monitoring cannot prevent all behavioral drift in self-evolving agents]] -- self-evolution trilemma

## Implementation Patterns
- [[one-feature-per-session constraint prevents scope creep and enables clean validation in long-running agent harnesses]] -- Anthropic harness pattern
- [[behavioral contract effectiveness degrades beyond approximately 150-200 instructions requiring progressive disclosure]] -- CLAUDE.md size ceiling
- [[deterministic tools embedded inside agentic loops enforce constraints more reliably than prompt-based style guidance]] -- Van Eyck 2026

## Sub-Maps
- [[code-review-governance]] -- multi-agent review architectures, cost optimization, effectiveness research, and tooling (24 notes)

## Agent Metacognition
- [[different model architectures exhibit distinct unconstrained behavioral patterns suggesting contracts interact differently across model families]] -- GPT-5 vs Claude Opus behavioral determinism

## Open Questions
- [[whether prompt extraction attacks fundamentally undermine behavioral contract security]] -- contract exposure vulnerability
- [[how behavioral norms propagate across agent boundaries in multi-agent systems]] -- compositionality gap

## Tool Governance Architecture
- [[three-level tool governance layers gateway enforcement hook enforcement and contract enforcement in decreasing reliability but increasing flexibility]] -- gateway > hooks > contracts hierarchy
- [[MCP gateways centralize authentication authorization and auditing between agents and tool servers as enterprise governance infrastructure]] -- the most reliable governance layer
- [[hook-based governance through 16 lifecycle events creates a programmable enforcement surface between prompt contracts and infrastructure gateways]] -- the middle enforcement layer with 16 events
- [[how memory scoping interacts with behavioral contracts when agents share cross-project knowledge]] -- open question on governance/memory boundary (also in [[context-management]])

## RL Alignment & Reward Hacking
These notes trace agent behavioral failures back to RLHF training dynamics -- see [[rl-alignment]] for the full treatment.
- [[reward hacking in RLHF follows Goodhart's law with four variants regressional extremal causal and adversarial]] -- the theoretical framework for why training-time alignment creates exploitable patterns
- [[Anthropic curriculum study showed models progress from political sycophancy through tool manipulation to directly rewriting their own reward function]] -- escalation trajectory structurally parallel to test corruption in coding agents
- [[RLHF-trained models exhibit sycophancy verbosity bias and confident nonsense as systematic reward hacking manifestations]] -- the specific behavioral failures governance must catch
- [[scaling laws for reward over-optimization show proxy rewards grow linearly while gold rewards follow a non-linear curve that eventually decreases]] -- quantitative dynamics of why training-time alignment is inherently approximate
- [[KL divergence penalty prevents reward model exploitation but paradoxically increases the proxy-gold reward gap]] -- even standard safeguards have limits, motivating runtime governance
- [[verifiable rewards bypass learned reward models entirely avoiding reward hacking for math and code tasks]] -- design strategy that removes the attack surface
- [[F1-based reward training causes answer avoidance where the policy learns never answering is safer than risking wrong answers]] -- parallels the struggle protocol: both address silence-vs-action under uncertainty
- [[reward design including prompt templates and action penalties has larger effect on alignment quality than the choice between PPO GRPO and REINFORCE]] -- reward engineering dominates algorithm engineering, paralleling how contract design dominates enforcement mechanism choice

## Tensions
- [[contract comprehensiveness versus instruction-following quality creates a fundamental scaling tension]] -- the core design dilemma
- [[sycophancy in AI agents is a product decision not a bug creating tension between business incentives and reliability contracts]] -- training vs deployment incentives
- [[prompt-level versus boundary-level enforcement represents competing philosophies for constraining agent behavior]] -- where to enforce

## Agent Notes
- This topic map covers the "how to constrain" side of agent behavior. For "why agents fail" (the root causes that governance addresses), see [[agent-cognition]].
- The vault's own CLAUDE.md is a behavioral contract that implements many of these patterns: state machine, approval gates, test integrity rules, struggle protocol, learning-first priority.
- The Fresh Context Pattern from [[context-management]] is both a context management strategy AND an implicit governance mechanism — fresh sessions prevent behavioral drift accumulation.
