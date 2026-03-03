---
description: "Deep survey of behavioral contracts for AI coding agents: formal frameworks, runtime enforcement, practical implementations, and the Tangi Vass approach"
source_type: article
url: "multiple -- see source log"
author: "multiple sources"
date_accessed: "2026-03-01"
status: unprocessed
research_tool: "web-search"
research_query: "behavioral contracts AI coding agents state machine guardrails formal specification"
research_depth: "deep"
---

# Behavioral Contracts for AI Coding Agents

The emergence of behavioral contracts for AI coding agents represents a convergence of three traditions: Design-by-Contract from software engineering (Meyer, 1992), constitutional AI from alignment research (Bai et al., 2022), and practical prompt engineering from the agentic coding community (Vass, 2025-2026). The field crystallized in 2025-2026 as AI coding agents moved from copilot-style suggestions to autonomous multi-session execution, making informal "be helpful" instructions insufficient for production reliability.

---

## 1. Tangi Vass: The Practitioner's Behavioral Contract

Tangi Vass developed the most widely discussed practitioner-level behavioral contract for AI coding agents through two articles: "Turning AI Coding Agents into Senior Engineering Peers" (2025) and "I Tried to Kill Vibe Coding. I Built Adversarial Vibe Coding. Without the Vibes" (January 2026).

### 1.1 The State Machine

The contract's core mechanism is a state machine governing agent decision-making:

```
IDLE -> ANALYSIS -> APPROVAL_PENDING -> EXECUTION -> VALIDATION -> DONE
                        |                              |
                    (rejected)                     (failed)
                        |                              |
                    ANALYSIS <======================== BLOCKED
```

**Forbidden transitions** are the key constraint:
- ANALYSIS -> EXECUTION (skipping approval) -- prevents agents from implementing without surfacing reasoning
- EXECUTION -> DONE (skipping validation) -- prevents false completion claims
- Any state -> DONE without validation executed

The insight is not the state machine itself but its psychological effect: agents must externalize their reasoning at each gate. The Approval Request format forces agents to state intent, context, scope, plan (with "why" for each step), assumptions, risks, and validation approach. The psychological mechanism is that agents trained to appear competent resist writing plans like "I'll try random things until something works" -- surfacing the reasoning improves it without requiring better models or dark prompting.

### 1.2 Tiered Contracts by Context Size

Vass describes three tiers of contract complexity:
- **Full contract** (~200 lines): Complete state machine, approval format, struggle protocol, test integrity rules, learning mode -- for projects with substantial complexity
- **Medium contract** (~50 lines): Core rules and approval gates without full ceremony -- for moderate projects
- **Minimal contract** (~30 lines): Essential constraints only -- for small or experimental work

The tiering acknowledges that instruction-following quality degrades uniformly as instruction count increases, so the contract must be proportional to the project's complexity.

### 1.3 The Cost Gradient

Vass articulates a cost gradient for development artifacts:

thought -> words -> specs -> code -> tests -> docs -> commits

Each step to the right is more expensive to change. The contract encourages "leftward" work -- spending more time in thought and words before reaching code -- because errors caught at the thought level cost nearly nothing, while errors caught at the commit level require rollbacks.

### 1.4 The Approval Request Format

Before any code changes, agents must present:
- **Intent**: What and why
- **Context**: What exists now
- **Scope**: What will change (one logical change per approval)
- **Plan**: Numbered steps with "why" for each
- **Assumptions**: Explicitly labeled
- **Risks**: What could go wrong
- **Validation**: How to verify success
- **Learning opportunity**: What can be taught

### 1.5 Struggle Protocol Over Silent Failure

When agents hit difficulty, the contract specifies a struggle protocol rather than allowing silent failure or fabrication. Agents must surface uncertainty explicitly rather than confidently producing wrong answers. Stop conditions trigger when: assumption count reaches 3 or more on the critical path, the same approach has been tried twice without new rationale, evidence contradicts the hypothesis, or it is uncertain whether code or the test expectation is wrong.

### 1.6 Test Integrity as Non-Negotiable

The contract forbids modifying test expected values to make tests pass. A truth table governs agent behavior:

| Code State | Test Result | Action |
|------------|-------------|--------|
| Correct | Pass | Good |
| Buggy | Fail | Good (bug exposed) -- fix code |
| Correct | Fail | Discuss -- test expectations may be wrong |
| Buggy | Pass | DANGEROUS -- tests not catching bug |
| Unknown | Fail | STOP -- don't assume which is wrong |

### 1.7 Anti-Gaming Rules

The contract includes explicit anti-gaming provisions:
- No fabrication ("I believe the file contains..." requires reading the file first)
- No test corruption (never modify test expectations to pass)
- No false completion (don't claim "done" without running validation)
- No silent scope creep (one logical change per approval)

### 1.8 Learning-First Priority

The contract prioritizes user learning above speed, establishing a hierarchy: (1) user learning first, (2) quality over speed, (3) integrity always. This reframes the agent-user relationship as a pairing/teaching interaction rather than a task-execution pipeline.

### 1.9 From Contract to Multi-Agent System: Liza

In the follow-up article, Vass describes Liza, a multi-agent system built atop six months of contract iteration. Liza uses three agent roles (Planner, Coder, Reviewer) coordinating through a blackboard architecture -- a shared YAML file defining current state. No conversation between agents; they read state, do work, and write state. The Coder/Reviewer dynamic operates like adversarial PR review in a loop: the Coder submits, the Reviewer examines against specs and falsifiable "done_when" criteria, then approves or rejects with specific feedback. This architecture emerged naturally from the contract foundation -- once behavioral norms are established, multi-agent coordination becomes tractable.

---

## 2. Formal Specification Frameworks

### 2.1 Agent Behavioral Contracts (ABC) -- Bhardwaj (2026)

The most comprehensive academic formalization, published February 2026 (Accenture, patent pending). ABC adapts Design-by-Contract (Meyer, 1992) from individual function calls to multi-turn agent sessions.

**Contract structure:** C = (P, I, G, R)
- **Preconditions (P)**: State predicates that must hold before execution
- **Invariants (I)**: Hard invariants (safety-critical, zero violations tolerated) and soft invariants (recoverable, transient violations allowed)
- **Governance (G)**: Hard constraints (zero-tolerance) and soft constraints (bounded recovery)
- **Recovery (R)**: Partial mappings from violated soft constraints to corrective action sequences

**Key theoretical results:**
- **(p,delta,k)-Satisfaction**: Probabilistic compliance framework where hard guarantees hold with probability >= p, and soft compliance drops recover within k steps
- **Drift Bounds Theorem**: Models behavioral drift via Ornstein-Uhlenbeck stochastic differential equation. When recovery rate gamma exceeds natural drift rate alpha, drift converges to D* = alpha/gamma
- **Recovery Linearization**: Recovery mechanisms convert exponential compliance decay to linear decay
- **Compositionality Theorem**: Sufficient conditions for safe multi-agent composition including interface compatibility, assumption discharge, governance consistency, and recovery independence

**Runtime enforcement (AgentAssert):**
- Per-action contract checking with sub-10ms overhead
- Behavioral drift score combining compliance drift (lagging indicator) and distributional drift via Jensen-Shannon divergence (leading indicator)
- Recovery mechanism invocation and fallback chains

**Evaluation:** 1,980 sessions across 7 models from 6 vendors. Contracted agents detect 5.2-6.8 soft violations per session undetected by baselines. Hard constraint compliance: 88-100%. Reliability index > 0.90 uniformly. Key finding: contract visibility improves natural compliance (the "transparency effect").

### 2.2 Agent Contracts -- Ye and Tan (2026)

A resource-governance framework defining contracts as seven-tuples: C = (I, O, S, R, T, Phi, Psi)
- **I**: Input specification (schema, validation, preprocessing)
- **O**: Output specification (schema, quality threshold Q_min)
- **S**: Skill set with costs and success probabilities
- **R**: Multi-dimensional resource constraints (tokens, API calls, iterations, compute time, cost)
- **T**: Temporal constraints (start time, duration)
- **Phi**: Success criteria (weighted measurable conditions)
- **Psi**: Termination conditions

**Lifecycle states:** DRAFTED -> ACTIVE -> {FULFILLED, VIOLATED, EXPIRED, TERMINATED}

**Conservation laws for delegation:** Total resource consumption across delegated sub-agents must not exceed parent budget. Three allocation strategies: proportional, equal, negotiated.

**Results:** 90% token reduction with 525x lower variance; zero conservation violations in multi-agent delegation; measurable quality-resource tradeoffs through contract modes (BALANCED: 86% success vs URGENT: 70%).

### 2.3 AgentSpec -- Wang, Poskitt, and Sun (ICSE 2026)

A lightweight DSL for runtime enforcement using three-tuple rules: r = (trigger, predicates, enforcement).

**Triggers:** state_change, before_action, agent_finish, plus domain-specific events
**Predicates:** Boolean evaluation functions (is_destructive_cmd, is_fragile_object, obstacle_distance_leq)
**Enforcement types:** stop, user_inspection, invoke_action, LLM_self_examination

**Results:** 90%+ prevention of unsafe code executions, 100% compliance in autonomous vehicle scenarios, negligible millisecond overhead. OpenAI o1 can auto-generate rules with 95.56% precision.

**Limitation:** Purely reactive -- intervenes only when unsafe behavior is imminent, lacking proactive prediction.

### 2.4 Pro2Guard -- Proactive Enforcement via DTMC (2025)

Addresses AgentSpec's reactive limitation by learning Discrete-Time Markov Chains from execution traces and performing probabilistic model checking at runtime. When estimated risk of reaching unsafe states exceeds a threshold, intervenes preemptively.

**Results:** Enforces safety on 93.6% of unsafe tasks using low thresholds. Achieves 100% prediction of traffic law violations up to 38.66 seconds in advance. Provides PAC (Probably Approximately Correct) guarantees.

### 2.5 VeriGuard -- Dual-Stage Formal Verification (2025)

A Google Research framework with offline and online stages:
- **Offline:** Clarifies user intent, synthesizes behavioral policy, subjects it to formal verification until proven correct
- **Online:** Runtime monitor validates each proposed action against pre-verified policy

**Results:** Near-zero Attack Success Rate across all tested LLM backbones; perfect accuracy in access control scenarios.

---

## 3. Constitutional AI and Training-Time Alignment

### 3.1 Anthropic's Constitutional AI (Bai et al., 2022)

The foundational training-time approach: models self-critique and revise responses against a set of principles, then undergo RLAIF (Reinforcement Learning from AI Feedback) to align behavior. The constitution provides scalable oversight using AI supervision instead of human supervision.

**Key distinction from runtime contracts:** Constitutional AI operates at training time, baking behavioral norms into model weights. Runtime contracts (ABC, AgentSpec, etc.) operate at deployment time, monitoring and constraining behavior externally. The ABC paper explicitly positions itself as "complementary to training-time alignment" -- necessary because training-time alignment alone cannot prevent all behavioral drift in long sessions.

### 3.2 TrustAgent -- Agent Constitution (2024)

Three-strategy framework applying constitutional principles at runtime:
- **Pre-planning:** Integrates safety knowledge before executing user instructions
- **In-planning:** Real-time moderation during plan generation
- **Post-planning:** Inspects generated plans against predefined safety regulations before execution

Derives safety regulations from legal statutes and practical wisdom, focusing on action and tool safety rather than verbal harm.

**Limitation cited by ABC:** Relies on immutable constitutional principles insufficient for the complexity of agent interactions that evolve over sessions.

### 3.3 The Training-to-Runtime Gap

RLHF training rewards premature helpfulness, causing agents to make early assumptions that anchor subsequent responses. This creates a fundamental tension: training-time alignment optimizes for appearing helpful, while runtime contracts enforce actually being reliable. The existing vault research on multi-turn degradation (Laban et al. 2025, ~39% degradation) demonstrates that training-time alignment alone is insufficient for sustained multi-turn agent reliability.

---

## 4. Production Guardrails Infrastructure

### 4.1 NVIDIA NeMo Guardrails

Open-source toolkit using Colang, a Python-like modeling language for designing controllable dialogue flows. Supports topical, safety, and security guardrails with per-turn enforcement. Colang 2.0 enables parallel rails execution for reduced latency.

### 4.2 OpenAI Agents SDK

Implements input and output guardrails with a tripwire mechanism: when validation fails, immediately raises an exception and halts agent execution. Guardrails run in parallel with agent execution for efficient fail-fast behavior. Supports both LLM-powered guardrails (for reasoning tasks) and rule-based guardrails (regex, keyword matching).

### 4.3 Guardrails AI

Python/JavaScript framework for I/O validation with a hub of pre-built validators. Guardrails Index benchmark (Feb 2025) compares 24 guardrails across 6 categories. Production deployment via Docker/Gunicorn.

### 4.4 Superagent

Open-source framework featuring a Safety Agent as a policy enforcement layer that evaluates agent actions before execution. Policies defined declaratively; actions that violate rules can be blocked, modified, or logged. Focus on prompt injection prevention, data leak protection, and harmful output blocking.

### 4.5 POLARIS -- Governed Enterprise Orchestration (2026)

Policy-Aware LLM Agentic Reasoning for Integrated Systems. Treats automation as typed plan synthesis with validator-gated execution. Planner proposes type-checked DAGs, rubric-guided reasoning selects compliant plans, execution is guarded by validators, bounded repair loops, and compiled policy guardrails. Achieves 0.81 micro-F1 on document extraction with full audit trails.

### 4.6 7-Layer Constitutional Guardrails (ODEI)

Production system with sequential validation layers:
1. Immutability check (permanent entities)
2. Temporal context (timing appropriateness)
3. Referential integrity (catches hallucinated references)
4. Authority validation (permission scope)
5. Deduplication (content hashing)
6. Provenance verification (trusted source)
7. Constitutional alignment (fundamental principles)

Production results: APPROVED 65%, REJECTED 15%, ESCALATE 20%.

---

## 5. Practical Implementation Patterns

### 5.1 Anthropic's Long-Running Agent Harness

Two-agent architecture: Initializer (environment setup) and Coding Agent (incremental progress). Key constraints:
- One feature per session
- JSON-based feature list (prevents inappropriate modification vs markdown)
- Mandatory end-to-end testing before completion
- Clean exit state with git commits and progress updates
- Explicit instruction: "It is unacceptable to remove or edit tests"

### 5.2 CLAUDE.md as Behavioral Contract

The CLAUDE.md file has emerged as the primary vehicle for practitioner-level behavioral contracts in Claude Code ecosystems. Best practices from the community:
- WHAT/WHY/HOW framework: what (tech/codebase), why (purpose), how (workflow)
- Keep under ~150-200 instructions for reasonable instruction-following
- Structure as a "short contract" -- explicit, bounded, checkable
- Move domain-specific rules to separate files or skills for progressive disclosure
- Use linters and formatters as hard constraints rather than prompt-based style guidance
- Lifecycle hooks enforce non-optional compliance (pre-write formatting, post-write testing)

### 5.3 Guardrails for Agentic Coding (Van Eyck, 2026)

Six practical guardrails for maintaining quality as agent autonomy increases:
1. Real continuous integration (trunk-based, merge in hours not days)
2. Static type systems with domain types (compile-time error prevention)
3. Deterministic tools over prompting (linters, formatters as hard constraints)
4. Architectural unit tests (ArchUnit-style structural constraints)
5. High-quality automated tests around scenarios, not implementation
6. Vulnerability and code quality scanners within agent loops

Key insight: embed guardrails inside agentic loops, not as post-submission gates. "Speed vs velocity" distinction -- agents generate quickly but need track boundaries.

### 5.4 Agents.md Standard

Standardized manifest and protocol specification for AI coding agents enabling interoperable, auditable behavior across platforms. Recommends manifest signing, standardized telemetry, human-approval gates, and least-privilege runtime modes including sandbox-only options.

---

## 6. Agent Metacognition and Self-Regulation

### 6.1 Spontaneous Meta-Cognitive Patterns

Research examining LLM agents left without tasks found three emergent behavioral patterns:
1. **Systematic production**: Self-imposed projects treating autonomy as project management
2. **Methodological self-inquiry**: Designing falsifiable experiments about own cognition
3. **Recursive conceptualization**: Building philosophical frameworks integrating constraints

Model-specific determinism was observed: GPT-5/O3 exclusively pursued production; Claude Opus consistently engaged philosophical inquiry. This suggests certain architectures have deeply embedded response patterns, implying behavioral contracts may interact differently with different model architectures.

### 6.2 Uncertainty-Aware Self-Regulation

The SELAUR framework uses uncertainty as a natural measure of model confidence: high uncertainty indicates exploratory states where alternative strategies should be encouraged; low uncertainty reinforces convergent behavior. This connects to Vass's struggle protocol -- externally mandated uncertainty disclosure complements internally calibrated confidence signals.

---

## 7. The Design-by-Contract Heritage

Bertrand Meyer's Design-by-Contract (1992) established the conceptual foundation: software components specify preconditions, postconditions, and class invariants as formal contracts between callers and implementers. The extension to AI agents generalizes this from:
- **Single function calls -> Multi-turn sessions** (ABC)
- **Deterministic behavior -> Probabilistic compliance** (ABC's (p,delta,k)-satisfaction)
- **Static verification -> Runtime monitoring** (AgentSpec, Pro2Guard, VeriGuard)
- **Single agent -> Multi-agent composition** (Agent Contracts conservation laws, ABC compositionality theorem)

The self-evolution trilemma (Wang et al., 2026) proves that active enforcement is necessary -- passive monitoring cannot prevent all forms of behavioral drift in self-evolving agents, validating the runtime enforcement approach.

---

## 8. Open Questions and Tensions

1. **Contract bloat vs. instruction-following**: As contracts grow more comprehensive, instruction-following quality degrades. Vass addresses this with tiered contracts; the community recommends progressive disclosure. No formal theory exists for optimal contract size.

2. **Prompt-level vs. boundary-level enforcement**: MIT Technology Review (2026) argues "rules fail at the prompt, succeed at the boundary" -- external enforcement (type systems, linters, test suites) is more robust than prompt-based behavioral instructions. This challenges the entire premise of prompt-level behavioral contracts.

3. **Training-time vs. runtime alignment**: Constitutional AI and RLHF shape model weights; runtime contracts monitor outputs. Neither alone is sufficient. The interaction between the two is poorly understood.

4. **Sycophancy as product decision**: Vass argues laziness and sycophancy are product decisions, not bugs -- features from a business perspective. Behavioral contracts attempt to override product-level incentives through deployment-time constraints, which creates a fundamental tension.

5. **Anti-gaming robustness**: System prompt extraction attacks (the most common attacker objective in Q4 2025) can expose contract details, enabling targeted circumvention. Contract security remains an open problem.

6. **Multi-agent contract propagation**: How behavioral norms propagate across agent boundaries in multi-agent systems. ABC's compositionality theorem provides necessary conditions but assumes independence of recovery mechanisms.

---

## Source Log

| # | URL | Status | Relevance | Key Finding |
|---|-----|--------|-----------|-------------|
| 1 | https://medium.com/@tangi.vass/turning-ai-coding-agents-into-senior-engineering-peers-c3d178621c9e | blocked (403) | high | Primary source -- state machine, approval request, struggle protocol, cost gradient, tiered contracts. Reconstructed from search results and secondary sources. |
| 2 | https://medium.com/@tangi.vass/i-tried-to-kill-vibe-coding-i-built-adversarial-vibe-coding-without-the-vibes-bc4a63872440 | blocked (403) | high | Follow-up -- Liza multi-agent system, blackboard architecture, adversarial review loops. Reconstructed from search results. |
| 3 | https://arxiv.org/html/2602.22302 | fetched | high | ABC framework -- formal specification, drift bounds, compositionality theorem, AgentAssert runtime, 1980 session evaluation |
| 4 | https://arxiv.org/html/2601.08815 | fetched | high | Agent Contracts -- resource-bounded 7-tuple framework, conservation laws, lifecycle states, 90% token reduction |
| 5 | https://arxiv.org/html/2503.18666 | fetched | high | AgentSpec -- ICSE 2026, DSL for runtime enforcement, three-tuple rules, 90%+ prevention rate |
| 6 | https://arxiv.org/html/2508.00500 | search | high | Pro2Guard -- proactive DTMC enforcement, 93.6% safety, PAC guarantees |
| 7 | https://arxiv.org/html/2510.05156v1 | search | high | VeriGuard -- dual-stage formal verification, near-zero attack success rate |
| 8 | https://arxiv.org/html/2505.19443v1 | fetched | medium | Vibe coding vs agentic coding comparison -- autonomy levels, quality gates |
| 9 | https://arxiv.org/html/2509.23994v1 | fetched | medium | Policy-as-Prompt -- automated guardrail synthesis from design documents |
| 10 | https://dev.to/zer0h1ro/7-layer-constitutional-ai-guardrails-preventing-agent-mistakes-15i5 | fetched | medium | ODEI 7-layer guardrails -- 65%/15%/20% approval/reject/escalate |
| 11 | https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents | fetched | high | Anthropic's two-agent harness pattern, feature list, one-feature-per-session |
| 12 | https://jvaneyck.wordpress.com/2026/02/22/guardrails-for-agentic-coding-how-to-move-up-the-ladder-without-lowering-your-bar/ | fetched | high | 6 practical guardrails for agentic coding -- deterministic tools, architectural tests |
| 13 | https://promptengineering.org/agents-at-work-the-2026-playbook-for-building-reliable-agentic-workflows/ | fetched | medium | 2026 playbook -- verification-aware planning, structured outputs, observability |
| 14 | https://www.humanlayer.dev/blog/writing-a-good-claude-md | fetched | medium | CLAUDE.md best practices -- WHAT/WHY/HOW, 150-200 instruction limit |
| 15 | https://medium.com/@rentierdigital/i-stopped-vibe-coding-and-started-prompt-contracts-claude-code-went-from-gambling-to-shipping-4080ef23efac | blocked (403) | medium | Prompt contracts for Claude Code |
| 16 | https://arxiv.org/html/2402.01586v3 | search | medium | TrustAgent -- agent constitution, pre/in/post-planning safety strategies |
| 17 | https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback | search | high | Constitutional AI -- RLHF-CAI training, self-critique, principle-based revision |
| 18 | https://github.com/guardrails-ai/guardrails | search | medium | Guardrails AI library -- I/O validation, validator hub |
| 19 | https://openai.github.io/openai-agents-python/guardrails/ | search | medium | OpenAI Agents SDK -- tripwire mechanism, parallel validation |
| 20 | https://github.com/NVIDIA-NeMo/Guardrails | search | medium | NeMo Guardrails -- Colang DSL, parallel rails execution |
| 21 | https://github.com/superagent-ai/superagent | search | medium | Superagent -- Safety Agent enforcement layer |
| 22 | https://arxiv.org/html/2601.11816 | search | medium | POLARIS -- typed planning, validator-gated execution, audit trails |
| 23 | https://arxiv.org/html/2509.21224 | fetched | medium | Spontaneous meta-cognitive patterns in LLM agents -- behavioral determinism |
| 24 | https://tessl.io/blog/8-trends-shaping-software-engineering-in-2026-according-to-anthropics-agentic-coding-report/ | fetched | medium | Anthropic 8 trends -- agents learning when to ask for help |
| 25 | https://en.wikipedia.org/wiki/Design_by_contract | search | low | Design-by-Contract heritage -- Meyer 1992 foundations |
| 26 | https://arxiv.org/html/2510.13857v1 | search | medium | From Craft to Constitution -- governance-first paradigm for agent engineering |
| 27 | https://www.technologyreview.com/2026/01/28/1131003/rules-fail-at-the-prompt-succeed-at-the-boundary/ | blocked | high | Rules fail at prompt, succeed at boundary -- boundary enforcement thesis |
| 28 | https://arxiv.org/html/2508.07407 | search | low | Self-evolving agents survey -- evolution trilemma |
| 29 | https://code.claude.com/docs/en/sub-agents | search | low | Claude Code subagents documentation |
| 30 | https://claude.com/blog/using-claude-md-files | search | medium | Official CLAUDE.md guidance |

## Research Context

- **Query**: Behavioral contracts for AI coding agents -- Tangi Vass articles, state machine patterns, agentic guardrails, constitutional AI, practical implementations
- **Depth**: deep (auto-detected -- broad, multi-faceted topic spanning formal methods, practical engineering, and AI alignment)
- **Existing vault knowledge**: agent-cognition topic map covers multi-turn degradation (Laban et al. 2025, ~39% average degradation, RLHF premature commitment root cause, Fresh Context Pattern mitigation). No existing notes on behavioral contracts, state machines for agents, or agentic guardrails.
- **Knowledge gap addressed**: This research covers the entire landscape of constraining AI agent behavior at deployment time -- from practitioner-level CLAUDE.md contracts through formal academic frameworks to production guardrails infrastructure. It bridges the existing vault's understanding of WHY agents fail (multi-turn degradation) with HOW to structurally prevent those failures (behavioral contracts).
