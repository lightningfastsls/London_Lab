---
description: Empirical findings on LLM multi-turn performance loss, RLHF root causes, task susceptibility framework, and Fresh Context mitigation pattern
type: moc
topics: "[[agent-cognition]]"
---

# multi-turn-degradation

Empirical findings on how LLMs degrade across conversational turns, why RLHF-driven premature commitment is the root cause, and what structural mitigations work. Split from [[agent-cognition]] to give this research cluster its own orientation surface. For vault-as-cognitive-architecture patterns, see [[agent-external-cognition]]. For within-session context window mechanisms, see [[context-management]].

## Synthesis

Multi-turn degradation (~39% average, Laban et al. 2025) traces to RLHF-driven premature commitment -- primarily unreliability (112% increase), not capability loss. The degradation is architecture-independent: since [[approximately 60 percent of relative multi-turn degradation is constant across model sizes suggesting scaling alone cannot solve it]], the fix must be structural, not just scaling. The convergent mitigation is the Fresh Context Pattern: since [[concatenating all requirements into a single prompt restores approximately 95 percent of single-turn performance]], gathering all requirements before engaging the model eliminates the temporal distribution that causes degradation. Multi-turn degradation also compounds with context window degradation in long sessions (see [[context-management]] for the CW side).

## Core Findings
- [[LLMs lose approximately 25 percentage points average performance when tasks are distributed across conversational turns]] -- the headline finding, 15 models, 6 tasks
- [[multi-turn degradation is primarily a 112 percent increase in unreliability rather than capability loss]] -- models CAN still solve tasks, they just don't RELIABLY
- [[even two conversational turns trigger multi-turn degradation regardless of task complexity]] -- temporal distribution, not information volume
- [[approximately 60 percent of relative multi-turn degradation is constant across model sizes suggesting scaling alone cannot solve it]] -- from Liu et al. 2026

## Task Susceptibility
- [[tasks vulnerable to multi-turn degradation are generative and non-episodic requiring information fusion across turns]] -- the three-property framework
- [[episodic multi-turn tasks that decompose into independent subtasks overestimate LLM multi-turn performance]] -- why existing benchmarks mislead
- [[most existing multi-turn benchmarks are episodic and overestimate real multi-turn performance]] -- the evaluation gap

## Root Causes
- [[RLHF training rewards premature helpfulness causing LLMs to make early assumptions that anchor subsequent responses]] -- the training incentive root cause
- [[LLMs attempt full solution generation on the first turn even when given only a vague initial shard]] -- the anchoring mechanism
- [[answer bloat compounds multi-turn errors as responses grow verbose without pruning incorrect assumptions]] -- the compounding anti-pattern
- [[LLMs over-adjust based on the last turn of conversation disproportionately weighting recent information]] -- recency bias at conversation level
- [[LLMs prematurely commit to incorrect solutions in early turns and fail to revise them producing cascading errors]] -- the revision failure pattern
- [[reasoning models produce longer responses and additional test-time compute does not solve multi-turn unreliability]] -- more thinking ≠ better multi-turn

## Theoretical Framing
- [[user utterances are a lossy compression of high-dimensional intent into low-dimensional surface forms]] -- Liu's information-theoretic foundation
- [[the principle of least effort drives conversational underspecification making ambiguity a fundamental feature not a bug]] -- Zipf's law applied to conversation

## Mitigations
- [[concatenating all requirements into a single prompt restores approximately 95 percent of single-turn performance]] -- validates Fresh Context Pattern
- [[snowball turn-by-turn accumulation mitigates 15-20 percent of the full-to-sharded performance deterioration]] -- realistic production strategy
- [[Mediator-Assistant framework separates intent inference from task execution recovering approximately 20 percentage points]] -- Liu et al. architecture
- [[temperature reduction has minimal effect on multi-turn unreliability because conversation structure introduces variation independently]] -- determinism doesn't help
- [[RAG-based memory provides only marginal improvement versus intent resolution demonstrating retrieval is not equivalent to resolving intent]] -- memory ≠ understanding
- [[fresh context per task preserves quality better than chaining phases]] -- spawning fresh subagents per phase prevents multi-turn degradation (also in [[agent-external-cognition]])

## Methods
- [[instruction sharding methodology enables controlled comparison between single-turn and multi-turn LLM performance]] -- Laban's core experimental technique

## Tensions
- [[the 39 percent degradation figure may overstate the problem for well-designed systems while understating it for messy real-world interactions]] -- Arani critique

## CW-MT Interaction
- [[context window and multi-turn degradation have distinct root causes but compound when both occur in long multi-turn sessions]] -- bridge to [[context-management]], resolves mechanism question

## Open Questions
- [[whether RLHF can be modified to reward clarification-seeking over premature helpfulness in multi-turn settings]] -- training objective redesign
- [[whether context window size and multi-turn degradation are independent or correlated phenomena]] -- largely resolved (see CW-MT Interaction)

## Agent Notes
- These findings directly inform the vault's session design: the orient-work-persist rhythm, /clear between phases, plan-file-first workflow, and subagent isolation all function as multi-turn mitigations.
- The Fresh Context Pattern is the single most effective mitigation (~95% recovery). The vault implements this through subagent isolation (/ralph) and session boundaries.
- For the RL training mechanisms that create premature helpfulness, see [[rl-alignment]] (bridged from parent [[agent-cognition]]).
