---
description: LLM multi-turn degradation mechanisms, root causes, mitigations, and theoretical framing for agent workflow design
type: moc
---

# agent-cognition

How LLMs degrade across conversational turns, why premature commitment is the root cause, and what mitigations work. This topic map covers the multi-turn behavioral side of agent performance. For context window mechanisms, benchmarks, and architectural management patterns, see [[context-management]].

## Synthesis

Multi-turn degradation (~39% average, Laban et al. 2025) traces to RLHF-driven premature commitment -- primarily unreliability (112% increase), not capability loss. The degradation is architecture-independent: since [[approximately 60 percent of relative multi-turn degradation is constant across model sizes suggesting scaling alone cannot solve it]], the fix must be structural, not just scaling. The convergent mitigation is the Fresh Context Pattern: since [[concatenating all requirements into a single prompt restores approximately 95 percent of single-turn performance]], gathering all requirements before engaging the model eliminates the temporal distribution that causes degradation. Multi-turn degradation also compounds with context window degradation in long sessions (see [[context-management]] for the CW side), which is why the vault's orient-work-persist rhythm, /clear between phases, and subagent isolation all serve as dual mitigations.

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

## Methods
- [[instruction sharding methodology enables controlled comparison between single-turn and multi-turn LLM performance]] -- Laban's core experimental technique

## Tensions
- [[the 39 percent degradation figure may overstate the problem for well-designed systems while understating it for messy real-world interactions]] -- Arani critique

## CW-MT Interaction
- [[context window and multi-turn degradation have distinct root causes but compound when both occur in long multi-turn sessions]] -- bridge to [[context-management]], resolves mechanism question

## RL Alignment Root Causes
These notes trace multi-turn degradation mechanisms back to their RLHF training origins -- see [[rl-alignment]] for the full RL alignment treatment.
- [[RL is needed for LLM alignment because no differentiable loss function captures the multi-dimensional quality of human preference judgments]] -- why RL training exists, and why its reward signal shapes failure modes
- [[RLHF follows a four-stage pipeline from pretraining through SFT to reward model training and RL fine-tuning]] -- the pipeline that creates premature helpfulness incentives
- [[credit assignment over hundreds of tokens from a single scalar reward is the central bottleneck of RLHF]] -- why per-token helpfulness feedback is impossible under standard RLHF
- [[RLHF-trained models exhibit sycophancy verbosity bias and confident nonsense as systematic reward hacking manifestations]] -- the behavioral consequences of reward optimization
- [[SFT suffers from exposure bias where teacher-forcing creates reliance on ground-truth context that degrades autoregressive generation]] -- why SFT alone cannot fix multi-turn issues
- [[DeepSeek-R1-Zero trained purely with GRPO produced emergent reasoning behaviors including self-reflection and verification without explicit training]] -- emergent cognitive behaviors from RL training parallel induction head emergence

## Open Questions
- [[whether RLHF can be modified to reward clarification-seeking over premature helpfulness in multi-turn settings]] -- training objective redesign
- [[whether context window size and multi-turn degradation are independent or correlated phenomena]] -- largely resolved (see CW-MT Interaction)

## ICL-to-Weights Knowledge Internalization
These notes bridge agent cognition with representation learning -- the progression from volatile context-based knowledge to persistent weight-based knowledge.
- [[the ICL to LoRA to Doc-to-LoRA progression represents a spectrum from implicit temporary to explicit persistent knowledge internalization]] -- the full knowledge persistence spectrum from ICL to weights
- [[LoRA makes explicit through gradient descent what ICL does implicitly through attention since both find task-relevant directions in weight space]] -- fundamental ICL-LoRA equivalence in weight modification space
- [[context distillation bridges ICL and fine-tuning by training a model to reproduce context-conditioned outputs without the context present]] -- the explicit process of transferring knowledge from context to weights

## Applied Cognition: Code Review
These notes extend agent-cognition findings into the code review domain -- see [[agent-governance]] for the full code review treatment.
- [[same-model generation and review creates confirmation bias producing 8x duplicated code and 72 percent Java security failures]] -- confirmation bias as measurable cognitive failure in self-review
- [[code review follows orientation then analytical phases where skipping orientation degrades analytical quality]] -- CRDM cognitive model of review as recognition-primed decision-making
- [[supervisory QA-Checker agent monitoring conversation prevents prompt drifting improving vulnerability confirmation from 73 to 93 percent]] -- prompt drifting as inter-agent cognitive degradation
- [[memory wipe per review turn prevents attention degradation treating each attempt as fresh start guided by coach feedback]] -- radical attention reset as cognitive intervention

## Agent Notes
- These findings directly inform the vault's session design: the orient-work-persist rhythm, /clear between phases, plan-file-first workflow, and subagent isolation all function as multi-turn mitigations.
- The [[bulk-source-processing-strategy]] in ops/methodology/ applies the Fresh Context principle to knowledge processing.
- Context window mechanisms, benchmarks, architectural patterns, and model-level techniques are now in [[context-management]].
- For the RL training mechanisms that create premature helpfulness and sycophancy, see [[rl-alignment]].
