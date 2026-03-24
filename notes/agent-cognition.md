---
description: Domain hub for agent cognitive architecture — routes to multi-turn degradation, external cognition, and bridges to RL alignment, ICL-to-weights, and code review
type: moc
topics: "[[index]]"
---

# agent-cognition

Domain hub for agent cognitive architecture. Routes to specialized sub-maps for the two major research clusters. For context window mechanisms, benchmarks, and architectural management patterns, see [[context-management]]. For cross-session memory infrastructure, see [[agent-memory]]. For behavioral contracts and governance, see [[agent-governance]].

## Sub-Maps

- [[multi-turn-degradation]] -- empirical findings on LLM multi-turn performance loss (~39% avg), RLHF root causes, task susceptibility framework, and Fresh Context mitigation pattern
- [[agent-external-cognition]] -- how vaults, hooks, and session design extend agent cognition beyond the context window: hook theory, traversal dynamics, between-session processing, external cognition

## Synthesis

Two complementary research clusters inform agent workflow design. First, since [[LLMs lose approximately 25 percentage points average performance when tasks are distributed across conversational turns]], multi-turn degradation is the central failure mode — see [[multi-turn-degradation]] for the full treatment. Second, the vault's response to this problem is externalized cognitive architecture: since [[external memory shapes cognition more than base model]], hooks, traversal patterns, and session design compensate for what the model cannot do alone — see [[agent-external-cognition]].

## RL Alignment Root Causes
These notes trace multi-turn degradation mechanisms back to their RLHF training origins -- see [[rl-alignment]] for the full RL alignment treatment.
- [[RL is needed for LLM alignment because no differentiable loss function captures the multi-dimensional quality of human preference judgments]] -- why RL training exists, and why its reward signal shapes failure modes
- [[RLHF follows a four-stage pipeline from pretraining through SFT to reward model training and RL fine-tuning]] -- the pipeline that creates premature helpfulness incentives
- [[credit assignment over hundreds of tokens from a single scalar reward is the central bottleneck of RLHF]] -- why per-token helpfulness feedback is impossible under standard RLHF
- [[RLHF-trained models exhibit sycophancy verbosity bias and confident nonsense as systematic reward hacking manifestations]] -- the behavioral consequences of reward optimization
- [[SFT suffers from exposure bias where teacher-forcing creates reliance on ground-truth context that degrades autoregressive generation]] -- why SFT alone cannot fix multi-turn issues
- [[DeepSeek-R1-Zero trained purely with GRPO produced emergent reasoning behaviors including self-reflection and verification without explicit training]] -- emergent cognitive behaviors from RL training parallel induction head emergence

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
- This is a domain-level hub. The two sub-maps cover distinct research clusters: multi-turn degradation (what goes wrong) and external cognition (how the vault compensates).
- The bridge sections above (RL Alignment, ICL-to-Weights, Code Review) remain here because they connect across multiple sub-maps and sibling topic maps.
- Context window mechanisms, benchmarks, architectural patterns, and model-level techniques are in [[context-management]].
- For the RL training mechanisms that create premature helpfulness and sycophancy, see [[rl-alignment]].
