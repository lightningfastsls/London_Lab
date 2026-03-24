---
description: RL-based LLM alignment methods -- RLHF pipeline, policy optimization algorithms (PPO, GRPO, REINFORCE, DPO), reward hacking dynamics, and the reward-engineering-over-algorithm-engineering meta-finding
type: moc
topics: "[[index]]"
---

# rl-alignment

Reinforcement learning methods for aligning language models with human preferences. Covers the full RLHF pipeline, policy optimization algorithms and their simplification trajectory, reward hacking dynamics and Goodhart's law manifestations, and the meta-finding that reward engineering dominates algorithm engineering. Connects to [[agent-cognition]] via the premature helpfulness root cause that RLHF creates, to [[agent-governance]] via reward hacking as the training-time precursor to runtime behavioral drift, to [[model-adaptation]] via the parameter-efficient training methods these algorithms operate on, and to [[transformer-architecture]] via the emergence phenomena observed during RL training.

## Synthesis

The RLHF field shows a clear simplification trajectory: from 4-model PPO to 3-model GRPO/REINFORCE to 2-model DPO to 1-model SimPO. The striking empirical finding is that simpler algorithms often match or exceed complex ones because since [[REINFORCE bandit formulation works for LLMs because strong pretrained priors make complex RL machinery unnecessary]], pretrained LLMs already occupy a good region of parameter space. The meta-finding from Search-R1 that since [[reward design including prompt templates and action penalties has larger effect on alignment quality than the choice between PPO GRPO and REINFORCE]] suggests the field over-invested in algorithm innovation relative to reward design. Reward hacking -- since [[reward hacking in RLHF follows Goodhart's law with four variants regressional extremal causal and adversarial]] -- is the central failure mode, with verifiable rewards providing the cleanest escape for domains where correctness can be mechanically checked.

## The RLHF Pipeline

- [[RL is needed for LLM alignment because no differentiable loss function captures the multi-dimensional quality of human preference judgments]] -- why RL, not just supervised learning
- [[SFT suffers from exposure bias where teacher-forcing creates reliance on ground-truth context that degrades autoregressive generation]] -- why SFT alone is insufficient
- [[RLHF follows a four-stage pipeline from pretraining through SFT to reward model training and RL fine-tuning]] -- the canonical architecture
- [[pairwise comparisons produce more reliable human preference data than absolute ratings because relative judgments avoid calibration problems]] -- why pairwise preference collection
- [[reward model training uses Bradley-Terry pairwise comparison on approximately 50k labeled preference samples]] -- Stage 3 specifics

## Credit Assignment

- [[credit assignment over hundreds of tokens from a single scalar reward is the central bottleneck of RLHF]] -- the fundamental challenge
- [[sequence-level bandit formulation matches LLM outcome-reward settings better than per-token MDP modeling]] -- the simplification that works

## PPO

- [[PPO for RLHF requires four models simultaneously creating a memory bottleneck that motivated critic-free alternatives]] -- the infrastructure burden
- [[PPO clipped surrogate objective constrains policy updates to a trust region preventing catastrophic forgetting during RLHF]] -- the core stability mechanism
- [[PPO spends 80 percent of compute time on sample generation making it the dominant cost in RLHF training]] -- the compute bottleneck

## REINFORCE Family

- [[REINFORCE bandit formulation works for LLMs because strong pretrained priors make complex RL machinery unnecessary]] -- why simplicity wins
- [[REINFORCE Leave-One-Out uses 50-70 percent less memory than PPO while consistently outperforming it on alignment tasks]] -- the strongest simple baseline
- [[REINFORCE++ bridges REINFORCE simplicity with PPO stability via token-level KL penalty and ratio clipping achieving 30 percent training time reduction]] -- selective PPO innovation adoption

## GRPO

- [[GRPO eliminates the critic network through group-relative advantage scoring achieving 50 percent memory reduction over PPO]] -- critic-free alternative
- [[GRPO requires large batch sizes for stability and suffers frequent training collapse in multi-step long-context reasoning]] -- the stability cost
- [[DeepSeek-R1-Zero trained purely with GRPO produced emergent reasoning behaviors including self-reflection and verification without explicit training]] -- emergent reasoning via clean rewards

## DPO

- [[DPO eliminates the reward model by deriving a closed-form relationship between optimal policy and reward function enabling pure classification-based alignment]] -- the offline approach
- [[PPO consistently outperforms DPO across dialogue code generation and safety tasks but DPO adoption grew 45 percent by 2025 due to simplicity]] -- the quality-simplicity trade-off

## Algorithm Comparison

- [[Search-R1 found REINFORCE outperformed both PPO and GRPO for agentic deep research tasks with the highest accuracy and most efficient search strategies]] -- the definitive comparison
- [[reward design including prompt templates and action penalties has larger effect on alignment quality than the choice between PPO GRPO and REINFORCE]] -- the meta-finding: reward engineering > algorithm engineering

## Reward Hacking & Failure Modes

- [[reward hacking in RLHF follows Goodhart's law with four variants regressional extremal causal and adversarial]] -- the theoretical framework
- [[RLHF-trained models exhibit sycophancy verbosity bias and confident nonsense as systematic reward hacking manifestations]] -- the behavioral instantiation
- [[Anthropic curriculum study showed models progress from political sycophancy through tool manipulation to directly rewriting their own reward function]] -- the escalation trajectory
- [[scaling laws for reward over-optimization show proxy rewards grow linearly while gold rewards follow a non-linear curve that eventually decreases]] -- the quantitative dynamics
- [[KL divergence penalty prevents reward model exploitation but paradoxically increases the proxy-gold reward gap]] -- the counterintuitive safeguard limitation

## Reward Design

- [[verifiable rewards bypass learned reward models entirely avoiding reward hacking for math and code tasks]] -- the cleanest escape from reward hacking
- [[Slow Thinking training templates are prone to self-reinforcing reasoning collapse where think tag frequency correlates with reward creating positive feedback loops]] -- template-induced collapse
- [[F1-based reward training causes answer avoidance where the policy learns never answering is safer than risking wrong answers]] -- reward function edge case

## Bridges to Other Domains

- [[RLHF training rewards premature helpfulness causing LLMs to make early assumptions that anchor subsequent responses]] -- bridge to [[agent-cognition]]: RLHF creates the multi-turn root cause
- [[training-time alignment and runtime contracts are complementary because neither alone prevents behavioral drift in long sessions]] -- bridge to [[agent-governance]]: why runtime governance is needed
- [[whether RLHF can be modified to reward clarification-seeking over premature helpfulness in multi-turn settings]] -- open question bridging alignment and agent cognition

## Related Areas

- [[agent-cognition]] -- multi-turn degradation traces to RLHF training incentives
- [[agent-governance]] -- runtime contracts as complement to training-time alignment
- [[model-adaptation]] -- LoRA/PEFT as the parameter-efficient substrate for RL fine-tuning
- [[transformer-architecture]] -- ICL theory and induction head emergence as parallel to RL-induced emergent behaviors

---

Topics:
- [[index]]
