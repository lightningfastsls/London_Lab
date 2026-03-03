---
description: Entry point to the knowledge system -- start here to navigate
type: moc
---

# index

Welcome to your USV research knowledge system.

## Topic Maps
- [[detection]] -- USV detection pipeline, energy detection, candidate generation
- [[classification]] -- CNN operational pipeline, labeling, training, performance baselines
  - [[classification-tools]] -- DeepSqueak, Python tools landscape, Raven interchange
  - [[classification-methodology]] -- clustering, repertoire comparison, few-shot learning, cross-population generalization
- [[representation-learning]] -- VQ-VAE pipeline, codebook discovery, information-theoretic analysis, null models, probing
- [[unsupervised-usv-discovery]] -- clustering methods, USV literature, alternative approaches for vocal repertoire discovery
- [[bioacoustic-ssl]] -- self-supervised learning paradigms, foundation models, cross-species transfer
- [[model-adaptation]] -- LoRA, PEFT variants, hypernetworks, ICL-LoRA theoretical bridge
- [[signal-processing]] -- STFT computation, frequency analysis, spectrogram rendering at 300 kHz
- [[experimental-methods]] -- hub: training data preparation, wild-lab vocal comparison, behavioral integration
  - [[training-methodology]] -- splits, augmentation, labeling, classification baselines
  - [[wild-lab-vocal-comparison]] -- courtship degradation hypotheses, repertoire statistics
  - [[behavioral-integration]] -- LMT synchronization, recording infrastructure
- [[agent-cognition]] -- LLM multi-turn degradation, root causes, mitigations, and theoretical framing
- [[context-management]] -- within-session context window degradation, attention mechanisms, benchmarks, and architectural patterns
- [[agent-memory]] -- cross-session memory architecture, forgetting strategies, MCP ecosystem, and multi-agent orchestration infrastructure
- [[agent-governance]] -- behavioral contracts, formal compliance frameworks, enforcement layers, and implementation patterns for constraining agent behavior
- [[transformer-architecture]] -- self-attention mechanics, positional encoding, MLP function, residual streams, and in-context learning theory
- [[rl-alignment]] -- RLHF pipeline, policy optimization algorithms (PPO, GRPO, REINFORCE, DPO), reward hacking dynamics, reward engineering meta-finding
- [[generative-modeling]] -- diffusion models, flow matching, stability analysis (Jensen Gap vs bounded gain), prediction targets, acceleration, production adoption

## Getting Started
1. Read ops/goals.md to orient on current threads
2. Capture your first note in notes/
3. Connect it to a topic map above
