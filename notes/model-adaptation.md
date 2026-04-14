---
description: LoRA, PEFT variants, hypernetworks, and the ICL-to-LoRA theoretical bridge -- parameter-efficient adaptation and instant knowledge internalization
type: moc
topics: "[[index]]"
---

# model-adaptation

Parameter-efficient fine-tuning (PEFT) methods for adapting large models without full retraining. Covers LoRA fundamentals and the low-rank hypothesis, practical variants (DoRA, QLoRA, rsLoRA), production serving patterns, hypernetwork-based instant adaptation (Doc-to-LoRA), and the theoretical bridge connecting in-context learning to weight-space modification.

## LoRA Fundamentals
- [[LoRA exploits low intrinsic rank of weight updates to match full fine-tuning with 10000x fewer trainable parameters]] -- foundational PEFT method decomposing weight updates as B*A with r << d
- [[pre-trained language models have low intrinsic dimension with larger models having even lower intrinsic dimension after pre-training]] -- theoretical foundation explaining why LoRA works
- [[LoRA adaptation amplifies existing underemphasized directions in pre-trained weights rather than learning entirely new features]] -- mechanistic insight: LoRA boosts, not creates
- [[adapting multiple LoRA weight matrices with lower rank outperforms single-matrix adaptation at higher rank for the same parameter budget]] -- distributed rank captures more diverse adaptation directions
- [[LoRA introduces no inference latency because adapter weights merge into base model weights unlike adapters and prefix tuning]] -- zero-overhead deployment via weight merging
- [[LoRA acts as implicit regularizer preserving base model capabilities with strong inverse linear relationship between adaptation and forgetting]] -- learns less AND forgets less than full fine-tuning
- [[multi-epoch LoRA training on static instruction data causes overfitting and capability degradation]] -- single-epoch on curated data is the practical recipe
- [[dataset quality exceeds quantity for LoRA fine-tuning as curated 1K LIMA matches 50K Alpaca performance]] -- data curation matters more than volume

## LoRA Variants
- [[DoRA weight decomposition into magnitude and direction consistently outperforms standard LoRA by 1-4 points across model sizes]] -- ICML 2024 Oral; separates magnitude and direction updates
- [[QLoRA 4-bit quantization enables 7B model fine-tuning on consumer GPUs with 33 percent memory savings at 39 percent runtime cost]] -- makes LoRA accessible on consumer hardware
- [[rsLoRA rank-stabilized scaling uses alpha over sqrt(r) instead of alpha over r preventing adaptation strength from depending on rank choice]] -- decouples scaling from rank hyperparameter

## Multi-LoRA Production
- [[multi-LoRA serving enables hundreds of concurrent adapters from a single base model with millisecond switching in production]] -- 2025 production reality across vLLM, TGI, Ray Serve

## Hypernetworks & Instant Adaptation
- [[hypernetworks learn functions that generate weights for other networks amortizing per-task training cost into a single meta-training phase]] -- meta-learning paradigm that enables Doc-to-LoRA
- [[Doc-to-LoRA hypernetwork generates LoRA adapters in a single forward pass via Perceiver cross-attention compressing documents into sub-50 MB weight updates]] -- sub-second document internalization
- [[Doc-to-LoRA chunk composition concatenates along rank dimension enabling extrapolation from 256 training tokens to 32K-plus context]] -- compositional scaling via rank concatenation
- [[Doc-to-LoRA reduces KV-cache memory from 12-plus GB to constant sub-50 MB regardless of document length by moving information from context to weights]] -- constant memory regardless of document length
- [[Text-to-LoRA generates task-specific LoRA adapters from natural language descriptions in a single forward pass replacing fine-tuning pipelines]] -- task-description-to-adapter generation
- [[Doc-to-LoRA transfers visual information from VLM to text-only LLM achieving 75 percent accuracy on image classification without visual training data]] -- cross-modal weight transfer

## ICL-LoRA Theoretical Bridge
- [[LoRA makes explicit through gradient descent what ICL does implicitly through attention since both find task-relevant directions in weight space]] -- fundamental ICL-LoRA equivalence
- [[the ICL to LoRA to Doc-to-LoRA progression represents a spectrum from implicit temporary to explicit persistent knowledge internalization]] -- the full knowledge internalization spectrum
- [[context distillation bridges ICL and fine-tuning by training a model to reproduce context-conditioned outputs without the context present]] -- the explicit transfer process from context to weights

## RL-Based Alignment Training
These notes cover the RL fine-tuning stage that operates on adapted model weights -- see [[rl-alignment]] for the full treatment.
- [[RLHF follows a four-stage pipeline from pretraining through SFT to reward model training and RL fine-tuning]] -- the canonical pipeline where Stage 4 (RL) operates on Stage 2 (SFT) adapted weights
- [[PPO for RLHF requires four models simultaneously creating a memory bottleneck that motivated critic-free alternatives]] -- memory constraints that parallel LoRA's motivation for parameter efficiency
- [[DPO eliminates the reward model by deriving a closed-form relationship between optimal policy and reward function enabling pure classification-based alignment]] -- alignment as classification rather than RL, requiring only 2 models
- [[PPO consistently outperforms DPO across dialogue code generation and safety tasks but DPO adoption grew 45 percent by 2025 due to simplicity]] -- simplicity-quality trade-off paralleling LoRA's own efficiency-quality balance
- [[SFT suffers from exposure bias where teacher-forcing creates reliance on ground-truth context that degrades autoregressive generation]] -- why RL fine-tuning is needed after SFT
- [[REINFORCE++ bridges REINFORCE simplicity with PPO stability via token-level KL penalty and ratio clipping achieving 30 percent training time reduction]] -- selective PPO innovation adoption: keeps clipping and KL, drops critic
- [[pairwise comparisons produce more reliable human preference data than absolute ratings because relative judgments avoid calibration problems]] -- the preference data methodology underlying reward model training
- [[reward model training uses Bradley-Terry pairwise comparison on approximately 50k labeled preference samples]] -- the scale of labeled data parallels the dataset quality finding for LoRA

## Open Questions
- Whether Doc-to-LoRA can be applied to bioacoustic domain adaptation
- How LoRA rank interacts with the intrinsic dimensionality of USV representation tasks
- Whether the ICL-LoRA spectrum extends to multi-modal settings beyond text

## Related Areas
- [[transformer-architecture]] -- the attention and MLP mechanics that ICL operates through
- [[representation-learning]] -- VQ-VAE pipeline that could benefit from LoRA-based adaptation
- [[bioacoustic-ssl]] -- foundation models that LoRA could efficiently adapt to USV tasks
- [[agent-cognition]] -- ICL-to-weights internalization in the context of agent knowledge management
- [[context-management]] -- Doc-to-LoRA as an alternative to long-context processing
- [[rl-alignment]] -- RL fine-tuning stage operates on the same weight adaptation substrate as LoRA/PEFT

---

Topics:
- [[index]]
