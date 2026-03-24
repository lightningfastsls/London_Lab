---
description: "Raschka's experiments confirm Zhou et al LIMA finding — 1K carefully curated examples match or exceed 50K Alpaca for LoRA instruction-tuning, making data curation the bottleneck"
type: finding
confidence: likely
created: 2026-03-02
topics:
  - "[[model-adaptation]]"
---

# dataset quality exceeds quantity for LoRA fine-tuning as curated 1K LIMA matches 50K Alpaca performance

Raschka's practical LoRA experiments confirmed a finding with significant implications for resource-constrained research: the curated 1K LIMA dataset (Zhou et al. 2023) matched or exceeded the much larger 50K Alpaca dataset for instruction-tuning via LoRA. The 50x difference in dataset size did not translate to meaningful quality improvement.

This makes sense in light of what LoRA actually does — since [[LoRA adaptation amplifies existing underemphasized directions in pre-trained weights rather than learning entirely new features]], the adaptation needs enough examples to identify *which* directions to amplify, not a massive corpus to learn new representations. A small number of high-quality, diverse examples suffices to signal the task-relevant directions.

The implication for our domain: LoRA fine-tuning of audio models for USV-specific tasks would require careful curation of relatively few high-quality examples rather than massive labeled datasets. Combined with the finding that since [[multi-epoch LoRA training on static instruction data causes overfitting and capability degradation]], the practical recipe is clear — curate a small, diverse, high-quality dataset and train for a single epoch.

---

Source: lora-doc-to-lora-hypernetworks-research-2026-03-02

Relevant Notes:
- [[LoRA adaptation amplifies existing underemphasized directions in pre-trained weights rather than learning entirely new features]] -- why small datasets suffice mechanistically
- [[multi-epoch LoRA training on static instruction data causes overfitting and capability degradation]] -- complementary finding about training duration
- [[pre-trained language models have low intrinsic dimension with larger models having even lower intrinsic dimension after pre-training]] -- the theoretical basis for why few examples suffice
- [[QLoRA 4-bit quantization enables 7B model fine-tuning on consumer GPUs with 33 percent memory savings at 39 percent runtime cost]] -- combines with small datasets for accessible fine-tuning
- [[reward model training uses Bradley-Terry pairwise comparison on approximately 50k labeled preference samples]] -- contrasts with the RL alignment domain where ~50k preference samples is standard; the quality-over-quantity principle likely applies to reward model training data as well
- [[scaling laws for reward over-optimization show proxy rewards grow linearly while gold rewards follow a non-linear curve that eventually decreases]] -- more reward model training data reduces Goodharting, paralleling how curated data quality reduces overfitting
- [[model size should scale with labeled dataset size to balance underfitting and overfitting]] -- the complementary principle: data quality determines what to train on, model-data scaling determines how much capacity to use

Topics:
- [[model-adaptation]]
