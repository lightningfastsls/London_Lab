---
description: "Raschka's practical finding — single-epoch training on instruction-following datasets preserves generalization; multiple epochs on static data degrades base model capabilities"
type: finding
confidence: likely
created: 2026-03-02
topics:
  - "[[model-adaptation]]"
---

# multi-epoch LoRA training on static instruction data causes overfitting and capability degradation

Raschka's extensive LoRA experiments (2023-2024) identified a practical anti-pattern: training LoRA adapters for multiple epochs on static instruction-following datasets causes overfitting and degrades the base model's general capabilities. This is distinct from the more general phenomenon where since [[LoRA acts as implicit regularizer preserving base model capabilities with strong inverse linear relationship between adaptation and forgetting]] — even LoRA's implicit regularization cannot compensate for repeated exposure to the same static data.

The mechanism is straightforward: instruction-following datasets like Alpaca are relatively small and homogeneous. Multiple passes cause the adapter to memorize surface patterns rather than learning generalizable instruction-following behavior. Since [[dataset quality exceeds quantity for LoRA fine-tuning as curated 1K LIMA matches 50K Alpaca performance]], the evidence points toward single-epoch training on high-quality, diverse data rather than multi-epoch grinding on larger but lower-quality collections.

This interacts with rank choice — higher-rank adapters have more capacity to memorize, making them more susceptible to multi-epoch overfitting. The practical guidance: prefer single-epoch training, invest effort in data curation rather than training duration.

---

Source: [[lora-doc-to-lora-hypernetworks-research-2026-03-02]]

Relevant Notes:
- [[LoRA acts as implicit regularizer preserving base model capabilities with strong inverse linear relationship between adaptation and forgetting]] -- the regularization that multi-epoch training overwhelms
- [[dataset quality exceeds quantity for LoRA fine-tuning as curated 1K LIMA matches 50K Alpaca performance]] -- the complementary finding about data quality
- [[adapting multiple LoRA weight matrices with lower rank outperforms single-matrix adaptation at higher rank for the same parameter budget]] -- higher-rank adapters are more susceptible to this overfitting

Topics:
- [[model-adaptation]]
