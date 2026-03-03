---
description: "Liu et al ICML 2024 Oral (1.5% acceptance) — decomposes weight into magnitude+direction, applies LoRA to direction only, closing the gap between LoRA and full fine-tuning learning patterns"
type: finding
confidence: proven
created: 2026-03-02
topics:
  - "[[model-adaptation]]"
---

# DoRA weight decomposition into magnitude and direction consistently outperforms standard LoRA by 1-4 points across model sizes

DoRA (Weight-Decomposed Low-Rank Adaptation, Liu et al. ICML 2024 Oral — 1.5% acceptance rate) addresses a subtle mismatch between how LoRA and full fine-tuning modify weights. Full fine-tuning changes both the magnitude and direction of weight vectors, but these changes follow different patterns. Standard LoRA couples magnitude and direction updates through its low-rank constraint, which limits its expressiveness.

DoRA decomposes pre-trained weights into magnitude and direction components, then applies LoRA specifically to the directional updates. Results are consistent across model scales: +3.7/+1.0 on Llama 7B/13B, +2.9 on Llama 2 7B, +4.4 on Llama 3 8B across commonsense reasoning, visual instruction tuning, and image/video-text understanding tasks.

The insight that full fine-tuning's magnitude and direction updates differ from LoRA's suggests that since [[LoRA adaptation amplifies existing underemphasized directions in pre-trained weights rather than learning entirely new features]], the amplification has both a "how much" (magnitude) and "which way" (direction) component. DoRA lets the model learn these independently, producing updates that more closely match what full fine-tuning would achieve.

QDoRA (Answer.AI, April 2024) combines DoRA's decomposition with QLoRA's 4-bit quantization, outperforming both full fine-tuning and QLoRA alone on Llama 2 and Llama 3. It is considered the 2025 PEFT standard by practitioners.

---

Source: lora-doc-to-lora-hypernetworks-research-2026-03-02

Relevant Notes:
- [[LoRA exploits low intrinsic rank of weight updates to match full fine-tuning with 10000x fewer trainable parameters]] -- the method DoRA improves upon
- [[LoRA adaptation amplifies existing underemphasized directions in pre-trained weights rather than learning entirely new features]] -- the mechanism DoRA refines by separating magnitude from direction
- [[QLoRA 4-bit quantization enables 7B model fine-tuning on consumer GPUs with 33 percent memory savings at 39 percent runtime cost]] -- combines with DoRA in QDoRA
- [[LoRA acts as implicit regularizer preserving base model capabilities with strong inverse linear relationship between adaptation and forgetting]] -- DoRA partially overcomes LoRA's adaptation ceiling while retaining regularization
- [[LoRA makes explicit through gradient descent what ICL does implicitly through attention since both find task-relevant directions in weight space]] -- DoRA further refines the direction-finding that both ICL and LoRA share, separating magnitude from directional adaptation

Topics:
- [[model-adaptation]]
