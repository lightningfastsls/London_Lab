---
description: "Hu et al ablation: W_q+W_v at r=4 matches all-four at r=2 (73.7% WikiSQL); Raschka 2024 updates to all layers including MLP (4.2M to 20.3M params, noticeable quality gain)"
type: finding
confidence: proven
created: 2026-03-02
topics:
  - "[[model-adaptation]]"
---

# adapting multiple LoRA weight matrices with lower rank outperforms single-matrix adaptation at higher rank for the same parameter budget

Hu et al.'s ablation study tested different configurations at a fixed 18M parameter budget on GPT-3:

| Configuration | Rank | WikiSQL |
|--------------|------|---------|
| W_q alone | r=8 | 70.4% |
| W_v alone | r=8 | 73.0% |
| W_q + W_v | r=4 | 73.7% |
| W_q + W_k + W_v + W_o | r=2 | 73.7% |

Spreading the parameter budget across more matrices with lower rank captures more diverse adaptation directions than concentrating rank in fewer matrices. The original paper recommended W_q + W_v as the best balance of simplicity and performance.

However, Raschka's extensive experiments (2023-2024) significantly update this guidance: apply LoRA across ALL transformer layers — attention AND MLP projections. Expanding from Q/V-only (4.2M params) to Q, K, V, O projections plus MLP layers (20.3M params) produces noticeable quality improvement with manageable memory overhead (14.18 GB to 16.62 GB). This reflects the broader principle that since [[LoRA adaptation amplifies existing underemphasized directions in pre-trained weights rather than learning entirely new features]], different weight matrices capture different task-relevant directions, and casting a wider net captures more of them.

---

Source: lora-doc-to-lora-hypernetworks-research-2026-03-02

Relevant Notes:
- [[LoRA exploits low intrinsic rank of weight updates to match full fine-tuning with 10000x fewer trainable parameters]] -- the method these ablations explore
- [[LoRA adaptation amplifies existing underemphasized directions in pre-trained weights rather than learning entirely new features]] -- why distributing across matrices helps
- [[rsLoRA rank-stabilized scaling uses alpha over sqrt(r) instead of alpha over r preventing adaptation strength from depending on rank choice]] -- decouples scaling from rank for cleaner ablations
- [[MLP layers store factual associations as distributed key-value memories where first-layer weights match patterns and second-layer weights output associated information]] -- why MLP layers benefit from LoRA adaptation alongside attention

Topics:
- [[model-adaptation]]
