---
description: "Analysis of deltaW vs W_0 shows amplification factors of ~21.5x (r=4) for directions already in the pre-trained weight space — LoRA selectively boosts, not creates"
type: finding
confidence: likely
created: 2026-03-02
topics:
  - "[[model-adaptation]]"
---

# LoRA adaptation amplifies existing underemphasized directions in pre-trained weights rather than learning entirely new features

Analysis of the learned LoRA adaptation matrices (deltaW = B*A) versus the original pre-trained weights (W_0) reveals a striking pattern: the adaptation matrices "amplify directions that are not emphasized in W" with amplification factors around 21.5x for r=4. The directions LoRA boosts already exist in the pre-trained weight space — they were learned during pre-training but underemphasized for the specific downstream task.

This explains several LoRA phenomena simultaneously. First, why low rank suffices: the model already "knows" the relevant directions, so adaptation only needs to adjust their relative magnitudes, not discover new geometry. Second, why since [[LoRA acts as implicit regularizer preserving base model capabilities with strong inverse linear relationship between adaptation and forgetting]], the regularization is natural — by amplifying existing directions rather than overwriting them, LoRA preserves the pre-trained structure.

The finding also connects to ICL theory. Since [[function vectors are compact single vectors encoding ICL task representations that can be transplanted across contexts and composed algebraically]], both ICL and LoRA appear to operate by selecting and amplifying existing representational directions — ICL does this implicitly through attention patterns, while LoRA does it explicitly through learned low-rank weight updates. The difference is persistence: ICL's amplification vanishes when context changes, while LoRA's is baked into the weights.

---

Source: [[lora-doc-to-lora-hypernetworks-research-2026-03-02]]

Relevant Notes:
- [[LoRA exploits low intrinsic rank of weight updates to match full fine-tuning with 10000x fewer trainable parameters]] -- the method this explains mechanistically
- [[function vectors are compact single vectors encoding ICL task representations that can be transplanted across contexts and composed algebraically]] -- the ICL parallel of direction amplification
- [[LoRA acts as implicit regularizer preserving base model capabilities with strong inverse linear relationship between adaptation and forgetting]] -- consequence of amplification vs. overwriting
- [[DoRA weight decomposition into magnitude and direction consistently outperforms standard LoRA by 1-4 points across model sizes]] -- refines the amplification by separating magnitude from direction
- [[Doc-to-LoRA transfers visual information from VLM to text-only LLM achieving 75 percent accuracy on image classification without visual training data]] -- cross-modal evidence that latent directions exist even for unseen modalities

Topics:
- [[model-adaptation]]
