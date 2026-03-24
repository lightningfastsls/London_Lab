---
description: "Aghajanyan et al 2021 — RoBERTa achieves 90% of full-parameter performance on MRPC with only 200 randomly-projected trainable params, explaining why LoRA works with tiny rank"
type: finding
confidence: proven
created: 2026-03-02
topics:
  - "[[model-adaptation]]"
---

# pre-trained language models have low intrinsic dimension with larger models having even lower intrinsic dimension after pre-training

Aghajanyan et al. (2021) demonstrated that pre-trained language models reside in remarkably low-dimensional subspaces. By optimizing only 200 trainable parameters randomly projected back into the full parameter space, RoBERTa achieves 90% of full-parameter performance on MRPC. This "intrinsic dimension" is not fixed — it *decreases* with model scale, which partly explains why larger models are so effective for downstream tasks despite having vastly more parameters.

The counterintuitive implication: larger models are *easier* to adapt, not harder. Their pre-training compresses the solution landscape into a smaller effective subspace. This is the theoretical foundation for LoRA — since [[LoRA exploits low intrinsic rank of weight updates to match full fine-tuning with 10000x fewer trainable parameters]], the low intrinsic dimension means the update space is genuinely low-rank, not just approximately so.

The finding also connects to the observation that since [[LoRA adaptation amplifies existing underemphasized directions in pre-trained weights rather than learning entirely new features]], pre-training already discovers the relevant feature directions. Fine-tuning's job is merely to adjust their relative emphasis, which requires far fewer degrees of freedom than learning features from scratch.

---

Source: lora-doc-to-lora-hypernetworks-research-2026-03-02

Relevant Notes:
- [[LoRA exploits low intrinsic rank of weight updates to match full fine-tuning with 10000x fewer trainable parameters]] -- the practical method that exploits this finding
- [[LoRA adaptation amplifies existing underemphasized directions in pre-trained weights rather than learning entirely new features]] -- the mechanistic explanation
- [[dataset quality exceeds quantity for LoRA fine-tuning as curated 1K LIMA matches 50K Alpaca performance]] -- low intrinsic dimension explains why small curated datasets suffice
- [[ICL is mathematically equivalent to query-dependent low-rank weight modifications of the MLP where each context token contributes a rank-1 update]] -- ICL also exploits this low-dimensional structure implicitly

Topics:
- [[model-adaptation]]
