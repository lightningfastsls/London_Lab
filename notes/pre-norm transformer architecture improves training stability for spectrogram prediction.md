---
description: Pre-norm (LayerNorm before attention/FFN) prevents gradient explosion in deep models, chosen over post-norm for 8-block 512-dim spectrogram prediction.
type: decision
confidence: proven
topics:
  - "[[classification]]"
---

# pre-norm transformer architecture improves training stability for spectrogram prediction

Pre-norm (LayerNorm before attention/FFN followed by residual addition) was chosen over post-norm (LayerNorm after residual addition) for training stability at the ~25-30M parameter scale of this model. In post-norm architectures, gradients must flow through the LayerNorm on the residual path, which can cause gradient explosion in deeper models. Pre-norm keeps the residual stream clean — gradients flow directly through the skip connections without passing through normalization — which dramatically stabilizes training for models with 8 or more transformer blocks.

The concrete architecture: 8 transformer blocks, each structured as LayerNorm→MultiheadAttention(dim=512, 8 heads)+residual, followed by LayerNorm→FFN(512→2048→512)+residual. The input projection Linear(170→512)→GELU→LayerNorm maps from the 170-bin frequency dimension to the model's internal representation space. The output head LayerNorm→Linear(512→170) maps back to frequency space for next-column prediction. The 4× expansion ratio in the FFN (512→2048) is standard for transformers and provides sufficient capacity for learning frequency-bin interactions.

The choice connects to [[transformer-first then VQ-VAE avoids forcing premature discretization]]: a stable training regime is a prerequisite for learning rich representations that can later be discretized meaningfully. Training stability is verified starting from Stage A of [[staged transformer training catches issues early by incrementally scaling from one bout to full dataset]], where loss monotonicity and absence of NaN/Inf confirm the architecture is well-conditioned. This is particularly important given [[HPC dependency for transformer training versus local-only development capability]], where debugging stability issues remotely on HPC hardware is far more costly than catching them locally. An unstable model might converge to degenerate solutions that would produce low-quality VQ-VAE codebooks. The architecture also informs decisions about model capacity relative to dataset size — compare [[three convolutional blocks with global average pooling suffice for USV classification on small datasets]], where the simpler architecture was chosen specifically because the labeled dataset is small. The transformer operates on unlabeled data and can afford more capacity.

---

Source: [[ROADMAP.md]], Phase 8
