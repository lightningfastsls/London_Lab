---
description: "Two-phase architecture: autoregressive transformer (~25-30M params) learns freely, then frozen hidden states are discretized by VQ-VAE"
type: decision
confidence: experimental
conditions: []
meta_state: current
topics:
  - "[[representation-learning]]"
---

# Transformer-first then VQ-VAE avoids forcing premature discretization

To discover language-like structure in USV repertoires, the project uses a two-phase architecture. Phase 1 trains an autoregressive transformer (d_model=512, 8 heads, 8 layers, ~25-30M params) that receives raw spectrogram columns (170-dim) and predicts the next column. The transformer develops internal representations freely without any discretization bottleneck. Phase 2 freezes the transformer and applies a VQ-VAE to hidden states from a middle layer (default layer 4 of 8), compressing continuous representations into a small discrete codebook (K=64). This separation is the key architectural insight: since [[separating representation learning from discretization enables richer feature discovery]]. End-to-end training would force discretization before the model knows what matters. VQ-VAE-first (DALL-E style) would only capture local spectral patterns, missing the contextual representations where "concepts" live. See also [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] and [[middle-layer hidden states capture mid-level concepts better than early or late layers for VQ-VAE input]].

### ROADMAP Context

ROADMAP specifies the v2 architecture in detail: Input projection is Linear(170→512)→GELU→LayerNorm, followed by learned positional embeddings via nn.Embedding(max_seq_len, d_model). The 8 transformer blocks use pre-norm (LayerNorm before attention and FFN) for training stability — see [[pre-norm transformer architecture improves training stability for spectrogram prediction]]. The output head is LayerNorm→Linear(512→170). Total parameter count is ~25-30M. The model uses causal attention masks matching the autoregressive objective — see [[causal attention in autoregressive transformer matches the scientific question of predicting what comes next in USV streams]]. return_hidden_states=False by default to save memory during Phase 1 training; set to True only during Phase 2 hidden-state extraction.

---

Source:
- DECISIONS.md (ADR-007) -- USV Spectrogram project architecture decisions

Relevant Notes:
- [[separating representation learning from discretization enables richer feature discovery]] -- the general principle
- [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] -- codebook sizing
- [[middle-layer hidden states capture mid-level concepts better than early or late layers for VQ-VAE input]] -- layer selection
- [[bout-level spectrograms preserve inter-USV timing context for transformer training]] -- the input data format
- [[post-hoc vector quantization substantially underperforms continuous representations motivating end-to-end VQ-VAE training]] -- empirical validation: post-hoc VQ (35% UAR) vs continuous features (49% UAR) shows the cost of the post-hoc approach our architecture avoids
- [[end-to-end VQ-VAE on animal vocalizations remains an open research gap as of February 2026]] -- this two-phase architecture fills an open gap in the field
- [[LoRA exploits low intrinsic rank of weight updates to match full fine-tuning with 10000x fewer trainable parameters]] -- if the 25-30M param transformer is pre-trained on general audio, LoRA could efficiently adapt it for USV spectrogram prediction without full retraining, reducing the HPC dependency for Phase 1
- [[speech pretrained SSL models transfer well to animal vocalizations with only marginal benefit from bioacoustic pretraining]] -- validates the bootstrap strategy: a speech-pretrained transformer backbone adapted via LoRA could replace training from scratch

Topics:
- [[representation-learning]]
