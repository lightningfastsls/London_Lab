---
description: Causal masking aligns with the research question by preventing attention to future positions, mirroring how USV vocalizations unfold sequentially.
type: decision
confidence: proven
topics:
  - "[[representation-learning]]"
---

# causal attention in autoregressive transformer matches the scientific question of predicting what comes next in USV streams

Causal masking (upper triangular mask preventing attention to future positions) is not merely a training trick but a principled design choice aligned with the core research question: "given what came before, what comes next?" The transformer sees spectrogram columns sequentially and predicts the next column, mirroring the temporal unfolding of USV vocalizations. Each position can only attend to itself and prior positions, enforcing a strict temporal ordering that reflects biological reality.

This design choice matters beyond implementation convenience. Mouse vocalizations occur sequentially and are influenced by prior acoustic context — a rising syllable constrains likely continuations differently than a flat-frequency call. By building this constraint into the architecture, the model is forced to learn representations that are genuinely predictive of future acoustic content rather than representations that merely reconstruct present context. The causal mask makes the scientific question structurally unavoidable.

The alignment between model structure and scientific question also benefits interpretability. When we later apply VQ-VAE to discretize hidden states, the resulting codes will represent "what this context predicts will come next" — a temporally grounded notion of acoustic concept. This connects directly to [[transformer-first then VQ-VAE avoids forcing premature discretization]], where the transformer's learned representations provide richer substrate for discretization than hand-crafted features. The temporal grounding is also why [[bout-level spectrograms preserve inter-USV timing context for transformer training]] — without bout-level continuity, the causal context would be severed at arbitrary boundaries.

---

Source: [ROADMAP](../ROADMAP.md), Phase 8
