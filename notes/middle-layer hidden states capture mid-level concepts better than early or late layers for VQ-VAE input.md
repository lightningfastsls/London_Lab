---
description: "Layer 4 of 8 as default VQ-VAE extraction point -- mid-level concepts, not raw spectral features or prediction-specialized features"
type: finding
confidence: experimental
conditions: []
meta_state: current
topics:
  - "[[representation-learning]]"
---

# Middle-layer hidden states capture mid-level concepts better than early or late layers for VQ-VAE input

The VQ-VAE in Phase 2 operates on hidden states extracted from the frozen transformer. The default extraction point is layer 4 of 8, chosen because middle layers are expected to encode mid-level concepts -- neither the raw spectral features captured by early layers nor the highly prediction-specialized representations in late layers. The hypothesis is that mid-level representations contain the most interpretable "concept-level" information: patterns that are abstract enough to generalize across instances but concrete enough to map to acoustic features. This needs empirical validation by comparing VQ-VAE performance and codebook interpretability across layers 2, 4, 6, and 8. The encoder architecture is Conv1d(512->256, k=5) -> GELU -> Linear(256->64) -> L2-norm, projecting hidden states into the codebook dimension.

### ROADMAP Context

ROADMAP specifies the multi-layer comparison protocol: train an identical VQ-VAE on hidden states from layers 2, 4, 6, and 8, then compare results across three metrics — perplexity (emphasis here as the primary interpretability signal), codebook utilization, and reconstruction loss. Storage estimate is ~1 GB per layer for 500K frames × 512 floats × 4 bytes. Outputs are saved as memory-mapped numpy arrays (.npy), with a companion metadata JSON mapping each frame index to (bout_id, chunk_id, frame_within_chunk, timestamp) for traceability. See [[comparing VQ-VAE across transformer layers reveals which abstraction level yields the most interpretable codebook]].

---

Source:
- DECISIONS.md (ADR-007) -- USV Spectrogram project architecture decisions

Relevant Notes:
- [[transformer-first then VQ-VAE avoids forcing premature discretization]] -- the parent architecture
- [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] -- codebook dimension matches this projection
- [[codebook collapse prevention requires simultaneous EMA updates plus dead code reset plus k-means init plus L2 normalization]] -- training stability for this extraction
- [[LoRA adaptation amplifies existing underemphasized directions in pre-trained weights rather than learning entirely new features]] -- layer selection for VQ-VAE extraction parallels LoRA's finding that task-relevant directions already exist in pre-trained weights; mid-layer hidden states may already contain the concept-level directions LoRA would amplify
- [[stacking transformer blocks creates hierarchical abstraction from syntax in lower layers through structure in middle layers to semantics in upper layers]] -- the probing evidence that motivates why layer 4 of 8 captures mid-level structure

Topics:
- [[representation-learning]]
