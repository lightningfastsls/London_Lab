---
description: "Probing studies show lower layers best for POS tagging, middle for dependency parsing, upper for semantic tasks — most clearly demonstrated in BERT, decoder-only models show less clean separation"
type: finding
confidence: likely
created: 2026-03-02
topics:
  - "[[transformer-architecture]]"
---

# stacking transformer blocks creates hierarchical abstraction from syntax in lower layers through structure in middle layers to semantics in upper layers

Stacking multiple attention+MLP blocks creates a hierarchy of processing that mirrors the classical NLP pipeline. Probing studies, particularly Tenney et al. (2019, "BERT Rediscovers the Classical NLP Pipeline") and Jawahar et al. (2019), demonstrate this gradient consistently:

**Lower layers** (1-4): Capture surface-level features and local syntax. Attention heads here tend to be positional, attending to adjacent tokens. MLPs learn basic token-level features. These layers are best for POS tagging — tasks requiring local syntactic understanding.

**Middle layers** (5-16): Capture syntactic structures and grammatical relationships — entity tracking, subject-verb agreement, coreference resolution. Attention heads specialize in dependency relations. These layers perform best on dependency parsing tasks.

**Upper layers** (17+): Capture high-level semantics, long-range dependencies, and task-specific features. Attention heads handle complex reasoning patterns. These layers are best for semantic similarity and natural language inference tasks.

Important caveat: this gradient was most clearly demonstrated in encoder models (BERT) via probing studies. Decoder-only models (GPT family) show a similar but less cleanly separable pattern, likely because causal masking changes the information flow. The specific layer assignments also vary by model size and architecture — a 12-layer and a 96-layer model will distribute these functions differently.

This directly connects to our USV work: since [[middle-layer hidden states capture mid-level concepts better than early or late layers for VQ-VAE input]], we observe a similar hierarchical pattern in spectrogram processing, where mid-level representations capture the most useful acoustic abstractions for discretization.

---

Source: transformer-architecture-icl-fundamentals-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[middle-layer hidden states capture mid-level concepts better than early or late layers for VQ-VAE input]] -- our USV observation of this hierarchical pattern
- [[transformers implement a gather-then-transform cycle where attention moves information between positions and MLP transforms it independently]] -- each block in the stack performs this cycle
- [[comparing VQ-VAE across transformer layers reveals which abstraction level yields the most interpretable codebook]] -- testing this hierarchy empirically

Topics:
- [[transformer-architecture]]
