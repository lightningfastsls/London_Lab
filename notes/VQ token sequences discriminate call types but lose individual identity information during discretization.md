---
description: "Sarkar and Magimai-Doss VQ on HuBERT with K=50 — call-type classification substantially outperformed caller identification, confirming identity is lost in the quantization bottleneck"
type: finding
confidence: likely
created: 2026-03-01
meta_state: current
topics:
  - "[[unsupervised-usv-discovery]]"
---

# VQ token sequences discriminate call types but lose individual identity information during discretization

Sarkar and Magimai-Doss (arXiv 2511.10190, November 2025) found that VQ token sequences from HuBERT embeddings (K=50) show an asymmetric information loss pattern: call-type classification substantially outperformed caller identification. This means the discretization bottleneck preferentially preserves categorical structure (what kind of call) while discarding individual variation (who made it).

This asymmetry has important implications for our VQ-VAE work. Since [[post-hoc vector quantization substantially underperforms continuous representations motivating end-to-end VQ-VAE training]], the 15-39% underperformance for call types and 15-71% for caller ID quantifies what is lost. The larger gap for identity information suggests that individual signatures live in fine-grained continuous variation that discrete codes cannot capture — at least with a single codebook of 50 entries.

This motivates multi-codebook approaches (RVQ) or larger codebook sizes for applications where individual identity matters. For mouse USV analysis, if the scientific question is about syllable types (categorical), VQ may suffice; if individual identity or strain differences matter, continuous representations may be essential.

---

Source: unsupervised-clustering-bioacoustic-vocalizations-2025-research-2026-02-27 (archived to archive/inbox/)

Relevant Notes:
- [[post-hoc vector quantization substantially underperforms continuous representations motivating end-to-end VQ-VAE training]] -- this finding quantifies the underperformance (15-39% for types, 15-71% for identity)
- [[single codebook with V=50 was insufficient for complex vocalization structure in discrete token experiments]] -- consistent: K=50 is too small
- [[Sarkar and Magimai-Doss 2025 applied post-hoc VQ to frozen HuBERT embeddings for marmoset and dog vocalizations]] -- same research group, deeper findings

Topics:
- [[unsupervised-usv-discovery]]
