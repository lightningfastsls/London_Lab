---
description: "General pattern -- permissive first stage maximizes recall, precise second stage filters for precision, each optimized independently"
type: pattern
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[detection]]"
---

# two-stage coarse-to-fine filtering is effective for imbalanced detection tasks

This project's USV detection pipeline instantiates a general pattern applicable to many imbalanced detection problems: a permissive first stage (energy detector) maximizes recall by casting a wide net, followed by a precise second stage (CNN classifier) that filters for precision. The pattern's strength is separation of concerns -- each stage can be independently optimized for its role without compromising the other. The first stage can use simple, fast heuristics (energy thresholds) while the second stage uses a more expensive but discriminative model (CNN). This is particularly effective when the target signal (USVs) is rare relative to noise, since [[recall versus precision tradeoff in two-stage USV detection]] shows how the stages complement each other. The pattern appears in many domains: cascaded classifiers in face detection (Viola-Jones), two-pass alignment in genomics, and candidate generation + reranking in information retrieval. The project's own representation learning pipeline embodies the same principle at a deeper level: since [[separating representation learning from discretization enables richer feature discovery]], the transformer first learns freely (coarse, continuous), then the VQ-VAE applies discrete structure (fine, discrete) — sequential stages optimized independently, just as in detection.

---

Source:
- DECISIONS.md (ADR-003) -- USV Spectrogram project architecture decisions

Relevant Notes:
- [[two-stage detection uses permissive energy detector followed by CNN precision filter]] -- the specific instantiation
- [[recall versus precision tradeoff in two-stage USV detection]] -- the designed tradeoff
- [[energy threshold at negative 60 dB is deliberately low to maximize recall in the first stage]] -- first stage configuration
- [[separating representation learning from discretization enables richer feature discovery]] -- the coarse-to-fine pattern recurs in representation learning: transformer first (coarse/continuous), then VQ-VAE (fine/discrete)
- [[post-hoc vector quantization substantially underperforms continuous representations motivating end-to-end VQ-VAE training]] -- a caveat to staged approaches: post-hoc discretization loses 14 percentage points vs continuous (Sarkar 2025), motivating tighter integration between stages
- [[error amplification near targets is a general instability pattern in iterative refinement systems beyond diffusion models]] -- the general instability risk: when the second stage amplifies errors from the first stage, the pipeline has unbounded gain; our CNN second stage is a classifier (bounded confidence output) which avoids this
- [[bounded gain in iterative refinement prevents error amplification while unbounded gain creates structural instability regardless of domain]] -- the abstracted design principle: our CNN classifier's bounded output (0-1 confidence) ensures the two-stage pipeline has bounded gain, unlike unbounded parameterizations in diffusion models

Topics:
- [[detection]]
