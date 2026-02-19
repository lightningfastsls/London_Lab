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

This project's USV detection pipeline instantiates a general pattern applicable to many imbalanced detection problems: a permissive first stage (energy detector) maximizes recall by casting a wide net, followed by a precise second stage (CNN classifier) that filters for precision. The pattern's strength is separation of concerns -- each stage can be independently optimized for its role without compromising the other. The first stage can use simple, fast heuristics (energy thresholds) while the second stage uses a more expensive but discriminative model (CNN). This is particularly effective when the target signal (USVs) is rare relative to noise, since [[recall versus precision tradeoff in two-stage USV detection]] shows how the stages complement each other. The pattern appears in many domains: cascaded classifiers in face detection (Viola-Jones), two-pass alignment in genomics, and candidate generation + reranking in information retrieval.

---

Source:
- DECISIONS.md (ADR-003) -- USV Spectrogram project architecture decisions

Relevant Notes:
- [[two-stage detection uses permissive energy detector followed by CNN precision filter]] -- the specific instantiation
- [[recall versus precision tradeoff in two-stage USV detection]] -- the designed tradeoff
- [[energy threshold at negative 60 dB is deliberately low to maximize recall in the first stage]] -- first stage configuration

Topics:
- [[detection]]
