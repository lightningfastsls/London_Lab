---
description: "Integration point where information theory, probing, and LMT behavioral grounding converge to test whether domestication altered vocal communication in mice"
type: hypothesis
confidence: speculative
meta_state: current
topics:
  - "[[representation-learning]]"
  - "[[experimental-methods]]"
---

# the converging research question asks whether transformer encodes behaviorally meaningful vocal categories differing between wild and lab populations

This note captures the overarching hypothesis that connects three independent workstreams into a single testable claim: if VQ-VAE code sequences exhibit language-like sequential structure (information theory workstream), and the transformer's hidden states encode behaviorally meaningful acoustic categories (probing workstream), and those categories differ systematically between wild and lab mouse populations (LMT integration workstream), then domestication has altered vocal communication in mice — not just the acoustic properties of individual calls, but the higher-order sequential organization of vocal behavior.

Each workstream provides a necessary but insufficient piece of evidence. The information theory workstream (null model hierarchy, entropy rate analysis, excess entropy, Zipf analysis) establishes whether the sequential structure in USV code sequences is real — whether it exceeds what trivial statistical properties or simple generative models can explain. Without this foundation, any downstream claim about "vocal syntax" or "compositional structure" rests on unvalidated assumptions. But demonstrating real sequential structure does not by itself connect to biology; the structure could be an artifact of the recording apparatus, the acoustic environment, or the VQ-VAE discretization itself.

The probing workstream (layer-property heatmaps, selectivity analysis) establishes whether the transformer has learned representations that correspond to biologically meaningful acoustic properties. If probes reveal that specific codebook entries map onto specific acoustic categories — and those categories align with known USV types or behavioral correlates — then the VQ-VAE codes are not arbitrary discretization artifacts but biologically grounded symbols. This is what transforms abstract information-theoretic patterns into interpretable claims about vocal behavior.

The LMT integration workstream grounds everything in behavior by testing whether the patterns discovered in the first two workstreams differ between wild and lab populations and correlate with observable behavioral outcomes (courtship success, social proximity, mounting behavior). This is where the hypothesis that [[inbreeding and absence of courtship selection pressure in captivity caused lab mice to degrade courtship vocal competence]] becomes testable: if wild mice show richer sequential structure, more diverse codebook usage, and stronger correlations between VQ-VAE codes and courtship outcomes, domestication has measurably degraded vocal communication.

The converging evidence is stronger than any single workstream because each addresses a different potential confound. Information theory without probing could be detecting non-biological patterns. Probing without information theory could be over-interpreting random probe successes. Either without LMT integration lacks the behavioral grounding to make claims about communication rather than mere acoustic variation. Together, they form a triangulated argument that is robust to the failure modes of any individual approach.

---

Source:
- vacation-master-plan-v2 (archived to archive/inbox/)

Relevant Notes:
- [[wild versus lab mouse USV comparison tests whether domestication altered vocal repertoires]] -- the population-level comparison that the converging hypothesis ultimately serves
- [[inbreeding and absence of courtship selection pressure in captivity caused lab mice to degrade courtship vocal competence]] -- the biological mechanism that would explain observed differences between populations
- [[information theory and null model foundation must precede probing and LMT integration]] -- the ordering constraint ensuring each workstream delivers validated results before convergence

Topics:
- [[representation-learning]]
- [[experimental-methods]]
