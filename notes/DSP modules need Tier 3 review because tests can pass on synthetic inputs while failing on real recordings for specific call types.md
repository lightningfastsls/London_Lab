---
description: "synthetic-input tests miss failure modes that only appear on specific call types in real recordings — standard code review catches syntax, only DSP-domain empirical review catches these"
type: method
confidence: likely
conditions:
  - "applies when DSP code processes natural recordings with domain-specific edge cases not captured in synthetic test fixtures"
meta_state: current
source: "inbox/sis-benchmark-design-2026-04-17.md"
topics:
  - "[[classification-methodology]]"
---

# DSP modules need Tier 3 review because tests can pass on synthetic inputs while failing on real recordings for specific call types

Standard code review (Tier 2) checks that code runs, handles types correctly, and passes unit tests. For DSP code, this is insufficient — the module can pass all Tier 2 checks while silently failing on specific call categories in real recordings.

**The pattern:** Synthetic test inputs (generated sine sweeps, clean spectrograms, uniform noise) exercise the code path but not the domain edge cases. Real recordings contain failure modes that synthetic data cannot replicate:

- Pre-filter threshold misses some noise bands → autoencoder wastes capacity learning noise
- Ridge tracker lacks continuity constraint → harmonic-jumping calls get mis-vectorized
- iMSA pitch-jump threshold wrong → Complex class absorbs everything, or nothing
- Autoencoder input shape mismatched to call distribution → reconstruction poor on tails

Each failure passes Tier 2: the test inputs don't trigger it. Only Tier 3 — empirical evaluation on real call subsets or DSP-domain review — catches them.

**Why this is specific to DSP:** Pure math modules (e.g., SIS computation) can be verified by first principles plus unit tests. Signal processing has distribution-dependent failure modes: the code is correct on the test distribution and wrong on the deployment distribution. Neither reviewer nor tests catch the gap; only empirical evaluation does.

**Implication for development process:** Modules that touch spectrograms, ridges, or acoustic features should be reviewed against real recordings *before* merge — not after a downstream module reports unexplained poor performance. The review cost is amortized by the debugging cost it prevents.

---

Source: [[sis-benchmark-design-2026-04-17]]

Relevant Notes:
- [[layered review depth tiering mirrors human review practice by matching investment to complexity]] — the review-tier framework this applies within
- [[adversarial builder-critic separation catches silent performance risks that pass all tests]] — related pattern: tests alone don't catch distribution-dependent failures
- [[pre-filtering layers each address a distinct ridge-extraction failure mode so removing any one layer likely reintroduces the failure it was blocking]] — concrete DSP example: each layer's value is verifiable only by Tier 3 ablation on real recordings, not by synthetic tests
- [[Oren marmoset ridge vectorization requires re-engineering not parameter tuning when adapted to mouse USVs because duration frequency band harmonics SNR and absolute-pitch relevance all differ]] — the silent assumptions listed in the species-port table are exactly what Tier 3 review surfaces before merge

Topics:
- [[classification-methodology]]
