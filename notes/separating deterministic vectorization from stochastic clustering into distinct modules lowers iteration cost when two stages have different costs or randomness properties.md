---
description: "when cheap/deterministic and expensive/stochastic pipeline stages are bundled, re-running one forces re-running the other — module boundaries should follow iteration cost gradients"
type: method
confidence: likely
conditions:
  - "applies when two pipeline stages differ substantially in cost, randomness, or parameter sensitivity"
meta_state: current
source: "inbox/sis-benchmark-design-2026-04-17.md"
topics:
  - "[[classification-tools]]"
---

# separating deterministic vectorization from stochastic clustering into distinct modules lowers iteration cost when two stages have different costs or randomness properties

Bundling vectorization and clustering into one "vectorize + cluster + benchmark" module looks efficient. It is not. The two stages have asymmetric properties:

- **Vectorization** is deterministic (one output per call at fixed config) and expensive per call
- **Clustering** is parameter-sensitive (many k values, random init) and cheap per run

When bundled, re-running a clustering sweep (natural iteration target) forces re-running vectorization (stable, should be cached). Worse: unit-testing the vectorizer requires clustering fixtures, and unit-testing the clusterer requires running the full vectorization pipeline.

**Why splitting is asymmetric in cost/value:** The downside of splitting is one extra module boundary and one extra data handoff. The upside is:
- Clustering sweeps run over multiple vectorizers (Oren, AMVOC) using the same clustering code
- Vectorization results cache across clustering parameter sweeps
- Unit tests isolate failures to one module
- A feature-change decision re-runs cheap clustering, not expensive vectorization

**When the principle generalizes:** Any pair of pipeline stages where one is fast/deterministic and the other is slow/stochastic (or vice versa) benefits from a boundary between them. The boundary matches the iteration cost gradient — you iterate on the cheap side without paying the expensive side's cost.

**When it doesn't apply:** Two stages with similar costs and similar randomness properties don't gain from the boundary; the extra handoff isn't paid for.

**General principle:** Module boundaries should follow iteration cost gradients, not semantic groupings. "These both do feature work" is a weaker argument for bundling than "these have different costs" is for splitting.

---

Source: [[sis-benchmark-design-2026-04-17]]

Relevant Notes:
- [[separating representation learning from discretization enables richer feature discovery]] — related principle: learned representation vs discrete token assignment is another asymmetric pair that gains from separation
- [[AMVOC 4-stage feature pipeline reduces 1280 bottleneck features through variance thresholding StandardScaler and PCA to cluster-ready dimensions]] — a multi-stage pipeline where each stage has different cost profiles
- [[decision-gate methodology requires computing free SIS baselines before committing to feature engineering]] — sibling methodology principle: both organize the SIS benchmark via cost-aware structure, gates and module boundaries respectively
- [[four-hypothesis framing organizes SIS maximization into rules plus handcrafted features plus learned features plus direct optimization]] — module separation supports the four-hypothesis benchmark by letting one clustering sweep run over multiple vectorizers (Oren, AMVOC) without re-running vectorization
- [[pre-filtering layers each address a distinct ridge-extraction failure mode so removing any one layer likely reintroduces the failure it was blocking]] — a concrete deterministic stage whose stable outputs benefit from caching across stochastic clustering sweeps

Topics:
- [[classification-tools]]
