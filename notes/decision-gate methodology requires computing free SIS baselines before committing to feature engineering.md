---
description: "cheap baseline computation determines whether 20+ hours of feature engineering can move the ceiling — insurance against sunk-cost commitment to untestable hypotheses"
type: method
confidence: likely
conditions:
  - "applies when proposed pipeline is substantially more expensive than the free-baseline diagnostic"
meta_state: current
source: "inbox/sis-benchmark-design-2026-04-17.md"
topics:
  - "[[classification-methodology]]"
---

# decision-gate methodology requires computing free SIS baselines before committing to feature engineering

Before any multi-day pipeline build, compute baselines that cost almost nothing and can determine whether the pipeline *can possibly* win. For SIS on our 5970 dataset, that baseline is computing depth-1 SIS on existing labelings (DeepSqueak k=27, HDBSCAN, Scattoni 7-type) before building any new features.

**Why this matters:** If all three existing labelings show MI < 0.05 bits at lag 1, the sequential structure is intrinsically weak. No feature engineering or optimization can create signal that isn't there. The entire 20+ hour build would be ill-conceived. One hour of free baseline computation buys the right to proceed — or the information needed to stop.

The failure mode this guards against is engineers (and LLMs) reflexively coding before checking. Once a plan is underway, sunk-cost bias defends it against evidence. Gate-first structure externalizes the stop criterion: "if the number is X, we don't build."

**This is not conservatism** — it is risk-adjusted prioritization. Cheap diagnostics that can redirect expensive work are net-positive even when they almost always pass. The ceiling question ("can the sequential structure exceed Y bits at all?") is answerable from raw sequences alone, independent of any labeling method.

**General principle:** When a proposed pipeline and a decision-relevant baseline differ in cost by 20x or more, the baseline runs first. Not because the pipeline is likely wrong, but because the pipeline's *value* depends on a quantity the baseline measures.

---

Source: [[sis-benchmark-design-2026-04-17]]

Relevant Notes:
- [[Hertz et al 2020 Syntax Information Score ranks classification schemes by how well syllable labels predict next syllable]] — SIS is the metric the baseline computes
- [[Syntax Information Maximization SIM algorithm iteratively perturbs cluster centroids to maximize SIS on training sequences]] — SIM is the expensive pipeline this gate can skip
- [[four-hypothesis framing organizes SIS maximization into rules plus handcrafted features plus learned features plus direct optimization]] — what the gate decides whether to build; the four hypotheses only matter if the baseline ceiling is non-trivial
- [[separating deterministic vectorization from stochastic clustering into distinct modules lowers iteration cost when two stages have different costs or randomness properties]] — sibling methodology principle: both organize the SIS benchmark via cost-aware structure, gates and module boundaries respectively

Topics:
- [[classification-methodology]]
