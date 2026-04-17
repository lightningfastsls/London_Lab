---
_schema:
  entity_type: "source-capture"
  applies_to: "inbox/*.md"
description: "Methodology claims about SIS benchmark decision-gate framing, four-hypothesis structure, feature vs label questions, and DSP review tier — captured from session reasoning on 2026-04-17"
source_type: conversation
date_accessed: 2026-04-17
status: unprocessed
topics: "[[classification-methodology]]"
---

# SIS Benchmark Decision-Gate Methodology

Session reasoning captured during SIS benchmark roadmap design. Nine distinct methodology claims that should be extracted as atomic notes and linked into the classification and classification-methodology topic maps.

## Key Points

1. **Decision-gate methodology** — compute free SIS baselines before committing to feature engineering budget. The cheap baselines (rules, k-means on simple features) tell you whether expensive feature engineering will pay off at all.

2. **Four-hypothesis framing** — SIS maximization space decomposes into: (a) rule-based labels, (b) handcrafted features + clustering, (c) learned features + clustering, (d) direct SIS optimization (SIM). This framing organizes the benchmark's experimental arms.

3. **Autoencoder bottleneck + PCA extracts concepts** — reconstruction loss forces the model to preserve axes of variation that matter, and PCA on the bottleneck isolates those axes. The combination is a concept extractor, not just a compressor.

4. **Low-dimensional intrinsic manifold argues FOR learned features** — not against them, because bottleneck compression is precisely how you discover low-dim structure. The intuition "USVs live on a low-dim manifold so simple features should work" gets the direction wrong.

5. **SIM optimization is structurally feature-independent** — it perturbs cluster centroids directly on whatever representation you hand it. So if SIM wins across feature sets, the finding is: labels matter more than features for sequential prediction.

6. **Oren marmoset ridge vectorization needs re-engineering, not tuning** — when adapted from marmoset to mouse USVs, duration, frequency band, harmonics structure, SNR, and absolute-pitch relevance all differ. Parameter tuning won't bridge the gap; the vectorization itself must be redesigned.

7. **Pre-filtering layers address distinct ridge-extraction failure modes** — each layer (noise gate, harmonic suppression, ridge smoothing, etc.) blocks a specific failure. Removing any one layer likely reintroduces the failure it was blocking. Ablation studies should be expected to degrade performance.

8. **Separate deterministic vectorization from stochastic clustering** — into distinct modules. Vectorization is deterministic and expensive (re-run rarely); clustering is stochastic and cheap (re-run many times with different seeds/K). Separating them lowers iteration cost.

9. **DSP modules need Tier 3 review** — tests can pass on synthetic inputs while failing on real recordings for specific call types. Synthetic-only tests create false confidence; Tier 3 review forces real-data verification.

## Raw Notes

These claims emerged from designing `ROADMAP_SIS_BENCHMARK.md` and the four-hypothesis decomposition (rules / handcrafted / learned / SIM). The decision-gate framing came from realizing that the cheap baselines (k-means on existing features, rule-based labels from pitch-jump detection) are essentially free to compute, so you should compute them first before deciding whether the autoencoder pipeline is worth building.

The "labels matter more than features" claim (#5) is the key methodological insight: SIM optimizes the label assignment directly, so if it beats handcrafted + clustering and learned + clustering with the same underlying features, you've learned that sequential prediction is bottlenecked by label quality, not feature quality.

The DSP Tier 3 review claim (#9) connects to the project-wide pattern that synthetic test inputs don't exercise the failure modes that real recordings hit — energy detector regressions, ridge extraction on noisy segments, etc.

## Processing Notes

Expected atomic notes after /reduce (one per claim, using title-as-claim convention):

- `SIS benchmark decision-gate methodology computes free baselines before committing to feature engineering budget.md`
- `Four-hypothesis framing decomposes SIS maximization into rules handcrafted features learned features and direct optimization.md`
- `Autoencoder bottleneck plus PCA extracts concepts by forcing reconstruction to preserve meaningful axes of variation.md`
- `Low-dimensional intrinsic manifold argues for learned features because bottleneck compression discovers low-dim structure.md`
- `SIM optimization is structurally feature-independent so winning across feature sets means labels matter more than features.md`
- `Oren marmoset ridge vectorization requires re-engineering not parameter tuning when adapted to mouse USVs.md`
- `Pre-filtering layers each address distinct ridge-extraction failure modes so removing any one reintroduces its failure.md`
- `Separating deterministic vectorization from stochastic clustering into distinct modules lowers iteration cost.md`
- `DSP modules require Tier 3 review because tests pass on synthetic inputs while failing on real recordings.md`

Linking targets:
- [[classification-methodology]] — claims 1, 2, 5, 6, 7, 8 (methodology for the benchmark)
- [[signal-processing]] — claims 6, 7, 9 (ridge extraction, DSP review)
- [[unsupervised-usv-discovery]] — claims 3, 4 (learned-feature approaches)

Existing vault notes to cross-link:
- `SIS equals entropy rate at depth zero minus entropy rate at depth D giving information gained from sequential context.md`
- `SIS normalized by log2 of cluster count removes dependency on number of labels enabling cross-Nc comparisons.md`
- `Syntax Information Maximization SIM algorithm iteratively perturbs cluster centroids to maximize SIS on training sequences.md`
- `iMSA rule-based pitch-jump classification produces the highest SIS among compared methods despite lower label entropy.md`
- `ridge extraction finds the dominant frequency bin with maximum energy at each time step creating a pitch contour trajectory.md`
- `Omer lab 80-dimensional FM plus AM ridge vectorization embeds each vocalization call in a fixed-length feature space.md`
- `per-caller normalization of AM and FM features to 0-1 prevents individual acoustic idiosyncrasies from dominating classification.md`
- `time-axis resampling to a fixed number of steps normalizes variable-duration vocalizations without discarding frequency information.md`
- `AMVOC convolutional autoencoder provides the best open-source Python tool for unsupervised USV feature extraction and clustering.md`
