# Design Rationale: USV Labeling SIS Benchmark (Phase 17)

**Date:** 2026-04-17
**Scope:** Design reasoning behind `ROADMAP_SIS_BENCHMARK.md` — a multi-method benchmark to determine which USV labeling scheme achieves the highest Syntax Information Score (SIS) on our 5970 dataset (7,518 calls, animal usv_lmt_034).
**Provenance:** Captures in-session design discussion (Opus 4.7) between user and Claude Code after the three-paper deep-read ingestion (Stoumpou 2022 AMVOC, Hertz 2020 SIS, Oren 2024 ridge vectorization). This file exists because the ROADMAP captures *what to build* but loses the *why these four approaches were chosen*.

---

## 1. The starting mistake and course-correction

**Initial proposal (rejected):** Implement only Oren et al. 2024's 80D ridge vectorization, cluster, compute SIS, compare against existing 0.093 bits.

**User pushback:** "Why 80D without pre-filtering? You went from Opus 4.6 to 4.7 — think harder. And I specifically wanted all three papers ingested *before* committing to a plan. I'm not sure Didi's (Omer's) method is best."

**What this exposed:** Pattern-matching on the most recently discussed paper rather than considering the goal (maximize SIS on our sequences) and asking which method each paper best supports. The three papers were ingested deliberately to frame the planning decision, not as background color.

**Course correction:** Restructure the plan around the GOAL (SIS maximization) rather than around one paper's technique. Every paper contributes a distinct hypothesis about what makes labels sequentially informative:

- **Hertz 2020 (London lab)** — two contributions: iMSA (rule-based pitch-jump classifier, the published top-SIS scorer for mouse USVs at 0.22 bits) and SIM (iterative SIS optimization on top of any existing labeling)
- **Oren 2024 (Omer lab)** — 80D FM+AM ridge vectorization (a handcrafted feature approach)
- **Stoumpou 2022 (AMVOC)** — convolutional autoencoder bottleneck + PCA (a learned feature approach)

---

## 2. The four-hypothesis framing

Maximizing SIS for our mouse USVs has four distinct mechanisms, each testable:

| Hypothesis | Mechanism | Paper | Status in prior art |
|-----------|-----------|-------|---------------------|
| Rule-based pitch-jump detection captures the sequence-informative structure | Detect pitch discontinuities, classify by trajectory shape | Hertz 2020 iMSA | **Highest published SIS: 0.22 bits on mouse USVs** |
| Continuous FM+AM ridge shape captures what matters | Extract ridge trajectory, resample, cluster | Oren 2024 | Not benchmarked on SIS (used for identity AUC) |
| Learned autoencoder features reveal the concepts | Train AE on spectrograms, bottleneck + PCA, cluster | Stoumpou 2022 AMVOC | Scored 37% higher on human evaluation but not on SIS |
| Direct SIS optimization dominates feature engineering | Start from any labeling, iteratively reassign to maximize SIS | Hertz 2020 SIM | **Matched or exceeded iMSA when started from iMUPET** |

Each corresponds to a distinct claim: sequential structure lives *in the rules* (iMSA), *in the pitch contour shape* (Oren), *in reconstruction-relevant features* (AMVOC), or *is discoverable by direct optimization* (SIM). These are complementary, not competing — we don't know which one is right for wild mouse dyadic recordings.

## 3. Why iMSA deserves priority equal to (or above) Oren

- iMSA is **published to work on mouse USVs** — Hertz 2020 reported 0.22 bits. Oren 2024 worked on marmoset phee calls (1-2 s, 6-9 kHz band) and never reported SIS.
- iMSA is from **Mickey London's own lab** — continuity of methodology matters for this project.
- iMSA is **rule-based** — no training, no hyperparameters to sweep beyond pitch-jump threshold. Fast to implement, fast to test.
- The vault note already flags that "iMSA's pitch-jump rules *implicitly perform ridge extraction*" — so once we have the ridge tracker (17.3), iMSA is a short step away.

Oren's method is worth implementing as a *feature-engineering hypothesis* test, but it is not the default top contender for mouse USV labeling SIS.

## 4. Why the AMVOC autoencoder belongs in the plan

Initial proposal dropped AMVOC with the argument: "Our manifold is low-dimensional (Goffinet 2021 ≤2 clusters, our HDBSCAN 3 clusters), so high-dim autoencoder features won't help."

**User correction:** "Autoencoders are one of the best forms of analysis. If I take the bottleneck layer and run PCA on it, I should get concepts."

**Why the user was right:**
- An autoencoder's bottleneck is forced to preserve reconstruction-relevant information. The 1280D bottleneck *discovers* which axes of variation matter for spectrogram reconstruction.
- PCA on the bottleneck finds the *dominant* axes among those — by construction, these are the concepts the model found the data has.
- Clustering in PCA-of-bottleneck space clusters on **learned concepts**, not on raw signal statistics.
- Low intrinsic dimensionality doesn't argue against autoencoders — it argues *for* them, because bottleneck compression is how you *find* low-dim structure. Goffinet 2021 used a VAE (same class of method) precisely to establish the low-dim manifold finding. The autoencoder IS the tool for this question, not the wrong tool.

## 5. Why SIM (direct SIS optimization) could dominate

SIM is structurally different from the other three approaches: it doesn't produce new features or new rules — it takes any existing labeling and iteratively perturbs labels to maximize SIS.

This means if our data's sequential structure is real but "hidden" behind arbitrary cluster boundaries, SIM can find it **regardless** of what features produced the initial clustering. Hertz showed SIM on iMUPET (0.13 bits initially) matched iMSA's 0.22 bits — the features didn't matter, the label refinement did.

**Implication:** If SIM on any starting labeling outperforms all of iMSA + Oren-kmeans + AMVOC-kmeans, the finding is: **"labels matter more than features for sequential USV prediction."** That's a meaningful scientific result — it means the difficulty isn't representation, it's label-space search.

## 6. The decision-gate methodology

Phase 17 is structured as a gated sequence, not a pipeline:

**17.1 (free SIS baselines on existing labels) runs first** and outputs a number. If all three existing labelings show MI < 0.05 bits at lag 1, the entire phase may be ill-conceived — the sequential structure may be intrinsically weak and no feature engineering or optimization will save it. The whole $20+ hour build is skipped in favor of reconsidering.

**Why this matters:** Engineers (and LLMs) reflexively want to start coding. But 1 hour of free baseline computation can determine whether 20 hours of feature engineering is worth doing. The decision gate is cheap insurance against sunk-cost commitment to a specific hypothesis before checking the ceiling.

## 7. Domain adaptation: marmoset phee ≠ mouse USV

Critical mismatches that affect every ported technique:

| Property | Marmoset phee (Oren 2024) | Mouse USV | Adjustment needed |
|----------|---------------------------|-----------|-------------------|
| Duration | 1-2 seconds | 10-100 ms | 40-step resampling may over-/under-sample our calls; sweep [20, 30, 40, 60] |
| Frequency band | 6-9 kHz | 30-110 kHz | Different band masking; different SNR profile |
| Harmonic prevalence | Rare | ~30% of calls | Ridge tracker MUST prevent harmonic jumping (DP, not argmax) |
| SNR profile | Controlled recordings | Wild-mouse dyadic cages (noise, scratching) | Pre-filtering (noise floor, median filter) mandatory |
| Absolute-pitch relevance | Low (per-caller normalization OK) | HIGH (50 kHz vs 90 kHz calls are distinct types) | Keep raw AND normalized versions; let clustering decide |

**Conclusion:** Porting Oren's method to mouse USVs requires *re-engineering* the pre-processing and possibly the feature set — not just parameter tuning. The initial "implement Oren's 80D" proposal glossed over this.

## 8. Pre-filtering rationale: why not just argmax the ridge

Naive argmax per column on unfiltered spectrograms produces ridges that latch onto:

1. **Silent columns** in fragmented calls → random-frequency ridges
2. **Harmonics** (30% of mouse USVs have them) → ridge jumps between fundamental and 2nd harmonic mid-call
3. **Broadband noise transients** (cage clicks, scratching) → ridge spikes to outlier frequencies
4. **Low-amplitude onset/offset columns** → unreliable ridge at call edges

**Defenses, layered:**

- **Amplitude threshold by local noise floor** (mask bins below 3× rolling-median magnitude per column) — removes silent-column noise
- **3×3 median filter** — removes isolated broadband transients without blurring smooth trajectories
- **Frequency band mask** (25-120 kHz for mouse USVs) — removes sub-USV equipment hum and super-USV sampling artifacts
- **Dynamic-programming ridge tracking** (Viterbi with transition penalty) — continuity constraint that prevents harmonic jumps

Each defense addresses a distinct failure mode. Removing any of them likely reintroduces the failure it was blocking.

## 9. Why per-caller normalization may be wrong for mouse USVs

Oren 2024 rescales AM and FM per-caller to [0,1]. This is appropriate for marmoset identity classification because they want callers to be comparable.

For mouse USV *type* classification, absolute pitch matters:
- A 50 kHz call and a 90 kHz call are different syllable types, not different callers of the same call type
- Scattoni's taxonomy uses absolute frequency ranges to distinguish Flat, Up, Chevron types
- Normalizing away absolute values throws out information that existing handcrafted features explicitly use

**Resolution:** The vectorization module (17.5) outputs *both* raw and normalized versions. Downstream clustering (17.7) can choose, or the benchmark (17.9) can compare both.

## 10. Duration as an explicit scalar feature

Time-resampling to a fixed number of steps (the core of Oren's method) preserves the full trajectory shape but **discards absolute duration**. This is the cost of the fixed-dim vectorization.

For mouse USVs, duration is a primary distinguishing feature (Short vs Long calls in Scattoni's taxonomy, different temporal contexts in Hertz's ISI analysis).

**Resolution:** After the 2 × n_steps trajectory features, append duration as a scalar 81st/121st feature. Let the clustering discover whether duration-based structure matters.

## 11. Why clustering is separate from vectorization (module split)

Originally, 17.5 was one module doing "vectorize + cluster + benchmark." The user's feedback: "I think there is a benefit in splitting that there is no equivalent negative in not splitting."

**Why splitting is correct:**
- Vectorization is deterministic (one output per call at fixed config)
- Clustering is parameter-sensitive (many k values, random init)
- Splitting lets the clustering sweep run over *both* Oren (17.5) and AMVOC (17.6) outputs using the same 17.7 code
- Re-running the clustering sweep after a feature change is cheap; re-running vectorization is expensive
- Split modules are independently testable (unit tests on vectorization don't need clustering fixtures)

**General principle:** When two stages have different costs (fast vs slow) or different randomness (deterministic vs stochastic), separating them into distinct modules lowers iteration cost and increases clarity of the comparison.

## 12. Tier assignments for DSP review

17.2 (pre-filter), 17.3 (ridge tracker), 17.4 (iMSA), 17.6 (autoencoder) are all Tier 3 — DSP/ML review required.

**Why:** These modules have subtle failure modes that only manifest on real recordings and only affect specific call types:
- Pre-filter misses some noise bands → autoencoder wastes capacity on noise
- Ridge tracker misses continuity constraint → harmonic-jumping calls get mis-vectorized
- iMSA wrong pitch-jump threshold → Complex class absorbs everything or nothing
- Autoencoder wrong input shape → capacity mismatch, poor reconstruction

A Tier 2 review (standard code review) misses these because the code *runs* and tests *pass* on synthetic inputs. Only empirical evaluation on real call subsets, or DSP-domain review, catches them. This is the same reason 17.8 (SIM optimization) is Tier 2 — it's math, but not signal processing math.

## 13. Effort budget awareness

Total estimated effort for Phase 17: ~20-26 hours across 9 modules. This is substantial, but:

- 17.1 alone (1 hour) may determine whether the rest is worth doing
- 17.2 + 17.3 (3-4 hours) is shared infrastructure used by 17.4, 17.5, 17.6 — the investment amortizes
- 17.4 (iMSA) is the highest expected-reward module per hour (published top scorer + simple rules + target-to-beat)
- 17.6 (AMVOC) is the highest cost per module (4-6 hours for training + integration) but tests the most distinct hypothesis (learned features)

If 17.4 or 17.8 produces a clear winner (>0.15 bits), the remaining modules may be curtailed. The Phase Gate explicitly supports this — we decide after seeing numbers, not before.

## 14. What this framework can and cannot tell us

**It CAN answer:**
- Which of four hypothesis classes best captures sequential USV structure on our data
- Whether any method beats Hertz's published 0.22 bits on a different (wild-mouse dyadic) population
- Whether labels or features are the bottleneck (via SIM results)
- Whether the DeepSqueak-27 k-means solution is sequentially superior to the 7-type Scattoni taxonomy

**It CANNOT answer:**
- Whether USV sequences carry "language-like" structure at depths >1 (our N=7,518 is too small for depth 2+ per Hertz's <10% zero-prob rule)
- Whether the winning labeling *means* anything behaviorally (need Phase B cross-dyad + LMT integration)
- Whether these findings transfer to 3452/9252 animals (Phase B explicit target)

The benchmark is descriptive, not explanatory. It ranks methods on one axis (depth-1 SIS). Explanation requires the follow-on Phase B work.

---

## Source papers

1. Hertz, S., Weiner, B., Perets, N. & London, M. (2020). *Temporal Structure of Mouse Courtship Vocalizations Facilitates Syllable Labeling.* Communications Biology, 3, 333.
2. Oren, G., Shapira, A., Lifshitz, R., Vinepinsky, E., Cohen, R., Fried, T., Hadad, G.P., & Omer, D. (2024). *Vocal labeling of others by nonhuman primates.* Science, 385(6712), 996-1003.
3. Stoumpou, V., Jarvis, E., Theodoridis, T. (2022). *AMVOC: Analysis of Mouse Vocal Communication.* Bioacoustics, 32(2), 199-229.
4. Goffinet, J. et al. (2021). *Low-dimensional learned feature spaces quantify individual and group differences in vocal repertoires.* eLife.

## Cross-references

- Parent ROADMAP: `ROADMAP_SIS_BENCHMARK.md`
- Paper ingestion handoff: `docs/handoffs/three-paper-deep-reads-2026-04-15.md`
- Prior analysis (0.093 bits baseline): `results/sequential_structure/sequential_structure_summary.csv`
- User goal context: `ops/goals.md` (USV Analysis Stage, Phase A2 DONE, Phase B upcoming)
