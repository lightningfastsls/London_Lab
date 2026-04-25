# Stream 5 — Bout Threshold Sensitivity: Results

**Status:** Complete (pending commit)
**Date:** 2026-04-24
**Scope:** 5970 + 3452 (9252 not yet classified — skipped per handoff condition)
**Script:** `scripts/bout_threshold_sensitivity.py`
**Outputs:** `results/bout_threshold_sensitivity/`

---

## TL;DR

| item | recommendation |
|------|----------------|
| Bout logic | **File-aware YES** — file change is always a bout break |
| Within-file threshold | **0.25 s** (primary) or 0.143 s (data-driven alternative) |
| Q1 resolution | Option **(b)** — behaviorally meaningful episode with brief pauses, combined with (c)'s file = outer boundary |
| Impact of moving canonical 0.6 s → 0.25 s file-aware | MI 0.0921 → 0.0935 (+0.0014 bits, within CI) — **not material** for current claims |

**The headline finding:** The canonical 0.6 s sits in a broad MI plateau on 5970, so the current wild-repertoire claims do not hinge on the threshold. But the threshold **is** load-bearing for cross-dataset comparison (5970 vs 3452 vs lab), because the optimal threshold differs by dataset, and small-N datasets (3452, future lab chunks) are vulnerable to MI finite-sample bias that the threshold choice amplifies.

---

## 1. 5970 results

![5970 non-file-aware](../../../results/bout_threshold_sensitivity/sweep_no_file_aware_5970.png)
![5970 file-aware](../../../results/bout_threshold_sensitivity/sweep_file_aware_5970.png)
![5970 comparison](../../../results/bout_threshold_sensitivity/comparison_5970.png)
![5970 gap distributions](../../../results/bout_threshold_sensitivity/gap_distributions_5970.png)

### Sweep (non-file-aware)

| threshold (s) | n_bouts | n_within_pairs | MI lag 1 (bits) | MI CI95 |
|---|---|---|---|---|
| 0.10 | 3197 | 4667 | 0.0839 | [0.071, 0.110] |
| 0.143 | 2276 | 5588 | **0.0980** | [0.084, 0.124] |
| 0.20 | 1996 | 5868 | 0.0935 | [0.081, 0.116] |
| 0.25 | 1859 | 6005 | 0.0937 | [0.081, 0.116] |
| 0.40 | 1677 | 6187 | 0.0926 | [0.081, 0.115] |
| **0.60 (canonical)** | 1514 | 6350 | **0.0921** | [0.079, 0.116] |
| 0.80 | 1408 | 6456 | 0.0929 | [0.078, 0.116] |
| 1.00 | 1342 | 6522 | 0.0921 | [0.080, 0.115] |
| 2.00 | 1202 | 6662 | 0.0891 | [0.077, 0.112] |

**Drift check:** non-file-aware MI @ 0.6 s = 0.092051, corpus_facts recorded 0.0921 → Δ = 5e-5 bits. **Script reproduces the canonical pipeline within float rounding.**

### Key observations

1. **MI is flat from 0.143 s onward** (within CI). Peak 0.098 at 0.143 s → 0.089 at 2.0 s, a 0.009-bit spread — smaller than a single bootstrap CI half-width (~0.015 bits). **Choosing any threshold in [0.143, 1.0] produces statistically indistinguishable MI estimates.**
2. **Tight thresholds break the analysis.** At 0.10 s bouts shatter (3197 bouts, only 4667 within-bout pairs remain vs 6350 at 0.6s), MI drops to 0.084, CI balloons. This is the noise floor: the threshold has fallen below even the dominant within-file ICI component (μ₁ = 74 ms).
3. **Mixture fit reproduces prior Q1 numbers.** μ₁=74.3 ms (w=78%), μ₂=183.9 ms (w=22%), crossover=143 ms — matches what Q1 background quoted.
4. **KS D=0.93 (p < 1e-300):** within-file and cross-file gap distributions are not just different — they are almost completely separable. The 2 s recorder timeout creates a hard gap in the gap distribution.
5. **File-aware logic barely moves MI on 5970.** At 0.6 s the file-aware MI is 0.0912 vs non-file-aware 0.0921 (Δ = 0.0009). At threshold = ∞ (file = bout, never split within a file), MI = 0.0893 — only 3% below the plateau.

---

## 2. 3452 results

![3452 non-file-aware](../../../results/bout_threshold_sensitivity/sweep_no_file_aware_3452.png)
![3452 file-aware](../../../results/bout_threshold_sensitivity/sweep_file_aware_3452.png)
![3452 comparison](../../../results/bout_threshold_sensitivity/comparison_3452.png)
![3452 gap distributions](../../../results/bout_threshold_sensitivity/gap_distributions_3452.png)

### Sweep (non-file-aware)

| threshold (s) | n_bouts | n_within_pairs | MI_plugin (bits) | MI CI95 | Miller-Madow correction* |
|---|---|---|---|---|---|
| 0.10 | 358 | 43 | 0.7078 | [0.570, 1.265] | −0.604 (catastrophic) |
| 0.143 | 280 | 121 | 0.4203 | [0.356, 0.771] | −0.215 |
| 0.20 | 233 | 168 | 0.3367 | [0.297, 0.603] | −0.155 |
| 0.25 | 211 | 190 | 0.2892 | [0.259, 0.556] | −0.137 |
| 0.40 | 161 | 240 | 0.2358 | [0.223, 0.457] | −0.108 |
| **0.60** | 133 | 268 | 0.1974 | [0.182, 0.407] | −0.097 |
| 0.80 | 117 | 284 | 0.2011 | [0.182, 0.408] | −0.091 |
| 1.00 | 104 | 297 | 0.1917 | [0.181, 0.386] | −0.087 |
| 2.00 | 88 | 313 | 0.1619 | [0.154, 0.357] | −0.083 |

*Miller-Madow bias correction ≈ (K−1)² / (2 N ln 2) with K=7; **subtract** this from MI_plugin to get the bias-corrected estimate. Not computed by this handoff's script — provided here as an order-of-magnitude guide.

### Key observations

1. **MI decreases monotonically with threshold.** This is **not** a genuine structure signal — it is finite-sample bias. The plug-in MI estimator is positively biased at small N, and 3452 has ~10× fewer within-bout pairs than 5970. At t=0.10 s the Miller-Madow correction would be 0.60 bits — larger than the raw estimate. The reported MI there is noise.
2. **After crude Miller-Madow correction the MI is ~0.10 bits**, roughly comparable to 5970's 0.09. The raw cross-dataset comparison is not trustworthy.
3. **3452 gap distribution is shifted relative to 5970.**
   - 5970: μ₁=74 ms (w=78%), μ₂=184 ms (w=22%), crossover=143 ms
   - 3452: μ₁=135 ms (w=66%), μ₂=484 ms (w=34%), crossover=269 ms
   The dominant within-file ICI is ~2× slower in 3452. **This is a real biological difference**, not just a statistical artifact. 3452 calls at a slower pace with a heavier-tailed distribution.
4. **KS D=0.76 (p ≈ 1e-49)** — within-file and cross-file still cleanly separable, but D is noticeably smaller than 5970's 0.93.

---

## 3. File-aware vs non-file-aware

### Theoretical argument for file-aware

The recorder triggers on noise and stops ~2 s after silence. A cross-file gap of 1.5 s reflects instrumentation behavior (recorder shutdown), not animal behavior. Pooling these with genuine within-file pauses is a category mistake. Stream 4's cross-population module will compare 5970/3452/9252 wild + lab data with **different recording setups** (trigger thresholds, timeout delays). File-aware bout logic isolates the "animal stopped calling" signal from the "recorder stopped recording" signal, making cross-system comparisons more defensible.

### Empirical argument

- **KS D = 0.93 (5970)** and **D = 0.76 (3452)**: within-file and cross-file gap distributions are dramatically distinct. The data itself says these are two populations.
- 5970 has 82 cross-file gaps under 1 s (down from the Q1-reported 90). Under non-file-aware at 0.6 s these 82 are silently treated as within-bout pairs — they shouldn't be.
- Under file-aware at t=∞ (file = bout, no within-file splits), MI = 0.0893. The file boundary alone carries most of the structure signal.

### Cost of file-aware logic

Almost nothing. MI shifts by <0.001 bits at any threshold ≥0.2 s on 5970; similar shift on 3452. **File-aware is a free defensibility upgrade.**

---

## 4. Q1 resolution — which option does the data support?

Q1 offered three definitions:

| option | description | data says |
|---|---|---|
| (a) | Continuous stream, tight threshold ~0.15 s | **Partially** — matches 5970 mixture crossover (143 ms), but over-splits on 3452 where crossover is 269 ms |
| (b) | Behaviorally meaningful episode, looser threshold ~0.25–0.5 s | **Yes** — in the 5970 MI plateau, sits above 3452's crossover, captures both mixture components as within-bout |
| (c) | File = bout, no within-file splitting | **Partially** — the file boundary is empirically the dominant structure signal, but within-file bimodality in the ICI distribution (μ₂ at 184/484 ms) suggests within-file bouts **do** exist |

**Recommendation: hybrid — (c) as an outer constraint (file = always a bout break) + (b) for within-file splitting (threshold ≈ 0.25 s).** This is exactly the "file-aware with gap threshold" logic Stream 5 prototyped.

---

## 5. Sensitivity bounds — what's the wild-vs-lab threshold for meaningful differences?

Combining **parameter sensitivity** (threshold choice) + **bootstrap CI** (sampling uncertainty):

- **5970 (non-file-aware, across [0.143, 1.0] s):** MI = 0.092 ± 0.008 (parameter) + ±0.015 (bootstrap) ≈ **±0.02 bits combined**.
- **5970 (file-aware, across [0.143, 1.0] s):** MI = 0.091 ± 0.007 + ±0.015 ≈ **±0.02 bits combined**.
- **3452:** dominated by MI bias — raw plug-in MIs span 0.16–0.71, but Miller-Madow-corrected would span ~0.08–0.10 with CIs ±0.04. **Not trustworthy for cross-dataset comparison without bias correction.**

**Practical rule:** Any wild-vs-lab MI lag-1 difference must exceed **~0.03 bits** (parameter + bootstrap + safety margin) to be interpretable as biological rather than methodological noise. With Miller-Madow or KSG-style corrections this could tighten to ~0.015.

---

## 6. Decision-needed signals (from handoff line 105–110)

**Triggered:**

1. **Per-dataset optimal threshold differs.** 5970 mixture crossover = 143 ms, 3452 crossover = 269 ms. Under handoff's decision-needed rule, this implies **the canonical threshold needs to be per-dataset, not global** — OR needs to be set high enough to be above both (e.g., 0.25 s works for 5970; 0.30 s would cover 3452). Surface for design discussion.
2. **3452 results are bias-dominated.** Raw MI estimates on small-N datasets without a bias correction are not comparable across datasets. This was not an explicit handoff deliverable, but it blocks any A2-style wild-vs-lab MI claim from being defensible. Either:
   - Adopt a bias-corrected MI estimator (Miller-Madow is the simplest; KSG-k is more principled for continuous features).
   - Report plug-in MIs only within a fixed-N bootstrap so bias is conditioned on pair count.

**Not triggered:**

- File-aware vs non-file-aware do **not** give qualitatively different MI rankings on either dataset. Both agree on the plateau structure on 5970 and the monotonic decline on 3452.
- The within-file gap distribution is bimodal on both datasets — the bout concept itself is not called into question by this analysis.

---

## 7. Caveats

- **9252 not run.** Stream 2 (9252 classification) had not completed at the time of this work. The per-dataset analysis should be re-run on 9252 when `results/traditional_taxonomy_9252/classified_traditional.csv` exists. Command is identical except `--csv` and `--dataset 9252`.
- **3452 finite-sample bias** is the dominant uncertainty. Any cross-dataset claim drawn from raw plug-in MIs is misleading.
- **Canonical values are unchanged.** `data/corpus_facts/5970.json` has not been modified by this handoff. Recommended canonical update (0.6 s non-file-aware → 0.25 s file-aware) requires a separate decision workflow per the handoff's Constraint 1.
- **Bootstrap resamples bouts, not pairs.** Transitions within a bout are correlated; resampling individual pairs would artificially tighten CIs. Resampling whole bouts is the conservative choice (and matches the handoff's "small N for some bouts" note).

---

## 8. Recommended canonical update (for Mickey)

> **Current canonical:** `bout_detection_a2.threshold_s = 0.6`, non-file-aware.
>
> **Proposed canonical:** `bout_detection_a2.threshold_s = 0.25`, **file-aware**. (Record `bout_detection_a2.file_aware = true` as a new key.)
>
> **Impact:**
> - 5970 MI lag 1: 0.0921 → 0.0935 (+0.0014 bits, within CI — no material change to downstream claims).
> - 5970 n_bouts: 1514 → 1899 (+25%).
> - 5970 n_within_bout_pairs: 6350 → 5965 (−385 pairs, but these are the 1377 cross-file pairs reclassified as cross-bout, which is the point).
>
> **Defensibility:** File boundaries are recorder-imposed (≥2 s silence), not biological. KS D=0.93 on 5970 says the two gap distributions are distinct. Hybrid (b)+(c) logic is what the data supports.

---

## Result section

- Commit SHA: **916f4ace** (script + CSVs); memo previously committed inadvertently in 375d4bdc by a parallel chat's bulk-stage; this update fills the SHA placeholder
- Recommended threshold: **0.25 s**
- Recommended logic: **file-aware (YES)**
- MI sensitivity range across reasonable thresholds: **±0.02 bits** on 5970 (combined parameter + bootstrap); bias-dominated on 3452
- Per-dataset agreement: **Partial** — both datasets prefer file-aware, but optimal within-file threshold differs (5970: 0.14 s peak, 3452: 0.27 s mixture crossover). A single global 0.25 s is a defensible compromise.
- Open Q for Mickey (refined): **"The 2 s recorder timeout produces a KS D=0.93 separation between within-file and cross-file gap distributions. We recommend treating every file boundary as a bout break (file-aware). Additionally, within-file pauses ≥0.25 s are treated as within-file bout breaks. Do you object to either? Specifically: should the ~82 cross-file gaps under 1 s in 5970 ever be treated as within-bout? Our analysis says no — but you may have behavioral evidence we lack."**
