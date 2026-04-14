# Progress Report — USV Vocalization Analysis

**Researcher:** Shachar
**PI:** Prof. London
**Date:** April 5, 2026
**Subject:** USV Detection, Classification & Early Analysis — Status Update

---

## 1. Detection Pipeline (Complete)

A complete CNN-based USV detection pipeline is operational and validated:

- **Model:** Custom CNN (207K parameters) trained on **15,444 labeled spectrogram windows** (5,790 USV + 9,654 non-USV). Training data was assembled in stages: initial hand-labeled USVs, then automated negative sampling from three sources (random positions in the recording, inter-USV gaps, and low-energy regions in the USV frequency band), and finally a hard-negative mining round where 620 common noise patterns misclassified as USV and 144 missed USVs from manual review were added. Current production model achieves **90.55% precision** and **98.7% USV rate** in manual review.
- **Post-processing:** Full pipeline includes hysteresis smoothing (F2=0.885), false-positive filtering (F2=0.850), temperature-calibrated confidence scoring (T=0.905), and automated triage.
- **Desktop application:** A PyQt6 GUI tool (similar to DeepSqueak/Audacity) provides interactive spectrogram viewing, threshold adjustment, and detection review. Includes a 0–30 kHz audible-range view alongside the USV-range view.
- **Datasets processed:**
  - **Cage 5970 (usv_lmt_034):** 6,400 WAV files → 1,328 with detected USVs → **7,575 total detections**
  - **Cage 3452 (usv_lmt_035):** ~5,400 WAV files processed (two batch runs: 5,409 sample + 841 reviewed, with partial overlap) → **~125 files with USVs → ~456 total detections.** Classification pending. Notably, this is roughly **16× fewer detections** than cage 5970 (456 vs 7,575), which may reflect genuine individual differences in vocalization rate.

### Cross-validation with DeepSqueak

To independently validate our CNN pipeline, we ran DeepSqueak's YOLO v2 mouse USV detector on a 197-file subset from cage 5970. DeepSqueak is a completely separate neural network (different architecture, different training data, different lab) — agreement between the two systems is strong evidence that detections are real USVs, not artifacts.

| | Our CNN | DeepSqueak | Both agree |
|---|---|---|---|
| **Detections** | 226 | 277 | 241 |
| **CNN-only** (DS missed) | 10 | — | — |
| **DS-only** (CNN missed) | — | 36 | — |

- **95.6% of our CNN detections were confirmed by DeepSqueak** (216/226 overlap). Mean IoU on agreed detections: 0.989 (near-perfect temporal alignment).
- **87% of DeepSqueak detections were confirmed by our CNN** (241/277). The 36 DS-only detections are mostly low-confidence calls or events in our FP filter's rejection zone.
- After retraining the FP filter without duration as a feature (see below), CNN confirmation of DeepSqueak rose to **91%** (25 DS-only, down from 36).

**Merged-call issue identified:** During dense bouts, our hysteresis smoothing merges adjacent calls into single long events (200-1000ms spanning 2-6 calls). The FP filter, trained mostly on short single-call events, learned "long = noise" and rejects some of these real merged bouts. A retrained filter without the duration feature (F2: 0.823 → 0.833) resolves this. See `docs/questions-for-mickey.md` Q5 for details on whether to adopt the new filter.

## 2. Classification (Complete for 5970)

Two independent classification approaches were applied to the 5970 dataset:

### A. DeepSqueak Bridge (k-means, 27 clusters)
- Built a Python↔MATLAB bridge: CNN detections exported as Raven selection tables → DeepSqueak MATLAB clustering → results imported back with 99.2% match rate (7,518 of 7,575 detections).
- DeepSqueak's k-means produced 27 acoustic clusters. However, subsequent analysis revealed these are likely **over-split** — an artifact of forcing k=27 on continuous data.

### B. Traditional Taxonomy (rule-based, 7 types)
- Applied a rule-based classifier using standard USV syllable categories (Scattoni et al.): **Flat (32%), Down (17%), Chevron (16%), Short (14%), Complex (9%), Frequency Jump (7%), Up (6%)**.
- Confidence scoring: 64% high confidence, 28% medium, 8% low.

### C. UMAP + HDBSCAN (data-driven, 3 density clusters)
- Dimensionality reduction (UMAP) on 10 acoustic features followed by density-based clustering (HDBSCAN).
- Result: **USV calls form a continuous manifold rather than discrete categories.** HDBSCAN found only 3 density clusters — one dominant cluster containing 7,598 of 7,864 calls (97%), plus a small long-duration cluster (131 calls) and an ultra-short cluster (98 calls). 37 outliers classified as noise.
- **Key finding:** This strongly suggests that traditional discrete categories (Flat, Down, Chevron, etc.) are convenient labels imposed on a continuous acoustic space, not natural groupings.

## 3. Temporal Dynamics Analysis (Complete for 5970)

Analyzed the temporal structure of calling behavior across the full ~32-hour continuous recording (Sep 30 11:18 → Oct 1 19:42, 2024):

- **Total:** 7,864 calls across 1,338 files over 32.4 hours
- **Calling pattern:** Highly bursty. Peak hour: 1,089 calls; 3 completely silent hours.
- **Bout structure:** ~1,500 calling bouts detected (median 4 calls/bout, median bout duration 0.48s). 19% were single-call "bouts." *Note: bout count depends on threshold — see Section 4 methodology note.*
- **Inter-call intervals:** Median within-file gap-based ICI = 0.078s; median cross-file gap = 15.3s. *(Earlier version reported 0.19s median ICI using onset-to-onset measurement; gap-based is more accurate.)*
- **Repertoire stability:** Type composition (Flat, Down, etc.) is remarkably stable during active vocalization periods — no evidence of repertoire shift over hours.
- **Bout-initial bias:** Down calls are significantly overrepresented as bout starters (~28% vs ~17% baseline), suggesting non-random sequential structure.
- **USV1–USV5 folders** were confirmed to be **download batches, not temporal sessions** — temporal analysis must use filename timestamps.

## 4. Sequential Structure Analysis (Complete for 5970)

**The big question:** We have 7,864 calls in a time-ordered sequence, each labeled as one of 7 types. Does knowing what the animal just said help predict what it says next? This matters because if sequences are structured, it suggests the vocalizations carry more information than individual calls alone — a prerequisite for anything resembling "syntax."

**Important methodological note:** All sequential analyses are **bout-aware** — we only count transitions between consecutive calls within the same vocal bout (inter-call interval < 0.6s). Calls separated by long silences (minutes to hours) are NOT treated as sequential. This excluded ~1,500 cross-bout gaps (~19% of all consecutive pairs) that would otherwise add noise.

**⚠ Open issue — bout threshold under review:** The 0.6s threshold was derived from `3 × median(onset-to-onset ICI)`, but a post-hoc audit found two problems: (1) ICI was measured start-to-start instead of gap-based (end-to-start), and (2) the WAV files are trigger-based recordings, so cross-file gaps are recording artifacts, not vocalization timing. A mixture model fit on within-file gap-based ICIs suggests the threshold should be **0.14–0.25s**. In practice the impact is modest (file boundaries already handle most bout breaks), but a final re-run is pending once the definition of "bout" is confirmed. See `docs/questions-for-mickey.md` Q1.

We applied six analysis layers, from simplest to most sophisticated:

### 1. Transition matrix — "What follows what?"

Count every consecutive within-bout pair (A→B) and compute P(B|A). If calls were random, every row would look like the overall type distribution (32% Flat, 18% Down, etc.). Deviations from that baseline = structure.

**What we found:** The strongest signal is the diagonal — self-transitions. Flat→Flat, Short→Short, etc. all happen more than chance predicts (25.7% observed vs 20.0% expected under independence). The animal tends to repeat itself, though the enrichment is modest (1.28×). Non-self transitions mostly just reflect base rates — the top non-self transitions (Frequency_Jump→Flat at 37%, Down→Flat at 30%, Chevron→Flat at 32%) are all transitions *to Flat*, because Flat is the most common type (32%), not because of a grammatical rule.

*Note: An earlier version reported 14.3% as the chance baseline (1/K assuming uniform distribution). The correct baseline is Σ(pᵢ²) = 20.0%, which accounts for the non-uniform type frequencies.*

### 2. Entropy rate — "How predictable is the sequence?"

Shannon entropy measures uncertainty. With 7 equally-likely types, maximum entropy = log₂(7) = 2.807 bits. Our marginal entropy is 2.544 bits (types aren't equally likely — Flat dominates). The key question is: does context reduce entropy?

The conditional entropy H(next | current) = 2.449 bits tells us directly: knowing the current call only reduces uncertainty by **0.095 bits (3.7%)**. For reference, in English text, knowing the previous letter reduces uncertainty by ~50%. So this is a very weak signal.

### 3. Mutual information at lag — "How far does memory reach?"

MI(T, T+k) asks: how much information does call T carry about call T+k?

- Lag 1: 0.093 bits (matches the entropy reduction — same information, different measure)
- Lag 2: 0.042 bits (drops by half — memory is very short-range)
- Lag 6+: essentially noise floor

The sequence has very short memory — mostly just the immediately preceding call, and even that is weak. The slow decay to lag ~10 is partly an artifact of self-repetition runs (during Short→Short→Short→Short, lag-3 correlations exist trivially because all calls are the same type).

### 4. Zipf distribution — "Does a power law govern type frequencies?"

In human language, word frequencies follow Zipf's law (the 2nd most common word appears ~half as often as the 1st, etc.) — a deep property of communicative systems.

Result: α=0, p=1 — **not a power law.** But with only 7 types, there isn't enough rank range for a meaningful fit. This would be more informative on a larger vocabulary (like the 27-cluster DeepSqueak classification).

### 5. Idiom detection — "Are there recurring phrases?"

The most sensitive test. For each n-gram (bigram through 5-gram), count how often it appears in the real sequence vs. 200 shuffled versions that preserve the overall type distribution but destroy sequential structure. N-grams significantly more common than shuffled = "idioms."

653 significant idioms found (after correcting a bug in the shuffle null model — an earlier version reported 1,843, which was inflated by ~65% due to a broken shuffle that didn't preserve bout boundaries). The top idioms by z-score are **same-type repetitions** (Complex×5 z=67.6, Down×5, Short×5) — perseverative runs, not heterogeneous motifs. 26 homogeneous idioms (all-same-type) are robust. The remaining 627 heterogeneous idioms are mostly low-count (55% observed only once) and should be interpreted cautiously.

### 6. What this means for the bigger picture

The sequence has one dominant pattern: **self-repetition.** The animal tends to produce runs of the same type. Beyond that, knowing the current call barely helps predict the next one. This is consistent with a model where:

1. The animal enters a "mode" (e.g., producing Short calls)
2. It stays in that mode for several calls (self-repetition)
3. When it switches, the next type is approximately random (weighted by base rates)

This is meaningful — **it rules out random independent production** — but it's a far cry from "syntax." The structure is more like a Markov chain with sticky states than a grammar with rules. The self-repetition may reflect sustained arousal states or motor planning constraints.

Sequential structure alone may not reveal "language-like" properties with only 7 coarse types. The UMAP analysis showed calls form a continuum — the traditional 7-type taxonomy may be too coarse to capture fine-grained sequential dependencies. A possible next step: re-run sequential analysis on the 27-cluster DeepSqueak vocabulary or on continuous acoustic features, where finer categories leave more room for sequential patterns to emerge.

### Figures produced

- `results/sequential_structure/transition_matrix.png` — 7×7 P(B|A) heatmap (within-bout)
- `results/sequential_structure/entropy_convergence.png` — entropy rate vs n-gram order
- `results/sequential_structure/mutual_information_lag.png` — MI at lags 1–10
- `results/sequential_structure/zipf_distribution.png` — rank-frequency plot
- `results/sequential_structure/idiom_report.csv` — full list of 653 significant idioms (corrected)
- `results/sequential_structure/bout_threshold_within_file.png` — bout threshold analysis (mixture model + sensitivity sweep)

## 5. Next Steps

| Priority | Analysis | Status |
|----------|----------|--------|
| **A3** | Acoustic feature deep-dive — PCA, feature correlations, UMAP overlays | Planned |
| **A4** | Detection confidence analysis | Planned |
| **B1** | Classify cage 3452 (usv_lmt_035) — apply taxonomy + UMAP to 456 detections | Detection done, classification pending |
| **B2** | Cross-animal comparison — repertoire, transitions, acoustic space overlap | Requires B1 |
| **C1** | Behavioral correlation (LMT integration) — PETH, type-specific event coupling | Waiting on LMT .sqlite behavioral database |

## 6. Tools & Infrastructure Built

- Full Python detection + post-processing pipeline (346 tests)
- PyQt6 desktop detection/review application
- Python↔DeepSqueak MATLAB bridge (Raven export/import)
- Rule-based and unsupervised classification modules
- Repertoire statistics module (Shannon entropy, PERMANOVA, JSD, transitions)
- Information theory module (Zipf, entropy rates, idiom detection, burstiness)
- Temporal dynamics analysis scripts
- LMT integration modules (PETH, DB loader, synchronizer — ready but awaiting behavioral data)

## 7. Open Questions

1. **Is the continuous manifold truly featureless?** A finer-grained density analysis might reveal sub-structure within the dominant cluster.
2. **Do bouts have functional signatures?** Does each bout use a consistent subset of types, or sample broadly?
3. **Are Short calls real USVs or detection artifacts?** The 14% Short calls overlap with HDBSCAN's ultra-short cluster — cross-referencing with CNN confidence would help distinguish.
4. **Does the second animal (3452) show a similar repertoire and continuum structure?**
5. **LMT behavioral data** — linking USV types to social behavior events would be the most impactful next analysis.
6. **Why does cage 3452 vocalize so much less?** With ~456 detections vs 5970's ~7,575 (a 16:1 ratio across similar recording volumes), this could reflect individual temperament, social context differences, or recording condition differences worth investigating before cross-animal comparison.
