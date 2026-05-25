# Sub-cluster morphology discovery + cleaning-pipeline artifact audit

**Status:** Plan / working document, opened 2026-05-24
**Scope:** Follow-up investigation triggered by exploring the K=20 centroid grid
in `results/latent_transitions/centroids_n20/` and the sub-K-means dive into
c07, c11, c08 in `results/latent_transitions/centroids_n20/sub_*`.
**Owner:** Shachar
**Related:** `docs/handoffs/2026-05-25_mickey_cleaning_pipeline_and_vae_briefing.html` §5.1, `docs/handoffs/2026-05-20_latent-analysis-b-a-c.md`

---

## 1. What we already established this session

### 1.1 Centroid panels re-rendered at N=20 exemplars
- Re-decoded every K=20 centroid with **20 latent-nearest exemplars** instead of the 9 used in the briefing.
- Output: `results/latent_transitions/centroids_n20/cluster_NN.png` (20 PNGs, 5×5 grid each).
- Contact sheet: `results/latent_transitions/centroids_n20/index.html`.
- Pipeline preserved: original 9-exemplar PNGs at `centroids/` left untouched so the briefing HTML's image references still resolve.
- Generator: `$CLAUDE_JOB_DIR/decode_centroids_n20.py` (one-shot wrapper around `_decode_centroid_examples` from `scripts/analyze_latent_transitions.py`).

### 1.2 K=20 alphabet is biologically balanced but rare-type poor
Cluster sizes range 2.2 % – 9.8 % — no <1 % "rare pocket." The alphabet over-resolves the flat-tone family (~10 of 20 cells) and under-resolves rare-type morphology (chevron, complex, frequency jump, etc.). Annotated reads from this session:

| Cluster | n | % | Read |
|---|---|---|---|
| c00 | 2869 | 4.1 % | Long sustained flat tones, mild ripple |
| c01 | 5107 | 7.4 % | Short upward sweeps; some short chevrons |
| c02 | 2593 | 3.7 % | Flat tone. **5970-signature (14.6 % vs <3 %)** — drives Move A JSD |
| c03–c05 | ~3700 | 5 % | Flat-tone variants |
| c06 | 6815 | 9.8 % | Flat. **3452+9252 shared (34 % / 28 %)** |
| c07 | 3339 | 4.8 % | **Closest to "complex"** — multi-segment / frequency-jump exemplars |
| c08 | 1505 | 2.2 % | **Smallest + 5970-skewed.** Short blobs with sub-harmonic shadow |
| c09 | 2842 | 4.1 % | Flat. **3452+9252 shared (21 % / 12 %)** |
| c10 | 4641 | 6.7 % | Short upward-tilted blobs |
| c11 | 5107 | 7.4 % | **The chevron cluster** — pronounced peaks |
| c12 | 4035 | 5.8 % | Short chevron-flavoured |
| c13 | 2157 | 3.1 % | **Noise/artifact bucket** — broadband streaks, vertical ringing |
| c14 | 3297 | 4.8 % | Steep downward sweeps |
| c15 | 2745 | 4.0 % | Flat-tone, lab-heavy |
| c16 | 2411 | 3.5 % | Nearly-flat with gentle downslope; 5970-skewed |
| c17 | 2592 | 3.7 % | Flat with curvature; mild chevron-ish; 5970-skewed |
| c18 | 5170 | 7.5 % | Flat. **Mild lab signature (8.7 %)** |
| c19 | 2571 | 3.7 % | Shallow chevrons. **Lab-only (3.8 % lab, ~0 % small-wild)** |

### 1.3 Sub-K-means on c07, c11, c08 (k_sub=4)
- Script: `$CLAUDE_JOB_DIR/subcluster_decode.py`
- Output: `results/latent_transitions/centroids_n20/sub_cNN/sub_S.png` + `metadata.json`
- Contact sheet: `results/latent_transitions/centroids_n20/sub_index.html`

#### c07 (parent 3339, "complex-ish") — cleanest split

| Sub | n | Cohort | Morphology |
|---|---|---|---|
| c07.0 | 1055 | lab=915 5970=136 | Asymmetric chevrons (peak + descent) |
| **c07.1** | 650 | lab=546 5970=103 | **Symmetric chevrons + frequency jumps** |
| **c07.2** | 656 | lab=557 5970=98 | **Sub-harmonic complex calls** (parallel band + frequency discontinuities) |
| c07.3 | 978 | lab=920 5970=57 | Kinked / step-down sweeps |

#### c11 (parent 3541, "chevron") — partial split

| Sub | n | Cohort | Morphology |
|---|---|---|---|
| c11.0 | 1209 | lab=1045 5970=162 | Shallow chevrons + gentle downslopes |
| c11.1 | 650 | lab=533 5970=141 | Left-asymmetric chevrons |
| **c11.2** | 710 | lab=465 **5970=245 (35 %)** | **Short peak + long descending tail — possible wild-chevron variant** |
| c11.3 | 946 | lab=833 5970=113 | Short / variable chevrons (overlaps with sub_0) |

#### c08 (parent 1505, "5970 short") — headline find

| Sub | n | Cohort | Morphology |
|---|---|---|---|
| c08.0 | 369 | 5970=279 (76 %) lab=87 | Short tonal + U-shaped valleys (inverted-chevron motif) |
| c08.1 | 501 | 5970=260 lab=232 (50/50) | Mixed exotics (short calls + sub-harmonics + chevrons + S-curves) |
| **c08.2** | 270 | 5970=231 (86 %) lab=34 | **S-curve / step-modulated short calls** |
| **c08.3** | 365 | **5970=344 (94 %)** lab=20 | **Frequency-jump pocket** — wild-specific syllable type erased by K=20 |

### 1.4 User observations on the panels (this session)
- c08.3 tiles 1, 2, 3, 4, 9, 11, 12, 18 are **frequency jumps** — corrects my earlier "fragmented" hedge.
- c11 confirmed as the chevron cluster.
- c07.2 also shows many jumps.
- **Open concern (the reason this audit exists):** could the cleaning pipeline be *creating* these jumps rather than revealing them?

---

## 2. Why the cleaning-pipeline question is well-founded

### 2.1 The mechanism that can produce fake jumps
`scripts/contour_mask_utils.py::apply_hard_bandwidth_mask` (in the
`contour-masked-vae-pipeline` worktree) operates per time-column:

- For each column `t`, look at the contour-extraction output for that column.
- If a contour ridge exists with `tonality >= tonality_threshold` (currently 0.0, so always satisfied if a ridge exists at all):
  - Keep S_pow within ±`bandwidth_kHz` (5 kHz) of the ridge frequency, zero everything else.
- **If no contour ridge exists for that column → zero the entire column.**

Consequence: a brief 2–3 column gap in contour extraction mid-call → those columns become solid black in the VAE input → the call appears split into "before-gap" and "after-gap" segments. The VAE decoder faithfully reproduces the gap because that's what it was trained on.

### 2.2 Why this would be cohort-asymmetric
Cage-specific noise floor, SNR, and sub-harmonic prominence all affect the contour extractor's per-column success rate. 5970 may have lower mid-call SNR than lab 131204 → more ridge dropouts → more apparent jumps → c08.3 becomes "5970 calls fail contour extraction more often" rather than "wild mice produce more frequency-jump calls." This would invert the biological interpretation entirely.

### 2.3 Prior corroborating evidence in the vault
Memory file `project_clustering_analysis_lab.md`: lab cluster c2 was previously shown to have "soft-notch bracketing" — its morphology was *produced* by an over-aggressive cleaning filter. Precision dropped to ~63 % vs claimed 85 % when manually verified. So "cleaning pipeline produces apparent morphology" is an established pattern in this codebase, not a hypothetical.

Memory file `project_cleaning_pipeline_inventory.md`: cleaning stack has 4 layers; soft-notch is only one. Other layers (`--subtract-baseline`, global MAD normalization, per-recording Z-norm) can also distort calls. **Check ALL layers before claiming a cleaning gap, per established rule.**

---

## 3. Planned investigations

Numbered in priority order. Each step has explicit cost, prerequisites, and decision gates.

### Phase A — Cheap statistical screen (1–2 hours)

#### A1. Zeroed-column rate per cluster
**Goal:** quantify how much of each cluster's patches the contour mask actually zeroed out. High zeroed-column rate = high artifact risk.

**Method:**
1. Load `results/masked_patches/combined_all_cohorts/patches.npz` (the VAE input).
2. For each patch, compute `frac_zero_cols = mean over columns of (patch.sum(axis=freq) == 0)`.
3. Group by (cluster, cohort), report mean ± IQR.
4. Compare c08.3, c07.2 (suspect clusters) vs c11.1, c11.0 (control — should be clean chevrons).

**Decision gate:**
- If c08.3 and c07.2 have meaningfully higher zeroed-column rates than c11 sub-cells → strong evidence of cleaning-induced apparent jumps.
- If similar → jumps are more likely real; proceed to Phase B for case-by-case confirmation.

**Cost:** ~10 min once `patches.npz` is located (may live in either this worktree or `contour-masked-vae-pipeline`).

#### A2. Inter-segment gap statistics
**Goal:** within each sub-cluster, measure the distribution of gap widths (consecutive zeroed columns mid-patch).

**Method:** for each patch, find runs of zero-columns interior to the call (skip leading/trailing silence). Report distribution of run lengths per sub-cluster.

**Decision gate:** if c08.3 gaps cluster at very short lengths (1–3 columns; convert to ms using `corpus.STFT_HOP / corpus.SAMPLE_RATE_HZ` — do NOT hardcode), that's suspicious — real frequency jumps are typically wider or sharper transitions in the raw signal. Real biology should show broader gap distributions.

> **Corpus-invariant note for any code implementing Phase A/B/C:** import all STFT and rate constants from `src/usv_spectrogram/corpus.py` (`SAMPLE_RATE_HZ`, `STFT_N_FFT`, `STFT_HOP`, `USV_FREQ_MIN_HZ`, `USV_FREQ_MAX_HZ`). Never redeclare. The cleaning pipeline's masking step depends on these exact values matching what the contour extraction used — divergence would invalidate every artifact-vs-biology comparison in this plan.

**Cost:** ~10 min, same input as A1.

### Phase B — Per-patch ground-truth comparison (2–3 hours)

#### B1. Three-panel side-by-side renders
For each candidate patch, render:
- **A. Raw power spectrogram** — STFT(WAV, sr=300000, no cleaning), log-dB scale. This is "ground truth biology."
- **B. Contour-masked spectrogram** — apply `apply_hard_bandwidth_mask` with the cohort's `contours.parquet`. This is what the VAE saw.
- **C. VAE-decoded centroid view** — for orientation.

**Selection (informed by A1, but tentative baseline):**
- 4 patches from c08.3 (the headline 94 % wild jumps)
- 2 patches from c07.2 (lab-skewed complex with jumps)
- **2 control patches from c11.1** (clean chevrons that should NOT show artifact gaps — proves the method can distinguish)
- 2 patches from c07.0 (asymmetric chevrons that should be clean) — second control

Total: 10 patches × 3 panels = 30 spectrograms in a single contact sheet.

**Decision criteria:**
- Gap in A AND B → **real biology** (frequency jump or multi-component call)
- No gap in A, gap in B → **cleaning artifact** (contour ridge dropout)
- Controls clean in A AND B → method validated

**Cost:** ~30–45 min including WAV path resolution and contour-parquet loading.

#### B2. If artifacts confirmed: scope the damage
If B1 shows artifacts:
- Sample 30 patches uniformly across c08.3, score "artifact vs real" per patch from the three-panel view.
- Estimate what fraction of c08.3 is artifact contamination.
- Decide: re-cluster after fixing cleaning, or annotate c08.3 with a warning in any downstream analysis.

### Phase C — Pipeline-level remediation (variable cost; conditional on Phase B)

#### C1. Contour extraction sanity check
Inspect `results/contour_extraction/<cohort>/contours.parquet` for c08.3 patches. Specifically:
- Per-call ridge continuity: how many columns have NO ridge?
- Are dropouts concentrated mid-call (suspicious) vs at edges (normal)?
- Compare per-cohort dropout rate.

#### C2. Try a "gap-tolerant" mask variant
Modify `apply_hard_bandwidth_mask` to bridge ridge gaps ≤ N columns by interpolating the ridge frequency. Re-extract patches for c08.3-suspicious calls, re-render. Does the apparent jump go away?

#### C3. Train-free sanity check
For each "candidate jump" patch, **load the raw WAV window** and check whether a human reviewing the raw spectrogram would label it a frequency jump.

---

## 4. Broader sub-clustering questions (parking lot — do NOT lose)

These came up while exploring the sub-clustering results. Captured here so future sessions can pick them up.

### 4.1 Run sub-K-means on the remaining 17 K=20 cells
**Rationale:** if c07 and c08 each hid 4 distinct morphologies, the other 17 clusters likely also hide structure — especially the lab-heavy flat-tone family which has 10 cells that may be more like 20–25 cells.

**Cost:** ~5 min for the full 17-cluster sub-K-means decode + ~30 min to write a master nested contact-sheet HTML.

**Priority:** medium — interesting but not gating any decision.

### 4.2 UMAP+HDBSCAN validity check inside each parent
**Rationale:** sub-K-means *imposes* k_sub sub-cells. UMAP+HDBSCAN can return "no, this is a continuum" — that's a more honest answer. Use as a validation step after sub-K-means flags a candidate.

**Method:** UMAP(parent_latents, 32→2) → HDBSCAN(min_cluster_size=30). Compare HDBSCAN's automatic K to sub-K-means' forced k=4.

**Cost:** ~1–2 min per parent cluster.

**Priority:** high if Phase B confirms c08.3 jumps are real — then we want to know if "frequency-jump morphology" is a single density mode or a continuum.

### 4.3 Global K bump (K=40 or K=60) — last-resort option
**Rationale:** if sub-clustering keeps finding rich structure, K=20 might just be too coarse globally.

**Cost:** ~30 min (re-fit K-means + re-assign labels + re-decode all centroids), but **invalidates everything downstream**: JSD heatmap, transition matrices, entropy bootstraps, idiom counts, briefing §5+§6 tables.

**Priority:** explicitly low — only do this if the K=20 narrative breaks and we have to start over. Most likely we'd keep K=20 as the "briefing alphabet" and use sub-K-means / HDBSCAN as a *complementary* analysis.

### 4.4 Other parent cells worth probing first
Based on the K=20 reads, candidates for next sub-K-means runs:
- **c13 (noise bucket)** — does it split into specific artifact types? Useful for diagnosing the cleaning pipeline.
- **c19 (shallow lab-only chevrons)** — does it have a "pure chevron" sub-cell vs a "noise" sub-cell?
- **c10 (short upward-tilted blobs)** — does it hide a frequency-jump variant analogous to c08.3?
- **c02 (5970-signature flat)** — does it actually split into multiple wild call types, or is it really one syllable?
- **c14 (steep downward sweeps)** — does the "downward + descent" character split by slope angle?

### 4.5 Cross-tab sub-clusters against the existing 7-type traditional taxonomy
**Rationale:** `classified_detections_full.csv` has the rule-based 7-type label for every 5970 call. If c08.3 patches are mostly labeled "Frequency-Jump" or "Complex" by the rule-based classifier, that's an independent confirmation. If c08.3 patches are scattered across taxonomy labels, that's a warning.

**Cost:** ~15 min.

**Priority:** medium — independent label corroboration is always valuable.

### 4.6 Examples-furthest-from-centroid view
**Rationale:** today's panels show the cluster *core* (20 nearest exemplars). The *furthest* exemplars show what's about to spill over to neighboring cells — useful for understanding where the cluster boundaries are loose.

**Cost:** ~5 min — flip the `argpartition` to take *largest* distances instead of smallest.

**Priority:** low — useful diagnostic but not gating any specific question.

---

## 5. Decision-gate flowchart

```
Phase A1 (zeroed-column rate)
    |
    +-- c08.3/c07.2 rates significantly higher than controls?
    |      |
    |     YES -> Phase B1 (per-patch ground truth) -> very high prior on artifact -> Phase C remediation
    |      |
    |     NO  -> Phase B1 anyway (cheap confirmation) -> very high prior on real biology
    |
    +-- Phase B1 result:
           |
           +-- raw shows jumps -> REAL -> update §5 narrative; c08.3 stays as a wild-specific syllable
           |
           +-- raw smooth, cleaned shows jumps -> ARTIFACT -> Phase C; re-evaluate the briefing's JSD interpretation
           |
           +-- mixed -> Phase B2 (scope the damage); cluster gets an "artifact-contaminated" annotation
```

---

## 6. Files / artifacts already produced this session

| Path | Purpose |
|---|---|
| `results/latent_transitions/centroids_n20/cluster_*.png` | 20-exemplar centroid panels (one per K=20 cluster) |
| `results/latent_transitions/centroids_n20/index.html` | Contact sheet for the K=20 alphabet |
| `results/latent_transitions/centroids_n20/sub_c07/sub_*.png` | c07 sub-K-means panels (k_sub=4) |
| `results/latent_transitions/centroids_n20/sub_c11/sub_*.png` | c11 sub-K-means panels |
| `results/latent_transitions/centroids_n20/sub_c08/sub_*.png` | c08 sub-K-means panels |
| `results/latent_transitions/centroids_n20/sub_*/metadata.json` | Per-sub-cluster patch counts + cohort breakdowns |
| `results/latent_transitions/centroids_n20/subclustering_summary.json` | Combined summary of all sub-K-means runs |
| `results/latent_transitions/centroids_n20/sub_index.html` | Contact sheet for the sub-K-means dives |
| `$CLAUDE_JOB_DIR/decode_centroids_n20.py` | One-shot generator for 20-exemplar centroids |
| `$CLAUDE_JOB_DIR/subcluster_decode.py` | Sub-K-means + decode + cohort tally |

(`$CLAUDE_JOB_DIR` resolves to `/home/shachar/.claude/jobs/04fa5e4d/` for this session.)

---

## 6.5 Phase A1 + visual inspection results (2026-05-24 session)

### Phase A1 — zeroed-column rate per cluster
Ran `$CLAUDE_JOB_DIR/phase_a1_zero_column_screen.py`.

**Verdict: the c08.3 jump morphology is NOT explained by column-dropout artifacts.**

Headline numbers from head-to-head comparison:

| Cluster | n | mean_frac_zero | %_with_gap | mean_max_gap_ms |
|---|---|---|---|---|
| c08.3 (94% wild "jumps") | 365 | 0.595 | 100 | **4.16** |
| c07.2 (lab "sub-harmonic complex") | 656 | 0.597 | 100 | **3.33** |
| c11.0 (control: shallow chevron) | 1209 | 0.685 | 100 | **4.33** |
| c11.1 (control: left-asym chevron) | 676 | 0.645 | 100 | **3.46** |
| c07.0 (control: asym chevron) | 1055 | 0.687 | 100 | **5.69** |

Key observations:
- **100% of patches in *every* cluster have interior gaps.** The contour mask is uniformly aggressive — gap *prevalence* doesn't distinguish suspect from control.
- **Median max gap length is ~0.85 ms (2 columns at hop=128, sr=300000) across all clusters.** That's sub-call-feature scale — far too small to produce the visible jumps seen in the c08.3 centroid panel.
- **Mean max gap length in c08.3 (4.16 ms) is *below* c11.0 control (4.33 ms) and well below c07.0 control (5.69 ms).** If the cleaning were creating fake jumps preferentially in c08.3, we'd expect c08.3 to have systematically wider gaps.
- **The clusters with the genuinely-largest gaps are c13 (14.9 ms, noise bucket) and c09 (14.3 ms, small-wild flat).** Neither was on the suspect list.

### Visual inspection
Ran `$CLAUDE_JOB_DIR/inspect_jump_mechanism.py` → `results/latent_transitions/centroids_n20/jump_mechanism_inspect/jump_mechanism_inspect.png`.

Rendered 4 c08.3 (5970) and 2 c07.2 (lab) raw VAE-input patches with a cyan peak-frequency trace overlay and zero-column markers. Observations:
- c08.3 patches show **smooth, continuous peak-frequency traces** through the visible "jumps." Where there's a vertical break in the patch, the ridge has snapped to a new frequency but signal exists on both sides — consistent with real frequency-jump morphology.
- c07.2 patches show **clean multi-band sub-harmonic structure** — the contour-mask's ±5 kHz band is wide enough to retain fundamental + first harmonic when both are within 5 kHz of each other (true at higher pitches).
- Neither cluster shows the "long mid-call column dropout then resumed signal" pattern that would indicate cleaning-induced fake jumps.

### Outstanding caveat
Phase A ruled out **column-dropout artifacts**. The mechanism *not yet ruled out* is **ridge-extraction failure mid-call** — where the contour extractor loses the ridge for several columns, the mask then zeros those columns out, and the visible "jump" is the result. This would be hard to distinguish from real biology without comparing to raw WAV. **Phase B (per-patch raw-vs-cleaned three-panel renders) is still the appropriate confirmation step** — but the prior for "real biology" is now substantially stronger than at session start.

### Decision-gate outcome
- Phase A signaled "low artifact risk" → Phase B is **optional rather than required**.
- Recommend Phase B as a 1-hour sanity check before *publishing* the c08.3 finding (e.g., updating the briefing or proposing K-bump rationale), but it's NOT blocking ongoing work.
- The c08.3 cluster can be cited cautiously as "candidate wild-specific frequency-jump morphology, pending Phase B raw-WAV confirmation."

## 7. Open questions surfaced but not yet investigated

1. **Does the cleaning pipeline create cohort-asymmetric jump artifacts?** (Section 2 — primary)
2. **If c08.3 jumps are real, do they appear in 3452/9252 as well?** (They have only 1 patch each in c08 currently — too few for inference. May indicate the morphology exists across all 3 wild dyads but only at high call volumes, where 5970 dominates due to its ~30× higher patch count.)
3. **Does c07.2 (sub-harmonic complex) correlate with `det_prob_max` or any detection-quality metric?** Real sub-harmonics tend to come from louder, more confident calls.
4. **Is there a syntactic difference?** I.e., do c08.3 jumps tend to occur at bout boundaries or in particular transition positions in the K=20 sequence model? This would distinguish "communicative jump" from "stochastic artifact."
5. **Does the briefing's c2 (5970-signature flat) also have sub-structure?** It's *the* JSD-driving cluster — if it hides 2–3 sub-types, the JSD interpretation changes.
