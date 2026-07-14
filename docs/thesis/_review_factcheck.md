The 1/17/4 detector-generation counts (Figure 5) come from a specific figure-generation run; they don't appear verbatim in the production docs I can find, but the qualitative claim (first under-sensitive, matched over-fires, hard-neg selective) is documented. The "4 detections" on the presentation clip is corroborated in a provenance audit. I'll flag the 1/17/4 as figure-derived but not independently traceable to a metrics file (they are scoped to one 3.6s clip).

I have completed verification. Here is the corrections list.

---

# CORRECTIONS LIST

## ERRORS REQUIRING CORRECTION

### 1. Precision rounded wrong: 90.6% should be 90.5%
- **Claim (Abstract):** "the production detector reached 90.6% precision"; also §3.3 "precision **90.6%**", §4 "90.6% precision", and Figure 7 caption "precision 90.6%".
- **WRONG value:** 90.6%
- **CORRECT value:** **90.5%** (or cite the unrounded 90.55% / 0.9055).
- **Source/computed:** `models/hard_neg_retrain/evaluation/test_metrics.json` → `precision = 0.9054820415879017` = 90.548%. To one decimal place the second decimal is 4, so it rounds **down to 90.5%**, not up to 90.6%. The production doc `docs/production/cnn_detection_pipeline.md:195` and the CLAUDE.md project context both cite **90.55%**. The thesis's 90.6% appears in at least four places (Abstract, §3.3, §4, Fig 7 caption) and is a systematic rounding error.

### 2. Bandwidth Mann–Whitney p = 0.014 should be 0.012 (≈0.0119)
- **Claim (§3.5a, Figure 8 caption, §4, §5.8):** "Mann–Whitney U one-sided **p = 0.014**" for the bandwidth IQR comparison.
- **WRONG value:** 0.014
- **CORRECT value:** **0.012** (exactly 0.0119)
- **Source/computed:** Recomputed `scipy.stats.mannwhitneyu(wild, lab, alternative='greater')` on the per-unit IQRs (wild={37.27, 43.11, 46.81}, lab 6 males) → **U=18.0, p=0.0119**, identical to the exact permutation p=0.0119. With N=3 vs N=6 and perfect rank separation, 1/84 = 0.0119 is the floor; the MWU and permutation p **cannot differ** here. The thesis already reports the permutation p as 0.012 for the *same* test, so the 0.014 is internally inconsistent. (The ground-phase verification flagged this same discrepancy.) Note: principal-frequency p = 0.012 is correct (0.0119 rounds to 0.012).

## ITEMS TO LEAVE AS-IS (verification sheet was WRONG — do NOT "fix")

### 3. Units are kHz, NOT Hz — the verification sheet's "fix" is incorrect
- **Context:** The ground-phase capstone re-verification's Caveat #1 asserts the IQR values "42.4 / 28.1 / 23.0 / 13.0" are really in **Hz** and the thesis should relabel them. **This is wrong; the thesis's kHz labeling is correct — do not change it.**
- **Evidence:** In `classified_detections_full.csv` (5970), `principal_freq_hz` has median ≈ **64.9** and `bandwidth_hz` median ≈ **38.5**, max ≈ **164.5**. A principal frequency of 64.9 sits squarely in the 20–120 **kHz** USV band; 64.9 *Hz* would be physically impossible for a mouse USV. The production doc states this explicitly: `docs/production/deepsqueak_bridge.md:206` — "**Frequencies are in kHz** despite `_hz`" — and `classify_traditional_taxonomy.py:60` carries the same warning. The thesis §5.7 already documents this correctly ("both are stored in **kHz** despite the `_hz` column suffix"). **Keep kHz.**

## VERIFIED CORRECT (no action needed — listed only to confirm coverage)

- **Architecture (32/96/192, 192-vector, Linear 192→64→1):** Confirmed from checkpoint state_dict — `features.0=(32,1,3,3)`, `features.4=(96,32,3,3)`, `features.8=(192,96,3,3)`, `classifier.1=(64,192)`, `classifier.4=(1,64)`. Spatial chain 256×100→128×50→64×25→32×12 verified arithmetically.
- **ROC-AUC 0.989:** Recomputed from `predictions.csv` (confidence = P(USV), positive = true_label=="USV") → **0.989115**. ✓
- **PR-AUC / average precision 0.974:** Recomputed → **0.973747**. ✓
- **Held-out n=1,829; TP 479 / FP 50 / TN 1238 / FN 62:** Confirmed both in `test_metrics.json` and by re-tallying `predictions.csv`. ✓
- **Accuracy 93.9% (0.9388), recall 88.5% (0.8854), specificity 96.1% (0.9612):** All confirmed from `test_metrics.json`. ✓
- **Temperature T = 0.902; NLL 0.16908 → 0.16809:** Confirmed from `temperature.json` (0.9019383780691683, nll_before 0.16908450, nll_after 0.16809051). ✓
- **Hysteresis onset 0.60 / sustain 0.40 / gap 0 / min-duration 3; CV mean F2 = 0.867 ± 0.051:** Confirmed from `hysteresis_optimization_v2.json` (best_params, best_f2_mean 0.8669, best_f2_std 0.0506). One-SE alternative (0.8/0.5/9) also correctly cited. ✓
- **Library defaults 0.75 / 0.40 / 5:** Cited consistently. ✓
- **FP filter: StandardScaler→LogisticRegression, class-balanced, 11 features, peak_probability dominant (1.65), CV F2 = 0.823 ± 0.035:** Confirmed from `fp_filter.json` (mean_f2 0.8233, std_f2 0.0352, 11 feature_importances, peak_probability 1.6501). The §5.5 11-feature list matches exactly. ✓
- **FP filter cost: keeps ~87.3% of GT USVs / ~7.67% interval loss / 100% noise-only / ~64% in-recording FPs:** Matches `docs/production/cnn_detection_pipeline.md:332` and memory `project_fp_filter_true_effect`. ✓
- **Triage tiers: auto_reject ≤0.10, auto_accept ≥0.90; QC flags p90>0.4, event count>10, >600 ms:** All confirmed (`cnn_detection_pipeline.md:120-136`). ✓
- **Energy gate 0.1 (batch) / 0.35 (app):** Confirmed (`corpus_constants`/`cnn_detection_pipeline.md:23,264`). ✓
- **STFT: n_fft 512, hop 128, 75% overlap, 585.9 Hz/bin, 1.71 ms frame, 0.427 ms hop, Nyquist 150 kHz, visualization n_fft 2048 ≈61 Hz/bin:** All confirmed from `corpus_constants.md` (585.9375 / 1.7067 / 0.4267). ✓
- **Hard negatives 620 + hard positives 144:** Consistent with CLAUDE.md project context and §5.3. ✓ (Counts not independently in a metrics JSON, but corroborated by project memory; **[partially sourced — corroborated by CLAUDE.md, not a metrics file]**.)
- **DeepSqueak bridge: 7,518 of 7,575 matched (99.2%), 75 ms tolerance, 18-column table:** Confirmed (`deepsqueak_bridge.md:269,273,156`). ✓
- **Cohort counts: 5970=7,921; 3452=401; 9252=604; lab=40,787:** Confirmed against `corpus_facts/*.json` (7921, 401, 604) and `classified_detections_lab_131204_clean.csv` (40,787 rows). Median durations 60.12 / 17.10 / 22.94 ms confirmed. ✓
- **Bandwidth IQR: wild 42.4 vs lab 28.1 kHz (+51%):** Reproduced exactly (wild mean 42.40, lab mean 28.07, +51.0%). ✓ (per-unit wild {37.27, 43.11, 46.81} confirmed.)
- **Principal-freq IQR: wild 23.0 vs lab 13.0 kHz (+77%), p=0.012:** Reproduced (wild 23.00, lab 12.99, +77.1%, MWU/perm p=0.0119≈0.012). ✓
- **UMAP matched sample sizes (5970=12,440; 9252=584; 3452=406; m6fm6=12,369; m4fm1=1,098; m1fm2=308):** All confirmed exactly against `results/contour_vae_combined/latents.parquet` cohort/wav_stem counts. ✓
- **32-D contour-VAE latent:** Confirmed (`production_vae.md`, latents.parquet has z_0..z_31). ✓

## UNSOURCED / FIGURE-SCOPED (flag for author awareness, not necessarily wrong)

- **Detector-generation counts "1 / 17 / 4" (Figure 5, §3.3, §5.4):** These are scoped to a single 3.6 s clip and produced by a figure-generation run; I found no metrics file recording 1/17/4. The "4 detections (production)" figure is corroborated by a provenance audit (`docs/handoffs/2026-05-03_presentation-png-provenance-audit.md:60`, 4 CNN detections on the presentation clip). The "1" and "17" are **[UNSOURCED beyond the figure script]** — verify the figure's underlying run still reproduces before final submission.
- **Lab per-male IQR breakdown (minor):** My recomputed per-male values differ trivially from the verification sheet (e.g. m1 bandwidth 34.81 vs sheet 34.6; m6 principal-freq 9.59 vs sheet 10.4) due to NaN/CSV-version handling, but the **means, percentages, U, and p-values reproduce exactly**, so no thesis number is affected. The thesis does not print per-male values, so no correction needed.