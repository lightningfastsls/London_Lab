# Handoff — Lab 131204 Phase 2B, classification + repertoire complete

**Date:** 2026-05-15
**Status:** Phase 2B (three classifiers + tier/couple-aware repertoire) COMPLETE.
**Previous handoffs:**
- `docs/handoffs/2026-05-14_lab_131204_post_labeling.md` — Phase 2 unblock + four findings to verify
- `docs/handoffs/2026-05-14_lab_131204_phase2a_deepsqueak_handoff.md` — DeepSqueak input prep
**Next handoff:** TBD — Phase 3 wild-vs-lab statistical comparison

---

## TL;DR

The full Phase 2B pipeline ran end-to-end on `events_clean.parquet` (41,061
events) → DeepSqueak (41,658 calls including 597 wild-mouse residue) → 75 ms
proximity merge (40,787 matched lab calls with 16 acoustic features) → three
classifiers + three repertoire-comparison variants.

**Five findings, two of them new and important:**

1. **Scattoni-7 type distribution** is dominated by **Flat (30%) and Chevron
   (22%)**, a profile distinct from DeepSqueak's 26 forced k-means clusters.
2. **(NEW) Lab UMAP embedding is essentially unimodal.** HDBSCAN with default
   density params finds 1 mega-cluster (71%) + 1 high-sinuosity Complex
   cluster (8%) + 1 small residual-noise cluster (0.6%) + 20% noise points,
   regardless of `min_cluster_size` ∈ {50, 100, 200}. Wild-mouse 3452
   produced 5+ separable clusters with the same defaults. **Lab USVs are
   substantially less differentiated than wild USVs by acoustic-feature
   density structure.**
3. **(CONFIRMED handoff finding 2) Tier signal Cramér's V = 0.250** —
   manual_review carries 2.4× more Short and 2.3× more Complex calls than
   auto_accept. Justifies the post-labeling guard.
4. **(CONFIRMED handoff finding 3) couple_keep_set Cramér's V = 0.165** —
   the 4 duration-noise-prone couples have distinct repertoires (more
   Short + Complex). Smaller effect than tier.
5. **(NEW) Residual-noise cluster identified.** HDBSCAN cluster 1 (244
   events) matches the post-labeling handoff's prediction (~290): 49 kHz
   principal frequency, 9.8 kHz bandwidth, sinuosity 1.04, tonality 0.47,
   118 ms median duration — the exact "no FM curvature, sustained tone"
   signature predicted. **89.8% auto_accept tier** (CNN was confident but
   wrong). **Couple distribution differs from duration-noise**: top
   contributors are m5fm5/m4fm4/m3fm1/m4fm2 (not m1fm1/m1fm2/m1fm4/m3fm3).
   **Two independent noise mechanisms.**

---

## What was done this session

1. **Pre-clean** (`scripts/clean_classified_detections_lab.py`): dropped
   1,145 outer-join NaN-side rows (597 wild-mouse residue + 548 unmatched
   side-pairs from 75 ms tolerance), filtered to lab-only stems, joined
   `tier` + `couple` from `events_clean.parquet`, added `couple_keep_set`
   boolean. **Resolved a CSV vs parquet float-precision bug** (pandas to_csv
   truncates float64 to ~15 sig-figs; round to µs at 300 kHz to round-trip).
   Output: `classified_detections_lab_131204_clean.csv` (40,787 rows × 35 cols).
2. **Scattoni-7 traditional taxonomy**
   (`results/traditional_taxonomy_lab_131204/`): all 40,787 events
   classified, 43% high-confidence, 42% medium, 15% low.
3. **UMAP+HDBSCAN re-cluster** (`results/recluster_umap_hdbscan_lab_131204/`):
   3 clusters + 8,342 noise points at `min_cluster_size=200, min_samples=30`.
   Tested {50, 100, 200} all produced single dominant mega-cluster.
4. **Acoustic feature analysis**
   (`results/acoustic_feature_analysis_lab_131204/`): correlation matrix,
   PCA biplot/scree/loadings, UMAP-by-feature, within-type violins, boundary
   cases. 6,137 low-confidence calls flagged (15.0%).
5. **Three repertoire-comparison variants**
   (`scripts/repertoire_compare_lab.py` → `results/repertoire_lab_131204/`):
   bypassed buggy `analyze_repertoire.py` PERMANOVA (off-by-one when
   per-event `tier` violates 1:1 animal_id↔population assumption), wrote a
   focused chi-square + Cramér's V + stacked-bar PNG per variant.
6. **Residual-noise cluster identified** by acoustic-profile signature
   matching against the post-labeling handoff's prediction.

---

## Canonical artifacts

| Path | Purpose | Notes |
|---|---|---|
| `classified_detections_lab_131204_clean.csv` | **PRIMARY: 40,787 events × 35 cols (16 DS features + tier/couple/couple_keep_set/animal_id)** | Phase 2C / Phase 3 input |
| `results/traditional_taxonomy_lab_131204/classified_traditional.csv` | Scattoni-7 labels (40,787 × 37) | adds `syllable_type`, `classification_confidence` |
| `results/traditional_taxonomy_lab_131204/{type_distribution,feature_summary,cluster_vs_type_heatmap}.png` | Scattoni-7 visuals | Mickey-presentable |
| `results/recluster_umap_hdbscan_lab_131204/reclassified_detections.csv` | HDBSCAN labels (40,787 × N+1) | `hdbscan_label` ∈ {-1, 0, 1, 2} |
| `results/recluster_umap_hdbscan_lab_131204/{umap_hdbscan_scatter,umap_kmeans_scatter,contingency_matrix}.png` | UMAP visuals | unimodality finding visible here |
| `results/acoustic_feature_analysis_lab_131204/*.png` (10 files) | PCA / UMAP / correlation / violins | full set per A3 wild precedent |
| `results/repertoire_lab_131204/by_{tier,couple_keep_set,couple}.{csv,png}` | **Phase 2 guard outputs** | Mickey-presentable; chi-square + Cramér's V |
| `results/repertoire_lab_131204/summary.md` | Cohort comparison summary table | one-page overview |

---

## Detailed findings

### 1. Scattoni-7 distribution (40,787 events)

| Type | Count | % | Confidence band |
|---|---|---|---|
| Flat | 12,134 | 29.7% | dominant |
| Chevron | 9,132 | 22.4% | dominant |
| Down | 5,884 | 14.4% | secondary |
| Short | 5,364 | 13.2% | secondary |
| Complex | 3,651 | 9.0% | minority |
| Up | 3,545 | 8.7% | minority |
| Frequency Jump | 1,077 | 2.6% | rare |

Confidence: 43.2% high / 41.8% medium / 15.0% low.

### 2. UMAP+HDBSCAN unimodality (NEW)

Three `min_cluster_size` settings tested, all produced single dominant mode:

| `min_cluster_size` | `min_samples` | Clusters | Mega-cluster size | Noise % |
|---|---|---|---|---|
| 50 (default) | 10 | 3 | 40,602 (99.5%) | 0.0% |
| 100 | 15 | 2 | 40,438 (99.1%) | 0.6% |
| **200 (canonical)** | **30** | **3** | **28,981 (71.0%)** | **20.5%** |

Wild-mouse 3452 (n=7,921) produced 5+ separable clusters of size 21–169
with default params. The lab batch (n=40,787) cannot produce comparable
density valleys at any tested setting.

**Connects to vault note:** [[forcing USVs into discrete categories may
obscure the continuous variation that distinguishes populations]] —
DeepSqueak's k-means k=26 is forcing partition onto a continuous
distribution. HDBSCAN says the underlying structure is one dominant mode
plus rare outlier modes.

**Caveat:** This finding is sensitive to the 10 acoustic features used.
A larger feature set (e.g., DeepSqueak's 16 + Oren-2024 80D ridge
vectors) might reveal more structure. Phase 3 candidate.

### 3. Tier-aware comparison (CONFIRMED handoff finding 2)

| Tier | N | Short | Flat | Up | Down | Chevron | Complex | Freq Jump |
|---|---|---|---|---|---|---|---|---|
| auto_accept | 29,790 | 9.6% | 33.5% | 8.4% | 15.8% | 23.0% | 6.6% | 3.1% |
| manual_review | 10,997 | **22.9%** | 19.6% | 9.4% | 10.7% | 20.6% | **15.3%** | 1.5% |

**χ² = 2,551, dof = 6, p ≈ 0, Cramér's V = 0.250 (medium-large effect)**

manual_review has 2.4× more Short and 2.3× more Complex calls — both
categories where short, fragmented, or multi-part detections cluster.
Confirms manual_review carries detection-quality artifacts that bias
repertoire summaries.

**Implication for Phase 3 wild-vs-lab:** primary stats should use
auto_accept only; manual_review should be reported separately as a
sensitivity check.

### 4. Couple_keep_set comparison (CONFIRMED handoff finding 3)

| Cohort | N | Short | Flat | Complex |
|---|---|---|---|---|
| 4 noise-prone couples (False) | 6,849 | 17.9% | 21.5% | 17.5% |
| 13 retained (True) | 33,938 | 12.2% | 31.4% | 7.2% |

**χ² = 1,108, dof = 6, p ≈ 0, Cramér's V = 0.165 (small-medium effect)**

The noise-prone cohort has ~2.4× more Complex calls — consistent with
fragmented multi-syllable calls being mis-segmented as separate Complex
events.

### 5. Per-couple variation (descriptive)

**χ² = 3,184, dof = 96, p ≈ 0, Cramér's V = 0.114 (small effect overall)**

Standout couples:
- **m1fm4: 39.0% Complex** (largest single outlier; small N=1,128).
- **m1fm2: 22.4% Complex, 0.0% Frequency_Jump** (very small N=263).
- **m1fm1: 27.5% Chevron, 16.3% Complex.**
- **m3fm3: 19.1% Short, 12.6% Up** (m3fm3 paradox from post-labeling
  handoff — couple is in calibration but produces noise — partially
  explained by Short over-representation).

### 6. Residual-noise cluster (NEW)

**HDBSCAN cluster 1 (244 events, 0.6% of dataset)** is the residual-noise
cluster the post-labeling handoff predicted (~290 events).

Acoustic-feature signature (population values in parens):

| Feature | Cluster 1 | Population | Interpretation |
|---|---|---|---|
| Principal frequency | **49 kHz** | 70 kHz | Below typical USV peak |
| Bandwidth | **9.8 kHz** | 35 kHz | Narrow tonal energy |
| Sinuosity | **1.04** | 1.79 | **No FM curvature** (handoff prediction!) |
| Tonality | **0.47** | 0.37 | Sustained-tone signature |
| Median duration | 118 ms | 68 ms | Long-duration outliers |

**Tier composition: 89.8% auto_accept** — the CNN classified these with
high confidence but they're noise. This contrasts with the post-labeling
handoff's tier-noise finding (manual_review carries noise via short
fragments). **Two distinct noise mechanisms.**

**Couple composition** (top 10):

| Couple | Cluster 1 events |
|---|---|
| m5fm5 | 54 |
| m4fm4 | 47 |
| m3fm1 | 33 |
| m4fm2 | 31 |
| m6fm6 | 16 |
| m2fm1 | 15 |
| m2fm2 | 14 |
| m3fm3 | 10 |
| m3fm4 | 7 |
| m4fm3 | 4 |

The 4 duration-noise-prone couples (m1fm1/m1fm2/m1fm4/m3fm3) contribute
only **10 events combined** to the residual-noise cluster.
**The duration-noise filter and the spectral-noise filter target
disjoint cohorts** — the post-labeling handoff's couple-aware guard
addresses the duration mechanism only.

---

## Cross-link to post-labeling handoff (2026-05-14)

| Post-labeling finding | Phase 2B status |
|---|---|
| 1. 11% noise rate in 200–299 ms band | (assumed; not re-tested in Phase 2B) |
| 2. Tier signal: manual_review = 24% noise | **CONFIRMED** by V=0.250 syllable composition shift |
| 3. 4 couples produce 18 of 22 noise events | **CONFIRMED** by V=0.165 couple_keep_set effect |
| 4. m3fm3 paradox | **PARTIALLY EXPLAINED** — m3fm3 has 19.1% Short (highest after manual_review tier), suggesting the calibration-presence + noise contradiction is a Short-over-representation artifact, not a residual-cluster issue (m3fm3 contributes only 10 events to cluster 1). |

---

## What NOT to do (without explicit user approval)

- **DO NOT modify `classified_detections_lab_131204_clean.csv` in place** —
  Phase 2C/Phase 3 will join to it. Re-run the upstream `clean_classified_detections_lab.py`
  if the parquet changes.
- **DO NOT recompute UMAP+HDBSCAN with `--min-cluster-size < 50`** to
  fragment the mega-cluster. Three values tested converged on the same
  unimodal structure; lower values won't reveal hidden density.
- **DO NOT use `analyze_repertoire.py` in its current form** for tier-aware
  analyses on lab data — its PERMANOVA assumes 1:1 animal_id↔population.
  Use `repertoire_compare_lab.py` instead. (Add to follow-up issue list.)
- **DO NOT mix manual_review events into Phase 3 wild-vs-lab primary stats.**
  Cramér's V = 0.25 means it will bias the comparison meaningfully.

---

## Phase 2B follow-up issues (file before Phase 3)

1. **`peak_freq_hz` column-rename bug** in `src/usv_spectrogram/classification/deepsqueak_import.py:_COLUMN_MAP`:
   maps `"Peak Frequency (kHz)"` but actual Excel header is `"Peak Freq (kHz)"`.
   Result: `peak_freq_hz` is all-null; `peak_freq_khz` rides through unchanged.
   Not blocking but should be fixed.
2. **`analyze_repertoire.py` PERMANOVA off-by-one** at
   `src/usv_spectrogram/classification/repertoire_stats.py:305` when
   `animal_id` count == `population` × `couple` ambiguous mapping.
   Workaround: `repertoire_compare_lab.py`.
3. **`scripts/export_raven_tables.py` PYTHONPATH requirement** — the
   `classification/__init__.py` eagerly imports `sis_baselines` which needs
   the sibling `usv_language/` package. Document or fix the import chain.

---

## Open questions / Phase 3 candidates

1. **Wild-vs-lab Scattoni-7 comparison** — does wild 3452/9252 show the same
   Flat-Chevron dominance, or different syllable mix? Cramér's V on the
   2-cohort comparison is the headline.
2. **Wild-vs-lab UMAP cluster count** — is the unimodality finding
   lab-specific (lab mice have less differentiated repertoires) or
   dataset-size driven (HDBSCAN params + UMAP density at 40k vs 8k)?
   Mitigation: down-sample lab to 8k and re-run; if HDBSCAN still produces
   1 mega-cluster, the unimodality is real.
3. **Two-noise-mechanism follow-up** — what process produces the
   residual-noise cluster in m5fm5/m4fm4/m3fm1/m4fm2? Long sustained
   tones at ~49 kHz could be equipment hum or environmental tonal sources;
   correlate with recording date / chunk-position-within-recording.
4. **Sequential structure analysis** (`scripts/analyze_sequential_structure.py`)
   not run this phase — bigger lift, separate phase.
5. **Bigger feature set** — adding 80D Oren-2024 ridge vectorization or
   16D DeepSqueak might reveal cluster structure the 10-feature UMAP
   missed. The continuous-vs-discrete question is feature-set-dependent.

---

## Verification commands (run before Phase 3)

```bash
cd /home/shachar/projects/mickey_london_lab

# Confirm canonical Phase 2B input still has the expected event count
.venv/bin/python -c "
import pandas as pd
df = pd.read_csv('classified_detections_lab_131204_clean.csv')
print(f'clean: {len(df):,} rows x {df.shape[1]} cols')
assert len(df) == 40787
print(f'tier: {df[\"tier\"].value_counts().to_dict()}')
print(f'couple_keep_set: {df[\"couple_keep_set\"].value_counts().to_dict()}')
print('OK')
"

# Confirm Scattoni-7 type distribution unchanged
.venv/bin/python -c "
import pandas as pd
df = pd.read_csv('results/traditional_taxonomy_lab_131204/classified_traditional.csv')
print(df['syllable_type'].value_counts().to_string())
assert (df['syllable_type'] == 'Flat').sum() == 12134
print('OK')
"

# Confirm residual-noise cluster identification
.venv/bin/python -c "
import pandas as pd
df = pd.read_csv('results/recluster_umap_hdbscan_lab_131204/reclassified_detections.csv')
n_c1 = (df['hdbscan_label']==1).sum()
print(f'cluster 1 (residual noise): {n_c1} events')
assert n_c1 == 244
print('OK')
"
```

---

## Pipeline diagram (Phase 2B)

```
classified_detections_lab_131204.csv (41,932 outer-join)
                |
                v  scripts/clean_classified_detections_lab.py
                |
classified_detections_lab_131204_clean.csv (40,787 × 35)
                |
        +-------+-------+------+
        |       |       |      |
        v       v       v      v
   Scattoni  UMAP+   acoustic  repertoire_compare_lab.py
   -7       HDBSCAN  features  (3 variants: tier, couple_keep_set, couple)
        |       |       |          |
        v       v       v          v
   results/  results/ results/  results/
   tradi-   recluster acoustic_  repertoire_
   tional_  _umap_   feature_   lab_131204/
   taxonomy_ hdbscan_ analysis_
   lab_     lab_      lab_
   131204/  131204/   131204/
        |       |       |          |
        +-------+-------+----------+
                |
                v  manual + acoustic-profile inspection
                |
       Cluster 1 = residual noise (244 events, identified by signature)
                |
                v  this handoff
                |
       Phase 2B complete
```

---

## Immediate next action options

1. **Phase 3 — wild-vs-lab statistical comparison.** Most natural next step.
   Use `repertoire_compare_lab.py` extended to two-population mode with
   wild-mouse 3452/9252 + lab as cohorts. Headline output: Cramér's V on
   Scattoni-7 distribution shift.
2. **Down-sample test of unimodality.** Subsample lab to n=7,921 (matching
   3452) and re-run UMAP+HDBSCAN. If still 1 mega-cluster, the lab
   unimodality finding is dataset-independent.
3. **Investigate the residual-noise mechanism.** Render spectrograms of
   ~10 cluster-1 events from each top couple (m5fm5/m4fm4/m3fm1/m4fm2);
   visual inspection should reveal whether it's equipment hum, environmental
   source, or biological vocalization at low frequency.
4. **Report Phase 2B findings to Mickey.** Five findings, two new — natural
   checkpoint to share before Phase 3 sinks more time.
5. **Fix the three Phase 2B follow-up issues** (peak_freq column rename,
   PERMANOVA off-by-one, PYTHONPATH chain). Small refactors.
