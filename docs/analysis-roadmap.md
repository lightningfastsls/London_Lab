# USV Analysis Roadmap

**Status:** Analysis stage. Detection pipeline complete, features extracted, two classification schemes applied.
**Date:** 2026-04-04
**Datasets:** 5970 (usv_lmt_034, 7,518 calls) | 3452 (usv_lmt_035, batch detection done, classification pending)

---

## What We Have

| Asset | Location | Description |
|-------|----------|-------------|
| Traditional taxonomy | `results/traditional_taxonomy/classified_traditional.csv` | 7 syllable types (Flat, Down, Chevron, Short, Complex, Freq Jump, Up) + confidence |
| UMAP + HDBSCAN | `results/recluster_umap_hdbscan/reclassified_detections.csv` | 3 density clusters (one dominant manifold) + 2D embedding |
| Original k-means | `classified_detections_full.csv` | 27 DeepSqueak clusters (now known to be over-split) |
| Acoustic features | All CSVs above | 10 features: duration, freq range, slope, sinuosity, tonality, power |
| CNN detection metadata | Merged in all CSVs | prob_max, prob_mean, user_action, match quality |
| Batch 3452 detections | `results/batch_3452_sample/`, `results/batch_3452_reviewed/` | Raw detections, not yet classified |
| LMT behavioral tools | `src/usv_spectrogram/lmt/` | PETH analysis, DB loader, synchronizer (ready but unused) |
| Repertoire stats module | `src/usv_spectrogram/classification/repertoire_stats.py` | Shannon entropy, proportions, PERMANOVA, JSD, transitions |
| Information theory | `usv_language/analysis/information_theory.py` | Zipf, entropy rates, idiom detection, burstiness |

### Data Structure

- **5970 dataset** = `usv_lmt_034` (one animal), recorded across 5 sessions (USV1–USV5), ~6,400 WAV files, 1,328 with USVs
- **3452 dataset** = `usv_lmt_035` (second animal), 4 sessions (USV_1–USV_4), ~5,400 WAV files
- Each WAV = one short recording segment. The filename timestamps provide temporal ordering.

---

## Analysis Directions

### 1. Temporal Dynamics (within 5970)

**Question:** How does vocal behavior change over time — within sessions and across the 5-session series?

| Analysis | What it tells you | How |
|----------|-------------------|-----|
| **Call rate over time** | When is the animal most vocal? Circadian rhythm? Habituation? | Parse timestamps from `file` column, bin call counts per hour/session |
| **Type composition over time** | Does the repertoire shift? More Complex calls early, more Flat later? | Stack syllable_type proportions by time bin |
| **Session-to-session comparison** | Is USV1 different from USV5? Learning? Habituation? | Group by USV session (from WAV path), compare type distributions |
| **Bout detection** | Are calls clustered into "bouts" with silent gaps? | Inter-call intervals from `begin_time_s`, threshold-based segmentation |
| **Within-bout structure** | Do bouts have internal structure (e.g., always start with Short)? | Sequence analysis within detected bouts |

**Existing tools:** `information_theory.py` has `BurstinessResult` (CV of inter-event intervals). The `begin_time_s` column gives within-file timing; cross-file ordering requires parsing the filename timestamps.

### 2. Sequential Structure

**Question:** Are call sequences random, or do certain types follow each other more than chance?

| Analysis | What it tells you | How |
|----------|-------------------|-----|
| **Transition matrix** | P(type_B follows type_A) — are there "syntax" rules? | `repertoire_stats.py` already computes this |
| **Bigram/trigram entropy** | How predictable is the next call given the current one? | `information_theory.py` → `entropy_rate()` |
| **Idiom detection** | Are there recurring "phrases" (e.g., Down→Up→Complex)? | `information_theory.py` → `detect_idioms()` with shuffle surrogates |
| **Zipf distribution** | Do a few call types dominate (Zipf-like), or is it uniform? | `information_theory.py` → `zipf_mle()` |
| **Mutual information at lag** | How far back does "memory" extend in the sequence? | `information_theory.py` → `mutual_information_at_lag()` |

**Key insight to watch for:** If transition probabilities are significantly non-uniform, it suggests the animal isn't just randomly producing calls — there's sequential structure, which is a prerequisite for any "syntax-like" interpretation.

### 3. Acoustic Feature Distributions

**Question:** What does the continuous acoustic space look like, beyond categorical labels?

| Analysis | What it tells you | How |
|----------|-------------------|-----|
| **Feature correlations** | Which features co-vary? Is slope related to duration? | Correlation matrix / pair plots of the 10 features |
| **PCA of features** | What are the main axes of variation? | PCA on standardized features, examine loadings |
| **UMAP colored by features** | How does each acoustic dimension map onto the manifold? | Use existing umap_x/umap_y, color by each feature |
| **Within-type variability** | How tight or loose is each traditional category? | Per-type feature distributions (box plots, violin plots) |
| **Boundary cases** | What do low-confidence classifications look like? | Filter `classification_confidence == "low"`, inspect spectrograms |

**Why this matters:** The UMAP result showed calls form a continuum. These analyses characterize the *shape* of that continuum — which dimensions vary most, where the density peaks are, and whether the traditional categories cut the space at meaningful boundaries.

### 4. Cross-Dataset Comparison (5970 vs 3452)

**Question:** Do two different animals (lmt_034 vs lmt_035) have different vocal repertoires?

| Analysis | What it tells you | How |
|----------|-------------------|-----|
| **Run classification on 3452** | Get comparable type labels for the second animal | Run traditional taxonomy + UMAP on 3452 detections |
| **Repertoire comparison** | Different type proportions? Different diversity? | Chi-squared, JSD, Shannon entropy comparison |
| **Acoustic space overlap** | Do they occupy the same region of feature space? | Project 3452 into 5970's UMAP space, or train joint UMAP |
| **Transition comparison** | Different sequential structure? | Compare transition matrices between animals |

**Prerequisite:** The 3452 dataset needs the DeepSqueak feature extraction pipeline run on it first (or extract features directly from WAVs using the CNN detection windows).

**Statistical tools ready:** `repertoire_stats.py` has PERMANOVA, chi-squared, and JSD — designed exactly for population comparisons.

### 5. Behavioral Correlation (LMT Integration)

**Question:** Do USV call types correlate with social behavior events?

| Analysis | What it tells you | How |
|----------|-------------------|-----|
| **PETH (peri-event time histogram)** | Does USV rate spike around specific behaviors? | `lmt/event_triggered.py` — already implemented |
| **Type-specific PETH** | Do *specific* call types increase around specific events? | Extend PETH to filter by syllable_type |
| **USV-behavior temporal coupling** | How tight is the temporal relationship? | Cross-correlation at varying lags |

**Prerequisite:** Need LMT behavioral annotation database (`.sqlite` from Live Mouse Tracker). The `lmt/db_loader.py` and `lmt/synchronizer.py` modules exist but haven't been used in production yet.

**This is potentially the most impactful analysis** — linking vocal output to behavior is the bridge from acoustics to ethology.

### 6. Call Quality & Detection Confidence

**Question:** How reliable are our detections, and does quality vary across call types?

| Analysis | What it tells you | How |
|----------|-------------------|-----|
| **CNN confidence by type** | Are some types harder to detect? | Box plot of `det_prob_max` grouped by syllable_type |
| **Match quality analysis** | Do poorly-matched calls cluster in specific types? | Cross-tab match_quality vs syllable_type |
| **HDBSCAN noise analysis** | What are the 37 noise points? Artifacts or rare calls? | Inspect spectrograms of hdbscan_label == -1 |
| **Confidence calibration** | Does high prob_max actually mean high accuracy? | Compare against user_action (reviewed calls) |

### 7. Power & Tonality Analysis

**Question:** Beyond frequency and duration, what do power and tonality patterns reveal?

| Analysis | What it tells you | How |
|----------|-------------------|-----|
| **Power by type** | Are some call types louder? | `mean_power_db` grouped by syllable_type |
| **Tonality distribution** | Which calls are most tonal vs noisy? | Tonality histogram, UMAP colored by tonality |
| **Power over time** | Does call intensity change across sessions? | mean_power_db binned by timestamp |
| **Tonality vs detection confidence** | Are tonal calls easier to detect? | Scatter: tonality vs det_prob_max |

### 8. Publication-Ready Outputs

| Figure | Purpose |
|--------|---------|
| **Repertoire pie/bar chart** | Traditional type distribution for Methods section |
| **UMAP with type overlay** | Show continuum structure with traditional boundaries |
| **Transition matrix heatmap** | Sequential structure visualization |
| **Temporal raster** | Call timing across recording session (like a spike raster) |
| **Cross-animal comparison** | Side-by-side repertoire distributions (if 3452 is classified) |
| **Example spectrogram panel** | One representative per type (galleries already exist) |

---

## Suggested Priority Order

```
Phase A — Immediate (use existing data + tools)
  A1. Temporal dynamics of 5970 (call rate, type shifts, session comparison)
  A2. Sequential structure (transitions, entropy, idioms)
  A3. Acoustic feature deep-dive (correlations, PCA, UMAP overlays)
  A4. Detection confidence analysis

Phase B — Requires one preprocessing step
  B1. Run traditional taxonomy on 3452 dataset
  B2. Cross-animal repertoire comparison (5970 vs 3452)

Phase C — Requires external data
  C1. LMT behavioral correlation (needs .sqlite behavioral database)
  C2. Type-specific PETH analysis

Phase D — Synthesis
  D1. Publication figures
  D2. Comprehensive repertoire report
```

---

## Open Questions Worth Investigating

1. **Is the "continuous manifold" truly continuous?** The UMAP showed one big blob, but are there density ridges or saddle points within it that suggest subtypes? A finer-grained density analysis (e.g., HDBSCAN with smaller min_cluster_size) could reveal hidden structure.

2. **Do bouts have signatures?** If calls cluster into temporal bouts, does each bout tend to use a consistent subset of types, or does it sample broadly? This could suggest functional "modes" of vocalization.

3. **Is there a calling "warmup"?** Many vocal animals show stereotyped bout-onset patterns. Does the first call in a bout predict the bout's type composition?

4. **Frequency drift:** Does principal_freq_hz drift systematically across the recording (hours/days)? Could indicate fatigue, temperature, or equipment drift.

5. **Short calls — real USVs or detection artifacts?** The 14% Short calls (< 15ms) and HDBSCAN cluster 0 (ultra-short, ~4ms) overlap. Are these genuine vocalizations or noise that passed the CNN? Cross-referencing with det_prob_max would help.

6. **Sinuosity as a complexity metric:** Complex calls (sinuosity > 3.5) are only 8.7% of the repertoire. Does complexity increase in specific temporal or behavioral contexts? This could suggest "arousal-dependent" call complexity.

7. **Are Flat calls a default state?** At 32% of the repertoire, Flat dominates. Is it uniformly distributed in time, or does it fill gaps between more "interesting" calls? Could be background vocalization vs. communicative signals.
