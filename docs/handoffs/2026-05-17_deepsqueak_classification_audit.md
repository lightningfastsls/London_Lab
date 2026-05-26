# Handoff: Audit DeepSqueak's classification quality on our data

**Mission:** Quantify how bad DeepSqueak's classification output is on the 5970 cohort
(and ideally 3452/9252 too). "Classification" here has **two layers**, both worth
auditing:

  1. **Acoustic feature extraction** — the 18 features DeepSqueak computes per call
     (slope, sinuosity, bandwidth_hz, tonality, principal_freq_hz, etc.). These
     feed our rule-based Scattoni-7 taxonomy and every downstream analysis.
  2. **k-means cluster labels** — the 27 clusters DeepSqueak assigns. These appear
     as the `label` column in `classified_detections_full.csv`.

Both are used throughout the analysis pipeline. We have **no ground-truth audit**
of either. The slide-14 audit (2026-05-17) surfaced strong evidence that at least
some features (slope, sinuosity) don't measure what their names imply on real calls.
That motivated this investigation.

---

## Why this matters — what the slide-14 audit just found

Quick context: I just audited the 7 syllable-type thumbnails on slide 14 of Mickey's
USV deck. Mickey reported 5 of 7 thumbnails didn't match their labels. The full
report is at `presentation/figures/thumbs/SLIDE14_AUDIT_REPORT.md`. The short version:

The rule-based Scattoni-7 classifier (`scripts/classify_traditional_taxonomy.py`)
fires correctly given its thresholds, but the **features themselves disagree with
the eye** on several diagnostic cases:

| Case | DS feature value | What the eye sees | Implication |
|------|-----------------|--------------------|-------------|
| Up slot-01 (`2024-09-30_11-18-27_0000003 @ 3.435s`) | slope = **+478** | trace appears flat or downward | **slope feature is unreliable on low-tonality calls** (ton = 0.23) |
| Chevron slot-01 (`2024-09-30_11-19-38_0000015 @ 3.468s`) | sin = 2.97, slope = **-1814** | a downward squiggle, not an arch | **sinuosity conflates clean direction-reversal with squiggle** |
| Complex slot-01 (`2024-09-30_11-20-14_0000022 @ 5.992s`) | sin = **5.69** | one clean smooth arch | **sinuosity > 5 fires on a single direction reversal** |
| FJ slot-01 (`2024-09-30_11-21-01_0000034 @ 3.979s`) | bw = 65 kHz, sin = 1.46 | continuous wide arch, no step | **bw + low-sin does not require a step discontinuity** |

These are individual cases, not statistics. The mission below is to get statistics.

**Bigger picture:** ~14.4% of all calls fall within 20% of a Scattoni threshold
(see `docs/handoffs/a3-acoustic-feature-deep-dive.md`). The above cases suggest
that boundary call labels are not just noise from rule design — they're at least
partly **noise from feature unreliability upstream of the rules**.

---

## What's already known about DeepSqueak on our data

- `docs/handoffs/deepsqueak-full-pipeline-results.md` (2026-04-03) — full pipeline
  run on 5970: **7,518 calls classified into 27 k-means clusters**, merged with CNN
  detections at 75ms tolerance. No quality audit was done at the time.
- `docs/handoffs/a3-acoustic-feature-deep-dive.md` — the 14.4% boundary-case finding,
  plus the table showing Short has slope SD = 2341 (slope is essentially
  uninformative on short calls).
- DeepSqueak's classification pipeline lives in MATLAB scripts:
  - `create_deepsqueak_mats.m`
  - `deepsqueak_batch_classify.m` (k-means, 27 clusters)
  - `deepsqueak_export_stats.m` (writes the 18 features per call to
    `deepsqueak_output_full/classified_Stats.xlsx`)
- Python side: `scripts/import_deepsqueak_results.py --batch-format` produces the
  31-column merged CSV `classified_detections_full.csv`.

---

## Investigation tracks (pick whichever subset fits the time budget)

### Track A — Feature reliability (highest priority, biggest blast radius)

Quantify how often each DS-computed feature agrees with a human-readable definition.

1. **Build a small validated set.** Pick ~200 random calls from 5970, stratified by
   call_length_s (so you cover Short / non-Short evenly). For each, hand-label by
   eye on the spectrogram: shape category (Flat / Up / Down / Chevron / Complex /
   FJ / Short) + visible-rise-or-fall sign + rough fundamental frequency.
2. **Compare to DS features per call:**
   - sign(slope) vs eye-direction → expect agreement; quantify confusion matrix
   - sinuosity buckets [<1.5, 1.5–3, >3] vs eye-shape (flat / arch / multi-segment)
   - bandwidth_hz vs eye-measured high-low frequency span
   - principal_freq_hz vs eye-estimate of fundamental
3. **Output:** confusion matrices + correlation tables. Specifically interested in:
   slope sign-flip rate, sinuosity vs eye-shape, and what threshold of `tonality`
   makes the features trustworthy.

**Hypothesis to test:** features below tonality 0.25–0.30 are unreliable enough
that the Scattoni-7 cascade is essentially noise on those calls. If true, that
recategorizes ~half the dataset as "unreliable label" and re-frames the deck.

### Track B — k-means cluster coherence

Quantify whether DeepSqueak's 27 k-means clusters are internally coherent.

1. **Per-cluster gallery audit.** For each of the 27 clusters, render N=10
   spectrograms (similar to the slide-14 audit — `presentation/figures/thumbs/
   candidates/` is the pattern). Spot-check: do the 10 calls in Cluster_5 look
   like the same kind of call?
2. **Quantitative:** compute silhouette score per cluster, intra-cluster feature
   variance, and the canonical "is the cluster real or random?" tests:
   - Shuffle labels within a stratum and compare intra-cluster distance — if
     real clusters are tighter than random, that's evidence of structure
   - Project to UMAP and color by cluster — well-formed clusters should occupy
     distinct manifold regions (we already did this for HDBSCAN — compare)
3. **Compare to HDBSCAN.** We have HDBSCAN-on-UMAP results in
   `results/recluster_umap_hdbscan/`. HDBSCAN found 1 main cluster (continuum).
   k-means forced 27. Where do the k-means cluster boundaries fall in UMAP space?
   Real boundaries or arbitrary cuts through a continuum?

**Hypothesis to test:** k-means at k=27 is over-clustering a continuum. Most pairs
of clusters will turn out to be statistically indistinguishable on held-out calls.
HDBSCAN's "continuum" answer is closer to truth.

### Track C — Cluster→type stability under reclassification

The Scattoni-7 rules are applied to DS-computed features. If we re-extract features
with a different tool (e.g., direct librosa spectral centroid + zero-crossing for
slope), do the labels stay stable?

1. Pick 200 calls. Re-extract slope, sinuosity, bandwidth from raw audio using a
   custom pipeline. Compare DS vs custom features pointwise.
2. Re-run the Scattoni-7 rules on custom features and compare type assignments.
3. Treat large disagreements as evidence of DS-side computational noise.

This is the most expensive track and probably skip-able unless Track A finds
disagreement big enough to motivate it.

---

## Concrete first steps for the new chat

1. Read the slide-14 audit report end-to-end:
   `presentation/figures/thumbs/SLIDE14_AUDIT_REPORT.md`
2. Sample 200 calls from 5970 stratified by call_length and tonality:
   `classified_detections_full.csv` is the source. Make sure WAVs are reachable.
3. Render inspection spectrograms (re-use `$CLAUDE_JOB_DIR/regenerate_slide14_thumbs.py`
   render functions; they import canonical constants from `usv_spectrogram.corpus`).
4. Build a human-labeling spreadsheet — one row per call, columns for eye-direction,
   eye-shape, eye-bandwidth-est, plus the DS feature values pre-filled.
5. Hand off the spreadsheet to me for labeling, or auto-label using a simple
   computer-vision heuristic on the spectrogram contour (less reliable but
   automatable) as a first pass.

---

## Files & paths to know

| File | Purpose |
|------|---------|
| `classified_detections_full.csv` | Master CSV — CNN + DS features + cluster labels, 7,518 rows |
| `results/traditional_taxonomy/classified_traditional.csv` | Adds Scattoni-7 type + confidence to above |
| `presentation/figures/thumbs/SLIDE14_AUDIT_REPORT.md` | Full audit findings that motivated this |
| `presentation/figures/thumbs/candidates/` | Top-3 candidates per type with feature dumps — reference pattern for cluster audit |
| `scripts/classify_traditional_taxonomy.py` | Rule classifier (consumes DS features) |
| `scripts/import_deepsqueak_results.py` | DS → Python merge pipeline |
| `docs/handoffs/deepsqueak-full-pipeline-results.md` | Original DS pipeline run, no audit |
| `docs/handoffs/a3-acoustic-feature-deep-dive.md` | 14.4% boundary cases, feature SDs |
| `results/recluster_umap_hdbscan/` | UMAP + HDBSCAN result for comparison |
| `data/corpus_facts/5970.json` | Canonical empirical facts for cohort 5970 |
| `src/usv_spectrogram/corpus.py` | Physical constants — sr, USV band, STFT |

---

## What "done" looks like

A short report at `docs/handoffs/deepsqueak-classification-audit-2026-05-17.md` answering:

- Sign-flip rate for `slope` (% of calls where eye-direction disagrees with sign(slope))
- Sinuosity vs eye-shape confusion matrix
- Bandwidth correlation with eye-measured span
- Tonality threshold below which features should not be trusted
- k-means cluster coherence verdict (silhouette + manual spot-check on 5 clusters
  minimum)
- A recommendation: keep DS labels as-is / re-extract features / drop the
  Scattoni-7 rules and rely only on HDBSCAN-continuum / something else

If Track A alone gets done in this chat, that's enough for now — Tracks B and C
can become their own handoffs.

---

## Constraints

- Use `src/usv_spectrogram/corpus.py` for sample rate, USV band, STFT params.
  Do not redeclare. (The corpus-invariant hook will catch you.)
- Don't modify production CNN paths (`scripts/run_batch_detection.py`,
  `app/core/sliding_inference.py`, `postprocessing/`) — this audit is read-only.
- Stay in cohort 5970 unless explicitly extending to 3452/9252; cross-cohort
  comparisons need separate setup.
- WAVs live in `5970 USV/` and `5970_reviewed/` (build via the `build_wav_lookup`
  pattern in `scripts/generate_cluster_gallery.py`).
- 5970 is one wild-mouse *couple* (M+F), not one individual — see
  `notes/project_wild_mice.md`. Doesn't change Track A but matters for any
  inferential statement.
