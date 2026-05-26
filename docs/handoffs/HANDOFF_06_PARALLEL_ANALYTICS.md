# HANDOFF 06 — Parallel Analytics Menu (Lab vs Wild USVs)

**Date:** 2026-05-16
**From:** Orchestrator session (post Q1/Q2/Q3 lab-vs-wild analysis)
**To:** Any fresh Claude session, a collaborator, or a separate machine
**Status:** Menu of independent follow-up analyses. None depend on resolving the pending Mickey-questions blocker.

---

## Executive Context

We just finished the first complete pass of Q1/Q2/Q3 lab-vs-wild USV analysis using `lab_131204` (lab matched + swap couples) against wild-derived cohorts `5970`, `3452`, `9252`. The picture from that pass is **small, qualified differences** — not a dramatic strain effect. Several follow-up analyses fall outside the Mickey-questions blocker (recording-protocol denominator asymmetry) and can be run in parallel sessions while we wait.

This document is the menu. Each entry below is sized for a single Claude session and can be picked up cold.

### Headline findings already in hand

- **Q1 within-bout (corpus-canonical, 0.6 s bout threshold):** lab_matched is close to wild 5970. Median IOI 145 ms (lab) vs 162 ms (5970). Median calls/bout 3 (lab) vs 4 (5970). Within-bout structure barely differs by strain.
- **Q1 calls/min (appendix only):** undefendable until Mickey clarifies recording protocol — lab = scheduled 10-min sessions; wild = LMT-triggered 2.147-s clips at ~3% duty cycle. The denominator asymmetry inflates apparent rate differences.
- **Q2 repertoire:** Scattoni-7 type proportions differ only modestly between strains (Cramér's V 0.10–0.19). Shannon entropy is preserved. **Wild-wild JSD exceeds lab-wild JSD** — individual variability within wild cohorts is larger than the strain gap.
- **Q3 acoustic features:** Small pitch upshift in lab (Cohen's d ≈ 0.4). Per-type, Complex and Chevron pitches differ more strongly (d ≈ 0.85, large). `mean_power_db` and `tonality` are **rig artifacts** (excluded from biology claims).

### What already exists (pointer inventory)

| Artifact | Path |
|---|---|
| Q1 within-bout canonical | `results/q1_within_bout_canonical/` (summary.md, IOI + calls/bout CSVs, `figure_within_bout_canonical.png`) |
| Q1 within-bout alternative | `results/q1_within_bout/` |
| Q1 call rate appendix | `results/q1_call_rate/` (per_couple.csv lives here) |
| Q2 repertoire | `results/q2_repertoire/` (JSD matrix, bootstrap, entropy table, stacked bar) |
| Q3 acoustic features | `results/q3_features/` (per-feature stats, per-type Cohen's d, violins, per-type heatmap, UMAP) |
| Per-cohort UMAP+HDBSCAN | `results/recluster_umap_hdbscan{,_3452,_9252,_lab_131204}/reclassified_detections.csv` |
| Per-cohort Scattoni-7 classification | `results/traditional_taxonomy{,_3452,_9252,_lab_131204}/classified_traditional.csv` |
| Sequential structure (some cohorts) | `results/sequential_structure{,_3452,_9252}/` |
| Lab classified detections | `classified_detections_lab_131204_clean.csv` (also `…_downsampled_n7921.csv` for balanced comparisons) |
| Production CNN | `models/hard_neg_retrain/best_model.pt` |
| Dashboard (visual context) | `results/dashboard/index.html` — served at `http://localhost:8765/` via `results/dashboard/serve.sh` |
| Pending Mickey questions | `reports/lab_131204_phase2b_mickey.html` |
| Corpus constants per cohort | `data/corpus_facts/{5970,3452,9252,lab_131204}.json` |

Recent relevant commits: `cf21c617` (soft-notch wired into corpus-constants), `8b2743d2` (soft-notch wired into batch detection), `d6c58de5` (first `lab_131204.json` calibration).

### Critical caveats inherited (read before starting any analysis)

1. **`mean_power_db` and `tonality` are rig artifacts** across cohorts (different microphones, gains, distances). Do **not** cite either as biology in any new analysis.
2. **Recording-protocol denominator asymmetry**: lab = 10-min scheduled blocks; wild = ~2.147-s LMT-triggered clips with ~3% duty cycle. Any "calls/min" or "session-level" metric is comparing apples to oranges until Mickey confirms the lab schedule.
3. **Bout threshold tension**: corpus-canonical value is 0.6 s (in `data/corpus_facts/<cohort>.json → bout_detection_a2.threshold_s`). A Stream-5 mutual-information plateau argued for 0.25 s on 5970. **Default to 0.6 s** for any new analysis unless you have an explicit reason to deviate, and document the choice at the top of your script.
4. **No single canonical WAV directory** — see `project_wav_directories` memory and the per-cohort batch result folders for file paths.
5. **Lab dataset has a known imbalance**: lab_matched has ~7× more events than lab_swap. Use `classified_detections_lab_131204_downsampled_n7921.csv` for like-with-like comparisons where balance matters.

---

## B. Parallel Analyses — the Menu

Each entry below is standalone. Pick whichever is most useful for the upcoming presentation. Estimated effort is for one focused Claude session including writing + verification.

---

### 1. CNN feature visualization — FLAGSHIP for the presentation

**Title:** Extract penultimate-layer CNN activations for every detected event and project to 2D.

**Why it matters:** Shows what the detection CNN has "learned." A 2D map where lab and wild populations either overlap (one acoustic manifold, two strains) or separate (lab really has a distinct acoustic identity) is the single most compelling visual we could produce for the talk. Currently we only have hand-engineered acoustic features → this would be the model's own representation.

**Inputs:**
- Model checkpoint: `models/hard_neg_retrain/best_model.pt`
- Model class: `src/usv_spectrogram/models/cnn_classifier.py → USVClassifierCNN`
  - Architecture: 3 conv blocks (32/64/128 filters) → `global_pool` → 128-dim flattened vector → 64-unit FC → 1 logit
  - **Hook target:** output of `model.global_pool` (128-dim) **or** output of the first `Linear(128 → 64)` in `model.classifier` (64-dim, richer post-feature mixing). Try both; the 64-dim usually clusters more cleanly.
- Per-event spectrogram patches: regenerate via `src/usv_spectrogram/app/core/sliding_inference.py` (the production inference path already builds the exact tensors the CNN expects — copy its windowing logic). The classified-detection CSVs (`results/traditional_taxonomy*/classified_traditional.csv`) give event start/end + WAV paths.
- Cohort metadata for coloring: `classified_traditional.csv` already carries `cohort`, `syllable_type`, `couple_id`, `max_probability`.

**Method outline:**
1. Verify no existing 128-dim CNN embeddings file is on disk. (Confirmed absent at handoff time: searches for `embeddings_all*` returned no parquet/CSV. The path `analysis/clustering/embeddings_all.csv` referenced in `scripts/clustering_extract_features.py` doesn't currently exist. Do not regenerate without confirming again.)
2. Load `best_model.pt`, register a `forward_hook` on `model.global_pool` (and optionally on the first FC layer).
3. Iterate over all classified events from all 4 cohorts (5970, 3452, 9252, lab_131204). Re-extract the spectrogram patch using `corpus.SAMPLE_RATE_HZ`, `corpus.N_FFT`, `corpus.HOP_LENGTH` (import — do not redeclare).
4. Stack into `(N_events, embed_dim)` and save with metadata columns.
5. Fit one UMAP on all events (stratified sample to ~30k if larger; the lab/wild ratio should be roughly preserved). `umap-learn` with `n_neighbors=30, min_dist=0.1, metric='euclidean'`.
6. Produce a 4-panel figure: same UMAP coordinates colored by (a) cohort, (b) Scattoni-7 type, (c) couple_id, (d) `max_probability` (continuous colormap).
7. Optional — gradient-ascent activation maximization on each of the 128 filters in the final conv block to produce "what does filter k respond to" tiles (use `torch-lucent` or hand-rolled).

**Expected outputs:**
- `results/cnn_features/embeddings_all_cohorts.parquet` — N_events rows × (embed_dim + metadata cols)
- `results/cnn_features/figure_cnn_umap.png` — 4-panel
- `results/cnn_features/figure_cnn_filter_responses.png` — optional, the activation-max tiles

**Estimated effort:** Medium-high (~1–2 sessions). Forward hooks + UMAP are straightforward; the time sink is verifying the spectrogram tensor layout matches what `sliding_inference.py` produces.

**Dependencies:** None — uses model + raw events.

**Risks / pitfalls:**
- The CNN was trained on **binary** USV-vs-noise discrimination, not on syllable types. The penultimate features may collapse all USVs into one blob. If so, that's still a useful finding for the presentation (frame as "binary detector treats lab and wild events as the same acoustic class") and the 64-dim layer often retains more structure than the 128-dim flatten output.
- Re-extracting spectrograms with the wrong frequency band silently corrupts the embedding. Use `corpus.FREQ_MIN_HZ=20000`, `corpus.FREQ_MAX_HZ=120000`.

**Libraries:** `torch`, `umap-learn`, `pandas`, `pyarrow`, `matplotlib`, optionally `torch-lucent` for filter visualization.

---

### 2. Spectrogram exemplars gallery — high visual payoff, lowest effort

**Title:** A 7×2×5 grid of representative spectrograms (Scattoni-7 type × strain × example).

**Why it matters:** Makes the dataset tangible in slides. Audience can see what "Chevron" and "Complex" actually look like and read lab vs wild side by side.

**Inputs:**
- Classified events: `results/traditional_taxonomy_lab_131204/classified_traditional.csv` (lab) and `results/traditional_taxonomy{,_3452,_9252}/classified_traditional.csv` (wild — pool or pick one cohort, recommend `5970`).
- Source WAVs: per `project_wav_directories` memory. For 5970 use the LMT folder linked in `data/corpus_facts/5970.json`; for lab use the path in `data/corpus_facts/lab_131204.json`.
- Spectrogram code: `src/usv_spectrogram/spectrogram.py` (canonical renderer).

**Method outline:**
1. For each of the 7 Scattoni types, filter to events with `max_probability ≥ 0.9` (high-confidence) and median pitch within ±10% of the type median (typical, not edge case).
2. Sample 5 events per (type × strain). Render each with `corpus.N_FFT=512, corpus.HOP_LENGTH=128`, viridis colormap, fixed dB range.
3. Stitch with `matplotlib.gridspec` into a 7-row × 10-column figure (5 lab + 5 wild per row), row labels = type names, top labels = "Wild (5970)" and "Lab (matched)".

**Expected outputs:** `results/exemplars/figure_exemplars_grid.png` (and per-tile PNGs for slide flexibility).

**Effort:** Low (~1 session).

**Dependencies:** None.

**Risks:** Don't use `mean_power_db` for selection (rig artifact). Use frequency-domain selection criteria only.

**Libraries:** `librosa`, `matplotlib`, `pandas`.

---

### 3. Saliency / Grad-CAM maps for CNN detections — presentation gold

**Title:** Heatmaps showing which time-frequency cells of each spectrogram drove the CNN's positive detection.

**Why it matters:** Shows the model is attending to biologically sensible features (the call sweep), not background. Cleanly answers "is the CNN trustworthy?" in one slide.

**Inputs:**
- Model: `models/hard_neg_retrain/best_model.pt`
- Same event patches as Analysis 1.
- 5–10 high-confidence (`max_probability ≥ 0.95`) detections per Scattoni-7 type, balanced across lab and wild.

**Method outline:**
1. Pick the last conv layer in `model.features` (output of the third `Conv2d(64→128)`) as the Grad-CAM target.
2. Use `captum.attr.LayerGradCam` or `torchcam.methods.GradCAM`. Both work directly on `torch.nn.Module`.
3. Upsample the CAM to the spectrogram resolution; overlay with alpha blending on the raw spectrogram (use jet for the CAM, viridis for the spectrogram).
4. Render a 7×10 grid (7 types × 10 examples) and a 2×10 lab-vs-wild contrast strip.

**Expected outputs:**
- `results/cnn_saliency/figure_gradcam_grid.png`
- `results/cnn_saliency/per_event/<event_id>.png` (individual overlays for the slide deck)

**Effort:** Medium (~1 session). The Captum + Grad-CAM boilerplate is short but verifying the upsampling alignment to the spectrogram axes takes care.

**Dependencies:** Analysis 1 helpful but not required (this only needs the model + event patches, not the embeddings).

**Risks:** Grad-CAM requires gradients through eval mode — make sure `torch.no_grad()` is not wrapping the forward pass during attribution. Watch out for `model.eval()` not turning off Dropout in some Captum versions.

**Libraries:** `captum` (preferred — more mature) or `torchcam`, `torch`, `matplotlib`.

---

### 4. Within-bout syllable transitions — deepens Q2

**Title:** Per-cohort Scattoni-type-to-type transition matrices within bouts (canonical 0.6 s threshold). Compare lab vs wild bigrams.

**Why it matters:** Q2 only compared marginal type proportions. Two cohorts can share marginals but use them in different sequences. This adds **syntax** to the repertoire story.

**Inputs:**
- Per-event classification with bout IDs: re-derive bouts from `results/q1_within_bout_canonical/` logic or rebuild from `classified_traditional.csv` + corpus threshold. The bout-IDing helper likely already lives in `src/usv_spectrogram/postprocessing/` — check `triage.py` and `hysteresis.py`.
- Utility candidate: `src/usv_spectrogram/information_theory.py` for entropy rate.

**Method outline:**
1. For each cohort, group events by bout. Build (Type_t, Type_{t+1}) bigram counts. Normalize per-row to get transition probability.
2. Compute conditional entropy H(Type_{t+1} | Type_t) and unconditional H(Type) per cohort. The gap is mutual information — how predictive is the previous type.
3. Compute matrix-wise Frobenius distance and JSD on the flattened transition vector between every cohort pair. Bootstrap CIs (1000 resamples).
4. Plot 4 transition heatmaps (one per cohort) and a difference heatmap (lab − wild_5970).

**Expected outputs:**
- `results/q4_transitions/transition_matrices_per_cohort.csv`
- `results/q4_transitions/figure_transition_heatmaps.png`
- `results/q4_transitions/entropy_table.csv`

**Effort:** Low (~1 session).

**Dependencies:** Q2 done (uses Scattoni-7 labels). No need for Q3.

**Risks:** Bouts with only 1 event contribute zero bigrams — make sure the denominator is "bouts with ≥2 events," not "all bouts." State this in the summary.

**Libraries:** `pandas`, `numpy`, `scipy.spatial.distance`, `seaborn` (heatmaps).

---

### 5. Per-couple repertoire fingerprints — concretizes the Q2 heterogeneity finding

**Title:** Small-multiples figure: one panel per couple (~20 panels), each showing Scattoni-7 proportions.

**Why it matters:** Q2's punchline was "wild-wild JSD > lab-wild JSD." This figure makes that visceral. The audience sees that wild couple A and wild couple B look as different from each other as wild and lab.

**Inputs:**
- `results/traditional_taxonomy*/classified_traditional.csv` (all 4 cohorts pooled)
- `results/q1_call_rate/per_couple.csv` for couple IDs

**Method outline:**
1. Compute per-couple Scattoni-7 proportion vectors (7 values summing to 1).
2. Order couples by hierarchical clustering on JSD between proportion vectors (so similar fingerprints sit next to each other).
3. Render a small-multiples grid: 4 rows × 5 cols of bar charts (or polar/radial charts). Color bars by type; outline border by strain (blue=wild, red=lab).
4. Add a row above the grid showing the JSD-ordered cohort labels.

**Expected outputs:** `results/q2_fingerprints/figure_per_couple_repertoire.png` plus the underlying CSV.

**Effort:** Low (~1 session).

**Dependencies:** Q2 done.

**Risks:** Couples with very low N (some wild couples have <20 events) will have noisy proportions. Annotate each panel with N; consider greying out panels with N<30.

**Libraries:** `matplotlib`, `seaborn`, `scipy.cluster.hierarchy`.

---

### 6. Cross-cohort shared UMAP — unified acoustic space

**Title:** Fit one UMAP on pooled acoustic features from all cohorts. Plot each cohort on the same 2D coordinates.

**Why it matters:** Q3 produced separate UMAPs per cohort, which can't be compared spatially. A shared embedding answers "do lab and wild events live in the same acoustic neighborhood?" visually.

**Inputs:** Combined acoustic features from `results/recluster_umap_hdbscan{,_3452,_9252,_lab_131204}/reclassified_detections.csv`. Use only the **non-rig** features (drop `mean_power_db`, `tonality`).

**Method outline:**
1. Concatenate the 4 reclassified CSVs. Add a `cohort` column.
2. Standardize features (`StandardScaler`). Stratified subsample to ~30k events if needed.
3. Fit one UMAP (`n_neighbors=30, min_dist=0.1`). Save the embedding.
4. Plot: top row faceted by cohort (4 panels, same axes); bottom row colored by Scattoni-7 type (one panel, all events overlaid).

**Expected outputs:**
- `results/q3_shared_umap/shared_umap_embedding.parquet`
- `results/q3_shared_umap/figure_shared_umap.png`

**Effort:** Low-medium (~1 session).

**Dependencies:** Q3 done.

**Risks:** UMAP on rig-contaminated features will silently encode microphone identity. Drop `mean_power_db` and `tonality` *before* fitting, not after.

**Libraries:** `umap-learn`, `pandas`, `pyarrow`, `matplotlib`.

---

### 7. Longitudinal stability of lab couples

**Title:** Do lab matched couples (m1fm1…m6fm6) drift over the ~9 days of recordings, or are their repertoires stable?

**Why it matters:** Tests whether "the lab repertoire" is a fixed phenotype or context-dependent. If repertoires drift, then the lab-wild comparison snapshot might shift depending on which day you sample.

**Inputs:** `classified_detections_lab_131204_clean.csv` — already carries couple, date, and `syllable_type` columns.

**Method outline:**
1. Per couple, group events by recording day. Compute Scattoni-7 proportion vector per (couple, day).
2. Compute JSD between each day and day-0 for that couple → per-couple timeseries.
3. Compute per-day Shannon entropy of the type distribution.
4. Plot: top panel per-couple JSD-vs-day timeseries (6 lines for matched, 6 for swap); bottom panel per-day entropy with couple-level error bars.
5. Test: per-couple linear regression of JSD-vs-day slope ≠ 0 (t-test). Bonferroni-correct across couples.

**Expected outputs:**
- `results/q7_longitudinal/per_couple_day_proportions.csv`
- `results/q7_longitudinal/figure_longitudinal_stability.png`
- `results/q7_longitudinal/summary.md`

**Effort:** Low-medium (~1 session).

**Dependencies:** None beyond the lab classification CSV.

**Risks:** Recording schedules are uneven across couples — some have day-0,1,3,5; others have day-0,4,8. Index by elapsed days from each couple's first session, not absolute calendar date.

**Libraries:** `pandas`, `scipy.stats`, `seaborn`.

---

### 8. Within-bout acoustic feature drift

**Title:** Does `principal_freq_hz` trend up or down across a bout? Does lab differ from wild?

**Why it matters:** Novel question. Could reveal a "warm-up," "arousal escalation," or "fatigue" signature. Even a null result is publishable as "USV acoustic features are stationary within bouts."

**Inputs:** `results/recluster_umap_hdbscan_{5970,3452,9252,lab_131204}/reclassified_detections.csv` (carries acoustic features and event timestamps) + bout IDs (derive from 0.6 s threshold).

**Method outline:**
1. For each bout with ≥5 calls, fit a linear regression of `principal_freq_hz` vs within-bout call index (0, 1, 2, …).
2. Collect per-bout slopes. Plot histogram per cohort.
3. Test: one-sample t-test of slopes ≠ 0 per cohort. Two-sample t-test lab vs wild slopes.
4. Repeat for `duration_ms` and `bandwidth_hz`.

**Expected outputs:**
- `results/q8_within_bout_drift/per_bout_slopes.csv`
- `results/q8_within_bout_drift/figure_slope_distributions.png`
- `results/q8_within_bout_drift/summary.md`

**Effort:** Low (~1 session).

**Dependencies:** None.

**Risks:** Bouts with very few calls (<5) give noisy slope estimates — enforce the threshold. Don't include `mean_power_db` or `tonality` (rig artifacts).

**Libraries:** `pandas`, `numpy`, `scipy.stats`, `statsmodels` (optional for mixed-effects).

---

### 9. Lab swap vs lab matched — within-male partner-novelty comparison

**Title:** The swap pairings reuse males. Compare each male's repertoire with his familiar female (matched) vs novel female (swap). Within-subject design.

**Why it matters:** Directly tests the "partner novelty" hypothesis the swap design was built for. Within-subject paired comparisons are statistically powerful even with N=4 males.

**Inputs:** `classified_detections_lab_131204_clean.csv` — filter to males that appear in both a matched and a swap couple (typically m1–m4).

**Method outline:**
1. Identify males appearing in both matched and swap couples. Build a per-male × condition (familiar vs novel) summary: Scattoni-7 proportions, mean pitch, mean duration, mean bandwidth, calls/bout (within-bout only).
2. Per male, compute familiar-vs-novel difference for each metric.
3. Paired Wilcoxon test across the N males (per metric).
4. Plot: per-male slope plot for each metric, lines connecting familiar→novel.

**Expected outputs:**
- `results/q9_partner_novelty/per_male_familiar_vs_novel.csv`
- `results/q9_partner_novelty/figure_slope_plots.png`
- `results/q9_partner_novelty/summary.md`

**Effort:** Medium (~1 session).

**Dependencies:** Q2 useful (uses Scattoni-7) but can run on raw acoustic features alone.

**Risks:** Recording-protocol denominator asymmetry within the lab is **not** an issue here (both conditions used the same scheduled protocol). But if matched and swap sessions had different durations, normalize before comparing rate metrics.

**Libraries:** `pandas`, `scipy.stats`, `seaborn`.

---

### 10. Sonification — audible exemplars for the talk

**Title:** Pitch-shift selected USVs from 60–90 kHz down to 1–4 kHz so humans can hear them.

**Why it matters:** People remember what they hear. A talk with audible USV exemplars is memorable; a talk with only spectrograms is not.

**Inputs:** 5–10 representative WAV segments per Scattoni-7 type (use the same selections as Analysis 2). Source WAV paths from `data/corpus_facts/<cohort>.json`.

**Method outline:**
1. Extract each event with a 50 ms pad on each side from the source WAV (`soundfile.read` with `start`/`stop` frames).
2. Bandpass filter to 20–120 kHz first (`scipy.signal.butter` + `sosfiltfilt`) to strip rig hum.
3. Pitch-shift down by ~5 octaves using `librosa.effects.pitch_shift(y, sr=300000, n_steps=-60)` **or** simply resample to a much lower sample rate (`soundfile.write(..., samplerate=12000)` while keeping the array as-is — this is the cheap "play slow" trick and preserves more morphology).
4. Optionally apply a soft compressor (`pedalboard.Compressor`) and normalize peak to -3 dBFS.
5. Bundle as MP3 or WAV per (type × strain).

**Expected outputs:**
- `results/sonification/{wild,lab}_<type>_<example_idx>.wav` (and .mp3)
- `results/sonification/README.md` describing the pitch-shift method so credit is honest in the talk

**Effort:** Low (~1 session).

**Dependencies:** None. Analysis 2 helps with example selection but isn't required.

**Risks:** Pitch-shifting via `librosa.effects.pitch_shift` is slow at sr=300000 — the resample trick is much faster and equally honest if you state the method. Do **not** describe these as "the actual mouse sounds" in the talk — they're slowed/shifted representations.

**Libraries:** `librosa`, `soundfile`, `scipy.signal`, optionally `pedalboard` (Spotify) for nicer compression.

---

## C. Orchestration Advice for a Fresh Session

- **Each entry is single-session scope.** None require re-deriving Q1/Q2/Q3 results — those CSVs are already on disk.
- **Recommended serial order if doing them sequentially:**
  1. Analysis 2 (exemplars) — fastest visual win, sanity-checks your spectrogram pipeline
  2. Analysis 1 (CNN features) — flagship; do this once Analysis 2 has confirmed the spectrogram extraction path
  3. Analysis 3 (Grad-CAM) — pairs naturally with Analysis 1
  4. Analyses 4–9 in any order
  5. Analysis 10 last (sonification depends on having picked exemplars)
- **Parallel execution across sessions / machines:** Analyses 1, 2, 3, 10 each touch a different stage of the pipeline (CNN internals / raw spectrograms / CNN gradients / WAV resampling) and are fully independent. Analyses 4, 5, 6 depend only on Q2/Q3 CSV outputs (already on disk) so they're independent of one another too. **Good parallel-launch sets:** {1, 4, 7}, {2, 3, 10}, {5, 6, 8, 9}.
- **State at start of each session:** read `ops/goals.md`, the pending Mickey questions in `reports/lab_131204_phase2b_mickey.html`, the inherited caveats at the top of this handoff, and `data/corpus_facts/<cohort>.json` for the cohort you're working with.

---

## D. Where to Look First — File-Path Index

| What | Path |
|---|---|
| Production CNN | `models/hard_neg_retrain/best_model.pt` |
| CNN class definition | `src/usv_spectrogram/models/cnn_classifier.py` (`USVClassifierCNN`) |
| Production inference path | `src/usv_spectrogram/app/core/sliding_inference.py` |
| Canonical DSP constants | `src/usv_spectrogram/corpus.py` (sample rate, freq band, STFT params — **import, never redeclare**) |
| Per-cohort corpus facts | `data/corpus_facts/{5970,3452,9252,lab_131204}.json` (bout threshold, mic info, recording protocol fields) |
| Per-cohort classified events (Scattoni-7) | `results/traditional_taxonomy{,_3452,_9252,_lab_131204}/classified_traditional.csv` |
| Per-cohort UMAP + acoustic features | `results/recluster_umap_hdbscan{,_3452,_9252,_lab_131204}/reclassified_detections.csv` |
| Lab cleaned per-event CSV | `classified_detections_lab_131204_clean.csv` |
| Lab downsampled (balanced) | `classified_detections_lab_131204_downsampled_n7921.csv` |
| Q1 within-bout (canonical 0.6 s) | `results/q1_within_bout_canonical/` |
| Q1 call-rate appendix | `results/q1_call_rate/` (incl. `per_couple.csv`) |
| Q2 repertoire | `results/q2_repertoire/` |
| Q3 acoustic features | `results/q3_features/` |
| Sequential structure (some cohorts) | `results/sequential_structure{,_3452,_9252}/` |
| Dashboard (visual context) | `results/dashboard/index.html` + `serve.sh` |
| Pending Mickey questions | `reports/lab_131204_phase2b_mickey.html` |
| Spectrogram renderer | `src/usv_spectrogram/spectrogram.py` |
| Information-theory utilities | `src/usv_spectrogram/information_theory.py` |
| Clustering helpers | `src/usv_spectrogram/clustering/` |
| Existing handoff context (lab pipeline) | `docs/handoffs/HANDOFF_05_LAB_DATA_PIPELINE.md` |
| Existing handoff context (analysis next steps) | `docs/handoffs/HANDOFF_06_ANALYSIS_NEXT_STEPS.md` |
| Memory: WAV directories | `~/.claude/projects/-home-shachar-projects-mickey-london-lab/memory/project_wav_directories.md` |
| Memory: CNN model lineage | `~/.claude/projects/-home-shachar-projects-mickey-london-lab/memory/project_cnn_retrain_matched_windows.md` |

---

## E. Standing Methodology Constraints (do not violate)

1. **Sample rate = 300 kHz, USV band = 20–120 kHz, STFT = n_fft 512 / hop 128.** Import from `src/usv_spectrogram/corpus.py`. Never redeclare. Never rely on library defaults.
2. **Bout threshold = 0.6 s canonical** (per `data/corpus_facts/<cohort>.json → bout_detection_a2.threshold_s`). Use this unless you have an explicit, documented reason to use the Stream-5 0.25 s alternative.
3. **`mean_power_db` and `tonality` are rig artifacts** across cohorts — do **not** cite as biology and do **not** include in UMAP/PCA inputs without explicit justification.
4. **Print parameters at the top of every analysis script.** Sample rate, bout threshold, filter cutoffs, sort keys, filter row counts. This is a project-wide invariant.
5. **Execution:** always use `.venv/bin/python` (Linux) or `.venv\Scripts\python.exe` (Windows). Never the system Python.
6. **Verification before reporting success:** `python -m py_compile <file>` then run any relevant test in `tests/`. State "no test coverage" explicitly when there is none.
7. **Never modify test expectations to pass.** Discuss instead.
8. **Never `git add -A` / `git add .`** in this repo. Stage by exact path. (See `feedback_no_bulk_stage_in_parallel_chats` memory — a parallel chat once swept Stream 5 memo into the wrong commit.)
9. **No emojis in code or reports.** Plain text only unless the user explicitly asks for emoji.

---

## Appendix — Quick-Start Snippet for Analysis 1 (CNN feature hook)

For the agent picking up the flagship analysis, here is the minimal hook pattern that should work without surprises:

```python
import torch
from src.usv_spectrogram.models.cnn_classifier import USVClassifierCNN

model = USVClassifierCNN()
state = torch.load("models/hard_neg_retrain/best_model.pt", map_location="cpu")
model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
model.eval()

activations = {}
def grab(name):
    def hook(mod, inp, out):
        activations[name] = out.detach().flatten(start_dim=1).cpu()
    return hook

model.global_pool.register_forward_hook(grab("global_pool"))         # 128-dim
model.classifier[1].register_forward_hook(grab("fc1"))               # 64-dim — usually richer
# (model.classifier is Sequential: Flatten, Linear(128,64), ReLU, Dropout, Linear(64,1))

with torch.no_grad():
    _ = model(spectrogram_batch)   # shape (B, 1, H, W) matching what sliding_inference produces
# activations["global_pool"], activations["fc1"] are now (B, 128) and (B, 64).
```

Verify the patch tensor shape matches `sliding_inference.py`'s windowing **before** running on all events — a silent shape mismatch will produce garbage embeddings without erroring.

---

**End of handoff.** Pick an analysis, read its row in section B end-to-end, then read sections D and E before writing any code.
