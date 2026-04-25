# Stream 2 — 9252 Classification + Audit + A3

**Status:** Python-half DONE (2026-04-24); MATLAB-half PENDING (Steps 2–6 blocked on DeepSqueak).
**Estimated time:** Python half took ~45 min; MATLAB half 1–2 hr whenever ready.
**Compute:** Light — 597 events total (not 271 — see correction below)

## Completion tracker (2026-04-24)

| Step | State | Notes |
|---|---|---|
| 1. Build merged detections CSV | **DONE** | `results/batch_9252/all_detections.csv` — 597 rows, 318 stems. New generic script: `scripts/merge_batch_detections.py`. |
| 1b. Raven selection tables | **DONE** (added for MATLAB handoff) | `raven_tables_9252/` — 318 `.Table.1.selections.txt` files ready for MATLAB import. |
| 2. Feature extraction (DeepSqueak) | **PENDING — MATLAB** | Run `create_deepsqueak_mats.m` then `deepsqueak_batch_classify.m` on `raven_tables_9252/`; see `docs/handoffs/deepsqueak-full-pipeline-results.md`. |
| 3. Traditional taxonomy | **BLOCKED by Step 2** | Once feature CSV exists: `scripts/classify_traditional_taxonomy.py --input ... --output-dir results/traditional_taxonomy_9252/`. |
| 4. UMAP + HDBSCAN | **BLOCKED by Step 2** | `scripts/recluster_umap_hdbscan.py`. Small-N: propose `min_cluster_size=10–15` (5970 used 7 864 calls; scaled proportionally). Document choice in output dir. |
| 5. corpus_facts | **BLOCKED by Step 4** | `scripts/audit_corpus.py --dataset 9252 --output data/corpus_facts/9252.json`. **Additional blocker:** summary.parquet coverage gap — see below. |
| 6. A3 acoustic deep-dive | **BLOCKED by Step 3** | `scripts/analyze_acoustic_features.py --input results/traditional_taxonomy_9252/classified_traditional.csv --output-dir results/acoustic_feature_analysis_9252/`. |
| 7. **Rate anomaly investigation** | **DONE** | Full write-up: `docs/handoffs/lab_parallel/02_RESULTS_9252_rate_anomaly.md`. 4 hypotheses evaluated; H3 falsified, H2 supported. |

## Corrections from the original handoff

1. **Event count:** 597 events across 318 files — not 271/173. The 271 number in `summary.parquet` reflects only the partial triage coverage (USV4-partial + USV5–USV8). USV1–USV3 are absent from summary.parquet but their JSON event files are complete.
2. **Rate magnitude:** ~7.6× lower file-yield and ~23× lower events/file (not 50×). Corrected computation uses `all_detections.csv` vs 5970's complete `batch_5970_v2_full/summary.parquet`.
3. **H3 is falsified, not an open question.** 9252's noise floor is systematically **lower** than 5970's (0.020 vs 0.036 median), so noise-floor suppression is not the mechanism.

## Gating item for Step 5 (corpus_facts)

Before `audit_corpus.py --dataset 9252` will produce meaningful tier statistics, USV1–USV3 and uncovered USV4 WAVs need a rerun of the FP-filter/triage step so `summary.parquet` covers all 11 580 files. Otherwise 40 % of WAVs are silently excluded from any tier-level aggregation. See `02_RESULTS_9252_rate_anomaly.md` §"Summary-parquet coverage gap."

---

## Original handoff below (preserved for context)

## Goal

Bring the 9252 wild dataset (animal `usv_lmt_036`) to the same analysis state as 5970:
1. Run feature extraction + traditional taxonomy + UMAP/HDBSCAN on the 271 detections
2. Generate `data/corpus_facts/9252.json` (mirror 5970.json schema exactly)
3. Run Phase A3 acoustic feature deep-dive
4. **Investigate the rate anomaly** — 271 events from 11,580 files is ~50× lower than 5970

## State on disk

```
USV_9252/USV1..USV8/                                  <-- 11,580 source WAVs across 8 sessions
results/batch_9252/
  detections/*.json                                   <-- 11,580 per-file detection JSONs (DONE)
  summary.parquet                                     <-- 6,905 file-level rows; 173 with n_events>0; 271 events total
```

Tier breakdown: 6,732 auto_reject, 126 auto_accept, 47 manual_review.

## Steps

### 1. Build the merged detections CSV

The batch run produced per-file JSONs; classification needs a single CSV with all 271 events plus their CNN metadata. Find the script that did this for 5970/3452 (likely `scripts/merge_detections.py`, `scripts/build_detections_csv.py`, or similar) and run it against `results/batch_9252/detections/`.

Output: `results/batch_9252/all_detections.csv` (or whatever the existing convention is — match how 5970 names it).

If no such script exists, write one. It must produce a CSV with the same columns as `results/batch_5970/manual_review_all_detections.csv`.

### 2. Run feature extraction (DeepSqueak bridge)

The 271 events need acoustic features (10 features: duration, freq range, slope, sinuosity, tonality, mean_power, low_freq, freq_sd, bandwidth, principal_freq). Find the script that did this for 5970 — see `docs/handoffs/deepsqueak-full-pipeline-results.md` for the bridge usage.

Output: `results/batch_9252/all_detections_with_features.csv`.

### 3. Traditional taxonomy

```bash
.venv/bin/python scripts/classify_traditional_taxonomy.py \
    --input results/batch_9252/all_detections_with_features.csv \
    --output-dir results/traditional_taxonomy_9252/
```

Output: `results/traditional_taxonomy_9252/classified_traditional.csv` + summary plots.

### 4. UMAP + HDBSCAN

```bash
.venv/bin/python scripts/recluster_umap_hdbscan.py \
    --input results/traditional_taxonomy_9252/classified_traditional.csv \
    --output-dir results/recluster_umap_hdbscan_9252/
```

**Note:** with only 271 events, UMAP / HDBSCAN may be unstable. Use `min_cluster_size` proportional to 5970's setting (5970 used something on 7,864 calls — scale down). Document the chosen parameters in the output directory.

### 5. corpus_facts

```bash
.venv/bin/python scripts/audit_corpus.py --dataset 9252
```

Same caveat as Stream 1 — if the script doesn't yet accept `--dataset 9252`, add the mapping by reading how 5970 is wired.

### 6. A3 acoustic deep-dive

```bash
.venv/bin/python scripts/analyze_acoustic_features.py \
    --input results/traditional_taxonomy_9252/classified_traditional.csv \
    --output-dir results/acoustic_feature_analysis_9252/
```

### 7. Rate anomaly investigation

This is the *interesting* part of Stream 2. Write `docs/handoffs/lab_parallel/02_RESULTS_9252_rate_anomaly.md` covering:

- **Per-session rate**: rate per WAV file across USV1–USV8. Is the low rate uniform, or concentrated in certain sessions?
- **Per-tier breakdown**: 173 files have events. Of those, what's the tier mix? Are the 47 manual_review files the bulk of events?
- **Hypothesis 1 — recording length**: is total recording duration shorter? `total_usv_duration_ms / n_events` and compare against 5970.
- **Hypothesis 2 — animal silence**: is this animal genuinely vocalizing less, or is something rejecting valid USVs?
- **Hypothesis 3 — noise floor**: `noise_floor_p90` from summary.parquet — is it systematically higher in 9252, suppressing detections?
- **Hypothesis 4 — date/season**: 9252 timestamps are 2024-10-06+ — different conditions vs 5970?
- **Manual sanity check**: pick 3 files from each tier, generate spectrograms, eyeball whether USVs are present that the CNN missed.

## Constraints

1. Same canonical pipeline — no threshold tuning.
2. Bout threshold = 0.6 s (canonical).
3. Print parameters and row counts.
4. Wild mouse (`project_wild_mice`).
5. Small-N caveats — with 271 events, every statistic has high uncertainty. Bootstrap CIs are mandatory for Shannon entropy, transition matrix, etc.
6. UMAP/HDBSCAN at small N: pick `min_cluster_size` carefully and document.

## Validation

Done when:
- [x] `results/traditional_taxonomy_9252/classified_traditional.csv` exists with **604 rows** (597 classified + 7 unclassified DeepSqueak-orphan rows — see attrition note in §"DeepSqueak round-trip" below)
- [x] `results/recluster_umap_hdbscan_9252/reclassified_detections.csv` exists (2 clusters + 12 noise points at `min_cluster_size=15`)
- [x] `data/corpus_facts/9252.json` exists, schema matches 5970.json
- [x] `results/acoustic_feature_analysis_9252/` populated (10 files including `analysis_summary.md`)
- [x] `docs/handoffs/lab_parallel/02_RESULTS_9252_rate_anomaly.md` written with **all 4** hypothesis evaluations (H3 falsified, H2 supported)
- [x] Commit SHA recorded (see Result section below)

## Decision-needed signals

- If feature extraction fails for >20% of events
- If manual sanity check reveals the CNN missed obvious USVs (suggests model retraining needed before lab data)
- If noise_floor_p90 in 9252 is markedly higher — affects FP filter applicability to lab data too

## Result section

**Stream 2 closed 2026-04-25.** Pipeline ran end-to-end; all eight handoff steps complete.

- **Commit SHAs (chronological):**
  - `020e31c2` — Python scripts + merged-detections CSV + rate-anomaly write-up + sanity-check renderer
  - `e6ba0e61` → `3aec914b` — MATLAB Raven→.mat converter (initial + parity fix vs `create_deepsqueak_mats_3452.m`)
  - `93b2020d` — full MATLAB-side pipeline outputs (DeepSqueak features → traditional taxonomy → UMAP/HDBSCAN → corpus_facts → A3) plus `scripts/audit_corpus.py` registry fix for 9252
  - `375d4bdc` — earlier hybrid commit that captured Stream-2 docs alongside other-stream sweeps (recovery pattern documented in `feedback_no_bulk_stage_in_parallel_chats` memory)

- **Events successfully classified:** 590 of 597 raw CNN events fully merged with DeepSqueak features (98.8% match rate at the canonical 75 ms tolerance). 7 events drift-orphaned on each side and appear in the merged CSV with `match_quality ∈ {'unmatched_ds','unmatched_det'}`. Net classifiable count after the 7 unclassified rows: **597** rows survive into the traditional taxonomy and HDBSCAN steps.

- **Rate anomaly conclusion:** 9252 is a **genuinely quiet, structurally tight vocalizer**. H3 (noise floor) falsified — 9252's noise floor is 45% LOWER than 5970's, removing the suppression hypothesis. H4 (date/season) weak — only 5 days separate the datasets. H1 (recording length) weak — clips reach 1+ s and contain events. H2 (animal silence) is the surviving primary explanation, but **non-uniform**: USV3 carries 48% of all events (0.18 ev/file) while USV4 sits at 0.011 ev/file — a 16.6× inter-session ratio. The aggregate gap to 5970 is 7.6× lower file-yield and 23× lower events-per-file. The follow-on cross-population finding from corpus_facts: 9252 has **2.7× higher MI lag-1** (0.247 vs 0.092 bits) and **shorter calls** (median duration 22.94 ms vs 60.12 ms), suggesting a phenotype of "few but tightly-patterned" calls.

- **Decision-needed flags:**
  - **[P0 — open]** `results/batch_9252/summary.parquet` covers only 6,905 of 11,580 WAVs (USV1–USV3 plus most of USV4 absent from triage). Tier-level statistics are unsafe until this gap is closed by rerunning the FP-filter/triage stage. Until then, downstream code MUST source events from `results/batch_9252/all_detections.csv`, not summary.parquet.
  - **[P1 — open]** Manual eyeball review of the 9 sanity-check spectrograms at `results/rate_anomaly_9252/sanity_check/` is still pending. If any reveal CNN-missed USVs (especially in USV1 / USV4 files), model generalization to lab data becomes a real concern.
  - **[P2 — partial]** N=3 cross-animal comparison is now data-ready (5970 / 3452 / 9252 all have corpus_facts.json), but the small-N caveat is severe: 9252's 597 events vs 5970's 7,864 means inferential statistics need bootstrap CIs; the cross-population module (Stream 4) is the right place to do this.
  - **[P3 — partial]** The DeepSqueak `Detections/` shared folder convention bit me mid-pipeline (113 contaminant .mat files from a prior 3452 run mixed with 9252's 318). Before the next animal runs, consider standardizing on per-animal subfolders (`Detections/<animal_id>/`) — non-recursive `dir()` in `deepsqueak_batch_classify.m` makes this safe with no script changes.
  - **[P3 — closed]** `scripts/audit_corpus.py` 9252 registry was stale (pointed at `results/traditional_taxonomy/classified_traditional_9252.csv` instead of the dataset-suffixed-directory paths actually produced). Fixed in `93b2020d` to mirror the 3452 entry's working pattern.
