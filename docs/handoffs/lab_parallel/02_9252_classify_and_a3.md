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
- [ ] `results/traditional_taxonomy_9252/classified_traditional.csv` exists with 271 rows (or fewer if some lacked extractable features — document attrition)
- [ ] `results/recluster_umap_hdbscan_9252/reclassified_detections.csv` exists
- [ ] `data/corpus_facts/9252.json` exists, schema matches 5970.json
- [ ] `results/acoustic_feature_analysis_9252/` populated
- [ ] `docs/handoffs/lab_parallel/02_RESULTS_9252_rate_anomaly.md` written with at least 3 hypothesis evaluations
- [ ] Commit SHA recorded

## Decision-needed signals

- If feature extraction fails for >20% of events
- If manual sanity check reveals the CNN missed obvious USVs (suggests model retraining needed before lab data)
- If noise_floor_p90 in 9252 is markedly higher — affects FP filter applicability to lab data too

## Result section

- Commit SHA:
- Events successfully classified:
- Rate anomaly conclusion:
- Decision-needed flags:
