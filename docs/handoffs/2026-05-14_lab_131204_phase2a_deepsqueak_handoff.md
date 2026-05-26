# Handoff — Lab 131204 Phase 2A, DeepSqueak inputs ready

**Date:** 2026-05-14
**Status:** Phase 2A (DeepSqueak input preparation) COMPLETE. MATLAB DeepSqueak run is the next step, owned by the user.
**Previous handoff:** `docs/handoffs/2026-05-14_lab_131204_post_labeling.md` (Phase 2 unblock, canonical events_clean.parquet)
**Next handoff:** `docs/handoffs/2026-05-14_lab_131204_phase2b_classification.md` (TBD, opens after DeepSqueak output exists)

---

## TL;DR

Phase 2A mirrored the wild-mouse (3452, 9252) DeepSqueak input pipeline onto
the lab 131204 batch with **zero changes to existing classification modules**.
Outputs:

- `results/batch_lab_full_softnotch_20260513_1538/detections_clean/` — 25,770
  per-stem JSON files, **41,061 events** (parity with `events_clean.parquet`).
- `raven_tables_lab_131204/` — **6,892 Raven selection tables**
  (`*.Table.1.selections.txt`), header byte-identical to `raven_tables_3452/`.
- `scripts/create_deepsqueak_mats_lab_131204.m` — MATLAB script for the user
  to run interactively. Structurally identical to `create_deepsqueak_mats_3452.m`,
  only path constants swapped.

**User action required:** run the MATLAB sequence on a Windows machine with
DeepSqueak installed (steps below). Phase 2B (Python: import + Scattoni-7 +
UMAP/HDBSCAN + tier/couple-aware analysis) opens once
`deepsqueak_output_lab_131204/classified_Stats.xlsx` exists on the WSL side.

---

## What was done this session

1. **Verified the wild-mouse pipeline shape** — wild-mouse detections JSONs
   already lived at `results/batch_*/detections/`. Lab batch has the same
   layout. Cleanest mirror: filter raw JSONs to match
   `events_clean.parquet`, then run the unchanged `export_raven_tables.py`.
2. **Wrote `scripts/filter_lab_detections_by_clean_events.py`** — reads
   `events_clean.parquet`, builds a per-stem set of `(start_s, end_s)` tuples,
   filters raw JSONs to that membership using exact float equality (verified
   both files store full IEEE-754 float64 precision, no tolerance needed).
3. **Ran the filter** → 41,061 events kept, 502 dropped (exactly the
   long-duration noise events the post-hoc filter targets), 6,892 non-empty
   output files, **zero orphans** (every parquet row matched a raw JSON entry).
4. **Ran `export_raven_tables.py --batch-format`** on the filtered JSONs →
   6,892 Raven tables. Header byte-identical to `raven_tables_3452/`, total
   selection rows across all tables = 41,061.
5. **Created `scripts/create_deepsqueak_mats_lab_131204.m`** — copy of the
   3452 MATLAB script with `ravenDir`, `wavDirs`, banner, and next-steps
   strings swapped to lab paths. Diff vs 3452: only those constants changed;
   loop body byte-identical.

---

## Canonical artifacts (Phase 2A outputs)

| Path | Purpose | Status |
|---|---|---|
| `results/batch_lab_full_softnotch_20260513_1538/detections_clean/` | **PRIMARY: 25,770 per-stem JSONs, 41,061 events, parquet-parity** | use this for any downstream JSON consumer |
| `raven_tables_lab_131204/` | **PRIMARY: 6,892 Raven selection tables for DeepSqueak ingest** | feed to MATLAB step 1 |
| `scripts/filter_lab_detections_by_clean_events.py` | Reproducible parquet→JSON filter (idempotent) | re-run if `events_clean.parquet` changes |
| `scripts/create_deepsqueak_mats_lab_131204.m` | **PRIMARY: MATLAB Raven → DeepSqueak `.mat` converter for lab** | run in MATLAB step 1 |
| `raven_tables_lab_131204/export_summary.json` | Per-stem export tally | audit only |

---

## What the user runs next (MATLAB, on Windows)

> **Pre-req:** MATLAB 2020a + DeepSqueak v3.x installed on a Windows machine
> with access to `\\wsl.localhost\Ubuntu\home\shachar\projects\mickey_london_lab\`.

### Step 1 — Convert Raven tables to DeepSqueak `.mat` files

```matlab
>> run('\\wsl.localhost\Ubuntu\home\shachar\projects\mickey_london_lab\scripts\create_deepsqueak_mats_lab_131204.m')
```

- Reads 6,892 `.txt` files from `raven_tables_lab_131204/`.
- Writes 6,892 `.mat` files to `<DeepSqueak_install_dir>/Detections/`.
- Per-file verification: post-save reload, table-shape and box-dimension sanity check.
- **Expected output:** `=== Done: 6892 saved, 0 failed ===`.
- **Estimated runtime:** ~30-60 min (one `audioinfo()` call per WAV; lab batch is ~60× larger than 3452).

### Step 2 — Headless classification

```matlab
>> deepsqueak_batch_classify( ...
       fullfile(fileparts(which('DeepSqueak')), 'Detections'), ...
       'C:\path\to\DeepSqueak', ...
       '\\wsl.localhost\Ubuntu\home\shachar\projects\mickey_london_lab\deepsqueak_output_lab_131204', ...
       'kmeans')
```

- Applies DeepSqueak's k-means classification (same model used for 3452/9252).
- Writes per-stem `.mat` results to `deepsqueak_output_lab_131204/`.

### Step 3 — Export Excel stats (16 acoustic features per call)

```matlab
>> deepsqueak_export_stats
```

- Produces `deepsqueak_output_lab_131204/classified_Stats.xlsx` — the file Phase 2B will consume.
- Features include `principal_freq_hz`, `low_freq_hz`, `high_freq_hz`,
  `bandwidth_hz`, `slope`, `sinuosity`, `tonality`, `mean_power_db`,
  `freq_std_dev_hz`, `call_length_s` (plus 6 more).

---

## Expected outputs after MATLAB run

| Path | Source | Phase 2B consumer |
|---|---|---|
| `deepsqueak_output_lab_131204/classified_Stats.xlsx` | MATLAB step 3 | `scripts/import_deepsqueak_results.py` |
| `deepsqueak_output_lab_131204/clustering_model_kmeans.mat` | MATLAB step 2 | (reference; not consumed by Python) |
| `deepsqueak_output_lab_131204/<stem>.mat` (×6,892) | MATLAB step 2 | (reference; not consumed by Python) |

---

## Phase 2B opens once Excel exists

Phase 2B will run the three classifiers (Scattoni-7 rule-based, DeepSqueak
k-means already inside the `.xlsx`, UMAP+HDBSCAN re-clustering) mirroring
`results/{traditional_taxonomy,recluster_umap_hdbscan,acoustic_feature_analysis}_3452/`.
Key adds beyond a literal mirror:

1. **Tier-aware reporting** — filter or tag `tier == 'manual_review'` (27% of
   events but 24% noise rate; will bias entropy / JSD / centroids if mixed).
2. **Couple-aware sensitivity** — produce primary stats twice: with all 17
   couples, and excluding the four noise-prone ones (m1fm1, m1fm2, m1fm4,
   m3fm3). Report whether conclusions change.
3. **Residual-noise cluster inspection** — ~290 events (~0.7% of 41,061) are
   likely noise that slipped the `<300 ms` filter. Manually inspect any tight
   UMAP+HDBSCAN cluster with no FM curvature.

Per-couple chunk distribution (top 8) for couple-aware analysis:

| Couple | Chunks |
|---|---|
| m6fm6 | 8,779 |
| m5fm5 | 5,300 |
| m2fm2 | 4,554 |
| m4fm4 | 4,104 |
| m3fm3 | 3,234 |
| m1fm1 | 2,264 |
| m2fm4 | 2,244 |
| m4fm2 | 1,663 |

---

## Verification commands (run before acting in Phase 2B)

```bash
# Confirm filtered JSONs still sum to 41,061
cd /home/shachar/projects/mickey_london_lab
.venv/bin/python -c "
import json
from pathlib import Path
total = sum(len(json.loads(p.read_text())) for p in
            Path('results/batch_lab_full_softnotch_20260513_1538/detections_clean').glob('*.json'))
print(f'detections_clean: {total} events')
assert total == 41061, f'Expected 41,061 got {total}'
print('OK')
"

# Confirm Raven tables still sum to 41,061 across 6,892 files
.venv/bin/python -c "
from pathlib import Path
files = list(Path('raven_tables_lab_131204').glob('*.Table.1.selections.txt'))
rows = sum(len(p.read_text().splitlines()) - 1 for p in files)
print(f'raven_tables_lab_131204: {len(files)} tables, {rows} rows')
assert len(files) == 6892 and rows == 41061
print('OK')
"
```

---

## What NOT to do (without explicit user approval)

- **DO NOT regenerate `events_clean.parquet`** — Phase 2A is pinned to its
  current 41,061-event content. Regeneration would silently desync the
  `detections_clean/` and Raven tables; rerun `filter_lab_detections_by_clean_events.py`
  if it changes.
- **DO NOT modify `scripts/export_raven_tables.py` or
  `src/usv_spectrogram/classification/raven_export.py`** — the entire point of
  Phase 2A's design was to leave them unchanged. Any future lab batch should
  reuse the same `filter_*` script + unchanged exporter.
- **DO NOT recalibrate `data/lab_tonal_lines/lab_131204.json`** — already
  established in the post-labeling handoff (counter-predictive audit flag).
- **DO NOT trust DeepSqueak's k-means cluster IDs as ground truth.** They are
  a starting bridge between detection and Scattoni-7 / UMAP+HDBSCAN in
  Phase 2B; the wild-mouse pipeline treats them as one input among three.

---

## Open issues / notes for Phase 2B

1. **PYTHONPATH note** — `scripts/export_raven_tables.py` and `import_deepsqueak_results.py`
   require `PYTHONPATH=.` from the repo root because
   `src/usv_spectrogram/classification/__init__.py` eagerly imports
   `sis_baselines.py`, which depends on the sibling `usv_language/` package.
   Documented here so Phase 2B doesn't rediscover it.
2. **45-chunk gap** — 6,937 raw JSONs were non-empty before filtering, only
   6,892 after. The 45-chunk gap corresponds to chunks whose **only**
   detections were ≥300 ms noise. Worth tracking if a future investigation
   asks "which chunks have *zero* surviving USVs?".
3. **Phase 2B should compute the acoustic-feature merge using `(stem,
   start_time_s)` timestamp proximity matching with configurable tolerance** —
   this is how `import_deepsqueak_results.py` already works for wild data.
   With exact-float parity (verified Phase 2A), the tolerance can stay at
   the existing default (~2 ms) safely.

---

## Pipeline diagram

```
events_clean.parquet (41,061)            raw detections/ (41,563)
       |                                        |
       +---> filter_lab_detections_by_  --------+
             clean_events.py
                     |
                     v
       detections_clean/ (41,061, 6,892 non-empty)
                     |
                     v
        export_raven_tables.py --batch-format
                     |
                     v
       raven_tables_lab_131204/ (6,892 .txt)
                     |
                     v  [MATLAB, user-run]
       create_deepsqueak_mats_lab_131204.m
                     |
                     v
       <DeepSqueak>/Detections/ (6,892 .mat)
                     |
                     v  [MATLAB, user-run]
       deepsqueak_batch_classify + export_stats
                     |
                     v
       deepsqueak_output_lab_131204/classified_Stats.xlsx
                     |
                     v  [Phase 2B opens here]
       import_deepsqueak_results.py
                     |
                     v
       classified_detections_lab_131204.csv
       (41,061 events with 16 acoustic features each)
                     |
                     +---> classify_traditional_taxonomy.py  (Scattoni-7)
                     +---> recluster_umap_hdbscan.py         (UMAP + HDBSCAN)
                     +---> analyze_acoustic_features.py      (PCA/correlation)
                     +---> tier-aware + couple-aware analyses (Phase 2 guards)
```

---

## Immediate next action

User: run the MATLAB sequence (steps 1–3 above). Estimated total runtime
1–2 hours including the per-file verification in step 1. When
`deepsqueak_output_lab_131204/classified_Stats.xlsx` exists, Phase 2B opens.
