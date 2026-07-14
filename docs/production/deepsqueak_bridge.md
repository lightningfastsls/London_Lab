# DeepSqueak Classification Bridge

> **What this is:** The round-trip pipeline that takes CNN-detected USVs, ships them
> through DeepSqueak (MATLAB) for unsupervised acoustic clustering + per-call feature
> extraction, and merges the result back into a single CSV keyed to our detections.
> It produced `classified_detections_full.csv` for the 5970 cohort (7,518 calls matched).
>
> **Status:** Production / complete for 5970, 3452, 9252, and lab_131204 cohorts.
> The Python halves (`raven_export.py`, `deepsqueak_import.py`) are stable and tested.
> The MATLAB middle (clustering + stats) is run by hand on a Windows MATLAB host.
>
> **Production artifact (5970):** `/home/shachar/projects/mickey_london_lab/classified_detections_full.csv`
> (7,921 rows, 31 columns). Other cohorts: `classified_detections_3452.csv`,
> `classified_detections_9252.csv`, `classified_detections_lab_131204.csv` (repo root).
>
> **Two taxonomies live here.** DeepSqueak's k-means writes 27 acoustic *clusters*
> (`label` column). A separate rule-based pass ([§2.6](#26-the-7-type-traditional-taxonomy-vs-the-continuum))
> writes 7 literature-standard *syllable types*. Neither is "ground truth" — see the
> continuum caveat. See also [labels & data](labels_and_data.md) and
> [CNN detection](cnn_detection_pipeline.md).

---

## Pipeline at a glance

```
CNN batch detection            results/batch_<cohort>/detections/<stem>.json   (run_batch_detection.py)
  │
  ▼  export_raven_tables.py --batch-format
Raven selection tables         raven_tables_full/<stem>.Table.1.selections.txt
  │
  ▼  MATLAB: create_deepsqueak_mats.m
DeepSqueak .mat detection files   <DeepSqueak>/Detections/<stem>.mat
  │
  ▼  MATLAB: deepsqueak_batch_classify.m  (k-means or artwarp)
Cluster labels written back to .mat + clustering_model_kmeans.mat
  │
  ▼  MATLAB: deepsqueak_export_stats.m  (called automatically by batch_classify)
deepsqueak_output_full/classified_Stats.xlsx   (18-column feature table)
  │
  ▼  import_deepsqueak_results.py --batch-format --tolerance-ms 75.0
classified_detections_full.csv   (DeepSqueak features + our detection metadata, merged)
  │
  ▼  (optional) classify_traditional_taxonomy.py
results/traditional_taxonomy/classified_traditional.csv   (+ syllable_type, confidence)
```

The two ends are **Python on WSL/Linux**; the middle three steps are **MATLAB on a
Windows host** that reaches the repo over a UNC path (`\\wsl.localhost\Ubuntu\...`).
DeepSqueak does not run headless on Linux, which is why the pipeline is split.

---

## 1. Operate

### 1.1 Prerequisites

| Requirement | Detail |
|-------------|--------|
| Python env | `.venv/bin/python` (Linux/WSL). Needs `pandas`, `openpyxl` (Excel reading). |
| MATLAB host | Windows machine with MATLAB + a DeepSqueak v3.1 checkout (commit `1be0267`). The Functions/ folder must exist. |
| Detections | A completed CNN batch-detection run, e.g. `results/batch_5970/detections/` (flat `<stem>.json` per WAV). Produce these with the 5-flag pipeline in [CNN detection](cnn_detection_pipeline.md). |
| WAV files | The source WAVs must be reachable from the MATLAB host (DeepSqueak re-renders spectrograms from raw audio). WAVs span multiple dirs — see [labels & data](labels_and_data.md). |

> **Re-score before you trust probabilities.** The `5970 USV/*_detections.json` files
> sitting *next to the WAVs* are frozen output from an **older CNN**. Their
> `max_probability` values do NOT match `models/hard_neg_retrain/best_model.pt`
> (example: a window scoring 0.9997 in the old JSON scores ≈0.001 under the current
> model). Always drive this bridge from a fresh `run_batch_detection.py` run
> (`results/batch_<cohort>/detections/`), never from the legacy companion JSONs.

### 1.2 Step 1 — Export Raven selection tables (Python)

Script: `scripts/export_raven_tables.py` → wraps
`src/usv_spectrogram/classification/raven_export.py`.

```bash
.venv/bin/python scripts/export_raven_tables.py \
    --detections-dir results/batch_5970/detections \
    --batch-format \
    --output-dir raven_tables_full
```

| Flag | Default | Meaning / when to change |
|------|---------|--------------------------|
| `--detections-dir` | `USV_Detections/` | Where detection JSONs live. For batch runs point at `results/batch_<cohort>/detections/`. |
| `--batch-format` | off | **Required for `run_batch_detection.py` output** (flat `<stem>.json`, each a list). Omit only for the legacy per-detection-subdirectory layout used by the PyQt6 review app. |
| `--wav-dir` | none | Required **only** when `--batch-format` is off (used to map subdir names → WAV stems). Batch JSONs encode their own stem, so leave unset. |
| `--output-dir` | `raven_tables/` | Where `.txt` tables are written. Convention: `raven_tables_full/` for full runs, `raven_tables/` for smoke tests. |
| `--low-freq` | `25000` (Hz) | Lower frequency bound written into **every** row. DeepSqueak treats the box as a region-of-interest, not a precise bound, so the fixed band is fine. |
| `--high-freq` | `125000` (Hz) | Upper frequency bound written into every row. |
| `--dry-run` | off | Print the WAV↔detection mapping and counts, write nothing. |
| `-v` / `--verbose` | off | DEBUG-level logging. |

**Output:** one tab-delimited file per WAV named `<stem>.Table.1.selections.txt`, plus
an `export_summary.json` in the output dir. Each row is one detection:

```
Selection	View	Channel	Begin Time (s)	End Time (s)	Low Freq (Hz)	High Freq (Hz)
1	Spectrogram 1	1	0.1417	0.2014	25000	125000
```

`Begin/End Time` come from each detection's `core_time` (or `start_time_s`/`end_time_s`
in batch JSONs), rounded to 4 decimals. Selection numbers are 1-indexed after sorting
by start time. Frequency columns are integers.

### 1.3 Steps 2–4 — DeepSqueak classification (MATLAB, Windows host)

Run these from the MATLAB console on the Windows machine. Paths are UNC into WSL.

**Step 2 — Raven TSV → DeepSqueak `.mat`** (`scripts/create_deepsqueak_mats.m`):

```matlab
run('\\wsl.localhost\Ubuntu\home\shachar\projects\mickey_london_lab\scripts\create_deepsqueak_mats.m')
```

This is a **script, not a function** — edit the config block at the top (lines 24–27)
to point `ravenDir`, `wavDirs`, and `outDir` at the right cohort before running:

| Variable | Set to |
|----------|--------|
| `ravenDir` | `...\raven_tables_full` (full) or `...\raven_tables` (smoke, 10 files) |
| `wavDirs` | Cell array of root dirs searched **recursively** (`**`) for WAVs by stem. |
| `outDir` | `fullfile(fileparts(which('DeepSqueak')), 'Detections')` — DeepSqueak's own Detections/ folder. |

For each Raven table it builds a `Calls` table with the box
`[BeginTime_s, LowFreq_kHz, DeltaTime_s, Bandwidth_kHz]` (`create_deepsqueak_mats.m:87-90`),
attaches `audioinfo(wavPath)` metadata, saves as v7.3 MAT, then **reloads and verifies**
(call count, variable names, audio path resolves, no zero-dimension boxes;
`create_deepsqueak_mats.m:111-170`). `Type` is initialized to the placeholder `'USV'`.

**Step 3 — classify** (`scripts/deepsqueak_batch_classify.m`), a real function:

```matlab
deepsqueak_batch_classify( ...
    '\\wsl.localhost\Ubuntu\home\shachar\projects\mickey_london_lab\deepsqueak_mats', ...
    'C:\path\to\DeepSqueak', ...
    '\\wsl.localhost\Ubuntu\home\shachar\projects\mickey_london_lab\deepsqueak_output_full', ...
    'kmeans')
```

Arguments (positional): `matDir`, `dsFolder` (DeepSqueak install), `outputDir`, `method`.

| `method` | Behavior | Source |
|----------|----------|--------|
| `'kmeans'` (default) | Contour-parameter k-means, **k chosen automatically** by elbow method (`kmeans_opt`, maxK=100). Default feature weights freq=2, slope=3, duration=1 (`deepsqueak_batch_classify.m:58-60`). This is what produced the 27 clusters. | `deepsqueak_batch_classify.m:256-312` |
| `'artwarp'` | Adaptive Resonance Theory + DTW; finds cluster count dynamically. Slower; better for non-spherical shapes. Settings at `deepsqueak_batch_classify.m:63`. | `deepsqueak_batch_classify.m:314-339` |

It writes cluster labels back into each `.mat` (`Calls.Type`), saves
`clustering_model_kmeans.mat` (centroids `C` + weights, for reuse), then **automatically
calls Step 4** to export the Excel. Per-call features are computed by DeepSqueak's
`CalculateStats` with `EntropyThreshold=0.215`, `AmplitudeThreshold=0.825`
(`deepsqueak_batch_classify.m:68-69`).

**Step 4 — export Excel** (`scripts/deepsqueak_export_stats.m`) — invoked by Step 3, not
run by hand. Writes `<outputDir>/classified_Stats.xlsx` with **18 columns**:
`File, ID, Label, Accepted, Score, Begin Time (s), End Time (s), Call Length (s),
Principal Frequency (kHz), Low Freq (kHz), High Freq (kHz), Delta Freq (kHz),
Frequency Standard Deviation (kHz), Slope (kHz/s), Sinuosity, Mean Power (dB/Hz),
Tonality, Peak Freq (kHz)` (`deepsqueak_export_stats.m:51-56`). The single consolidated
`classified_Stats.xlsx` (one file for all WAVs) carries a per-row `File` column — this
matters for the merge (see Gotchas).

> `Label` in the Excel = the DeepSqueak cluster name (`Cluster_1`…`Cluster_27`), NOT a
> behavioral label. `Tonality` is exported from `stats.SignalToNoise`
> (`deepsqueak_export_stats.m:122`). Frequency columns are in **kHz** despite the
> `_hz` suffix they later get in Python.

### 1.4 Step 5 — Import & merge back (Python)

Script: `scripts/import_deepsqueak_results.py` → wraps
`src/usv_spectrogram/classification/deepsqueak_import.py`.

```bash
.venv/bin/python scripts/import_deepsqueak_results.py \
    --results-dir deepsqueak_output_full \
    --detections-dir results/batch_5970/detections \
    --batch-format \
    --output classified_detections_full.csv \
    --tolerance-ms 75.0
```

| Flag | Default | Meaning / when to change |
|------|---------|--------------------------|
| `--results-dir` | (required) | Directory of DeepSqueak `.xlsx` output (e.g. `deepsqueak_output_full/`). All `*.xlsx` in it are loaded and concatenated. |
| `--detections-dir` | `USV_Detections/` | The **same** detections used in Step 1. Must match, or timestamps won't line up. |
| `--batch-format` | off | **Required** when detections came from `run_batch_detection.py`. Pair it with the same flag you used in Step 1. |
| `--output` | `classified_detections.csv` | Merged CSV destination. Use `classified_detections_full.csv` for full runs. |
| `--tolerance-ms` | `5.0` | Max `\|ds.begin_time − det.start\|` to accept a match. **The 5.0 default is wrong for this pipeline — use `75.0`.** See Gotcha #1. |
| `--dry-run` | off | Load + match, print counts, write nothing. |
| `-v` / `--verbose` | off | DEBUG logging. |

**Outputs:**
- The merged CSV (default `classified_detections.csv`).
- `import_summary.json` written **next to the CSV** (`deepsqueak_import.py:768`) with
  `total_ds_calls`, `total_detections`, `matched`, `unmatched_ds`, `unmatched_det`,
  `match_rate`, `per_file_counts`.

### 1.5 Reading the merged CSV

`classified_detections_full.csv` has **31 columns** (verified against the file). Grouped:

| Group | Columns | Notes |
|-------|---------|-------|
| DeepSqueak passthrough | `file`, `id`, `label`, `accepted`, `score` | `label` = k-means cluster (`Cluster_N`). `file` = per-row WAV stem from the Excel. |
| DeepSqueak features | `begin_time_s`, `end_time_s`, `call_length_s`, `principal_freq_hz`, `low_freq_hz`, `high_freq_hz`, `bandwidth_hz`, `freq_std_dev_hz`, `slope`, `sinuosity`, `mean_power_db`, `tonality`, `peak_freq_khz`, `peak_freq_hz` | **Frequencies are in kHz** despite `_hz` (`classify_traditional_taxonomy.py:60` warns of this). `bandwidth_hz` = Delta Freq. `peak_freq_hz`/`peak_freq_khz` both appear due to column-name variation across DS Excel versions. |
| Merge metadata | `source_file`, `wav_stem`, `match_quality`, `match_distance_ms` | `source_file` = Excel filename; `wav_stem` = resolved stem used for matching. |
| Our detection metadata | `det_start_s`, `det_end_s`, `det_duration_ms`, `det_index`, `det_prob_max`, `det_prob_mean`, `det_user_action`, `det_json_path` | From the detection JSON. `det_prob_*` are the **current production CNN** scores (because you fed fresh batch detections). |

`match_quality` values: `exact` (distance == 0.0), `fuzzy` (matched within tolerance),
`unmatched_ds` (a DS call with no detection within tolerance), `unmatched_det` (a
detection with no DS call). **For analysis, keep `exact` + `fuzzy` only.** A non-trivial
count of `unmatched_ds` rows means tolerance is too tight or you mixed mismatched
detection/Excel sets.

> **Two duration columns, different meanings.** `call_length_s` is DeepSqueak's
> ridge-derived tonal sweep length; `det_duration_ms` is our hysteresis-event duration.
> They can differ by up to ~10×. Visual/PNG verdicts must use `det_duration_ms` (the
> PNGs show the hysteresis event), not `call_length_s`. See
> [labels & data](labels_and_data.md).

### 1.6 The 7-type traditional taxonomy (vs the continuum)

After the merge you can optionally tag each call with a literature-standard syllable
type (Holy & Guo 2005 / Scattoni et al. 2008) using deterministic feature thresholds —
no ML, no training data:

```bash
.venv/bin/python scripts/classify_traditional_taxonomy.py \
    --csv classified_detections_full.csv \
    --output-dir results/traditional_taxonomy
```

| Flag | Default |
|------|---------|
| `--csv` | `classified_detections_full.csv` |
| `--output-dir` | `results/traditional_taxonomy` |
| `--n-per-type` | `5` (gallery examples per type) |
| `--seed` | `42` |
| `--skip-gallery` | off |

Priority-ordered cascade (first rule that fires wins; `classify_traditional_taxonomy.py:74-131`):

| # | Type | Rule | Threshold const |
|---|------|------|-----------------|
| 1 | Short | `call_length_s < 0.015` (15 ms) | `THRESH_SHORT_DURATION_S` |
| 2 | Complex | `sinuosity > 3.5` | `THRESH_COMPLEX_SINUOSITY` |
| 3 | Chevron | `sinuosity > 1.8` AND `bandwidth_hz > 25.0` (kHz) | `THRESH_CHEVRON_*` |
| 4 | Frequency_Jump | `bandwidth_hz > 55.0` (kHz) AND `sinuosity < 1.8` | `THRESH_FREQJUMP_*` |
| 5 | Up | `slope > 200` | `THRESH_SLOPE_DIRECTIONAL` |
| 6 | Down | `slope < -200` | `THRESH_SLOPE_DIRECTIONAL` |
| 7 | Flat | default (low slope, low sinuosity) | — |

Calls with any NaN feature → `unclassified`. Output adds `syllable_type` +
`classification_confidence` (`high`/`medium`/`low`/`none`) and writes
`classified_traditional.csv`, distribution/feature/cross-tab PNGs+CSVs, and a per-type
spectrogram gallery.

**The two taxonomies disagree on purpose.** UMAP+HDBSCAN on the same 10 features
(`scripts/recluster_umap_hdbscan.py`, `results/recluster_umap_hdbscan/`) finds **one
continuous manifold**, not discrete types: ~96.6% of calls fall in a single cluster.
The 27 k-means clusters and the 7 rule-based types are **discretizations of a
continuum** (consistent with Goffinet et al. 2021). Use the 7 types for cross-study
comparability and publication; do not treat the k-means clusters as biological
categories. See `notes/` clustering record and [labels & data](labels_and_data.md).

### 1.7 Worked example (5970, full run)

1. CNN batch detection → `results/batch_5970/detections/` (1,328 WAVs with USVs, 7,575 detections).
2. `export_raven_tables.py --batch-format --output-dir raven_tables_full` → 1,328 tables.
3. MATLAB `create_deepsqueak_mats.m` → 1,328 `.mat` files.
4. MATLAB `deepsqueak_batch_classify.m ... 'kmeans'` → 27 clusters, 7,864 classified calls, `classified_Stats.xlsx`.
5. `import_deepsqueak_results.py --batch-format --tolerance-ms 75.0 --output classified_detections_full.csv` → **7,518 matched (99.2%)**, mean match distance 20.28 ms, median 18.60 ms.

Result: `classified_detections_full.csv`, 7,921 rows / 31 columns. (Row count =
7,518 matched + 346 `unmatched_ds` + 57 `unmatched_det`. Of the 346 unmatched DS
calls, 289 trace to smoke-test stem overlap and 57 are edge cases. Note: every
matched row is `fuzzy` — there are zero `exact` matches in this run because
DeepSqueak's ridge recompute perturbs every timestamp at least slightly.) Full
record: `docs/handoffs/deepsqueak-full-pipeline-results.md`.

### 1.8 Troubleshooting / Gotchas

1. **`--tolerance-ms 75.0` is mandatory; the 5.0 default silently drops almost
   everything.** DeepSqueak's `CalculateStats` *recomputes* Begin Time from spectrogram
   ridge analysis instead of using the box value we imported, so timestamps drift:
   most ~0.37 ms (STFT bin quantization), some 8–68 ms (ridge shifting). P95=42 ms,
   P99=55 ms, max=74 ms. 75 ms is safe because the minimum inter-call gap is 39.6 ms,
   so there's no cross-match risk (`docs/handoffs/deepsqueak-full-pipeline-results.md` §1).
   At 5 ms you'd mark most calls `unmatched_ds`.
2. **`--batch-format` must match on both ends.** Use it in Step 1 *and* Step 5 when
   detections come from `run_batch_detection.py`. Forgetting it makes the loader expect
   per-detection subdirectories and find nothing.
3. **Don't drive from `5970 USV/*_detections.json`.** Those are an older CNN's frozen
   output; probabilities are stale. Always start from a fresh batch-detection run.
4. **Consolidated Excel needs the per-row `File` column.** The headless pipeline emits
   one `classified_Stats.xlsx` for all WAVs. The importer detects this and uses each
   row's `file` value as `wav_stem` rather than the Excel filename
   (`deepsqueak_import.py:253-260`). If `wav_stem` is wrong, matching collapses.
5. **MATLAB can't `save`/`load`/`writetable` to UNC paths.** All three MATLAB scripts
   write to `tempdir` then `copyfile` to the WSL UNC path
   (`deepsqueak_batch_classify.m:306-311`, `deepsqueak_export_stats.m:151-157`).
   Keep that pattern if you edit them.
6. **Zero-dimension boxes are silently skipped** by both `deepsqueak_batch_classify.m`
   (line 186) and `deepsqueak_export_stats.m` (line 84). A WAV whose detections all have
   zero width/height yields no DS rows → all `unmatched_det`.
7. **Frequency columns are kHz, not Hz.** The `_hz`-suffixed columns hold kHz values
   downstream. `classify_traditional_taxonomy.py` thresholds (25.0, 55.0) are in kHz
   accordingly.
8. **Re-running with overlapping smoke + full sets inflates `unmatched_ds`.** The 5970
   run shows 289 spurious unmatched DS calls from smoke-test stems sharing names. Clear
   stale `.mat`/Excel between runs or use disjoint output dirs.

---

## 2. Internals

### 2.1 Module layout

| Layer | File | Role |
|-------|------|------|
| CLI (export) | `scripts/export_raven_tables.py` | Arg parsing, dry-run, calls `export_raven_tables`. |
| Lib (export) | `src/usv_spectrogram/classification/raven_export.py` (525 lines) | Detection JSON → Raven TSV. `# VAULT:` canary — run `/kcheck` before editing. |
| MATLAB | `scripts/create_deepsqueak_mats.m`, `deepsqueak_batch_classify.m`, `deepsqueak_export_stats.m` | Raven → `.mat` → cluster → Excel. |
| CLI (import) | `scripts/import_deepsqueak_results.py` | Arg parsing, dry-run, calls `import_deepsqueak_results`. |
| Lib (import) | `src/usv_spectrogram/classification/deepsqueak_import.py` (773 lines) | Excel + detections → merged CSV. `# VAULT:` canary — run `/kcheck` before editing. |
| Taxonomy | `scripts/classify_traditional_taxonomy.py` (444 lines) | Rule-based 7-type tagging on the merged CSV. |

Both `raven_export.py` and `deepsqueak_import.py` carry `# VAULT:` canary comments and
`# Run /kcheck before modifying this file.` headers — treat them as constrained.

### 2.2 Raven export — key signatures

- `RavenExportConfig` (`raven_export.py:57-111`) — frozen dataclass. Validates
  `low_freq_hz < high_freq_hz`, non-negative freqs, and that `wav_dir` is present unless
  `batch_format=True`. Defaults: `low_freq_hz=25_000.0`, `high_freq_hz=125_000.0`,
  `output_dir=Path("raven_tables")`.
- `load_detection_json(json_path) -> dict` (`raven_export.py:142-177`) — reads only the
  `core_time` block (`start_s`, `end_s`, `duration_ms`). Raises `ValueError` on bad JSON
  or missing `core_time`. Deliberately ignores `saved_region` (padded extraction window).
- `discover_wav_detection_mapping(detections_dir, wav_dir)` (`raven_export.py:189-258`) —
  per-detection layout. Maps each subdir to a WAV stem by exact, else **longest-prefix**,
  match.
- `discover_batch_detection_mapping(detections_dir)` (`raven_export.py:261-323`) — batch
  layout. Each flat `<stem>.json` is a list; normalizes `start_time_s`/`end_time_s`/
  `duration_s` → `start_s`/`end_s`/`duration_ms`.
- `detections_to_raven_table(detections, low, high)` (`raven_export.py:326-364`) — builds
  the 7-column DataFrame (`_RAVEN_COLUMNS`, line 42), sorted by start, 1-indexed
  `Selection`, times rounded to 4 dp, freqs cast to int.
- `export_raven_tables(config)` (`raven_export.py:367-483`) — orchestrator; writes
  `<stem>.Table.1.selections.txt` (tab-delimited, `\n` line terminator) and
  `export_summary.json`. Batch path delegates to `_export_from_preloaded` (line 486).

`_SKIP_FILENAMES = {"_saved_tracking.json", "detections_summary.csv"}` and
`_SKIP_SUFFIXES = {".png", ".csv"}` (`raven_export.py:38-39`) filter non-detection files;
`_is_detection_json` (line 180) is reused by the importer.

### 2.3 DeepSqueak import — key signatures

- `DeepSqueakImportConfig` (`deepsqueak_import.py:94-128`) — frozen dataclass. Fields:
  `results_dir`, `detections_dir`, `output_path`, `tolerance_ms=5.0`, `batch_format=False`.
  Validates `tolerance_ms > 0`.
- `_COLUMN_MAP` (`deepsqueak_import.py:48-65`) — maps the 16 raw DS Excel headers to
  snake_case. Note `"Label"` **and** `"Type"` both → `label` (DS-version tolerance).
  `_DS_OUTPUT_COLUMNS` (line 68) is the canonical ordered set.
- `_extract_wav_stem(filename)` (`deepsqueak_import.py:165-186`) — strips DS suffixes
  `_DS_SUFFIXES = ("_Detections","_detections","_calls","_Calls","_classified")`.
- `load_deepsqueak_excel(path)` (`deepsqueak_import.py:221-261`) — reads via
  `engine="openpyxl"`, normalizes columns, sets `source_file`, and resolves `wav_stem`
  from the per-row `file` column when present (the consolidated-Excel fix, lines 253-260).
- `load_all_deepsqueak_results(results_dir)` (line 264) — concatenates all `*.xlsx`.
- `_load_detection_json_extended(json_path)` (`deepsqueak_import.py:308-351`) — like the
  raven loader but also pulls `detection_index`, `probabilities.max/mean`, `user_action`,
  `json_path`. `load_detections_for_merge` (line 354) / `load_batch_detections_for_merge`
  (line 406) group by stem.
- `merge_with_detections(ds_df, detections_by_stem, tolerance_ms)`
  (`deepsqueak_import.py:477-614`) — the matcher (§2.4).
- `_resolve_detection_stem_mapping(ds_stems, detection_stems)`
  (`deepsqueak_import.py:617-649`) — exact-then-longest-prefix stem resolution mirroring
  Raven export, so suffixed detection folders still round-trip.
- `export_classified_detections(df, output_path)` (line 693) and
  `import_deepsqueak_results(config)` (line 721, the orchestrator that also writes
  `import_summary.json`).

### 2.4 Matching algorithm (the load-bearing invariant)

`merge_with_detections` does **greedy 1:1 nearest-neighbor** matching per WAV stem
(`deepsqueak_import.py:526-588`):

1. Resolve each DS `wav_stem` to a detection stem (exact, else longest prefix).
2. Within a stem, sort DS calls by `begin_time_s`; for each, pick the nearest *available*
   detection by `|ds.begin_time_s − det.start_s|`.
3. Accept if `best_dist <= tolerance_ms/1000`. `exact` if distance == 0.0, else `fuzzy`.
4. Remove the matched detection from the pool (prevents double-assignment).
5. Leftover detections → `unmatched_det`; DS calls with no match → `unmatched_ds`.
6. Detection stems with no DS results at all → all `unmatched_det` (lines 601-611).

**Invariant:** greedy 1:1 is correct *only because* each stem has the same number of DS
calls and detections (DeepSqueak read our Raven tables) and the min inter-call gap
(39.6 ms) exceeds the max timestamp drift (74 ms). Break either assumption and greedy
matching can mis-pair. This is why tolerance is 75 ms, not larger.

### 2.5 MATLAB internals

- `create_deepsqueak_mats.m` builds `Calls` with box
  `[BeginTime_s, LowFreq_kHz, DeltaTime_s, Bandwidth_kHz]` (lines 87-90), matching
  DeepSqueak's `import_raven_Callback.m:49` variable order. `Type` starts as `'USV'`.
- `deepsqueak_batch_classify.m` Phase 1 (lines 149-250) re-renders each call's
  spectrogram via `CreateFocusSpectrogram` and extracts features with `CalculateStats`.
  Phase 2 (lines 252-344): k-means feature vector = z-scored freq(12) + slope(12) +
  duration(12), weighted 2/3/1; `kmeans_opt` picks k (elbow). Phase 3 writes labels back
  via `UpdateCluster`. Phase 4 calls `deepsqueak_export_stats`.
- `deepsqueak_export_stats.m` maps `stats.SignalToNoise → Tonality` (line 122); skips
  zero-dimension boxes (line 84).

These are pinned to **DeepSqueak v3.1, commit `1be0267`** (stated in both script headers).
A different DeepSqueak version may rename Excel columns — the `_COLUMN_MAP` fallback
(`deepsqueak_import.py:211-217`) will pass unknown columns through with basic
snake-casing, which is why stray columns like `peak_freq_khz` appear alongside
`peak_freq_hz`.

### 2.6 The 7-type traditional taxonomy vs the continuum

`classify_traditional_taxonomy.py` is a pure cascade (`classify_call`, lines 74-131) with
constants at lines 62-71. It is independent of the DeepSqueak k-means labels — it reads
`call_length_s`, `slope`, `sinuosity`, `bandwidth_hz` from the merged CSV. The
cross-tabulation figure (`generate_cross_tabulation`, line 264) shows k-means clusters
scatter across all 7 types — direct evidence the bins are arbitrary cuts of a continuum.
The competing data-driven view (UMAP+HDBSCAN → one manifold) lives in
`scripts/recluster_umap_hdbscan.py`; keep both framings and never present either cluster
set as biological ground truth.

### 2.7 Tests

| File | Count |
|------|-------|
| `tests/test_classification/test_raven_export.py` | 33 |
| `tests/test_classification/test_deepsqueak_import.py` | 23 |
| `tests/test_classification/test_deepsqueak_import_hardened.py` | (hardening set) |

```bash
.venv/bin/python -m pytest tests/test_classification/ -v
```

### 2.8 Where to change things

- **Frequency band in Raven tables:** `--low-freq`/`--high-freq` flags, or
  `RavenExportConfig` defaults (`raven_export.py:84-85`). Cosmetic — DeepSqueak ignores
  exact bounds.
- **Match tolerance:** `--tolerance-ms`. Do not raise above the inter-call-gap floor
  (39.6 ms gap vs 74 ms max drift → 75 ms is the ceiling of safety).
- **New DeepSqueak Excel column:** add to `_COLUMN_MAP` (`deepsqueak_import.py:48`) and
  `_DS_OUTPUT_COLUMNS` (line 68).
- **Clustering method/weights:** `deepsqueak_batch_classify.m:58-60` (k-means weights) or
  `:63` (ARTwarp settings). Changing weights changes the cluster geometry — re-run the
  whole MATLAB middle.
- **Taxonomy thresholds:** `classify_traditional_taxonomy.py:62-71`. These are calibrated
  to the 5970 distributions; recalibrate per cohort if porting.
