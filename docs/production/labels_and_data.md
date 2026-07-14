# Labels & Data — Where Everything Lives

> **What this is.** The "where is everything" map for the USV lab: detection-label
> outputs (`USV_Detections/`), the master classified-detection tables
> (`classified_detections_*.csv`), WAV recording locations, human label sets, and
> the rig-vs-box split for large artifacts.
> **Status.** CURRENT (verified against the working tree 2026-06-21).
> **Production artifact paths.** Wild master table: `classified_detections_full.csv`
> (7,921 rows). Lab master table: `classified_detections_lab_131204_clean.csv`
> (40,787 rows). Human shape labels: `data/manual_shape_labels.csv` (758 rows;
> originally 204 — see [§1.4](#14-human-label-sets)).
> **Sibling docs.** Large non-git artifacts: [DATA_LOCATIONS.md](../DATA_LOCATIONS.md).
> Detection pipeline that produces these files: see `scripts/run_batch_detection.py`
> and `docs/handoffs/v2-full-pipeline-results.md`.

---

## 1. Operate

This component is **data, not a program** — there is no single command to "run". This
section tells you how to *read* each file, what every column means, and the traps that
have bitten people before.

### 1.1 The detection-label store — `USV_Detections/`

`USV_Detections/` is the output of the **PyQt6 desktop app's** detection/review loop
(written by `src/usv_spectrogram/app/core/detection_exporter.py`). Each WAV recording
that has been reviewed gets one subdirectory named after the WAV stem, e.g.
`USV_Detections/2024-09-30_20-59-55_0002986/`.

```
USV_Detections/
├── <wav_stem>/                     # one dir per reviewed recording (ACCEPTED detections)
│   ├── detection_000_0.986s-1.097s.json   # per-detection metadata
│   ├── detection_000_0.986s-1.097s.png    # cropped spectrogram image (the human view)
│   ├── ...
│   ├── detections_summary.csv             # flat table of THIS recording's detections
│   └── _saved_tracking.json               # append-only log of every save action (accept + reject)
├── rejected_detections/            # MIRROR of the above, for detections the user DELETED
│   ├── <wav_stem>/ ...
│   └── delete/                     # scratch bucket for ad-hoc single deletions
└── noise_labeled_files/            # whole-file "this file is pure noise" labels (lab)
    └── 131204_1400_m1fm1_chunk_000.json ...   (127 files)
```

- **Accepted vs rejected.** A detection the reviewer kept stays under
  `USV_Detections/<wav_stem>/`. A detection the reviewer deleted is written under
  `USV_Detections/rejected_detections/<wav_stem>/` instead. The same PNG/JSON naming
  applies in both trees. The `_saved_tracking.json` inside an accepted dir logs **both**
  outcomes with an `output_path` pointing into whichever tree the file landed in (see the
  real example below — note rows whose `output_path` contains `rejected_detections/`).
- The naming convention `detection_{NNN}_{start:.3f}s-{end:.3f}s` is built at
  `detection_exporter.py:77`.

#### Per-detection JSON schema

Written by `DetectionExporter._save_json_metadata` (method defined at
`detection_exporter.py:210`; the `metadata` dict is built at `:235`). Real
accepted example (`USV_Detections/2024-09-30_20-59-55_0002986/detection_000_0.986s-1.097s.json`):

| Field | Type | Meaning |
|---|---|---|
| `detection_index` | int | Index of this detection within the recording's detection list |
| `core_time.start_s` / `core_time.end_s` | float | The detection's tight onset/offset, in seconds from file start |
| `core_time.duration_ms` | float | `(end_s - start_s) * 1000` (`detection_exporter.py:240`) |
| `saved_region.start_s` / `saved_region.end_s` | float | The crop window actually saved — core time **plus context padding** |
| `saved_region.context_ms` | float | Padding added each side. Default **20.0 ms** (`DetectionExporter.__init__`, `detection_exporter.py:21`) |
| `probabilities.max` | float | Max CNN probability over the detection window |
| `probabilities.mean` | float | Mean CNN probability over the window |
| `probabilities.original_cnn_probability` | float \| null | Pre-recalibration probability; `null` for app-detected events that were never re-scored |
| `spectrogram_columns.start_col` / `end_col` | int | STFT frame-column indices of the event in the full spectrogram |
| `user_action` | str \| null | `null` = kept as-is; `"deleted_by_user"` = rejected; other review verbs possible |
| `session.session_id` | str (UUID) | Review-session identifier |
| `session.threshold_preset` | str \| null | Named threshold preset if one was used |
| `session.threshold_high` / `threshold_low` | float | Hysteresis onset/exit thresholds active when saved |
| `timestamp` | ISO-8601 str | When the JSON was written |
| `user_adjusted` *(optional)* | bool | Present only if the reviewer dragged the boundaries (`detection_exporter.py:267`) |
| `original_boundaries` *(optional)* | object | Pre-adjustment `start_s`/`end_s`, present only when `user_adjusted` |

> **Threshold values are context-dependent, not a bug.** Accepted app detections show
> `threshold_high: 0.04, threshold_low: 0.03` (the low-threshold operating point the app
> was run at). The `delete/` scratch example shows `0.6 / 0.4`. These are whatever the
> reviewer's session used; they are NOT the batch pipeline's calibrated thresholds.

#### `detections_summary.csv` (per recording)

Rewritten from the on-disk JSONs by `DetectionExporter._rewrite_csv_summary`
(`detection_exporter.py:277`). Header (verified):

```
wav_file,detection_index,start_time_s,end_time_s,duration_ms,max_prob,mean_prob,user_action,timestamp
```

| Column | Source field |
|---|---|
| `wav_file` | recording stem |
| `detection_index` | `detection_index` |
| `start_time_s` / `end_time_s` | `core_time.start_s` / `core_time.end_s` (6 dp) |
| `duration_ms` | `core_time.duration_ms` (2 dp) |
| `max_prob` / `mean_prob` | `probabilities.max` / `.mean` (6 dp) |
| `user_action` | `user_action` (empty string if `null`) |
| `timestamp` | JSON `timestamp` |

#### `_saved_tracking.json`

Append-only list of save events for the recording (managed by
`src/usv_spectrogram/app/core/saved_detection_tracker.py`). Empty file = `[]`. Each entry:
`start_time_s`, `end_time_s`, `save_timestamp`, `output_path` (absolute path to the saved
PNG — **this is how you tell accept from reject**), `threshold_preset`, `threshold_high`,
`threshold_low`, `session_id`, `user_action`.

#### `noise_labeled_files/` (whole-file noise labels)

127 JSON files, one per **2-second lab chunk** judged to be pure noise (`file_label: "noise"`).
Schema differs from per-detection JSONs — top-level keys are `metadata`
(`wav_file`, `model_file`, `created_at`, `duration_s`, `sample_rate: 300000`,
`n_detections`, `file_label`), `detection_params` (`high_threshold: 0.04`,
`low_threshold: 0.03`), `detections` (list), and `probability_curve`
(`times` + probability arrays). These are produced by the **noise-review app**
(`src/usv_spectrogram/labeling/noise_review_app.py`) using the production model
`models/hard_neg_retrain/best_model.pt`. Used as a **noise ground-truth** set
(see [§1.4](#14-human-label-sets)).

### 1.2 The master classified-detection tables

These wide CSVs are the analysis workhorses. They join the **DeepSqueak acoustic
features** (one row per DS tonal sweep) against the **production detection events** (one
row per hysteresis event) via a fuzzy time-overlap match.

| File | Rows (excl. header) | Cohort | Notes |
|---|---|---|---|
| `classified_detections_full.csv` | **7,921** | 5970 (wild) | The wild master table |
| `classified_detections_3452.csv` | — | 3452 (wild) | per-cohort |
| `classified_detections_9252.csv` | — | 9252 (wild) | per-cohort |
| `classified_detections_lab_131204_clean.csv` | **40,787** | lab 131204 | The lab master table (see clean note below) |
| `classified_detections_lab_131204.csv` | 40,787 + residue | lab 131204 | RAW outer-join (has NaN-side rows + 597 wild residue) — do not analyze directly |
| `classified_detections_lab_131204_clean_filtered.csv` | — | lab 131204 | clean + extra filtering |
| `classified_detections_lab_131204_downsampled_n7921.csv` | 7,921 | lab 131204 | wild-matched downsample |
| `classified_detections.csv` | — | 5970 subset | older/partial — prefer `_full` |

**Clean lab table provenance.** `classified_detections_lab_131204_clean.csv` is produced
from the raw import by `scripts/clean_classified_detections_lab.py`, which (per its
docstring): drops outer-join NaN-side rows, filters to lab stems only (removes 597
wild-mouse 3452 residue), joins `tier`/`couple` from `events_clean.parquet`, adds
`couple_keep_set`, and **asserts the final count is exactly 40,787**.

#### Columns common to all tables (header verified)

```
file,id,label,accepted,score,begin_time_s,end_time_s,call_length_s,
principal_freq_hz,low_freq_hz,high_freq_hz,bandwidth_hz,freq_std_dev_hz,
slope,sinuosity,mean_power_db,tonality,peak_freq_khz,source_file,wav_stem,
det_start_s,det_end_s,det_duration_ms,det_index,det_prob_max,det_prob_mean,
det_user_action,det_json_path,match_quality,match_distance_ms,peak_freq_hz
```

| Column | Source | Meaning |
|---|---|---|
| `file` | DS | recording stem the DS call came from |
| `id` | DS | DeepSqueak call id within the file (1-based float) |
| `label` | classifier | cluster/type label, e.g. `Cluster_27` (empty string if unlabeled) |
| `accepted` | DS | DeepSqueak accept flag |
| `score` | DS | DeepSqueak detection score |
| `begin_time_s` / `end_time_s` | DS | DS tonal-sweep onset/offset |
| **`call_length_s`** | **DS** | **DeepSqueak tonal-sweep duration (seconds). See the duration gotcha below.** |
| `principal_freq_hz` | DS | dominant frequency |
| `low_freq_hz` / `high_freq_hz` | DS | frequency extent |
| `bandwidth_hz` | DS | `high - low` |
| `freq_std_dev_hz` | DS | frequency spread |
| `slope` | DS | mean spectral slope |
| `sinuosity` | DS | contour wiggliness |
| `mean_power_db` | DS | **cage/recording-environment artifact — NEVER cite as biology without cross-cage calibration** |
| `tonality` | DS | tonal-vs-noise measure (same cage caveat) |
| `peak_freq_khz` / `peak_freq_hz` | DS | peak frequency (kHz and Hz forms) |
| `source_file` | provenance | e.g. `classified_Stats.xlsx` |
| `wav_stem` | provenance | recording stem (join key) |
| `det_start_s` / `det_end_s` | **detection event** | hysteresis-event onset/offset (empty if the DS call had no matched detection) |
| **`det_duration_ms`** | **detection event** | **hysteresis-event duration (ms). USE THIS for visual-verdict filters.** |
| `det_index` | detection event | detection index within the recording |
| `det_prob_max` / `det_prob_mean` | detection event | CNN probabilities from the matched detection |
| `det_user_action` | detection event | review verb if any |
| `det_json_path` | detection event | path to the source detection JSON (lab rows point into `results/batch_lab_full_softnotch_.../detections_clean/`) |
| `match_quality` | join | `fuzzy` (matched), `unmatched_ds` (DS call with no detection), `unmatched_det` (detection with no DS call). In `_full`: 7,518 fuzzy / 346 unmatched_ds / 57 unmatched_det |
| `match_distance_ms` | join | time gap between the DS call and matched detection |

#### Lab-only extra columns

`classified_detections_lab_131204_clean.csv` appends 4 columns (added by
`clean_classified_detections_lab.py`):

| Column | Meaning |
|---|---|
| `tier` | triage tier: `auto_accept` (29,790 rows) or `manual_review` (10,997 rows) |
| `couple` | partner-pairing id, e.g. `m1fm1` (17 distinct couples — this is a partner-swap matrix, NOT one couple) |
| `couple_keep_set` | bool: `True` for the 13 retained couples; `False` for the 4 noise-prone ones {m1fm1, m1fm2, m1fm4, m3fm3} |
| `animal_id` | animal identifier — currently **identical to `couple`** in every row (verified: 0 rows where `animal_id != couple`), i.e. it carries the pairing id, not a distinct per-animal id |

> ### ⚠ GOTCHA — the TWO duration columns (most common analysis bug)
>
> Each table has **two** duration columns that can differ **by up to 10×**:
> - `call_length_s` — the **DeepSqueak tonal-sweep** duration (the bright contour DS traced).
> - `det_duration_ms` — the **hysteresis-detection-event** duration (the full event the
>   CNN+hysteresis fired on).
>
> **The PNGs in `USV_Detections/` show the HYSTERESIS EVENT.** Therefore any filter whose
> result will be eyeballed against a PNG — exemplar selection, "show me calls < X ms",
> visual-verdict review — **MUST filter on `det_duration_ms`, not `call_length_s`.**
> Filtering on `call_length_s` selects a window that does not match what you see.
> (Unit mismatch too: one is seconds, one is milliseconds.)

#### Quick read example

```bash
# Count lab rows per triage tier
.venv/bin/python -c "
import csv, collections
r = csv.DictReader(open('classified_detections_lab_131204_clean.csv'))
c = collections.Counter(row['tier'] for row in r)
print(c)
"
# -> Counter({'auto_accept': 29790, 'manual_review': 10997})
```

### 1.3 WAV recording locations — there is NO single canonical dir

Recordings span multiple top-level directories. Never assume one. Counts verified in the
working tree (2026-06-21):

| Directory | WAV count | Cohort |
|---|---|---|
| `5970 USV/` | 12 | 5970 (wild) — note the **space** in the name |
| `USV_3452_sample/` | 5,409 | 3452 (wild) |
| `USV_3452_sample_reviewed/` | 853 | 3452 reviewed subset |
| `USV_9252/` | 11,580 | 9252 (wild) |
| `USV_lab_131204/` | 83 | lab 131204 (un-chunked) |
| `USV_2379_sample/` | 1,280 | 2379 |
| `USV_2379_sample_reviewed/` | 77 | 2379 reviewed |

The lab detection JSONs reference chunked 2-second WAVs under
`USV_lab_131204_chunked_2s_full/` (e.g. `131204_1400_m1fm1_chunk_000.wav`). **That
chunked directory is NOT present in the box working tree** — regenerate from
`USV_lab_131204/` or pull from the rig if needed.

**Resolving WAV paths in scripts.** Use `--wav-search-dirs` (one or more dirs, searched
recursively) in `scripts/unify_labels.py` (`scripts/unify_labels.py:305`; example at
`:19`). Legacy fallback: the `USV_WAV_DIR` environment variable (referenced in CLAUDE.md;
NOT referenced inside `unify_labels.py` itself).

```bash
.venv/bin/python scripts/unify_labels.py \
    --wav-search-dirs "USV5/usv_lmt_034" "USV_3452_sample_reviewed" \
    --output data/unified_labels.json
```

> Note: `unify_labels.py` produces `data/unified_labels.json` (default
> `--output`, `unify_labels.py:309`) — it is the label-unifier, **not** the script that
> builds the classified-detection master CSVs.

### 1.4 Human label sets

#### Shape labels — `data/manual_shape_labels.csv`

The lab's shape labels were made by a human on **NORMAL spectrogram patches**.

- **Substrate (correct one):** `scripts/experiments/render_human_view_patches.py` →
  output `data/alpha3_human_patches/<call_id>.png` + `manifest.csv`. This is a tight,
  per-call crop on a normal spectrogram (see the script's module docstring,
  `render_human_view_patches.py:1`).
- **NOT the substrate:** the contour-masked VAE render
  (`render_vocalmat_style_patches.py` / `alpha3_patches`) produces illegible vertical
  bands and was **not** used for human shape labeling.

`data/manual_shape_labels.csv` header: `call_id,cohort,shape_label,labeled_at_index`.
`call_id` form is `<wav_stem>__det<N>` (e.g. `131204_1400_m1fm1_chunk_033__det9`).

> **204 → 758.** The label set started at **204** rows (preserved at
> `data/manual_shape_labels.backup_204_pre_relabel.csv`, 204 data rows). It has since
> been expanded; the live `data/manual_shape_labels.csv` currently holds **758** rows.
> If a doc or note says "204 labels", it refers to the original frozen set, not the
> current file. Other backups: `data/manual_shape_labels.backup_20260603_183046_pre_drop3wild.csv`.

#### Noise ground-truth sets

- `USV_Detections/noise_labeled_files/` — 127 whole-file lab-chunk noise labels
  (`file_label: "noise"`), produced by `noise_review_app.py` (see [§1.1](#11-the-detection-label-store--usv_detections)).
- `results/batch_5970/1960_problem_examples/` — 12 curated 5970 problem PNGs
  (the historical "1960 problem" known-noise / spectral-flatness examples; some
  suffixed `_GOOD.png`). Plan context: `docs/plans/1960-spectral-flatness-plan.md`.

**Why this matters:** "this is noise" claims should be anchored on these
human-confirmed sets, not on a visual read of a single spectrogram — short/faint USVs
are not reliably eyeball-separable from broadband-noise transients.

### 1.5 Rig vs box — where the LARGE artifacts live

Big, regeneratable artifacts (masked `patches.npz`, the contour-VAE, latent parquets,
registered ridges) are **deliberately kept out of git** and live ONLY on the GPU rig:

```
shachar@100.113.224.57:/data/shachar/contour_vae/        (cloudyclaude, 3× RTX 3060 Ti)
```

Examples (full table in [DATA_LOCATIONS.md](../DATA_LOCATIONS.md)):
`results/masked_patches/<cohort>_focus/patches.npz`,
`results/masked_patches/combined_all_cohorts/patches.npz` (**16 G**),
`models/contour_vae_combined/best.pt`,
`results/contour_vae_combined/latents.parquet`,
`models/shape_kmeans/k20.joblib`.

> **Do not "re-create" the rig patches from memory** — a 2026-05-29 chat did exactly
> that and diverged on ~6 axes. The render is canonical: global Hann 512/128 @ 300 kHz,
> fixed 234-bin windows, raw-power `(N, 257, 234)`. **Check the rig first.** And
> **never full-scan the 16 G combined `patches.npz` on the box** — it OOM-crashed WSL.
> See [DATA_LOCATIONS.md](../DATA_LOCATIONS.md) and auto-memory
> `feedback_cleaning_pipeline_impl_on_rig`.

### Troubleshooting / Gotchas

| Symptom | Cause / Fix |
|---|---|
| Filtered-by-duration calls don't match their PNGs | You used `call_length_s` (DS sweep). PNGs show the hysteresis event — filter on **`det_duration_ms`** ([§1.2](#12-the-master-classified-detection-tables)). |
| Lab CSV has NaN-filled rows / 597 stray wild rows | You opened the RAW `classified_detections_lab_131204.csv`. Use the `_clean.csv` (40,787 rows, NaN-side rows dropped). |
| `mean_power_db` / `tonality` shows a "strain difference" | Both are cage/recording-environment artifacts. Do not call it biology without cross-cage calibration. |
| Can't find the WAV for a stem | No single WAV dir. Search 5970 USV/, USV_3452_sample(_reviewed)/, USV_9252/, USV_lab_131204/, USV_2379_sample/. Use `--wav-search-dirs`. |
| Lab detection JSON references a WAV that doesn't exist | It points at `USV_lab_131204_chunked_2s_full/` (chunked 2 s) — not in the box tree; regenerate from `USV_lab_131204/` or pull from rig. |
| Shape-label patches look like illegible vertical bands | You rendered the contour-masked VAE patches. Human labels were made on **normal** spectrograms via `render_human_view_patches.py`. |
| "We only have 204 shape labels" contradicts the file | 204 is the frozen original (`backup_204_pre_relabel.csv`). Live file is 758 rows. |
| `patches.npz` missing locally | It only exists on the rig (`/data/shachar/contour_vae/`). It is git-ignored by design. |
| Detection has `threshold_high: 0.6` not `0.04` | Thresholds are whatever the review session used; not the calibrated batch values. Not a corruption. |

---

## 2. Internals

### Producers (who writes each file)

| Output | Writer | Key location |
|---|---|---|
| `USV_Detections/<stem>/detection_*.json` | `DetectionExporter._save_json_metadata` | `src/usv_spectrogram/app/core/detection_exporter.py:210` (metadata dict at `:235`) |
| `USV_Detections/<stem>/detections_summary.csv` | `DetectionExporter._rewrite_csv_summary` | `detection_exporter.py:277` |
| `_saved_tracking.json` | `saved_detection_tracker.py` | `src/usv_spectrogram/app/core/saved_detection_tracker.py` |
| `noise_labeled_files/*.json` | noise-review app | `src/usv_spectrogram/labeling/noise_review_app.py` |
| `classified_detections_lab_131204_clean.csv` | clean script | `scripts/clean_classified_detections_lab.py` |
| `data/alpha3_human_patches/*.png` + `manifest.csv` | human-view renderer | `scripts/experiments/render_human_view_patches.py` |
| `data/unified_labels.json` | label unifier | `scripts/unify_labels.py:288` (`main`) |

### Data flow

```
WAV ──CNN+hysteresis (app Detect or run_batch_detection)──▶ detections (events)
                                                                  │
        DeepSqueak ──acoustic features (tonal sweeps)──┐         │
                                                        ▼         ▼
                                   fuzzy time-overlap JOIN (match_quality, match_distance_ms)
                                                        │
                                                        ▼
                                    classified_detections_<cohort>.csv
                                       │ (lab: + clean script)
                                       ▼
                            classified_detections_lab_131204_clean.csv
                                   (+ tier, couple, couple_keep_set, animal_id)

PyQt6 app review loop ──accept/reject──▶ USV_Detections/{<stem>/, rejected_detections/<stem>/}
                                                  (detection_*.json + .png + detections_summary.csv + _saved_tracking.json)
```

### Key signatures

- `DetectionExporter.__init__(self, output_dir: Path, context_ms: float = 20.0)` —
  `detection_exporter.py:21`. The **20 ms** default context padding lives here.
- `DetectionExporter._save_json_metadata(...)` — `detection_exporter.py:210` (metadata
  dict at `:235`). The JSON schema is defined inline here; the optional `user_adjusted` /
  `original_boundaries` keys are added at `:267`.
- `DetectionExporter._rewrite_csv_summary(self, wav_filename: str) -> Path` —
  `detection_exporter.py:277`. The CSV is **regenerated from the JSONs on disk**, so the
  JSONs are the source of truth; the CSV is a derived view.

### Invariants

- **JSONs are canonical; `detections_summary.csv` is derived.** Editing the CSV by hand
  is pointless — it is rewritten from the JSONs on the next save.
- **Accept/reject is encoded by directory + `user_action`/`output_path`.** A file's tree
  (`USV_Detections/<stem>/` vs `.../rejected_detections/<stem>/`) and its `user_action`
  field together determine its disposition; cross-check via `_saved_tracking.json`'s
  `output_path`.
- **`classified_detections_lab_131204_clean.csv` row count == 40,787** is *asserted* by
  the producer (`clean_classified_detections_lab.py` docstring). A different count means
  the upstream import changed.
- **Signal-processing constants are not in these data files.** Sample rate, band, and STFT
  are enforced once in `src/usv_spectrogram/corpus.py`:
  `SAMPLE_RATE_HZ = 300_000` (`corpus.py:30`), `USV_FREQ_MIN_HZ = 20_000` (`:32`),
  `USV_FREQ_MAX_HZ = 120_000` (`:33`), `STFT_N_FFT = 512` (`:35`), hop 128 (ADR-002,
  `corpus.py:12`). Import from `corpus`; never restate these in analysis code.

### Where to change things

- **Add a field to the detection JSON / CSV:** edit the `metadata` dict in
  `_save_json_metadata` (`detection_exporter.py:235`) AND the row builder in
  `_rewrite_csv_summary` (`detection_exporter.py:290`) — both, or the CSV will drift.
- **Change context padding:** the `context_ms` default in `DetectionExporter.__init__`
  (`detection_exporter.py:21`); callers may override it.
- **Change the lab clean/filter logic or the keep-set:** `scripts/clean_classified_detections_lab.py`
  (the {m1fm1, m1fm2, m1fm4, m3fm3} noise-prone exclusion lives there).
- **Change WAV resolution:** `build_wav_index` / the `--wav-search-dirs` handling in
  `scripts/unify_labels.py:332`.
