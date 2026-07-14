# USV Labeling — Candidate Labeling & Noise Review

> **What this is:** The tools a human uses to mark USV candidates as *USV / Not USV / Uncertain*
> and to triage whole recordings as *USV-bearing vs. noise*.
> **Status (2026-06):** The two Streamlit apps under `src/usv_spectrogram/labeling/`
> (`labeling_app.py`, `noise_review_app.py`) are **DEPRECATED and retired** — they
> `sys.exit(1)` on launch. **All labeling now happens in the PyQt6 desktop app.**
> **Launch the live tool:** `.venv/bin/python scripts/run_app.py`
> **Labels are written as JSON** (per-WAV) by `app/core/label_storage.py`.

---

## TL;DR — which tool do I actually run?

| You want to… | Use | Command |
|---|---|---|
| Review CNN detections on a WAV, adjust/add/delete them, mark USV vs noise, save labels | **PyQt6 desktop app** | `.venv/bin/python scripts/run_app.py` |
| Bulk-label thousands of pre-rendered candidate PNGs (old workflow) | ~~Streamlit `labeling_app.py`~~ | **RETIRED — do not use** |
| Triage noise-sample PNGs (Clean / Contains USV / Skip) | ~~Streamlit `noise_review_app.py`~~ | **RETIRED — do not use** |

The Streamlit `run()` functions print a deprecation notice and exit
(`labeling_app.py:633`, `noise_review_app.py:395`). The bodies below those lines are
unreachable legacy code, kept only for reference. The launcher script the README points
at, `scripts/usv_labeling_tool.py`, **no longer exists**. Do not try to revive these;
the supported path is the desktop app.

This document covers the **live** desktop-app labeling workflow first (§1, §2), then
documents the retired Streamlit apps (§3) so you can read old code and old output files
without confusion.

Sibling docs: [CNN detection pipeline](../modules/cnn-classifier.md) ·
[batch detection](../handoffs/v2-full-pipeline-results.md) ·
[PyQt6 app implementation plan](../plans/USV_DETECTION_APP_IMPLEMENTATION.md) ·
keyboard shortcuts: [USV_DETECTION_APP_SHORTCUTS.md](../USV_DETECTION_APP_SHORTCUTS.md)
(note: that file's "default thresholds 0.40/0.28" is **stale** — see §1.4) ·
labeling criteria: [labeling_guide.md](../labeling_guide.md).

---

## 1. Operate

### 1.1 Launch

```bash
.venv/bin/python scripts/run_app.py
```

`scripts/run_app.py` imports `torch` **before** PyQt6 (a Windows DLL-ordering
workaround, `scripts/run_app.py:10-13`), then calls
`usv_spectrogram.app.main.main()`.

There is **no argparse** and no positional arguments. Everything — WAV file, output
folder, thresholds — is chosen interactively or restored from saved settings. The model
is hard-wired:

- **Default CNN model:** `models/hard_neg_retrain/best_model.pt`
  (set at `src/usv_spectrogram/app/main.py:24-27`). This is the current production model
  (precision 90.55%). The two older models
  (`models/matched_windows/best_model.pt`, `models/production/best_model.pt`) are
  deprecated and are **not** what the app loads.

**Environment:** the repo `.venv` with PyQt6, torch, librosa, soundfile, matplotlib.
On WSL the app needs an X server / WSLg to display a window.

### 1.2 What the app does, end to end

1. **Open a WAV** (`Ctrl+O` or the *Load* button). Audio loads via `AudioLoader`, a
   spectrogram is computed and displayed. When you open a *new* file while one is already
   open, the previous file is moved to a parallel `*_reviewed` folder — the first
   path component containing `USV`/`usv` gets a `_reviewed` suffix, e.g.
   `USV_1/usv_lmt_034/file.wav` → `USV_1_reviewed/usv_lmt_034/file.wav`
   (`main_window.py:630-676`, `_move_reviewed_file`).
2. **Run detection** (`Space`). The CNN slides over the spectrogram producing a
   probability curve; hysteresis thresholding turns that into detected USV intervals.
3. **Review / edit** each detection: adjust boundaries, add a manual detection, or delete
   one (`Delete`). Edits are tracked with `user_adjusted` / `user_action` metadata.
4. **Classify the whole file** as noise if appropriate (*Label File as Noise* button) —
   this auto-writes a JSON into `noise_labeled_files/`.
5. **Save labels** (`Ctrl+S`) to a per-WAV JSON, and/or **export** an annotated PNG
   (`Ctrl+E`).

> This app's *Detect* is **CNN + hysteresis only** — it does **not** apply the
> FP-filter, temperature calibration, or soft-notch that the offline batch pipeline uses.
> Applying the FP-filter here would wrongly kill real detections. To reproduce the full
> production triage on a folder, use `scripts/run_batch_detection.py` (the 5-flag
> pipeline in `CLAUDE.md`), not this app.

### 1.3 Keyboard shortcuts (read from code)

Menu accelerators are registered in `main_window.py:238-271`; the threshold/detection
hotkeys in `_setup_keyboard_shortcuts` at `main_window.py:1448-1470`.

| Key | Action | Source |
|---|---|---|
| `Ctrl+O` | Open WAV file | `main_window.py:239` |
| `Ctrl+S` | Save labels to JSON | `main_window.py:244` |
| `Ctrl+L` | Load labels from JSON | `main_window.py:250` |
| `Ctrl+E` | Export annotated image (PNG) | `main_window.py:256` |
| `Ctrl+Q` | Quit | `main_window.py:269` |
| `Space` | Run detection (`_run_detection`) | `main_window.py:1465-1466` |
| `↑` Up | Increase **high** threshold by 0.01 | `main_window.py:1451-1452`, `:1472` |
| `↓` Down | Decrease **high** threshold by 0.01 | `main_window.py:1454-1455`, `:1479` |
| `→` Right | Increase **low** threshold by 0.01 | `main_window.py:1461-1462`, `:1486` |
| `←` Left | Decrease **low** threshold by 0.01 | `main_window.py:1458-1459`, `:1493` |
| `Delete` | Remove the selected detection | `main_window.py:1469-1470` |

The threshold steps move QSlider values by 1 unit out of 100, i.e. 0.01 in
probability units (`main_window.py:1474`), and re-apply thresholds live if a detection
result already exists.

### 1.4 Thresholds and detection parameters (defaults)

Loaded from `QSettings` with these fallbacks at `main_window.py:147-151`. On an app
**version bump** (`APP_VERSION = "1.2.0"`, `main_window.py:44`) the threshold settings are
wiped so these defaults are forced (`main_window.py:103-125`).

| Parameter | Default | Meaning | Source |
|---|---|---|---|
| `high_threshold` (onset) | **0.60** | A USV interval *starts* when CNN prob crosses this | `main_window.py:147` |
| `low_threshold` (sustain) | **0.40** | The interval *continues* until prob drops below this | `main_window.py:148` |
| `min_sustained_prob` | **0.0** | Continuity gate (disabled) | `main_window.py:149` |
| `exclude_start_sec` | **0.0** | Seconds excluded at file start | `main_window.py:150` |
| `exclude_end_sec` | **0.0** | Seconds excluded at file end | `main_window.py:151` |

These defaults are deliberately aligned to the production batch hysteresis (onset 0.60 /
sustain 0.40) so the live view matches `run_batch_detection.py` output
(`main_window.py:142-146`). **Note:** [USV_DETECTION_APP_SHORTCUTS.md](../USV_DETECTION_APP_SHORTCUTS.md)
still lists "High = 0.40, Low = 0.28" — that document is **stale**; the code value is
0.60 / 0.40.

Settings are stored under `QSettings("USV Lab", "USV Detection")`
(`main_window.py:101`) — Windows Registry on Windows, an INI/plist elsewhere. Window
geometry, last thresholds, and the output directory persist across sessions
(`_save_settings`, `main_window.py:1510-1519`).

### 1.5 Output locations and file formats

**Default output directory:** `<repo_root>/USV_Detections/`
(`main_window.py:138-140`). Change it via *File → Set Output Directory…* (`Ctrl`-less
menu action, `main_window.py:262-264`); the choice persists.

What lands where:

| Output | Path | Written by |
|---|---|---|
| Per-WAV saved labels (manual `Ctrl+S`) | wherever you point the Save dialog; defaults to `<wav>.json` next to the WAV | `_save_labels`, `main_window.py:1257-1297` |
| Whole-file noise label (auto) | `<output_dir>/noise_labeled_files/<wav_stem>.json` | `_auto_save_noise_label`, `main_window.py:1111-1135` |
| Rejected detections | `<output_dir>/rejected_detections/` | `main_window.py:1008-1009` |
| Annotated image (`Ctrl+E`) | wherever you point the Export dialog; defaults to `<wav>.png` | `_export_image` → `LabelStorage.export_annotated_image` |
| Reviewed WAVs (moved on file switch/close) | parallel folder with the first `USV`/`usv` path component suffixed `_reviewed` (e.g. `USV_1/usv_lmt_034/file.wav` → `USV_1_reviewed/usv_lmt_034/file.wav`); falls back to renaming the WAV's parent folder if no `USV`/`usv` component is found | `_move_reviewed_file`, `main_window.py:630-676` |

#### Labels JSON schema

Written by `LabelStorage.save` (`app/core/label_storage.py:27-111`). Top-level keys:

```json
{
  "metadata": {
    "wav_file":   "<absolute path to WAV>",
    "model_file": "<absolute path to best_model.pt, or null>",
    "created_at": "<ISO-8601 timestamp>",
    "duration_s": 30.0,
    "sample_rate": 300000,
    "n_detections": 7,
    "file_label": null
  },
  "detection_params": { "high_threshold": 0.60, "low_threshold": 0.40 },
  "detections": [ /* one object per USV, see below */ ],
  "probability_curve": { "times": [...], "probabilities": [...], "column_indices": [...] }
}
```

Each entry in `detections` (`label_storage.py:70-99`):

| Field | Type | Meaning |
|---|---|---|
| `start_time_s`, `end_time_s` | float | Detection interval in seconds |
| `duration_s` | float | `end - start` |
| `start_col`, `end_col` | int | Spectrogram column indices |
| `max_probability`, `mean_probability` | float | CNN prob over the interval |
| `user_adjusted` | bool | Present only if the human moved a boundary |
| `original_start_time_s`, `original_end_time_s` | float | Pre-edit boundaries (only when adjusted) |
| `user_action` | str | e.g. `"added_manually"` (only when set) |
| `save_state` | str | `"unsaved"` / reviewed state (only when set) |
| `original_cnn_probability` | float | CNN prob before a manual edit/deletion (only when set) |

`probability_curve` always carries the full CNN trace so a reload reconstructs the exact
view (`LabelStorage.reconstruct_detection_result`, `label_storage.py:162-193`).

#### Annotated PNG export

`LabelStorage.export_annotated_image` (`label_storage.py:195-314`): a 2-panel figure —
spectrogram (`cmap='magma'`, frequency axis in **kHz**) with green start / red end lines
per USV, and below it the probability curve with high (red dashed) and low (orange
dashed) threshold lines and green-shaded detected regions. Defaults: `dpi=100`,
`figsize=(16, 6)` (`label_storage.py:202-203`).

### 1.6 Worked example

```bash
# 1. Launch
.venv/bin/python scripts/run_app.py

# 2. In the GUI:
#    Ctrl+O  -> pick e.g.  5970 USV/0004954.wav
#    Space   -> run CNN + hysteresis detection
#    Up/Down -> nudge the onset (high) threshold; Left/Right -> sustain (low)
#    click a detection, drag its edge to fix boundaries, or Delete to remove it
#    if the whole file is noise: click "Label File as Noise"
#        -> auto-writes USV_Detections/noise_labeled_files/0004954.json
#    Ctrl+S  -> save labels JSON (defaults to 0004954.json beside the WAV)
#    Ctrl+E  -> export annotated PNG
```

### 1.7 Label categories

The live app is built around **per-detection USV intervals** plus a **whole-file noise
flag** (the *Label File as Noise* button, `main_window.py:323-329`). It does not use the
three-way *USV / Not USV / Uncertain* button set — that taxonomy belongs to the retired
Streamlit app (§3.1). For the human criteria distinguishing a real USV from noise/
artifact (narrow-band, coherent contour, 25–110 kHz, 10–500 ms, etc.), follow
[labeling_guide.md](../labeling_guide.md).

### 1.8 Troubleshooting / Gotchas

- **"The Streamlit labeling tool exits immediately."** Expected — it is retired. Use
  `scripts/run_app.py`. The launcher `scripts/usv_labeling_tool.py` the old README cites
  is gone.
- **App detections differ from `run_batch_detection.py`.** Expected: the app is CNN +
  hysteresis only; the batch pipeline adds temperature, FP-filter, and soft-notch. Do not
  "fix" the app to match — they are different stages by design.
- **Thresholds reset after an update.** A version bump clears threshold settings on
  purpose (`main_window.py:106-125`); they reload to 0.60 / 0.40.
- **A WAV "disappeared" from its folder.** Files are moved to a parallel `*_reviewed`
  folder when you switch/close (`main_window.py:584`, `:1530`; logic at `:630-676`) —
  the first `USV`/`usv` path component is renamed with a `_reviewed` suffix (e.g.
  `USV_1/...` → `USV_1_reviewed/...`), not a subfolder of the WAV's directory. Look there.
- **Save is greyed out / "No Detections" warning.** You must run detection (`Space`)
  before Save/Export (`main_window.py:1259-1265`, `:1301-1307`).
- **Stale noise label.** Toggling the noise flag off removes
  `noise_labeled_files/<stem>.json` (`_remove_noise_label_file`, `main_window.py:1146-1159`).
- **No window on WSL.** Ensure WSLg / an X server is running; PyQt6 needs a display.

---

## 2. Internals (PyQt6 app)

### 2.1 Architecture / data flow

```
scripts/run_app.py
  └─ import torch (before PyQt6)            # DLL ordering, run_app.py:10-13
  └─ app.main.main()                        # main.py:13
       └─ MainWindow(default_model_path=models/hard_neg_retrain/best_model.pt)
```

`MainWindow` (`src/usv_spectrogram/app/main_window.py`, **1819 lines** — read in
≤500-line chunks per the Large File Protocol) owns the whole GUI. Core logic is delegated
to `app/core/`:

| Module | Role |
|---|---|
| `core/audio_loader.py` | WAV → `AudioData` (samples, spectrogram, times, frequencies) |
| `core/sliding_inference.py` | CNN sliding-window probability curve |
| `core/detection_logic.py` | Hysteresis → `DetectionResult` / `DetectedUSV` |
| `core/label_storage.py` | JSON save/load + annotated-PNG export |
| `core/detection_exporter.py` | Per-detection clip/PNG export (`context_ms=20.0`) |
| `core/saved_detection_tracker.py` | Tracks which detections are saved per WAV |
| `core/notch.py`, `core/denoise.py`, `core/preset_config.py` | DSP options & presets |

Views live in `app/widgets/`: `spectrogram_view.py`, `probability_view.py`,
`sonic_view.py`. Horizontal scrolling is synchronized across the spectrogram and
probability panels.

### 2.2 Key signatures (file:line)

- `MainWindow.__init__(self, default_model_path: Path | None = None)` — `main_window.py:91`
- `LabelStorage.save(output_path, audio_data, detection_result, wav_path, model_path, high_threshold, low_threshold) -> None` — `label_storage.py:27`
- `LabelStorage.load(input_path) -> Dict[str, Any]` — `label_storage.py:113`
- `LabelStorage.reconstruct_detected_usv(detection_dict) -> DetectedUSV` — `label_storage.py:137`
- `LabelStorage.reconstruct_detection_result(data) -> DetectionResult` — `label_storage.py:162`
- `LabelStorage.export_annotated_image(output_path, audio_data, detection_result, high_threshold, low_threshold, dpi=100, figsize=(16, 6)) -> None` — `label_storage.py:195`
- `MainWindow._setup_keyboard_shortcuts(self)` — `main_window.py:1448`
- `MainWindow._save_labels`, `_export_image`, `_load_labels` — `main_window.py:1257`, `:1299`, `:1339`
- `MainWindow._auto_save_noise_label` — `main_window.py:1111`

### 2.3 Signal/slot wiring (where to change shortcuts & buttons)

- **Menu actions** (Open/Save/Load/Export/Quit) are `QAction`s built in `_init_ui`
  around `main_window.py:238-271`; each `.setShortcut(...)` and `.triggered.connect(...)`
  there.
- **Hotkeys** (arrows, Space, Delete) are `QShortcut(QKeySequence(...))` objects created
  in `_setup_keyboard_shortcuts`, `main_window.py:1448-1470`; rebind here.
- **Noise button:** `self.label_noise_btn` created at `main_window.py:323`, connected to
  `_toggle_noise_label` at `:324`.

### 2.4 Invariants

- **Sample rate is 300 kHz**, enforced via `corpus.py`
  (`SAMPLE_RATE_HZ = 300_000`, `src/usv_spectrogram/corpus.py:30`;
  `USV_FREQ_MIN_HZ = 20_000` `:32`, `USV_FREQ_MAX_HZ = 120_000` `:33`,
  `STFT_N_FFT = 512` `:35`). Never restate these — import from `corpus`.
- The model path is the **production** CNN; do not point the app at the deprecated
  `matched_windows` / `production` checkpoints.
- The app's detection is CNN + hysteresis **only** (no FP-filter / temperature /
  soft-notch). Keep that boundary; the offline pipeline owns those stages.

---

## 3. The retired Streamlit apps (historical)

Kept under `src/usv_spectrogram/labeling/`. **Both refuse to run.** Their `run()`
functions write a deprecation banner and `sys.exit(1)`, pointing to
`scripts/run_app.py`:

- `labeling_app.py:633-642` (`run()` deprecation guard; legacy body `:644-853` is
  unreachable).
- `noise_review_app.py:395-404` (same pattern; legacy body `:406-492` unreachable).

Documented here so you can interpret **old output files** and old code.

### 3.1 `labeling_app.py` — three-way candidate labeling (retired)

- **Label classes:** `LABEL_OPTIONS = ["USV", "Not USV", "Uncertain"]`
  (`labeling_app.py:22`). Keyboard 1/2/3 mapped via injected JS, not real Qt hotkeys
  (`labeling_app.py:445`).
- **Inputs (repo root):** `candidates_optimized.csv` and a `spectrograms_review/`
  folder of `<candidate_id>.png` images (`labeling_app.py:653-654`).
- **Output:** `labels.csv` at repo root (`labeling_app.py:655`), rewritten in full on
  every label (`save_label`, `labeling_app.py:122-152`). Columns:

  | Column | Meaning |
  |---|---|
  | `candidate_id` | matches the PNG / CSV row |
  | `label` | `USV` / `Not USV` / `Uncertain` |
  | `labeled_at` | ISO timestamp |
  | `expand_ms` | signed ms of boundary expansion (>0 = start earlier, <0 = end later) |

- **Expanded PNGs:** `spectrograms_review/expanded/` (and `…/expanded/preview/`),
  re-extracted via `SpectrogramExtractor` with `ExtractionConfig(sample_rate=300000,
  default_render_mode="review")` (`labeling_app.py:116-119`, README lines 59-62).
- **Archive/reset:** moves labeled PNGs and `labels.csv` into
  `labeling_archives/labeling_archive_<timestamp>/` (`labeling_app.py:155-213`).

### 3.2 `noise_review_app.py` — noise/USV triage (retired)

- **Triage classes:** `LABEL_OPTIONS = ["Clean", "Contains USV", "Skip"]`
  (`noise_review_app.py:25`); "Contains USV" can be **Trimmed** to remove the USV portion,
  yielding a 4th status (`noise_review_app.py:350`, `:485`).
- **Inputs:** `noise_samples/noise_samples.csv` + `noise_samples/*.png`; WAVs from
  `5970 USV/` (`noise_review_app.py:411-414`).
- **Output:** `noise_samples/noise_reviews.csv` (`save_review`,
  `noise_review_app.py:50-74`). Columns: `candidate_id`, `status`
  (`Clean`/`Contains USV`/`Trimmed`/`Skip`), `trim_ms` (signed; >0 trims start, <0 trims
  end), `reviewed_at`.
- **Min trimmed segment:** 10 ms (`noise_review_app.py:151`).

### 3.3 Relationship between the two retired apps

`labeling_app` assigned **syllable-level USV/Not-USV/Uncertain** to candidate
spectrograms; `noise_review_app` did **whole-sample Clean/Contains-USV/Skip** triage with
optional trimming. Both have been superseded by the single PyQt6 desktop app, where
per-detection review and the whole-file noise flag now live together.

> The **stable-ID review-folder convention** (copying the current batch into one folder
> with prefixed filenames like `typ01_*.png` for interactive Option-A labeling) is an
> operator workflow used when staging PNGs for review — it is **not** implemented by
> either retired app. If you adopt it, stage PNGs by stable prefix yourself; don't dump
> absolute paths.
