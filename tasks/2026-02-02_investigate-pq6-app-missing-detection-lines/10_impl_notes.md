# Implementation Notes

## Summary of implementation
- Stage 1 baseline CLI check: `scripts/test_detection_backend.py` on `5970 USV\\2024-09-30_11-18-17_0000001.wav` with `--threshold 0.10` produced **0 detections**; probability range `[0.000, 0.032]` and per-window normalization warning. Energy filter skipped 0 windows.
- pq6 app UI run could not be confirmed in this environment (see limitations).
- Stage 2 config/model verification: pq6 app default model path points to `models/production/best_model.pt` (set in `src/usv_spectrogram/app/main.py`), and thresholds default to high=0.10 / low=0.05 (from `src/usv_spectrogram/app/main_window.py`). Model file exists on disk and was successfully loaded by the CLI backend in Stage 1.
- Stage 3 preprocessing trace: app inference uses `SlidingInference` with `energy_threshold=0.35` and `enable_per_window_norm=False`; core inference pipeline applies **MAD normalization** to the full spectrogram, **colormap + grayscale**, **resize to 256px height**, **per-image min/max normalization**, and **pads width to 512px** before CNN. These steps are implemented inside `src/usv_spectrogram/app/core/sliding_inference.py`. Audio loading uses `ExtractionConfig` with `sample_rate=300_000`, `n_fft=512`, `hop=128`, and bandpass 20–120 kHz.
- Stage 4 UI rendering trace: `_apply_thresholds()` computes detections and calls `self.spectrogram_view.set_detections(self.detection_result.usvs)` and updates `detection_info_label` to "Detected: N USVs". `SpectrogramView` draws vertical lines per detection in `SpectrogramCanvas.paintEvent`. `ProbabilityView` shades detection regions and draws threshold lines. No explicit "show/hide detections" toggle found in UI path; rendering is direct from `detection_result.usvs`.
- Stage 5 findings: current evidence shows the break point is upstream of UI in inference output. CLI baseline produced probabilities in `[0.000, 0.032]`, which are below low/high thresholds (0.05/0.10), resulting in zero detections. UI rendering path appears correct; it will display lines if detections exist.

## Decisions and tradeoffs
- Used `--output NUL` to avoid writing JSON while still exercising the full pipeline.
- Used default WAV location (`5970 USV`) because `USV_WAV_DIR` was not set.
- Used on-disk file metadata + Stage 1 model load as evidence that `models/production/best_model.pt` is present and loadable.
- Treated the CLI test script as a pipeline reference: it uses `SlidingInference` defaults (energy_threshold=0.1, per-window norm enabled), which differs from pq6 app settings.

## Commands used during development
- `Write-Output $env:USV_WAV_DIR`
- `Get-ChildItem -Path "5970 USV" -Filter "*.wav" | Select-Object -First 1 | Select-Object FullName,Length,LastWriteTime`
- `Get-Content -Path src\usv_spectrogram\app\main.py`
- `Get-Content -Path src\usv_spectrogram\app\main_window.py | Select-String -Pattern "default_model_path|high_threshold|low_threshold" -Context 2,4`
- `Get-Item -Path "models\production\best_model.pt" | Select-Object FullName,Length,LastWriteTime`
- `.\.venv\Scripts\python.exe scripts\test_detection_backend.py --help`
- `.\.venv\Scripts\python.exe scripts\test_detection_backend.py --wav "5970 USV\2024-09-30_11-18-17_0000001.wav" --threshold 0.10 --output NUL`
- `.\.venv\Scripts\python.exe -m usv_spectrogram.app.main` (failed: ModuleNotFoundError)
- `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m usv_spectrogram.app.main` (launched, timed out waiting for GUI)
- `Get-Content -Path src\usv_spectrogram\app\core\sliding_inference.py | Select-String -Pattern "pad|padding|render_mode|normalize|normalization|mad" -Context 2,4`
- `Get-Content -Path src\usv_spectrogram\app\core\sliding_inference.py | Select-String -Pattern "class SlidingInference|def __init__|enable_per_window_norm|energy_threshold" -Context 2,4`
- `Get-Content -Path src\usv_spectrogram\app\main_window.py | Select-String -Pattern "SlidingInference" -Context 2,6`
- `Get-Content -Path scripts\test_detection_backend.py | Select-String -Pattern "SlidingInference|enable_per_window_norm|energy_threshold|window_width" -Context 2,4`
- `Get-Content -Path src\usv_spectrogram\app\core\audio_loader.py`
- `Get-Content -Path src\usv_spectrogram\detection\extraction_config.py | Select-String -Pattern "sample_rate|render_mode|freq_min|freq_max|n_fft|hop_length" -Context 2,4`
- `rg -n "ProbabilityView|SpectrogramView|show detections|detection|toggle|checkbox" src\usv_spectrogram\app`
- `Get-Content -Path src\usv_spectrogram\app\main_window.py | Select-String -Pattern "set_detections|ProbabilityView|SpectrogramView|detection_info_label|_apply_thresholds" -Context 2,4`
- `Get-Content -Path src\usv_spectrogram\app\widgets\spectrogram_view.py`
- `Get-Content -Path src\usv_spectrogram\app\widgets\probability_view.py`
- `Get-Content -Path src\usv_spectrogram\app\main_window.py | Select-String -Pattern "_run_detection|_on_inference_finished|_apply_thresholds" -Context 2,6`

## How to run
- CLI baseline: `.\.venv\Scripts\python.exe scripts\test_detection_backend.py --wav "5970 USV\2024-09-30_11-18-17_0000001.wav" --threshold 0.10 --output NUL`
- pq6 app (requires GUI + manual file load): `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m usv_spectrogram.app.main`

## Known limitations / TODOs
- Unable to confirm UI behavior in this environment; need a manual pq6 app run to load the WAV and verify whether detections are shown/hidden.
- CLI script only uses a single high threshold; pq6 app uses low/high thresholds (expected 0.05/0.10) so a second pipeline check may be needed for low-threshold behavior.
- App inference config differs from CLI test defaults (energy threshold 0.35 vs 0.1; per-window normalization disabled vs enabled). This may impact probability range comparisons.
- UI rendering appears to be direct from detection results; if detections are zero upstream, UI will show no lines.
- Next-step recommendation (minimal): run a CLI inference that matches pq6 app settings (window_width=100, hop=10, energy_threshold=0.35, per-window normalization disabled) and record the probability range. If probabilities remain <0.05, consider lowering thresholds temporarily or revisiting preprocessing alignment with training.

## Files changed
- None.
