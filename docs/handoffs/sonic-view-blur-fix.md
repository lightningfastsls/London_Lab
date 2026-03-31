# Handoff: Sonic Spectrogram View — Fix Horizontal Blur

## Status
The sonic (0-15 kHz) spectrogram view is implemented and working, but the image appears **horizontally smeared/blurry** compared to the crisp USV (20-120 kHz) view above it. Layout, scroll sync, and audio playback all work correctly. Only the rendering sharpness needs fixing.

## The Problem

The USV spectrogram is crisp. The sonic spectrogram is horizontally blurry. Both use the same `_spectrogram_to_image()` → `QPixmap` → `paintEvent` pipeline. The difference is in **how many native pixels the STFT produces** along the time axis.

### Why the USV view is crisp
- Runs FFT directly on 300kHz signal: `n_fft=512, hop=128`
- 3-second clip (900,000 samples) → **~7,029 time columns**
- Canvas is 7,029 pixels wide → displayed through a QScrollArea
- **No scaling ever happens** — you scroll through native-resolution pixels

### Why the sonic view is blurry
- Decimates 300kHz → 60kHz first (needed for frequency resolution)
- Current config: `n_fft=1024, hop=26` on 180,000 decimated samples
- Should produce ~6,884 time columns — **similar to USV**
- Canvas renders at native resolution (no `QPixmap.scaled()` call)
- **Yet it still looks blurry** — the blur must be coming from somewhere else

### What we tried (all in this session)
1. ✅ `QPixmap.scaled()` with SmoothTransformation — blurry (expected)
2. ✅ `QPixmap.scaled()` with FastTransformation — blocky/pixelated
3. ✅ Native resolution (no scaling) with hop=256 — crisp but only 700px wide (wrong width)
4. ✅ Native resolution with hop=26 — should match USV width, but **still blurry**
5. The `_spectrogram_to_image()` code in SonicCanvas is identical to SpectrogramCanvas

### Likely root cause (not yet investigated)
The blur may NOT be a rendering issue. It may be inherent to the **STFT computation**:
- `scipy.signal.decimate()` applies a low-pass anti-aliasing filter before downsampling. This filter might be smearing the time resolution.
- With hop=26 and n_fft=1024, each frame overlaps 998/1024 samples with the next (~97.5% overlap). This extreme overlap means adjacent columns are nearly identical, producing a naturally "smooth" (blurry-looking) image.
- The USV STFT has 75% overlap (hop=128, n_fft=512) which is standard and doesn't blur.
- **Test**: Try printing `sonic_spectrogram_db.shape` to verify it actually has ~7000 columns. If it does, the blur is from the decimation filter or overlap, not from pixel scaling.

### Alternative approaches to consider
1. **Skip decimation entirely** — run FFT on the raw 300kHz signal with `n_fft=4096` (gives 73 Hz/bin) or `n_fft=2048` (146 Hz/bin), `hop=128` to match USV columns exactly. Frequency resolution won't be as fine but avoids the decimation filter blur. This is the simplest fix.
2. **Use a sharper decimation filter** — `scipy.signal.decimate(x, 5, ftype='fir', n=30)` with a shorter FIR filter has less temporal smearing.
3. **Reduce overlap** — use hop=128 on 60kHz (gives ~1,400 columns) and accept that the sonic canvas is narrower, scrolling independently.

## Files Involved

| File | What it does |
|------|-------------|
| `src/usv_spectrogram/app/core/audio_loader.py` | `SonicConfig` dataclass (line ~23), `_compute_sonic_spectrogram()` (line ~186), `_compute_playback_audio()` (line ~237) |
| `src/usv_spectrogram/app/widgets/sonic_view.py` | `SonicCanvas` + `SonicSpectrogramView` — rendering widget |
| `src/usv_spectrogram/app/widgets/spectrogram_view.py` | USV `SpectrogramCanvas` — reference for how crisp rendering works |
| `src/usv_spectrogram/app/main_window.py` | Layout, scroll sync, Play button, sonic data feeding |
| `src/usv_spectrogram/_stft_core.py` | Shared `extract_frames()` and `compute_stft_frames_db()` used by both views |

## Current SonicConfig values
```python
target_sr: int = 60_000
decimation_factor: int = 5
n_fft: int = 1024
hop_length: int = 26        # Meant to match USV column count
window: str = "hann"
freq_min_hz: int = 0
freq_max_hz: int = 15_000
```

## What's working fine (don't change)
- Layout: buttons tight to top, USV view capped at 210px, sonic view gets stretch=2, probability view at bottom
- 3-way scroll sync (spectrogram → sonic + probability) via normalized position
- Play/Stop button with sounddevice playback (48kHz downsampled)
- `sounddevice` added to requirements.txt
- Audio playback stops on new file load

## To verify your fix
1. `python -m py_compile` on all modified files
2. `python -m pytest tests/test_stft_core.py tests/test_energy_detector.py tests/test_config.py -v` (existing tests)
3. Manual: Open app → Load WAV → Compare USV and sonic sharpness visually
