# Event Features Module

**Module:** `src/usv_spectrogram/postprocessing/event_features.py`
**Tests:** `tests/test_event_features.py` (17 tests)

## Purpose

Extracts discriminative features from each `USVEvent` for second-stage false-positive filtering. After hysteresis detection, some noise events that sustained above the low threshold remain. Event-level features enable a classifier to separate true USVs (smooth probability curves, tonal spectral content) from false positives (spiky probabilities, broadband noise).

## Public API

### `EventFeatures` (frozen dataclass)

| Field | Type | Description |
|-------|------|-------------|
| `peak_probability` | float | Max probability in the event |
| `mean_probability` | float | Mean probability in the event |
| `prob_std` | float | Population std of probabilities |
| `prob_kurtosis` | float | Excess kurtosis (spiky noise > 0, plateau USV ~ 0) |
| `prob_roughness` | float | Mean |second derivative| of prob curve; high = jagged |
| `duration_windows` | int | Number of windows in the event |
| `tonality` | float | 1 - SFM; high = tonal, low = broadband |
| `mean_peak_freq_bin` | float | Mean argmax frequency bin across columns |
| `freq_range_bins` | float | Max - min peak frequency bin (FM extent) |
| `freq_modulation_rate` | float | Mean |delta peak_freq| between columns; high = FM sweep |
| `snr_db` | float | Mean (peak_dB - 10th percentile_dB) per column |

### `extract_event_features(event, spectrogram, hop_px=10) → EventFeatures`

Extracts hop-spaced spectrogram columns — one per event window at positions `(start_window + i) * hop_px` — and computes all 11 features. This samples across the full event duration while maintaining 1:1 correspondence with probability values.

## Usage

```python
from usv_spectrogram.postprocessing import extract_event_features

features = extract_event_features(event, spectrogram_db, hop_px=10)
if features.tonality > 0.3 and features.snr_db > 5.0:
    # Likely a real USV
    ...
```

## Key Decisions

- **Hop-spaced column sampling**: Extracts one column per window at `(start_window + i) * hop_px` rather than consecutive columns from `start_col`. This ensures frequency features (range, modulation rate) reflect the full event duration, not just a narrow initial slice. Decision documented in `docs/handoffs/event-features-column-mapping.md`.
- **Tonality = 1 - SFM** (inverted spectral flatness): tonal signals score high, broadband noise scores low. Follows DeepSqueak convention where > 0.3 suggests tonal content.
- **GM computed in log domain**: `exp(mean(log(power)))` avoids underflow with many small values.
- **SNR in dB space**: `peak_dB - noise_floor_dB` (10th percentile) — equivalent to `10*log10(peak/noise)` but avoids dB→linear→dB round-trip.
- **Population std (ddof=0)**: Consistent with the event being the full population, not a sample.

## Integration Points

- **Input:** `USVEvent` from `hysteresis.py`, spectrogram from `AudioLoader.load().spectrogram_db`
- **Output:** `EventFeatures` — consumed by future FP filter / second-stage classifier
- **Parameters:** `hop_px` must match `SlidingInference` stride (default 10)

## ADR References

- ADR-002: STFT parameters (n_fft=512, hop=128, sr=300000) determine the 170 freq bins and ~586 Hz/bin resolution
- Frequency band: 20-120 kHz. To convert `mean_peak_freq_bin` to Hz: `freq_hz = 20000 + bin * 586`
