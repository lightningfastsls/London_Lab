# Acoustic Property Extractors

**Phase:** Probing targets for linear probing classifiers
**ADRs:** ADR-001 (sr=300kHz), ADR-002 (hop=128)
**Tests:** `usv_language/tests/test_acoustic_properties.py` -- 22 tests across 9 test classes
**Dependencies:** NumPy only (no torch/scipy/matplotlib)

## Purpose

Extract ground-truth acoustic properties from dB-scaled spectrogram columns. These serve as **probing targets** — labels that simple linear classifiers predict from transformer hidden states, testing whether the transformer has learned to represent specific acoustic features.

## Public Interface

### AcousticPropertyConfig (frozen dataclass)

| Field | Default | Description |
|-------|---------|-------------|
| `freq_min_hz` | 20,000.0 | Lower frequency bound (Hz) |
| `freq_max_hz` | 120,000.0 | Upper frequency bound (Hz) |
| `n_freq_bins` | 170 | Number of frequency bins per column |
| `sample_rate` | 300,000 | Audio sample rate (ADR-001) |
| `hop_length` | 128 | STFT hop length (ADR-002) |
| `voiced_threshold_db` | -50.0 | Max dB above which a frame is voiced |
| `direction_threshold` | 500.0 | Min Hz delta for rising/falling classification |

Note: `n_freq_bins` corresponds to `AnalysisConfig.n_freq` — same concept, standalone name for independent use.

### Single-Column Functions (6)

| Function | Signature | Returns | Notes |
|----------|-----------|---------|-------|
| `peak_frequency` | `(col, cfg)` | float Hz | `freqs[argmax(col)]` |
| `spectral_centroid` | `(col, cfg)` | float Hz | `sum(f*E)/sum(E)`, E=10^(dB/10), center-freq fallback on silence |
| `energy` | `(col)` | float | `sum(10^(dB/10))` — linear power |
| `is_voiced` | `(col, cfg)` | bool | `max(col) > threshold_db` |
| `frequency_direction` | `(prev, curr, cfg)` | str | `'rising'`/`'falling'`/`'flat'` based on peak_frequency delta |
| `bout_position` | `(idx, length)` | float | `idx/length` in [0, (T-1)/T], 0.0 for single-frame |

### Temporal Functions (1)

| Function | Signature | Returns | Notes |
|----------|-----------|---------|-------|
| `time_since_last_usv` | `(idx, onsets, cfg)` | float ms | Binary search; -1.0 if no preceding onset; defensively sorts unsorted onsets |

### Batch Extractor (1)

| Function | Signature | Returns |
|----------|-----------|---------|
| `extract_all_properties` | `(spec, cfg, onsets=None)` | `dict[str, ndarray]` |

Returns 7 arrays of length T: `peak_frequency`, `spectral_centroid`, `energy`, `is_voiced`, `frequency_direction`, `bout_position`, `time_since_last_usv`.

Vectorized where possible (peak/centroid/energy/voiced/direction/position via NumPy ops). The `time_since_last_usv` uses an O(T+N) linear scan with pointer advancement.

**Input contract:** `spec` must be 2-D `(n_freq_bins, T)`, raises `ValueError` otherwise. Onsets are defensively sorted if not already ascending.

## dB-to-Linear Conversion

The conversion `10^(S_db/10)` is deliberate. Since the upstream STFT stores `S_db = 20*log10(|X|)` (amplitude dB), inverting with `/10` gives `|X|^2` (power). This is physically correct for energy summation and spectral centroid weighting.

## Frequency Axis

`np.linspace(freq_min_hz, freq_max_hz, n_freq_bins)` — consistent with `codebook_viz.py` line 150.

## Relationship to AnalysisConfig

`AcousticPropertyConfig` is independent of `AnalysisConfig` — it can be used without the full VQ-VAE analysis suite. The frequency parameters share the same defaults.
