"""Tests for usv_spectrogram.classifier.resample — Module 18.2b resample.py.

Written by test-architect BEFORE implementation exists. All tests will fail at
collection (ImportError) until resample.py is created. That is the expected TDD
red phase.

ROADMAP §18.2b test plan coverage:
  1. len(out) ≈ len(in) * 5/6  ->  test_duration_preserved_within_one_sample
  2. Stereo raises ValueError   ->  test_stereo_input_raises_value_error
  3. Returns float32 dtype      ->  test_output_dtype_is_float32
  4. 60 kHz tone passes clean   ->  test_60khz_tone_passes_unscathed
  5. 140 kHz tone is aliased    ->  test_140khz_tone_anti_aliased_below_40db

Additional coverage (recurring gap patterns):
  - Empty 1-D array             ->  test_empty_array_returns_empty
  - Single-sample input         ->  test_single_sample_input
  - Float64 input → float32 out ->  test_float64_input_coerced_to_float32
  - 2-D row vector raises       ->  test_2d_row_vector_raises_value_error

Total: 9 tests (5 from ROADMAP, 4 additional)

DSP note on test 5:
  300 kHz source, 250 kHz target.  New Nyquist = 125 kHz.
  A 140 kHz tone would alias to |250 - 140| = 110 kHz after naive decimation.
  resample_poly uses a Kaiser-window FIR anti-aliasing filter that attenuates
  content above the new Nyquist before decimation.  The test asserts the alias
  at 110 kHz in the output is ≥ 40 dB below the original 140 kHz tone power
  (measured in the INPUT frame, since the alias itself may sit in the 0-125 kHz
  passable band after resampling).  40 dB is the ROADMAP spec.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Repo-root sys.path bootstrap (matches pattern in test_cleaning_real_data_loader.py)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Import under test — will fail until resample.py exists (that's correct).
# ---------------------------------------------------------------------------
pytest.importorskip("scipy", reason="scipy required for resample_poly")

from usv_spectrogram.classifier.resample import (  # noqa: E402
    resample_to_vocalmat,
    SOURCE_SAMPLE_RATE_HZ,
    TARGET_SAMPLE_RATE_HZ,
    RESAMPLE_UP,
    RESAMPLE_DOWN,
)

# ---------------------------------------------------------------------------
# Constant sanity — verify the module exports the right values.
# These are load-time assertions (not pytest test functions) because if the
# constants are wrong every test below would have incorrect expectations.
# ---------------------------------------------------------------------------
assert SOURCE_SAMPLE_RATE_HZ == 300_000, "SOURCE_SAMPLE_RATE_HZ must be 300 000 (ADR-001)"
assert TARGET_SAMPLE_RATE_HZ == 250_000, "TARGET_SAMPLE_RATE_HZ must be 250 000 (VocalMat)"
assert RESAMPLE_UP == 5
assert RESAMPLE_DOWN == 6
assert SOURCE_SAMPLE_RATE_HZ * RESAMPLE_UP / RESAMPLE_DOWN == TARGET_SAMPLE_RATE_HZ


# ===========================================================================
# Test 1 (ROADMAP item 1) — Duration preservation
# ===========================================================================

def test_duration_preserved_within_one_sample():
    """Spec: len(out) ≈ len(in) * 5 / 6 within 1 sample tolerance.

    1 second of audio at 300 kHz = 300 000 samples.
    After 5/6 resampling = 250 000 samples exactly.
    The polyphase filter may add or drop ≤1 sample at the boundary.
    """
    duration_samples = 300_000  # exactly 1 second at 300 kHz
    rng = np.random.default_rng(42)
    samples = rng.standard_normal(duration_samples).astype(np.float32)

    out = resample_to_vocalmat(samples)

    expected_len = 250_000
    assert abs(len(out) - expected_len) <= 1, (
        f"Expected output length ≈{expected_len} ± 1, got {len(out)}"
    )


# ===========================================================================
# Test 2 (ROADMAP item 2) — Stereo raises ValueError
# ===========================================================================

def test_stereo_input_raises_value_error():
    """Spec: stereo input (shape (N, 2)) must raise ValueError.

    The function only supports mono (1-D) audio.  Accepting stereo silently
    would corrupt the output — a 2-D array is not a legitimate mono signal.
    """
    stereo = np.zeros((300_000, 2), dtype=np.float32)
    with pytest.raises(ValueError, match=r"mono"):
        resample_to_vocalmat(stereo)


# ===========================================================================
# Test 3 (ROADMAP item 3) — float32 dtype guarantee
# ===========================================================================

def test_output_dtype_is_float32():
    """Spec: output dtype is np.float32 regardless of input dtype.

    Downstream spectrogram generation and CNN inference both assume float32.
    This test explicitly passes float64 to exercise the dtype coercion.
    """
    samples_f64 = np.ones(300_000, dtype=np.float64)
    out = resample_to_vocalmat(samples_f64)
    assert out.dtype == np.float32, (
        f"Expected float32, got {out.dtype}"
    )


def test_float64_input_coerced_to_float32():
    """Complementary dtype check using a non-trivial signal (not just ones).

    Ensures the coercion happens on the resampled result, not just the input.
    """
    rng = np.random.default_rng(7)
    samples_f64 = rng.standard_normal(60_000).astype(np.float64)  # 0.2 s at 300 kHz
    out = resample_to_vocalmat(samples_f64)
    assert out.dtype == np.float32


# ===========================================================================
# Test 4 (ROADMAP item 4) — 60 kHz tone passes clean through resampler
# ===========================================================================

def test_60khz_tone_passes_unscathed():
    """Spec: 60 kHz sine at 300 kHz resampled to 250 kHz keeps peak at 60 kHz,
    amplitude preserved within 2 dB.

    60 kHz is well below the new Nyquist of 125 kHz.  No aliasing or
    significant attenuation is expected: the Kaiser FIR anti-alias filter
    has its transition band near 125 kHz, not 60 kHz.

    Hand-computed expected values:
      - Input tone amplitude: 1.0
      - FFT peak bin at output sample rate 250 000 Hz:
          bin_index = round(60_000 / (250_000 / N_out))
        where N_out ≈ 250 000 / 3 ≈ 25 000 (0.3 s * 250 kHz / 3 doesn't divide cleanly;
        we locate the actual peak bin rather than pinning it exactly).
      - 2 dB tolerance = amplitude ratio ≥ 10^(-2/20) ≈ 0.794.
    """
    # 0.1 second at 300 kHz
    n_in = 30_000  # 0.1 s * 300 000 Hz
    sr_in = SOURCE_SAMPLE_RATE_HZ  # 300 000
    sr_out = TARGET_SAMPLE_RATE_HZ  # 250 000
    t = np.arange(n_in) / sr_in
    tone_freq = 60_000  # Hz
    amplitude = 1.0
    samples = (amplitude * np.sin(2 * np.pi * tone_freq * t)).astype(np.float64)

    out = resample_to_vocalmat(samples)

    n_out = len(out)
    freqs = np.fft.rfftfreq(n_out, d=1.0 / sr_out)
    spectrum = np.abs(np.fft.rfft(out))

    # Locate peak frequency in output
    peak_bin = int(np.argmax(spectrum))
    peak_freq = freqs[peak_bin]

    # Frequency resolution of output FFT
    freq_resolution = sr_out / n_out  # Hz per bin

    # Assert peak is within 1 FFT bin of 60 kHz
    assert abs(peak_freq - tone_freq) <= freq_resolution, (
        f"Expected peak at {tone_freq} Hz ± {freq_resolution:.1f} Hz, "
        f"got peak at {peak_freq:.1f} Hz"
    )

    # Assert amplitude is preserved within 2 dB.
    # For a real-valued tone, rfft peak magnitude ≈ amplitude * n_out / 2.
    expected_magnitude = amplitude * n_out / 2
    actual_magnitude = spectrum[peak_bin]
    ratio = actual_magnitude / expected_magnitude
    ratio_db = 20 * np.log10(ratio)
    assert abs(ratio_db) <= 2.0, (
        f"Amplitude changed by {ratio_db:.2f} dB (limit ±2 dB) — "
        "resample may be attenuating 60 kHz unexpectedly"
    )


# ===========================================================================
# Test 5 (ROADMAP item 5) — 140 kHz tone is anti-aliased
# ===========================================================================

def test_140khz_tone_anti_aliased_below_40db():
    """Spec: 140 kHz tone at 300 kHz → resample → alias energy ≥40 dB below input.

    DSP reasoning:
      140 kHz is above the new Nyquist (125 kHz). Without an anti-alias filter
      it would fold to |250 000 - 140 000| = 110 kHz in the resampled output.
      scipy's resample_poly applies a Kaiser-window FIR low-pass filter before
      decimation, designed to attenuate above Nyquist/2 of the new rate.

    Strategy:
      1. Measure input tone power (in input frame, single-frequency, known).
      2. Compute output FFT at 250 kHz sample rate.
      3. Find energy in a narrow band around the expected alias frequency (110 kHz).
      4. Assert alias_power_db ≤ input_tone_power_db - 40.

    Because 140 kHz > 125 kHz Nyquist, we cannot expect ANY 140 kHz peak in the
    output — we test the alias at 110 kHz instead.
    """
    n_in = 30_000  # 0.1 s at 300 kHz
    sr_in = SOURCE_SAMPLE_RATE_HZ   # 300 000
    sr_out = TARGET_SAMPLE_RATE_HZ  # 250 000
    t = np.arange(n_in) / sr_in
    tone_freq = 140_000  # Hz — above new Nyquist
    amplitude = 1.0
    samples = (amplitude * np.sin(2 * np.pi * tone_freq * t)).astype(np.float64)

    out = resample_to_vocalmat(samples)
    n_out = len(out)

    # Input tone power: amplitude^2 / 2 for a pure sine.
    input_tone_power = (amplitude ** 2) / 2.0
    input_tone_power_db = 10 * np.log10(input_tone_power)

    freqs = np.fft.rfftfreq(n_out, d=1.0 / sr_out)
    spectrum_power = (np.abs(np.fft.rfft(out)) ** 2) / (n_out ** 2)

    # Expected alias frequency after naive decimation: |sr_out - tone_freq| = 110 kHz.
    alias_freq = abs(sr_out - tone_freq)  # 110 000 Hz
    freq_resolution = sr_out / n_out

    # Sum energy in ±5 bins around alias center (conservative window to avoid
    # missing a slightly-shifted alias peak due to spectral leakage).
    alias_center_bin = int(round(alias_freq / freq_resolution))
    window_bins = 5
    lo = max(0, alias_center_bin - window_bins)
    hi = min(len(spectrum_power) - 1, alias_center_bin + window_bins)
    alias_power = float(spectrum_power[lo:hi + 1].sum())

    if alias_power <= 0:
        # If there is literally zero energy in that band, the filter is perfect —
        # definitely passes the ≥40 dB spec.
        return

    alias_power_db = 10 * np.log10(alias_power)
    attenuation_db = input_tone_power_db - alias_power_db

    assert attenuation_db >= 40.0, (
        f"Anti-alias attenuation at 110 kHz = {attenuation_db:.1f} dB "
        f"(required ≥ 40 dB). Input tone at {tone_freq} Hz not sufficiently "
        "attenuated by resample_poly Kaiser FIR."
    )


# ===========================================================================
# Additional tests (gap patterns)
# ===========================================================================

def test_empty_array_returns_empty():
    """Empty input (0 samples) must return an empty float32 array, not raise.

    This is a boundary condition: a caller could legitimately encounter a
    zero-length audio segment (e.g., a trimmed silent region).
    """
    empty = np.array([], dtype=np.float32)
    out = resample_to_vocalmat(empty)
    assert out.shape == (0,), f"Expected empty 1-D array, got shape {out.shape}"
    assert out.dtype == np.float32


def test_single_sample_input():
    """Single-sample input must not crash and must return float32.

    Not a common case but exercises the boundary between 'mono 1-D' (valid)
    and 'empty' (also valid).  The output length is not prescribed precisely
    for a single sample because polyphase filtering semantics are ambiguous,
    but it must be a 1-D float32 array.
    """
    single = np.array([0.5], dtype=np.float32)
    out = resample_to_vocalmat(single)
    assert out.ndim == 1
    assert out.dtype == np.float32


def test_2d_row_vector_raises_value_error():
    """A (1, N) row-vector (accidentally 2-D) must raise ValueError.

    This catches the common numpy mistake of shape (1, N) instead of (N,).
    Both are 'mono' in spirit but the spec explicitly requires 1-D.
    """
    row_vector = np.zeros((1, 300_000), dtype=np.float32)
    with pytest.raises(ValueError):
        resample_to_vocalmat(row_vector)
