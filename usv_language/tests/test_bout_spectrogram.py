"""Tests for bout spectrogram computation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usv_language.data.spectrogram import BoutSpectrogramConfig, compute_bout_spectrogram


@pytest.fixture
def bout_wav(tmp_path):
    """Create a synthetic WAV file at 300 kHz with a 60 kHz tone."""
    sr = 300_000
    duration_s = 0.5
    n_samples = int(sr * duration_s)
    t = np.arange(n_samples) / sr
    signal = (
        0.01 * np.random.RandomState(42).randn(n_samples).astype(np.float32)
        + 0.5 * np.sin(2 * np.pi * 60_000 * t).astype(np.float32)
    )
    wav_path = tmp_path / "bout_test.wav"
    sf.write(str(wav_path), signal, sr, subtype="FLOAT")
    return wav_path, sr, duration_s


class TestBoutSpectrogramConfig:
    def test_default_values(self):
        config = BoutSpectrogramConfig()
        assert config.sample_rate == 300_000
        assert config.n_fft == 512
        assert config.hop_length == 128
        assert config.freq_min_hz == 20_000
        assert config.freq_max_hz == 120_000
        assert config.window == "hann"
        assert config.eps == 1e-10

    def test_frozen(self):
        config = BoutSpectrogramConfig()
        with pytest.raises(AttributeError):
            config.sample_rate = 44100  # type: ignore[misc]

    def test_freq_bins_count(self):
        """20-120 kHz at sr=300k, n_fft=512 should give ~170 bins."""
        config = BoutSpectrogramConfig()
        n_bins = config.get_n_freq_bins()
        assert n_bins == 170

    def test_validation_invalid_freq_range(self):
        with pytest.raises(ValueError, match="freq_min_hz"):
            BoutSpectrogramConfig(freq_min_hz=120_000, freq_max_hz=20_000)


class TestComputeBoutSpectrogram:
    def test_output_shape(self, bout_wav):
        """Output should be (170, T) where T depends on audio length."""
        wav_path, sr, duration_s = bout_wav
        config = BoutSpectrogramConfig()
        spec = compute_bout_spectrogram(wav_path, 0.0, duration_s, config)

        assert spec.shape[0] == 170  # freq bins
        # T = 1 + (n_samples - n_fft) // hop_length
        expected_samples = int(sr * duration_s)
        expected_frames = 1 + (expected_samples - config.n_fft) // config.hop_length
        assert spec.shape[1] == expected_frames

    def test_frequency_bins_match_20_120khz(self, bout_wav):
        """Band mask should select exactly 170 bins for 20-120kHz."""
        config = BoutSpectrogramConfig()
        freqs = config.get_freq_bins()
        mask = config.get_band_mask()
        selected_freqs = freqs[mask]
        assert selected_freqs[0] >= 20_000
        assert selected_freqs[-1] <= 120_000
        assert len(selected_freqs) == 170

    def test_short_audio_handled(self, tmp_path):
        """Audio shorter than n_fft should return empty spectrogram."""
        sr = 300_000
        # Only 100 samples — much less than n_fft=512
        signal = np.random.randn(100).astype(np.float32)
        wav_path = tmp_path / "short.wav"
        sf.write(str(wav_path), signal, sr, subtype="FLOAT")

        config = BoutSpectrogramConfig()
        spec = compute_bout_spectrogram(wav_path, 0.0, 0.001, config)
        assert spec.shape[0] == 170
        assert spec.shape[1] == 0

    def test_db_values_in_expected_range(self, bout_wav):
        """dB values should be finite and in a reasonable range."""
        wav_path, _, duration_s = bout_wav
        config = BoutSpectrogramConfig()
        spec = compute_bout_spectrogram(wav_path, 0.0, duration_s, config)

        assert np.all(np.isfinite(spec))
        # With eps=1e-10, floor is 20*log10(1e-10) = -200 dB
        assert np.all(spec >= -200.0)

    def test_no_negative_infinity(self, bout_wav):
        """eps floor should prevent -inf in dB output."""
        wav_path, _, duration_s = bout_wav
        config = BoutSpectrogramConfig(eps=1e-10)
        spec = compute_bout_spectrogram(wav_path, 0.0, duration_s, config)
        assert not np.any(np.isinf(spec))

    def test_subsegment_extraction(self, bout_wav):
        """Should correctly extract only the requested time range."""
        wav_path, sr, duration_s = bout_wav
        config = BoutSpectrogramConfig()

        # Full spectrogram
        full = compute_bout_spectrogram(wav_path, 0.0, duration_s, config)
        # Half spectrogram
        half = compute_bout_spectrogram(wav_path, 0.0, duration_s / 2, config)

        # Half should have roughly half the columns
        ratio = half.shape[1] / full.shape[1]
        assert 0.4 < ratio < 0.6

    def test_output_dtype_float32(self, bout_wav):
        """Output should be float32."""
        wav_path, _, duration_s = bout_wav
        spec = compute_bout_spectrogram(wav_path, 0.0, duration_s)
        assert spec.dtype == np.float32

    def test_sample_rate_mismatch_raises(self, tmp_path):
        """Should raise if WAV sample rate doesn't match config."""
        sr = 44100  # Wrong sample rate
        signal = np.random.randn(44100).astype(np.float32)
        wav_path = tmp_path / "wrong_sr.wav"
        sf.write(str(wav_path), signal, sr, subtype="FLOAT")

        config = BoutSpectrogramConfig(sample_rate=300_000)
        with pytest.raises(ValueError, match="sample rate"):
            compute_bout_spectrogram(wav_path, 0.0, 1.0, config)
