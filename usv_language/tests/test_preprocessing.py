"""Tests for data preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from usv_language.src.data.preprocessing import (
    STFTConfig,
    compute_full_spectrogram,
    preprocess_single_wav,
    preprocess_all_wavs,
)


class TestSTFTConfig:
    def test_defaults(self):
        cfg = STFTConfig()
        assert cfg.sample_rate == 300_000
        assert cfg.n_fft == 512
        assert cfg.hop_length == 128

    def test_n_freq_bins(self):
        cfg = STFTConfig()
        # 20-120 kHz at 300k SR, 512 FFT => freq_resolution ~585 Hz
        # Should be approximately 170 bins
        assert 150 < cfg.n_freq_bins < 200

    def test_freq_resolution(self):
        cfg = STFTConfig()
        assert abs(cfg.freq_resolution_hz - 300_000 / 512) < 1.0

    def test_hop_duration(self):
        cfg = STFTConfig()
        assert abs(cfg.hop_duration_s - 128 / 300_000) < 1e-8

    def test_validation_errors(self):
        with pytest.raises(ValueError, match="sample_rate"):
            STFTConfig(sample_rate=-1)
        with pytest.raises(ValueError, match="n_fft"):
            STFTConfig(n_fft=0)
        with pytest.raises(ValueError, match="hop_length"):
            STFTConfig(hop_length=600)  # >= n_fft
        with pytest.raises(ValueError, match="freq_min_hz.*freq_max_hz"):
            STFTConfig(freq_min_hz=130_000, freq_max_hz=120_000)
        with pytest.raises(ValueError, match="Nyquist"):
            STFTConfig(freq_max_hz=200_000)

    def test_frozen(self):
        cfg = STFTConfig()
        with pytest.raises(AttributeError):
            cfg.sample_rate = 250_000


class TestComputeFullSpectrogram:
    def test_shape(self, synthetic_wav):
        cfg = STFTConfig()
        spec = compute_full_spectrogram(synthetic_wav, cfg)
        assert spec.ndim == 2
        assert spec.shape[0] == cfg.n_freq_bins
        assert spec.shape[1] > 0
        assert spec.dtype == np.float32

    def test_contains_values(self, synthetic_wav):
        cfg = STFTConfig()
        spec = compute_full_spectrogram(synthetic_wav, cfg)
        # dB values should be finite
        assert np.all(np.isfinite(spec))


class TestPreprocessSingleWav:
    def test_output_keys(self, synthetic_wav):
        result = preprocess_single_wav(synthetic_wav, STFTConfig())
        assert "spectrogram" in result
        assert "wav_stem" in result
        assert "n_frames" in result
        assert "n_freq_bins" in result

    def test_spectrogram_shape(self, synthetic_wav):
        cfg = STFTConfig()
        result = preprocess_single_wav(synthetic_wav, cfg)
        assert result["spectrogram"].shape[0] == cfg.n_freq_bins
        assert result["n_frames"] == result["spectrogram"].shape[1]


class TestPreprocessAllWavs:
    def test_batch_process(self, wav_dir_with_files, tmp_path):
        output_h5 = tmp_path / "output.h5"
        summary = preprocess_all_wavs(wav_dir_with_files, output_h5, STFTConfig())
        assert summary["n_files"] == 3
        assert summary["total_frames"] > 0

        import h5py
        with h5py.File(output_h5, "r") as f:
            assert len(f.keys()) == 3
            for stem in f:
                assert "spectrogram" in f[stem]

    def test_empty_dir(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            preprocess_all_wavs(empty_dir, tmp_path / "out.h5")
