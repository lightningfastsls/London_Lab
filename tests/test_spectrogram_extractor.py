"""Tests for spectrogram extraction module.

Tests cover:
- ExtractionConfig validation and defaults
- SpectrogramExtractor single extraction
- SpectrogramExtractor batch extraction
- Review vs training render modes
- Frequency band limiting
- dB conversion accuracy
- Edge cases (missing WAV, short audio)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.detection.candidate import Candidate
from usv_spectrogram.detection.extraction_config import ExtractionConfig
from usv_spectrogram.detection.spectrogram_extractor import SpectrogramExtractor


class TestExtractionConfigDefaults:
    """Test that ExtractionConfig has correct default values."""

    def test_extraction_config_defaults(self):
        """Verify all default values are as expected."""
        config = ExtractionConfig()

        # STFT parameters
        assert config.sample_rate == 300_000
        assert config.n_fft == 512
        assert config.hop_length == 128
        assert config.window == "hann"

        # Frequency range
        assert config.freq_min_hz == 20_000
        assert config.freq_max_hz == 120_000

        # Image dimensions
        assert config.image_height_px == 256
        assert config.pixels_per_ms == 2.0
        assert config.min_width_px == 128
        assert config.max_width_px == 512

        # Color scale
        assert config.db_floor == -80.0
        assert config.db_ceiling == 0.0
        assert config.colormap == "magma"

        # Dynamic range
        assert config.dynamic_range_method == "mad"
        assert config.mad_vmin_scale == 2.0
        assert config.mad_vmax_scale == 4.0

        # Render mode
        assert config.default_render_mode == "review"


class TestExtractionConfigValidation:
    """Test ExtractionConfig parameter validation."""

    def test_extraction_config_validation_sample_rate(self):
        """Invalid sample_rate raises ValueError."""
        with pytest.raises(ValueError, match="sample_rate"):
            ExtractionConfig(sample_rate=0)

        with pytest.raises(ValueError, match="sample_rate"):
            ExtractionConfig(sample_rate=-1)

    def test_extraction_config_validation_n_fft(self):
        """Invalid n_fft raises ValueError."""
        with pytest.raises(ValueError, match="n_fft"):
            ExtractionConfig(n_fft=0)

        with pytest.raises(ValueError, match="n_fft"):
            ExtractionConfig(n_fft=-512)

    def test_extraction_config_validation_hop_length(self):
        """Invalid hop_length raises ValueError."""
        with pytest.raises(ValueError, match="hop_length"):
            ExtractionConfig(hop_length=0)

        with pytest.raises(ValueError, match="hop_length"):
            ExtractionConfig(n_fft=512, hop_length=512)

        with pytest.raises(ValueError, match="hop_length"):
            ExtractionConfig(n_fft=512, hop_length=1024)

    def test_extraction_config_validation_frequency_bounds(self):
        """freq_min >= freq_max raises ValueError."""
        with pytest.raises(ValueError, match="freq_min_hz"):
            ExtractionConfig(freq_min_hz=80_000, freq_max_hz=50_000)

        with pytest.raises(ValueError, match="freq_min_hz"):
            ExtractionConfig(freq_min_hz=60_000, freq_max_hz=60_000)

    def test_extraction_config_validation_nyquist(self):
        """freq_max > sample_rate/2 raises ValueError."""
        with pytest.raises(ValueError, match="Nyquist"):
            ExtractionConfig(sample_rate=200_000, freq_max_hz=120_000)  # Nyquist=100kHz

    def test_extraction_config_validation_window(self):
        """Invalid window type raises ValueError."""
        with pytest.raises(ValueError, match="window"):
            ExtractionConfig(window="invalid")

        with pytest.raises(ValueError, match="window"):
            ExtractionConfig(window="kaiser")

    def test_extraction_config_validation_image_dims(self):
        """Invalid image dimensions raise ValueError."""
        with pytest.raises(ValueError, match="image_height_px"):
            ExtractionConfig(image_height_px=0)

        with pytest.raises(ValueError, match="pixels_per_ms"):
            ExtractionConfig(pixels_per_ms=0)

        with pytest.raises(ValueError, match="min_width_px"):
            ExtractionConfig(min_width_px=0)

        with pytest.raises(ValueError, match="max_width_px"):
            ExtractionConfig(min_width_px=128, max_width_px=64)

    def test_extraction_config_validation_db_range(self):
        """db_floor >= db_ceiling raises ValueError."""
        with pytest.raises(ValueError, match="db_floor"):
            ExtractionConfig(db_floor=-40.0, db_ceiling=-80.0)

        with pytest.raises(ValueError, match="db_floor"):
            ExtractionConfig(db_floor=0.0, db_ceiling=0.0)

    def test_extraction_config_validation_render_mode(self):
        """Invalid default_render_mode raises ValueError."""
        with pytest.raises(ValueError, match="default_render_mode"):
            ExtractionConfig(default_render_mode="invalid")

        with pytest.raises(ValueError, match="default_render_mode"):
            ExtractionConfig(default_render_mode="export")

    def test_extraction_config_validation_dynamic_range_method(self):
        """Invalid dynamic_range_method raises ValueError."""
        with pytest.raises(ValueError, match="dynamic_range_method"):
            ExtractionConfig(dynamic_range_method="invalid")

    def test_extraction_config_validation_mad_scales(self):
        """Invalid mad scale values raise ValueError."""
        with pytest.raises(ValueError, match="mad_vmin_scale"):
            ExtractionConfig(mad_vmin_scale=0)

        with pytest.raises(ValueError, match="mad_vmin_scale"):
            ExtractionConfig(mad_vmin_scale=-1)

        with pytest.raises(ValueError, match="mad_vmax_scale"):
            ExtractionConfig(mad_vmax_scale=0)


class TestExtractionConfigMethods:
    """Test ExtractionConfig helper methods."""

    def test_extraction_config_width_calculation(self):
        """width_px_for_duration_ms() clamps correctly."""
        config = ExtractionConfig(
            pixels_per_ms=2.0,
            min_width_px=128,
            max_width_px=512,
        )

        # Below minimum
        assert config.width_px_for_duration_ms(50.0) == 128  # 50*2=100 -> clamp to 128

        # Within range
        assert config.width_px_for_duration_ms(150.0) == 300  # 150*2=300

        # Above maximum
        assert config.width_px_for_duration_ms(500.0) == 512  # 500*2=1000 -> clamp to 512

        # Exactly at minimum
        assert config.width_px_for_duration_ms(64.0) == 128  # 64*2=128

        # Exactly at maximum
        assert config.width_px_for_duration_ms(256.0) == 512  # 256*2=512

    def test_freq_resolution_hz(self):
        """freq_resolution_hz() returns correct value."""
        config = ExtractionConfig(sample_rate=300_000, n_fft=512)
        expected = 300_000 / 512  # ~586 Hz
        assert abs(config.freq_resolution_hz() - expected) < 1.0

    def test_hop_ms(self):
        """hop_ms() returns correct value."""
        config = ExtractionConfig(sample_rate=300_000, hop_length=128)
        expected = (128 / 300_000) * 1000.0  # ~0.427 ms
        assert abs(config.hop_ms() - expected) < 0.001


class TestExtractSingle:
    """Test single candidate extraction."""

    def test_extract_single_creates_png(self, create_tone_wav, tmp_path):
        """Extracts a single candidate and creates PNG file."""
        # Create a WAV with a 50ms tone at 55 kHz
        wav_path = create_tone_wav(
            freq_hz=55_000,
            duration_ms=50.0,
            amplitude=0.8,
            sample_rate=300_000,
            start_offset_ms=100.0,
        )

        # Create a candidate for this tone
        candidate = Candidate.create(
            source_file=wav_path,
            start_ms=100.0,
            end_ms=150.0,
            peak_freq_hz=55_000,
            peak_energy_db=-20.0,
            context_before_ms=50.0,
            context_after_ms=50.0,
        )

        # Extract spectrogram
        extractor = SpectrogramExtractor()
        output_path = extractor.extract_single(
            candidate=candidate,
            wav_dir=wav_path.parent,
            output_dir=tmp_path,
            render_mode="review",
        )

        # Verify PNG was created
        assert output_path is not None
        assert output_path.exists()
        assert output_path.suffix == ".png"
        assert output_path.name == f"{candidate.candidate_id}.png"

    def test_extract_single_review_mode(self, create_tone_wav, tmp_path):
        """Review mode includes axes and labels."""
        wav_path = create_tone_wav(
            freq_hz=60_000,
            duration_ms=50.0,
            amplitude=0.8,
            sample_rate=300_000,
        )

        candidate = Candidate.create(
            source_file=wav_path,
            start_ms=50.0,
            end_ms=100.0,
            peak_freq_hz=60_000,
            peak_energy_db=-25.0,
        )

        extractor = SpectrogramExtractor()
        output_path = extractor.extract_single(
            candidate=candidate,
            wav_dir=wav_path.parent,
            output_dir=tmp_path,
            render_mode="review",
        )

        assert output_path is not None
        assert output_path.exists()

        # Review mode creates larger files with matplotlib axes
        # We can't easily verify the axes without parsing the image,
        # but we can check the file exists and is larger than a minimal PNG
        assert output_path.stat().st_size > 1000  # At least 1KB

    def test_extract_single_training_mode(self, create_tone_wav, tmp_path):
        """Training mode creates raw image."""
        wav_path = create_tone_wav(
            freq_hz=60_000,
            duration_ms=50.0,
            amplitude=0.8,
            sample_rate=300_000,
        )

        candidate = Candidate.create(
            source_file=wav_path,
            start_ms=50.0,
            end_ms=100.0,
            peak_freq_hz=60_000,
            peak_energy_db=-25.0,
        )

        config = ExtractionConfig(
            image_height_px=256,
            min_width_px=128,
            max_width_px=512,
        )
        extractor = SpectrogramExtractor(config)
        output_path = extractor.extract_single(
            candidate=candidate,
            wav_dir=wav_path.parent,
            output_dir=tmp_path,
            render_mode="training",
        )

        assert output_path is not None
        assert output_path.exists()

        # Verify image dimensions
        img = Image.open(output_path)
        assert img.height == 256
        # Width should be between min and max
        assert 128 <= img.width <= 512

    def test_extract_single_missing_wav(self, tmp_path):
        """Returns None if WAV file not found."""
        # Create a candidate pointing to a non-existent WAV
        nonexistent_wav = tmp_path / "nonexistent.wav"
        candidate = Candidate.create(
            source_file=nonexistent_wav,
            start_ms=0.0,
            end_ms=50.0,
            peak_freq_hz=55_000,
            peak_energy_db=-20.0,
        )

        extractor = SpectrogramExtractor()
        output_path = extractor.extract_single(
            candidate=candidate,
            wav_dir=tmp_path,
            output_dir=tmp_path / "output",
        )

        # Should return None when WAV is missing
        assert output_path is None

    def test_extract_single_too_short_audio(self, create_tone_wav, tmp_path):
        """Returns None if audio segment is shorter than n_fft."""
        # Create a very short tone
        wav_path = create_tone_wav(
            freq_hz=60_000,
            duration_ms=1.0,  # Very short
            amplitude=0.8,
            sample_rate=300_000,
        )

        # Try to extract a segment that's too short
        candidate = Candidate.create(
            source_file=wav_path,
            start_ms=50.0,
            end_ms=50.5,  # Only 0.5ms
            peak_freq_hz=60_000,
            peak_energy_db=-20.0,
            context_before_ms=0.2,
            context_after_ms=0.2,
        )

        extractor = SpectrogramExtractor()
        output_path = extractor.extract_single(
            candidate=candidate,
            wav_dir=wav_path.parent,
            output_dir=tmp_path,
        )

        # Should return None because segment < n_fft samples
        assert output_path is None


class TestComputeSpectrogram:
    """Test spectrogram computation."""

    def test_compute_spectrogram_frequency_band(self, create_tone_wav):
        """Spectrogram limited to configured frequency range."""
        wav_path = create_tone_wav(
            freq_hz=60_000,
            duration_ms=50.0,
            amplitude=0.8,
            sample_rate=300_000,
        )

        # Load audio
        from usv_spectrogram.io_wav import load_wav_mono

        samples, sample_rate = load_wav_mono(wav_path)

        # Compute spectrogram with specific frequency band
        config = ExtractionConfig(
            sample_rate=300_000,
            freq_min_hz=40_000,
            freq_max_hz=80_000,
        )
        extractor = SpectrogramExtractor(config)
        spec_db, freqs_hz, times_s = extractor._compute_spectrogram(samples, sample_rate)

        # Verify frequency range
        assert freqs_hz.min() >= 40_000
        assert freqs_hz.max() <= 80_000
        assert spec_db.shape[0] == len(freqs_hz)

    def test_compute_spectrogram_db_conversion(self, create_tone_wav):
        """Correct dB conversion (20*log10)."""
        wav_path = create_tone_wav(
            freq_hz=60_000,
            duration_ms=50.0,
            amplitude=0.5,
            sample_rate=300_000,
            noise_level=0.001,
        )

        from usv_spectrogram.io_wav import load_wav_mono

        samples, sample_rate = load_wav_mono(wav_path)

        config = ExtractionConfig(sample_rate=300_000)
        extractor = SpectrogramExtractor(config)
        spec_db, freqs_hz, times_s = extractor._compute_spectrogram(samples, sample_rate)

        # Verify spec_db is in reasonable dB range
        # With 0.5 amplitude tone, we expect values around -6 to -10 dB at peak
        assert np.max(spec_db) > -20.0  # Should have some strong energy
        assert np.min(spec_db) < -40.0  # Should have some quiet regions

        # Verify dB conversion formula: 20*log10
        # This is hard to test directly, but we can verify the range is reasonable
        assert not np.any(np.isnan(spec_db))
        assert not np.any(np.isinf(spec_db))

    def test_compute_spectrogram_empty_audio(self):
        """Empty audio returns empty arrays."""
        config = ExtractionConfig()
        extractor = SpectrogramExtractor(config)

        # Very short audio (less than n_fft samples)
        samples = np.zeros(100, dtype=np.float32)
        spec_db, freqs_hz, times_s = extractor._compute_spectrogram(samples, 300_000)

        # Should return empty arrays
        assert spec_db.size == 0
        assert freqs_hz.size == 0
        assert times_s.size == 0


class TestExtractBatch:
    """Test batch extraction from CSV."""

    def test_extract_batch_processes_all(self, create_multi_tone_wav, tmp_path):
        """Batch extraction processes all candidates in CSV."""
        # Create a WAV with multiple tones
        tones = [
            {"freq_hz": 50_000, "start_ms": 50, "duration_ms": 50, "amplitude": 0.8},
            {"freq_hz": 60_000, "start_ms": 200, "duration_ms": 50, "amplitude": 0.8},
            {"freq_hz": 70_000, "start_ms": 350, "duration_ms": 50, "amplitude": 0.8},
        ]
        wav_path = create_multi_tone_wav(
            tones=tones,
            total_duration_ms=500.0,
            sample_rate=300_000,
        )

        # Create candidates CSV
        candidates = [
            Candidate.create(
                source_file=wav_path,
                start_ms=50.0,
                end_ms=100.0,
                peak_freq_hz=50_000,
                peak_energy_db=-20.0,
            ),
            Candidate.create(
                source_file=wav_path,
                start_ms=200.0,
                end_ms=250.0,
                peak_freq_hz=60_000,
                peak_energy_db=-22.0,
            ),
            Candidate.create(
                source_file=wav_path,
                start_ms=350.0,
                end_ms=400.0,
                peak_freq_hz=70_000,
                peak_energy_db=-21.0,
            ),
        ]

        # Write candidates to CSV
        csv_path = tmp_path / "candidates.csv"
        fieldnames = list(candidates[0].to_dict().keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for c in candidates:
                writer.writerow(c.to_dict())

        # Extract batch
        extractor = SpectrogramExtractor()
        output_dir = tmp_path / "spectrograms"
        results = list(
            extractor.extract_batch(
                candidates_csv=csv_path,
                wav_dir=wav_path.parent,
                output_dir=output_dir,
                render_mode="training",
            )
        )

        # Verify all candidates were processed
        assert len(results) == 3

        # Verify all spectrograms were created
        for candidate, output_path in results:
            assert output_path is not None
            assert output_path.exists()

    def test_extract_batch_handles_missing_wav(self, tmp_path):
        """Batch extraction handles missing WAV files gracefully."""
        # Create candidates CSV with non-existent WAV
        nonexistent_wav = tmp_path / "nonexistent.wav"
        candidates = [
            Candidate.create(
                source_file=nonexistent_wav,
                start_ms=0.0,
                end_ms=50.0,
                peak_freq_hz=55_000,
                peak_energy_db=-20.0,
            ),
        ]

        csv_path = tmp_path / "candidates.csv"
        fieldnames = list(candidates[0].to_dict().keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for c in candidates:
                writer.writerow(c.to_dict())

        # Extract batch
        extractor = SpectrogramExtractor()
        results = list(
            extractor.extract_batch(
                candidates_csv=csv_path,
                wav_dir=tmp_path,
                output_dir=tmp_path / "output",
            )
        )

        # Should return result with None output_path
        assert len(results) == 1
        candidate, output_path = results[0]
        assert output_path is None

    def test_save_updated_csv(self, create_tone_wav, tmp_path):
        """save_updated_csv() writes CSV with spectrogram paths."""
        wav_path = create_tone_wav(
            freq_hz=55_000,
            duration_ms=50.0,
            amplitude=0.8,
            sample_rate=300_000,
        )

        candidate = Candidate.create(
            source_file=wav_path,
            start_ms=50.0,
            end_ms=100.0,
            peak_freq_hz=55_000,
            peak_energy_db=-20.0,
        )

        extractor = SpectrogramExtractor()
        output_path = extractor.extract_single(
            candidate=candidate,
            wav_dir=wav_path.parent,
            output_dir=tmp_path / "spectrograms",
        )

        # Save updated CSV
        results = [(candidate, output_path)]
        csv_path = tmp_path / "updated.csv"
        extractor.save_updated_csv(results, csv_path)

        # Verify CSV was created
        assert csv_path.exists()

        # Read back and verify spectrogram_path is populated
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["spectrogram_path"] != ""
        assert Path(rows[0]["spectrogram_path"]).name == f"{candidate.candidate_id}.png"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_different_window_functions(self, create_tone_wav, tmp_path):
        """Extraction works with different window functions."""
        wav_path = create_tone_wav(
            freq_hz=60_000,
            duration_ms=50.0,
            amplitude=0.8,
            sample_rate=300_000,
        )

        candidate = Candidate.create(
            source_file=wav_path,
            start_ms=50.0,
            end_ms=100.0,
            peak_freq_hz=60_000,
            peak_energy_db=-20.0,
        )

        # Test different window functions
        for window in ["hann", "hamming", "blackman", "boxcar"]:
            config = ExtractionConfig(window=window)
            extractor = SpectrogramExtractor(config)
            output_path = extractor.extract_single(
                candidate=candidate,
                wav_dir=wav_path.parent,
                output_dir=tmp_path / window,
            )

            assert output_path is not None
            assert output_path.exists()

    def test_different_colormaps(self, create_tone_wav, tmp_path):
        """Extraction works with different colormaps."""
        wav_path = create_tone_wav(
            freq_hz=60_000,
            duration_ms=50.0,
            amplitude=0.8,
            sample_rate=300_000,
        )

        candidate = Candidate.create(
            source_file=wav_path,
            start_ms=50.0,
            end_ms=100.0,
            peak_freq_hz=60_000,
            peak_energy_db=-20.0,
        )

        # Test different colormaps
        for cmap in ["magma", "viridis", "gray", "hot"]:
            config = ExtractionConfig(colormap=cmap)
            extractor = SpectrogramExtractor(config)
            output_path = extractor.extract_single(
                candidate=candidate,
                wav_dir=wav_path.parent,
                output_dir=tmp_path / cmap,
                render_mode="training",
            )

            assert output_path is not None
            assert output_path.exists()

    def test_width_clamping(self, create_tone_wav, tmp_path):
        """Image width is clamped to min/max."""
        # Create very short tone (should pad to min_width)
        wav_path_short = create_tone_wav(
            freq_hz=60_000,
            duration_ms=10.0,  # Very short
            amplitude=0.8,
            sample_rate=300_000,
        )

        candidate_short = Candidate.create(
            source_file=wav_path_short,
            start_ms=50.0,
            end_ms=60.0,
            peak_freq_hz=60_000,
            peak_energy_db=-20.0,
            context_before_ms=5.0,
            context_after_ms=5.0,
        )

        config = ExtractionConfig(min_width_px=128, max_width_px=512)
        extractor = SpectrogramExtractor(config)

        output_path = extractor.extract_single(
            candidate=candidate_short,
            wav_dir=wav_path_short.parent,
            output_dir=tmp_path / "short",
            render_mode="training",
        )

        assert output_path is not None
        img = Image.open(output_path)
        assert img.width >= 128  # Should be padded to min
