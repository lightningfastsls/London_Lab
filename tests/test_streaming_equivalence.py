"""Streaming vs in-memory spectrogram equivalence tests."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.config import SpectrogramConfig
from usv_spectrogram.io_wav import load_wav_mono
from usv_spectrogram.spectrogram import compute_spectrogram_db
from usv_spectrogram.stft_stream import stream_wav_spectrogram_db


class TestStreamingEquivalence(unittest.TestCase):
    def test_streaming_matches_in_memory(self) -> None:
        sample_rate_hz = 250_000
        duration_s = 0.05
        t = np.linspace(0, duration_s, int(sample_rate_hz * duration_s), endpoint=False)
        samples = (
            0.1 * np.sin(2 * np.pi * 50_000 * t)
            + 0.05 * np.sin(2 * np.pi * 80_000 * t)
        ).astype(np.float32)

        cfg = SpectrogramConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "test.wav"
            sf.write(wav_path, samples, sample_rate_hz)

            file_samples, file_rate = load_wav_mono(wav_path)
            mem_spec, mem_freqs, mem_times = compute_spectrogram_db(
                file_samples, file_rate, cfg
            )

            spec_chunks = []
            time_chunks = []
            stream_freqs = None
            for spec_db, freqs_hz, times_s in stream_wav_spectrogram_db(
                wav_path, cfg, block_size_samples=4096, progress=False
            ):
                if stream_freqs is None:
                    stream_freqs = freqs_hz
                else:
                    np.testing.assert_allclose(freqs_hz, stream_freqs)
                spec_chunks.append(spec_db)
                time_chunks.append(times_s)

            self.assertIsNotNone(stream_freqs)
            stream_spec = np.concatenate(spec_chunks, axis=1)
            stream_times = np.concatenate(time_chunks)

            np.testing.assert_allclose(mem_freqs, stream_freqs)
            self.assertEqual(mem_spec.shape, stream_spec.shape)
            np.testing.assert_allclose(mem_times, stream_times, atol=1e-9, rtol=0.0)

            max_abs_diff = float(np.max(np.abs(mem_spec - stream_spec)))
            self.assertLessEqual(max_abs_diff, 1e-3)


if __name__ == "__main__":
    unittest.main()
