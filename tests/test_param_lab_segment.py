"""Segment read helper tests."""

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

from usv_spectrogram.io_wav import load_wav_segment_mono


class TestParamLabSegment(unittest.TestCase):
    def test_load_wav_segment_mono(self) -> None:
        sample_rate_hz = 1000
        samples = np.linspace(-1.0, 1.0, 1000, dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "segment.wav"
            sf.write(wav_path, samples, sample_rate_hz, subtype="FLOAT")

            segment, rate = load_wav_segment_mono(wav_path, start_s=0.2, duration_s=0.3)

        self.assertEqual(rate, sample_rate_hz)
        expected = samples[200:500]
        np.testing.assert_allclose(segment, expected, atol=1e-6, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
