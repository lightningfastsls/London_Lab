"""Heuristic candidate detection tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.param_lab.heuristic_detect import (
    HeuristicConfig,
    detect_candidates,
    summarize_candidates,
)


class TestParamLabHeuristic(unittest.TestCase):
    def test_detect_candidates(self) -> None:
        spec_db = np.full((10, 20), -80.0, dtype=float)
        spec_db[2:5, 3:8] = -20.0
        spec_db[6:9, 12:16] = -10.0

        freqs_hz = np.linspace(30000.0, 90000.0, 10)
        times_s = np.linspace(0.0, 0.2, 20)

        cfg = HeuristicConfig(
            threshold_db=20.0,
            min_area_bins=4,
            min_frames=2,
            min_bins=2,
        )

        candidates = detect_candidates(spec_db, freqs_hz, times_s, cfg)
        summary = summarize_candidates(candidates)

        self.assertEqual(len(candidates), 2)
        self.assertEqual(summary["count"], 2.0)
        self.assertGreater(summary["total_area_bins"], 0.0)


if __name__ == "__main__":
    unittest.main()
