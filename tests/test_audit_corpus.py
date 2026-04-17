"""Tests for scripts/audit_corpus.py.

The script ingests real CSVs + npy artifacts (not synthetic fixtures)
because the *point* of the script is to produce a byte-stable record of
the real corpus. These tests therefore only run when the 5970 inputs
exist — on a fresh clone without the pipeline outputs they are skipped.

Anchors verified against real data 2026-04-17 (see docs/handoffs/
corpus-constants-unification-2026-04-17.md). If any anchor drifts, a
preprocessing step upstream has changed — STOP and investigate before
updating these numbers.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "audit_corpus.py"
CLASSIFIED_CSV = REPO_ROOT / "results/traditional_taxonomy/classified_traditional.csv"
ICI_GAP_NPY = REPO_ROOT / "results/sequential_structure/ici_gap.npy"
SEQ_SUMMARY = REPO_ROOT / "results/sequential_structure/sequential_structure_summary.csv"

REQUIRED_FOR_5970 = (CLASSIFIED_CSV, ICI_GAP_NPY, SEQ_SUMMARY)


def _all_inputs_present() -> bool:
    return all(p.exists() for p in REQUIRED_FOR_5970)


@unittest.skipUnless(_all_inputs_present(), "5970 source artifacts not present")
class TestAuditCorpus5970(unittest.TestCase):
    """End-to-end test — runs the real script, inspects the real JSON."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp_dir = Path(REPO_ROOT / "tests" / "_tmp_audit")
        cls._tmp_dir.mkdir(exist_ok=True)
        cls._output = cls._tmp_dir / "5970_test.json"

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--dataset",
                "5970",
                "--output",
                str(cls._output),
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        cls._stdout = result.stdout
        cls._stderr = result.stderr
        cls._returncode = result.returncode

        if cls._returncode == 0 and cls._output.exists():
            with open(cls._output) as f:
                cls._payload = json.load(f)
        else:
            cls._payload = None

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._output.exists():
            cls._output.unlink()
        if cls._tmp_dir.exists() and not any(cls._tmp_dir.iterdir()):
            cls._tmp_dir.rmdir()

    def test_script_exits_zero(self) -> None:
        self.assertEqual(
            self._returncode, 0, msg=f"stderr: {self._stderr}"
        )

    def test_parameters_block_printed(self) -> None:
        """Every analysis run must state its parameters (feedback rule)."""
        self.assertIn("audit_corpus.py — Parameters", self._stdout)
        self.assertIn("[inputs]", self._stdout)
        self.assertIn("[methodology]", self._stdout)
        self.assertIn("[literature references]", self._stdout)

    def test_top_level_keys(self) -> None:
        self.assertIsNotNone(self._payload)
        for key in (
            "dataset",
            "generated_at_utc",
            "sources",
            "counts",
            "timing",
            "bout_detection_a2",
            "labeling_distributions",
            "references",
        ):
            self.assertIn(key, self._payload)
        self.assertEqual(self._payload["dataset"], "5970")

    # ── Anchors (verified against real data 2026-04-17) ────────────────────

    def test_counts_anchors(self) -> None:
        counts = self._payload["counts"]
        self.assertEqual(counts["n_calls_raw"], 7921)
        self.assertEqual(counts["n_calls_after_dropna_file"], 7864)

    def test_timing_anchors(self) -> None:
        timing = self._payload["timing"]
        self.assertAlmostEqual(timing["median_ici_gap_ms"], 86.68, places=1)
        self.assertAlmostEqual(timing["median_ioi_ms"], 192.99, places=1)
        self.assertAlmostEqual(timing["q25_ici_gap_ms"], 65.14, places=1)
        self.assertAlmostEqual(timing["q75_ici_gap_ms"], 209.11, places=1)
        self.assertEqual(timing["n_negative_gaps"], 10)
        self.assertEqual(timing["n_cross_file_pairs_over_10s"], 829)

    def test_bout_detection_anchors(self) -> None:
        bout = self._payload["bout_detection_a2"]
        self.assertEqual(bout["threshold_s"], 0.6)
        self.assertEqual(bout["n_bouts"], 1238)
        self.assertEqual(bout["n_within_bout_pairs"], 6350)
        self.assertEqual(bout["n_cross_bout_pairs_excluded"], 1513)

    def test_scattoni_7_labels_present(self) -> None:
        scattoni = self._payload["labeling_distributions"]["scattoni_7"]
        self.assertEqual(
            set(scattoni.keys()),
            {"Flat", "Down", "Chevron", "Short", "Complex", "Frequency_Jump", "Up"},
        )
        self.assertEqual(sum(scattoni.values()), 7864)

    def test_hdbscan_labels_present(self) -> None:
        hdb = self._payload["labeling_distributions"]["hdbscan"]
        self.assertEqual(set(hdb.keys()), {"2", "1", "0", "-1"})
        # 2 is the dominant cluster; -1 is HDBSCAN noise (unclusterable)
        self.assertGreater(hdb["2"], hdb["-1"])

    def test_literature_references_present(self) -> None:
        refs = self._payload["references"]
        self.assertEqual(refs["hertz_2020_imsa_sis_bits"], 0.22)
        self.assertEqual(refs["median_within_bout_silent_gap_ms"], 90)


class TestAuditCorpusMissingInputs(unittest.TestCase):
    """If required inputs are missing the script prints a clear skip, not a crash."""

    def test_3452_skip_is_graceful(self) -> None:
        """3452 has no classified CSV yet — --dataset 3452 must exit 1 with a skip message."""
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--dataset",
                "3452",
                "--output",
                "/tmp/should_not_be_written.json",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("[skip] dataset=3452", result.stderr)
        self.assertIn("missing required inputs", result.stderr)


if __name__ == "__main__":
    unittest.main()
