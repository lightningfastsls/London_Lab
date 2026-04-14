"""Tests for the data preparation pipeline (prepare_data.py).

Uses mocking to avoid real WAV I/O. Focuses on pipeline integration,
normalization data-leakage prevention, and the file naming compatibility
bug between prepare_data and load_bout_spectrograms.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usv_language.data.bout_extractor import Bout
from usv_language.data.prepare_data import main as prepare_data_main
from usv_language.training.train_transformer import load_bout_spectrograms

N_FREQ = 170


def _make_synthetic_bouts(n: int = 5) -> list[Bout]:
    """Create synthetic Bout objects from 3 different source files."""
    sources = ["rec_001.wav", "rec_001.wav", "rec_002.wav", "rec_002.wav", "rec_003.wav"]
    bouts = []
    for i in range(min(n, len(sources))):
        bouts.append(Bout(
            source_file=sources[i],
            start_time_s=i * 1.0,
            end_time_s=i * 1.0 + 0.5,
            usv_count=3,
            usv_times=[(i * 1.0 + 0.1, i * 1.0 + 0.15)],
        ))
    return bouts


def _mock_compute_spectrogram(source_file, start, end, config=None):
    """Return a deterministic synthetic spectrogram based on filename.

    Different recordings get different distributions so we can detect
    normalization data leakage.
    """
    rng = np.random.RandomState(hash(source_file) % 2**31)
    n_frames = 200
    # Shift mean by recording to make distributions distinguishable
    rec_num = int(Path(source_file).stem.split("_")[-1])
    offset = rec_num * 10.0
    return (rng.randn(N_FREQ, n_frames).astype(np.float32) * 5 + offset)


# ---------------------------------------------------------------------------
# Test 24: E2E pipeline produces expected directory structure
# ---------------------------------------------------------------------------


@patch("usv_language.data.prepare_data.compute_bout_spectrogram", side_effect=_mock_compute_spectrogram)
@patch("usv_language.data.prepare_data.BoutExtractor")
def test_prepare_data_e2e_produces_outputs(
    mock_extractor_cls: MagicMock,
    mock_spec_fn: MagicMock,
    tmp_path: Path,
) -> None:
    """Pipeline creates expected files: normalization_stats, splits, manifests, config."""
    # Mock the extractor
    mock_extractor = MagicMock()
    mock_extractor.extract_from_detection_dir.return_value = _make_synthetic_bouts()
    mock_extractor_cls.return_value = mock_extractor

    detection_dir = tmp_path / "detections"
    detection_dir.mkdir()
    output_dir = tmp_path / "output"

    prepare_data_main([
        "--detection-dir", str(detection_dir),
        "--output-dir", str(output_dir),
        "--seed", "42",
    ])

    # Check directory structure
    assert (output_dir / "normalization_stats.npz").exists()
    assert (output_dir / "pipeline_config.json").exists()
    assert (output_dir / "train").is_dir()
    assert (output_dir / "train" / "manifest.json").exists()

    # At least one split should have .npy files
    all_npy = list(output_dir.rglob("*.npy"))
    assert len(all_npy) > 0, "No spectrogram .npy files were saved"


# ---------------------------------------------------------------------------
# Test 25: Normalization computed from training split only
# ---------------------------------------------------------------------------


@patch("usv_language.data.prepare_data.compute_bout_spectrogram", side_effect=_mock_compute_spectrogram)
@patch("usv_language.data.prepare_data.BoutExtractor")
def test_normalization_from_train_only(
    mock_extractor_cls: MagicMock,
    mock_spec_fn: MagicMock,
    tmp_path: Path,
) -> None:
    """Normalization stats must be computed only from training data."""
    mock_extractor = MagicMock()
    mock_extractor.extract_from_detection_dir.return_value = _make_synthetic_bouts()
    mock_extractor_cls.return_value = mock_extractor

    detection_dir = tmp_path / "detections"
    detection_dir.mkdir()
    output_dir = tmp_path / "output"

    prepare_data_main([
        "--detection-dir", str(detection_dir),
        "--output-dir", str(output_dir),
        "--seed", "42",
    ])

    # Load the saved normalization stats
    from usv_language.data.normalization import load_normalization_stats
    stats = load_normalization_stats(output_dir / "normalization_stats.npz")

    # Load the training manifest to see which recordings are in training
    train_manifest = json.loads((output_dir / "train" / "manifest.json").read_text())
    train_rec_ids = set(train_manifest["recording_ids"])

    # Recompute stats from training data only
    from usv_language.data.normalization import compute_normalization_stats
    train_specs = []
    for bout in _make_synthetic_bouts():
        rec_stem = Path(bout.source_file).stem
        if rec_stem in train_rec_ids:
            spec = _mock_compute_spectrogram(bout.source_file, 0, 1)
            train_specs.append(spec)

    if train_specs:
        expected_stats = compute_normalization_stats(train_specs)
        np.testing.assert_array_almost_equal(stats.mean, expected_stats.mean, decimal=3)
        np.testing.assert_array_almost_equal(stats.std, expected_stats.std, decimal=3)


# ---------------------------------------------------------------------------
# Test 26: Manifest recording IDs are correct
# ---------------------------------------------------------------------------


@patch("usv_language.data.prepare_data.compute_bout_spectrogram", side_effect=_mock_compute_spectrogram)
@patch("usv_language.data.prepare_data.BoutExtractor")
def test_manifest_recording_ids_correct(
    mock_extractor_cls: MagicMock,
    mock_spec_fn: MagicMock,
    tmp_path: Path,
) -> None:
    """Each manifest's recording_ids only contains IDs from that split."""
    mock_extractor = MagicMock()
    mock_extractor.extract_from_detection_dir.return_value = _make_synthetic_bouts()
    mock_extractor_cls.return_value = mock_extractor

    detection_dir = tmp_path / "detections"
    detection_dir.mkdir()
    output_dir = tmp_path / "output"

    prepare_data_main([
        "--detection-dir", str(detection_dir),
        "--output-dir", str(output_dir),
        "--seed", "42",
    ])

    # Collect all recording IDs across splits
    all_seen = set()
    for split_name in ("train", "val", "test"):
        manifest_path = output_dir / split_name / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            split_ids = set(manifest["recording_ids"])
            # No overlap with other splits
            overlap = all_seen & split_ids
            assert len(overlap) == 0, (
                f"Recording IDs in '{split_name}' overlap with earlier splits: {overlap}"
            )
            all_seen.update(split_ids)


# ---------------------------------------------------------------------------
# Test 27: No bouts → graceful exit
# ---------------------------------------------------------------------------


@patch("usv_language.data.prepare_data.BoutExtractor")
def test_no_bouts_exits_gracefully(
    mock_extractor_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """Empty bout list exits without crash or creating output files."""
    mock_extractor = MagicMock()
    mock_extractor.extract_from_detection_dir.return_value = []
    mock_extractor_cls.return_value = mock_extractor

    detection_dir = tmp_path / "detections"
    detection_dir.mkdir()
    output_dir = tmp_path / "output"

    # Should not raise
    prepare_data_main([
        "--detection-dir", str(detection_dir),
        "--output-dir", str(output_dir),
    ])

    # No normalization stats or pipeline config should exist
    assert not (output_dir / "normalization_stats.npz").exists()
    assert not (output_dir / "pipeline_config.json").exists()


# ---------------------------------------------------------------------------
# Test 28: File naming compatibility (catches Bug #3)
# ---------------------------------------------------------------------------


@patch("usv_language.data.prepare_data.compute_bout_spectrogram", side_effect=_mock_compute_spectrogram)
@patch("usv_language.data.prepare_data.BoutExtractor")
def test_npy_naming_vs_load_compatibility(
    mock_extractor_cls: MagicMock,
    mock_spec_fn: MagicMock,
    tmp_path: Path,
) -> None:
    """Files saved by prepare_data must be loadable by load_bout_spectrograms.

    BUG EXPOSURE: prepare_data saves as 'spec_00000.npy' but
    load_bout_spectrograms globs for '*_bout*.npy'. The npy fallback
    path cannot find files produced by the data preparation script.
    """
    mock_extractor = MagicMock()
    mock_extractor.extract_from_detection_dir.return_value = _make_synthetic_bouts()
    mock_extractor_cls.return_value = mock_extractor

    detection_dir = tmp_path / "detections"
    detection_dir.mkdir()
    output_dir = tmp_path / "output"

    prepare_data_main([
        "--detection-dir", str(detection_dir),
        "--output-dir", str(output_dir),
        "--seed", "42",
    ])

    # Find the training split directory (should have .npy files)
    train_dir = output_dir / "train"
    assert train_dir.is_dir()
    npy_files = list(train_dir.glob("*.npy"))
    assert len(npy_files) > 0, "No .npy files in training split"

    # Try to load using load_bout_spectrograms
    # This should work but currently FAILS because of naming mismatch
    try:
        specs, rec_ids = load_bout_spectrograms(train_dir)
        # If we get here, the bug is fixed
        assert len(specs) > 0
    except FileNotFoundError:
        pytest.fail(
            "load_bout_spectrograms cannot load files from prepare_data output. "
            f"Files saved as: {[f.name for f in npy_files]}. "
            "But load_bout_spectrograms globs for '*_bout*.npy'. "
            "The naming conventions are incompatible."
        )


# ---------------------------------------------------------------------------
# Test 29: Pipeline config JSON has all expected keys
# ---------------------------------------------------------------------------


@patch("usv_language.data.prepare_data.compute_bout_spectrogram", side_effect=_mock_compute_spectrogram)
@patch("usv_language.data.prepare_data.BoutExtractor")
def test_pipeline_config_json_complete(
    mock_extractor_cls: MagicMock,
    mock_spec_fn: MagicMock,
    tmp_path: Path,
) -> None:
    """pipeline_config.json contains all expected top-level and nested keys."""
    mock_extractor = MagicMock()
    mock_extractor.extract_from_detection_dir.return_value = _make_synthetic_bouts()
    mock_extractor_cls.return_value = mock_extractor

    detection_dir = tmp_path / "detections"
    detection_dir.mkdir()
    output_dir = tmp_path / "output"

    prepare_data_main([
        "--detection-dir", str(detection_dir),
        "--output-dir", str(output_dir),
    ])

    config = json.loads((output_dir / "pipeline_config.json").read_text())

    # Top-level keys
    assert "bout_extraction" in config
    assert "spectrogram" in config
    assert "dataset" in config
    assert "n_bouts" in config
    assert "n_spectrograms" in config
    assert "n_recordings" in config

    # Spectrogram sub-keys (critical for reproducibility)
    spec = config["spectrogram"]
    assert spec["sample_rate"] == 300_000
    assert spec["n_fft"] == 512
    assert spec["hop_length"] == 128
    assert spec["freq_min_hz"] == 20_000
    assert spec["freq_max_hz"] == 120_000

    # Dataset sub-keys
    ds = config["dataset"]
    assert "max_seq_len" in ds
    assert "train_split" in ds
    assert "seed" in ds


# ---------------------------------------------------------------------------
# Test 30: Skipped spectrograms are counted correctly
# ---------------------------------------------------------------------------


@patch("usv_language.data.prepare_data.BoutExtractor")
def test_skipped_spectrograms_counted(
    mock_extractor_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """Bouts that fail spectrogram computation are skipped, valid ones counted."""
    mock_extractor = MagicMock()
    mock_extractor.extract_from_detection_dir.return_value = _make_synthetic_bouts(3)
    mock_extractor_cls.return_value = mock_extractor

    detection_dir = tmp_path / "detections"
    detection_dir.mkdir()
    output_dir = tmp_path / "output"

    call_count = 0

    def flaky_spectrogram(source_file, start, end, config=None):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("Simulated WAV read failure")
        return _mock_compute_spectrogram(source_file, start, end, config)

    with patch(
        "usv_language.data.prepare_data.compute_bout_spectrogram",
        side_effect=flaky_spectrogram,
    ):
        prepare_data_main([
            "--detection-dir", str(detection_dir),
            "--output-dir", str(output_dir),
        ])

    config = json.loads((output_dir / "pipeline_config.json").read_text())
    # 3 bouts, 1 failed -> 2 valid spectrograms
    assert config["n_spectrograms"] == 2
    assert config["n_bouts"] == 2
