"""Tests for LabelStorage save/load behavior."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.app.core.audio_loader import AudioData
from usv_spectrogram.app.core.detection_logic import DetectionResult, DetectedUSV
from usv_spectrogram.app.core.label_storage import LabelStorage


def _make_audio_data() -> AudioData:
    times = np.array([0.0, 0.1, 0.2], dtype=float)
    frequencies = np.array([40_000.0, 50_000.0], dtype=float)
    spectrogram_db = np.zeros((2, 3), dtype=float)
    return AudioData(
        audio=np.zeros(30, dtype=np.float32),
        sample_rate=300_000,
        spectrogram_db=spectrogram_db,
        frequencies=frequencies,
        times=times,
        duration_s=0.2,
    )


def _make_detection_result() -> DetectionResult:
    return DetectionResult(
        usvs=[
            DetectedUSV(
                start_time_s=0.05,
                end_time_s=0.10,
                start_col=0,
                end_col=1,
                max_probability=0.9,
                mean_probability=0.8,
                user_adjusted=True,
                original_start_time_s=0.04,
                original_end_time_s=0.11,
                save_state="saved_current",
                user_action="added_manually",
                original_cnn_probability=0.95,
            )
        ],
        probabilities=np.array([0.1, 0.9, 0.2], dtype=float),
        column_indices=np.array([0, 1, 2], dtype=int),
        times=np.array([0.0, 0.1, 0.2], dtype=float),
        file_label=None,
    )


class TestLabelStorage:
    def test_save_allows_missing_model_path(self, tmp_path: Path) -> None:
        output_path = tmp_path / "labels.json"

        LabelStorage.save(
            output_path=output_path,
            audio_data=_make_audio_data(),
            detection_result=_make_detection_result(),
            wav_path=tmp_path / "recording.wav",
            model_path=None,
            high_threshold=0.4,
            low_threshold=0.3,
        )

        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["metadata"]["model_file"] is None
        assert data["metadata"]["sample_rate"] == 300_000

    def test_round_trip_reconstructs_detection_result_when_model_path_is_missing(
        self, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "labels.json"
        original = _make_detection_result()

        LabelStorage.save(
            output_path=output_path,
            audio_data=_make_audio_data(),
            detection_result=original,
            wav_path=tmp_path / "recording.wav",
            model_path=None,
            high_threshold=0.4,
            low_threshold=0.3,
        )

        loaded = LabelStorage.load(output_path)
        reconstructed = LabelStorage.reconstruct_detection_result(loaded)

        assert reconstructed.file_label is None
        assert len(reconstructed.usvs) == 1
        assert reconstructed.usvs[0].start_time_s == original.usvs[0].start_time_s
        assert reconstructed.usvs[0].end_time_s == original.usvs[0].end_time_s
        assert reconstructed.usvs[0].user_adjusted is True
        assert reconstructed.usvs[0].original_start_time_s == 0.04
        assert reconstructed.usvs[0].original_end_time_s == 0.11
        assert reconstructed.usvs[0].save_state == "saved_current"
        assert reconstructed.usvs[0].user_action == "added_manually"
        assert reconstructed.usvs[0].original_cnn_probability == 0.95
        assert np.array_equal(reconstructed.probabilities, original.probabilities)
