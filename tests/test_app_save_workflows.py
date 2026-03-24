"""Workflow regressions for app save/export behavior."""

from __future__ import annotations

import csv
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.app.core.audio_loader import AudioData
from usv_spectrogram.app.core.detection_exporter import DetectionExporter
from usv_spectrogram.app.core.detection_logic import DetectionResult, DetectedUSV
from usv_spectrogram.app.core.saved_detection_tracker import SavedDetectionTracker


class _StatusBar:
    def showMessage(self, _message: str, _timeout: int | None = None) -> None:
        return None


class _DummyProgressDialog:
    def __init__(self, *_args, **_kwargs) -> None:
        self._canceled = False

    def setWindowTitle(self, *_args, **_kwargs) -> None:
        return None

    def setWindowModality(self, *_args, **_kwargs) -> None:
        return None

    def wasCanceled(self) -> bool:
        return self._canceled

    def setValue(self, *_args, **_kwargs) -> None:
        return None

    def close(self) -> None:
        return None


class _DummyMessageBox:
    class StandardButton:
        Yes = 1
        No = 2

    @staticmethod
    def question(*_args, **_kwargs) -> int:
        return _DummyMessageBox.StandardButton.Yes

    @staticmethod
    def information(*_args, **_kwargs) -> None:
        return None

    @staticmethod
    def warning(*_args, **_kwargs) -> None:
        return None

    @staticmethod
    def critical(*_args, **_kwargs) -> None:
        return None


def _make_audio_data() -> AudioData:
    times = np.linspace(0.0, 2.0, 201, dtype=float)
    frequencies = np.array([40_000.0, 50_000.0], dtype=float)
    spectrogram_db = np.zeros((2, times.size), dtype=float)
    return AudioData(
        audio=np.zeros(2_000, dtype=np.float32),
        sample_rate=300_000,
        spectrogram_db=spectrogram_db,
        frequencies=frequencies,
        times=times,
        duration_s=2.0,
    )


def _make_detection(
    start: float,
    end: float,
    *,
    user_adjusted: bool = False,
    original_start: float = 0.0,
    original_end: float = 0.0,
    save_state: str = "unsaved",
    user_action: str | None = None,
) -> DetectedUSV:
    times = _make_audio_data().times
    start_col = int(np.searchsorted(times, start))
    end_col = int(np.searchsorted(times, end))
    return DetectedUSV(
        start_time_s=start,
        end_time_s=end,
        start_col=start_col,
        end_col=end_col,
        max_probability=0.9,
        mean_probability=0.8,
        user_adjusted=user_adjusted,
        original_start_time_s=original_start,
        original_end_time_s=original_end,
        save_state=save_state,
        user_action=user_action,
        original_cnn_probability=0.95,
    )


def _make_fake_window(
    tmp_path: Path,
    detections: list[DetectedUSV],
    main_window_cls,
) -> types.SimpleNamespace:
    audio_data = _make_audio_data()
    output_dir = tmp_path / "exports"
    current_wav_path = tmp_path / "recording.wav"

    fake = types.SimpleNamespace(
        detection_result=DetectionResult(
            usvs=detections,
            probabilities=np.linspace(0.0, 1.0, audio_data.times.size, dtype=float),
            column_indices=np.arange(audio_data.times.size, dtype=int),
            times=audio_data.times,
            file_label=None,
        ),
        audio_data=audio_data,
        output_dir=output_dir,
        current_wav_path=current_wav_path,
        session_id="session-123",
        current_preset="test",
        high_threshold=0.4,
        low_threshold=0.3,
        detection_exporter=DetectionExporter(output_dir, context_ms=20.0),
        saved_tracker=SavedDetectionTracker(current_wav_path.stem, output_dir),
        statusBar=_StatusBar(),
    )
    fake._refresh_detection_views = lambda: None
    fake._remove_existing_saved_exports = types.MethodType(
        main_window_cls._remove_existing_saved_exports, fake
    )
    fake._export_detection_and_mark_saved = types.MethodType(
        main_window_cls._export_detection_and_mark_saved, fake
    )
    return fake


def _load_main_window_module(monkeypatch):
    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_stub.device = lambda value: value
    torch_stub.load = lambda *args, **kwargs: {}

    cnn_stub = types.ModuleType("usv_spectrogram.models.cnn_classifier")

    class _DummyCNN:
        def load_state_dict(self, *_args, **_kwargs) -> None:
            return None

        def to(self, *_args, **_kwargs) -> None:
            return None

        def eval(self) -> None:
            return None

    cnn_stub.USVClassifierCNN = _DummyCNN

    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "usv_spectrogram.models.cnn_classifier", cnn_stub)

    sys.modules.pop("usv_spectrogram.app.main_window", None)
    sys.modules.pop("usv_spectrogram.app.core.sliding_inference", None)

    module = importlib.import_module("usv_spectrogram.app.main_window")
    return module


def _read_summary_rows(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestAppSaveWorkflows:
    def test_save_current_view_replaces_stale_export_for_adjusted_detection(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        main_window_module = _load_main_window_module(monkeypatch)
        monkeypatch.setattr(main_window_module, "QMessageBox", _DummyMessageBox)

        original = _make_detection(0.50, 0.60, save_state="saved_current")
        adjusted = _make_detection(
            0.52,
            0.61,
            user_adjusted=True,
            original_start=0.50,
            original_end=0.60,
        )
        fake = _make_fake_window(tmp_path, [adjusted], main_window_module.MainWindow)
        fake._get_visible_detections = lambda: [adjusted]

        old_png, _, summary_path = fake.detection_exporter.export_detection(
            detection=original,
            audio_data=fake.audio_data,
            wav_filename=fake.current_wav_path.stem,
            detection_index=0,
            session_id=fake.session_id,
            threshold_preset=fake.current_preset,
            threshold_high=fake.high_threshold,
            threshold_low=fake.low_threshold,
        )
        fake.saved_tracker.mark_saved(original, str(old_png))

        main_window_module.MainWindow._save_current_view(fake)

        new_png = (
            fake.output_dir
            / fake.current_wav_path.stem
            / "detection_000_0.520s-0.610s.png"
        )
        old_json = old_png.with_suffix(".json")

        assert not old_png.exists()
        assert not old_json.exists()
        assert new_png.exists()

        rows = _read_summary_rows(summary_path)
        assert len(rows) == 1
        assert rows[0]["detection_index"] == "0"
        assert rows[0]["start_time_s"] == "0.520000"
        assert rows[0]["end_time_s"] == "0.610000"

    def test_save_all_detections_resyncs_indices_after_reorder(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        main_window_module = _load_main_window_module(monkeypatch)
        monkeypatch.setattr(main_window_module, "QMessageBox", _DummyMessageBox)
        monkeypatch.setattr(main_window_module, "QProgressDialog", _DummyProgressDialog)

        original_first = _make_detection(0.30, 0.40, save_state="saved_current")
        original_second = _make_detection(0.70, 0.80, save_state="saved_current")
        remaining = _make_detection(0.70, 0.80, save_state="saved_current")
        manual = _make_detection(
            1.10,
            1.20,
            save_state="unsaved",
            user_action="added_manually",
        )
        fake = _make_fake_window(tmp_path, [remaining, manual], main_window_module.MainWindow)

        first_png, _, _ = fake.detection_exporter.export_detection(
            detection=original_first,
            audio_data=fake.audio_data,
            wav_filename=fake.current_wav_path.stem,
            detection_index=0,
            session_id=fake.session_id,
            threshold_preset=fake.current_preset,
            threshold_high=fake.high_threshold,
            threshold_low=fake.low_threshold,
        )
        second_png, _, summary_path = fake.detection_exporter.export_detection(
            detection=original_second,
            audio_data=fake.audio_data,
            wav_filename=fake.current_wav_path.stem,
            detection_index=1,
            session_id=fake.session_id,
            threshold_preset=fake.current_preset,
            threshold_high=fake.high_threshold,
            threshold_low=fake.low_threshold,
        )
        fake.saved_tracker.mark_saved(original_first, str(first_png))
        fake.saved_tracker.mark_saved(original_second, str(second_png))
        fake.saved_tracker.mark_saved(
            _make_detection(0.30, 0.40, user_action="deleted_by_user"),
            str(tmp_path / "rejected.png"),
            user_action="deleted_by_user",
        )

        main_window_module.MainWindow._save_all_detections(fake)

        output_subdir = fake.output_dir / fake.current_wav_path.stem
        json_files = sorted(path.name for path in output_subdir.glob("detection_*.json"))
        png_files = sorted(path.name for path in output_subdir.glob("detection_*.png"))

        assert json_files == [
            "detection_000_0.700s-0.800s.json",
            "detection_001_1.100s-1.200s.json",
        ]
        assert png_files == [
            "detection_000_0.700s-0.800s.png",
            "detection_001_1.100s-1.200s.png",
        ]

        rows = _read_summary_rows(summary_path)
        assert [row["detection_index"] for row in rows] == ["0", "1"]
        assert [row["start_time_s"] for row in rows] == ["0.700000", "1.100000"]
        assert [row["user_action"] for row in rows] == ["", "added_manually"]
        assert fake.saved_tracker.get_saved_count() == 3

    def test_load_labels_restores_saved_thresholds(self, tmp_path: Path, monkeypatch) -> None:
        main_window_module = _load_main_window_module(monkeypatch)

        audio_data = _make_audio_data()
        labels_path = tmp_path / "loaded_labels.json"
        detection_result = DetectionResult(
            usvs=[_make_detection(0.40, 0.50, save_state="saved_current")],
            probabilities=np.linspace(0.0, 1.0, audio_data.times.size, dtype=float),
            column_indices=np.arange(audio_data.times.size, dtype=int),
            times=audio_data.times,
            file_label=None,
        )

        from usv_spectrogram.app.core.label_storage import LabelStorage

        LabelStorage.save(
            output_path=labels_path,
            audio_data=audio_data,
            detection_result=detection_result,
            wav_path=tmp_path / "recording.wav",
            model_path=None,
            high_threshold=0.67,
            low_threshold=0.21,
        )

        monkeypatch.setattr(
            main_window_module.QFileDialog,
            "getOpenFileName",
            staticmethod(lambda *_args, **_kwargs: (str(labels_path), "JSON")),
        )
        monkeypatch.setattr(main_window_module, "QMessageBox", _DummyMessageBox)

        class _Slider:
            def __init__(self) -> None:
                self.value = None
                self.blocked = []

            def blockSignals(self, blocked: bool) -> None:
                self.blocked.append(blocked)

            def setValue(self, value: int) -> None:
                self.value = value

        class _Label:
            def __init__(self) -> None:
                self.text = None

            def setText(self, text: str) -> None:
                self.text = text

            def setStyleSheet(self, _style: str) -> None:
                return None

            def setEnabled(self, _enabled: bool) -> None:
                return None

        fake = types.SimpleNamespace(
            audio_data=audio_data,
            current_wav_path=tmp_path / "recording.wav",
            output_dir=tmp_path / "exports",
            detection_result=None,
            inference_result=None,
            saved_tracker=None,
            detection_exporter=None,
            high_threshold=0.10,
            low_threshold=0.05,
            high_threshold_slider=_Slider(),
            low_threshold_slider=_Slider(),
            high_threshold_label=_Label(),
            low_threshold_label=_Label(),
            label_noise_btn=_Label(),
            apply_btn=_Label(),
            save_current_btn=_Label(),
            save_all_btn=_Label(),
            remove_detection_btn=_Label(),
            statusBar=_StatusBar(),
            spectrogram_view=types.SimpleNamespace(
                canvas=types.SimpleNamespace(clear_selection=lambda: None)
            ),
        )
        fake._enable_threshold_controls = lambda _enabled: None
        fake._refresh_detection_views = lambda: None
        fake._update_detection_info = lambda: None

        main_window_module.MainWindow._load_labels(fake)

        assert fake.high_threshold == 0.67
        assert fake.low_threshold == 0.21
        assert fake.high_threshold_slider.value == 67
        assert fake.low_threshold_slider.value == 21
        assert fake.high_threshold_label.text == "0.67"
        assert fake.low_threshold_label.text == "0.21"

    def test_clearing_noise_label_removes_stale_noise_json(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        main_window_module = _load_main_window_module(monkeypatch)
        monkeypatch.setattr(main_window_module, "QMessageBox", _DummyMessageBox)

        noise_dir = tmp_path / "exports" / "noise_labeled_files"
        noise_dir.mkdir(parents=True)
        noise_path = noise_dir / "recording.json"
        noise_path.write_text(json.dumps({"metadata": {"file_label": "noise"}}), encoding="utf-8")

        class _Button:
            def __init__(self) -> None:
                self.text = None
                self.style = None

            def setText(self, text: str) -> None:
                self.text = text

            def setStyleSheet(self, style: str) -> None:
                self.style = style

        fake = types.SimpleNamespace(
            detection_result=DetectionResult(
                usvs=[],
                probabilities=np.array([0.1, 0.2], dtype=float),
                column_indices=np.array([0, 1], dtype=int),
                times=np.array([0.0, 0.1], dtype=float),
                file_label="noise",
            ),
            audio_data=_make_audio_data(),
            output_dir=tmp_path / "exports",
            current_wav_path=tmp_path / "recording.wav",
            label_noise_btn=_Button(),
            statusBar=_StatusBar(),
        )
        fake._enable_threshold_controls = lambda _enabled: None
        fake._refresh_detection_views = lambda: None
        fake._update_detection_info = lambda: None
        fake._remove_noise_label_file = types.MethodType(
            main_window_module.MainWindow._remove_noise_label_file, fake
        )

        main_window_module.MainWindow._toggle_noise_label(fake)

        assert fake.detection_result.file_label is None
        assert not noise_path.exists()
        assert fake.label_noise_btn.text == "Label File as Noise"

    def test_load_labels_rejects_mismatched_wav_file(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        main_window_module = _load_main_window_module(monkeypatch)

        audio_data = _make_audio_data()
        labels_path = tmp_path / "wrong_labels.json"
        detection_result = DetectionResult(
            usvs=[_make_detection(0.20, 0.30)],
            probabilities=np.linspace(0.0, 1.0, audio_data.times.size, dtype=float),
            column_indices=np.arange(audio_data.times.size, dtype=int),
            times=audio_data.times,
            file_label=None,
        )

        from usv_spectrogram.app.core.label_storage import LabelStorage

        LabelStorage.save(
            output_path=labels_path,
            audio_data=audio_data,
            detection_result=detection_result,
            wav_path=tmp_path / "other_recording.wav",
            model_path=None,
            high_threshold=0.5,
            low_threshold=0.2,
        )

        monkeypatch.setattr(
            main_window_module.QFileDialog,
            "getOpenFileName",
            staticmethod(lambda *_args, **_kwargs: (str(labels_path), "JSON")),
        )

        warnings: list[str] = []

        class _CapturingMessageBox(_DummyMessageBox):
            @staticmethod
            def warning(_parent, title: str, text: str) -> None:
                warnings.append(f"{title}: {text}")

        monkeypatch.setattr(main_window_module, "QMessageBox", _CapturingMessageBox)

        class _Slider:
            def blockSignals(self, _blocked: bool) -> None:
                return None

            def setValue(self, _value: int) -> None:
                return None

        class _Label:
            def setText(self, _text: str) -> None:
                return None

            def setStyleSheet(self, _style: str) -> None:
                return None

            def setEnabled(self, _enabled: bool) -> None:
                return None

        fake = types.SimpleNamespace(
            audio_data=audio_data,
            current_wav_path=tmp_path / "recording.wav",
            output_dir=tmp_path / "exports",
            detection_result="unchanged",
            inference_result=None,
            saved_tracker=None,
            detection_exporter=None,
            high_threshold=0.10,
            low_threshold=0.05,
            high_threshold_slider=_Slider(),
            low_threshold_slider=_Slider(),
            high_threshold_label=_Label(),
            low_threshold_label=_Label(),
            label_noise_btn=_Label(),
            apply_btn=_Label(),
            save_current_btn=_Label(),
            save_all_btn=_Label(),
            remove_detection_btn=_Label(),
            statusBar=_StatusBar(),
            spectrogram_view=types.SimpleNamespace(
                canvas=types.SimpleNamespace(clear_selection=lambda: None)
            ),
        )
        fake._enable_threshold_controls = lambda _enabled: None
        fake._refresh_detection_views = lambda: None
        fake._update_detection_info = lambda: None

        main_window_module.MainWindow._load_labels(fake)

        assert fake.detection_result == "unchanged"
        assert warnings
        assert "different WAV file" in warnings[0]
