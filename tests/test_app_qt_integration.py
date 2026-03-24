"""Qt-instantiated integration tests for main window workflow state."""

from __future__ import annotations

import importlib
import os
import sys
import types
import csv
from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.app.core.audio_loader import AudioData
from usv_spectrogram.app.core.detection_logic import DetectionResult, DetectedUSV


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class _FakeSettings:
    def __init__(self, *_args, **_kwargs) -> None:
        self._values: dict[str, object] = {}

    def value(self, key: str, default=None, type=None):
        value = self._values.get(key, default)
        if type is not None and value is not None:
            return type(value)
        return value

    def setValue(self, key: str, value) -> None:
        self._values[key] = value

    def remove(self, key: str) -> None:
        self._values.pop(key, None)


def _make_audio_data(duration_s: float = 20.0, n_cols: int = 2000) -> AudioData:
    times = np.linspace(0.0, duration_s, n_cols, dtype=float)
    frequencies = np.array([40_000.0, 50_000.0], dtype=float)
    spectrogram_db = np.zeros((2, n_cols), dtype=float)
    return AudioData(
        audio=np.zeros(int(duration_s * 300_000 / 100), dtype=np.float32),
        sample_rate=300_000,
        spectrogram_db=spectrogram_db,
        frequencies=frequencies,
        times=times,
        duration_s=duration_s,
    )


def _make_detection(start: float, end: float, *, save_state: str = "unsaved") -> DetectedUSV:
    return DetectedUSV(
        start_time_s=start,
        end_time_s=end,
        start_col=0,
        end_col=1,
        max_probability=0.9,
        mean_probability=0.8,
        save_state=save_state,
    )


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
    monkeypatch.setattr(module, "QSettings", _FakeSettings)
    return module


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestMainWindowQtIntegration:
    def test_load_labels_clears_stale_canvas_selection(
        self, qapp, tmp_path: Path, monkeypatch
    ) -> None:
        main_window_module = _load_main_window_module(monkeypatch)
        monkeypatch.setattr(main_window_module.QMessageBox, "critical", staticmethod(lambda *args, **kwargs: None))
        monkeypatch.setattr(main_window_module.QMessageBox, "warning", staticmethod(lambda *args, **kwargs: None))

        audio_data = _make_audio_data(duration_s=6.0, n_cols=600)
        monkeypatch.setattr(
            main_window_module.AudioLoader,
            "load",
            lambda self, _wav_path: audio_data,
        )

        window = main_window_module.MainWindow(default_model_path=None)
        window.resize(1000, 700)
        window.show()

        wav_path = tmp_path / "reload.wav"
        wav_path.write_bytes(b"")
        window._load_wav_file(wav_path)
        qapp.processEvents()

        initial_detection = DetectedUSV(
            start_time_s=1.0,
            end_time_s=1.2,
            start_col=100,
            end_col=120,
            max_probability=0.9,
            mean_probability=0.8,
        )
        window.detection_result = DetectionResult(
            usvs=[initial_detection],
            probabilities=np.linspace(0.0, 1.0, audio_data.times.size, dtype=float),
            column_indices=np.arange(audio_data.times.size, dtype=int),
            times=audio_data.times,
        )
        window._refresh_detection_views()
        qapp.processEvents()

        start_x = window.spectrogram_view.canvas._col_to_pixel(initial_detection.start_col)
        QTest.mouseClick(
            window.spectrogram_view.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPoint(start_x + 1, 20),
        )
        qapp.processEvents()

        assert window.spectrogram_view.canvas._selected_detection_idx == 0

        labels_path = tmp_path / "reload.json"
        from usv_spectrogram.app.core.label_storage import LabelStorage

        reloaded_detection = DetectedUSV(
            start_time_s=2.5,
            end_time_s=2.8,
            start_col=250,
            end_col=280,
            max_probability=0.85,
            mean_probability=0.75,
        )
        LabelStorage.save(
            output_path=labels_path,
            audio_data=audio_data,
            detection_result=DetectionResult(
                usvs=[reloaded_detection],
                probabilities=np.linspace(0.0, 1.0, audio_data.times.size, dtype=float),
                column_indices=np.arange(audio_data.times.size, dtype=int),
                times=audio_data.times,
            ),
            wav_path=wav_path,
            model_path=None,
            high_threshold=0.4,
            low_threshold=0.3,
        )
        monkeypatch.setattr(
            main_window_module.QFileDialog,
            "getOpenFileName",
            staticmethod(lambda *_args, **_kwargs: (str(labels_path), "JSON")),
        )

        info_messages: list[str] = []
        monkeypatch.setattr(
            main_window_module.QMessageBox,
            "information",
            staticmethod(lambda _parent, title, text: info_messages.append(f"{title}: {text}")),
        )

        window._load_labels()
        qapp.processEvents()

        assert window.spectrogram_view.canvas._selected_detection_idx is None

        window._remove_detection()
        qapp.processEvents()

        assert len(window.detection_result.usvs) == 1
        assert info_messages
        assert "Please select a detection first" in info_messages[-1]

        window.detection_result = None
        window.close()

    def test_load_wav_file_clears_review_ui_state(self, qapp, tmp_path: Path, monkeypatch) -> None:
        main_window_module = _load_main_window_module(monkeypatch)
        monkeypatch.setattr(main_window_module.QMessageBox, "critical", staticmethod(lambda *args, **kwargs: None))

        stale_audio = _make_audio_data(duration_s=10.0, n_cols=1000)
        fresh_audio = _make_audio_data(duration_s=5.0, n_cols=500)
        monkeypatch.setattr(
            main_window_module.AudioLoader,
            "load",
            lambda self, _wav_path: fresh_audio,
        )

        window = main_window_module.MainWindow(default_model_path=None)
        window.resize(1200, 800)
        window.show()
        qapp.processEvents()

        window.audio_data = stale_audio
        window.current_wav_path = tmp_path / "old.wav"
        window.detection_result = DetectionResult(
            usvs=[_make_detection(1.0, 1.2)],
            probabilities=np.array([0.1, 0.9], dtype=float),
            column_indices=np.array([0, 1], dtype=int),
            times=np.array([0.0, 1.0], dtype=float),
        )
        window.inference_result = object()
        window.saved_tracker = object()
        window.detection_exporter = object()
        window.spectrogram_view.set_data(stale_audio.spectrogram_db, stale_audio.times, stale_audio.frequencies, total_columns=stale_audio.spectrogram_db.shape[1])
        window.spectrogram_view.set_detections(window.detection_result.usvs)
        window.probability_view.set_data(
            window.detection_result.times,
            window.detection_result.probabilities,
            0.4,
            0.3,
            window.detection_result.usvs,
            column_indices=window.detection_result.column_indices,
            target_width=window.spectrogram_view.get_canvas_width(),
        )
        window.apply_btn.setEnabled(True)
        window.save_current_btn.setEnabled(True)
        window.save_all_btn.setEnabled(True)
        window.remove_detection_btn.setEnabled(True)
        window.label_noise_btn.setEnabled(True)

        new_wav_path = tmp_path / "new.wav"
        new_wav_path.write_bytes(b"")
        window._load_wav_file(new_wav_path)
        qapp.processEvents()

        assert window.current_wav_path == new_wav_path
        assert window.audio_data is fresh_audio
        assert window.detection_result is None
        assert window.inference_result is None
        assert window.saved_tracker is None
        assert window.detection_exporter is None
        assert window.probability_view.canvas.probabilities is None
        assert window.probability_view.canvas.detections == []
        assert window.spectrogram_view.canvas.detections == []
        assert window.detect_btn.isEnabled() is True
        assert window.apply_btn.isEnabled() is False
        assert window.save_current_btn.isEnabled() is False
        assert window.save_all_btn.isEnabled() is False
        assert window.remove_detection_btn.isEnabled() is False
        assert window.label_noise_btn.isEnabled() is False
        assert window.detection_info_label.text() == "No detections"

        window.close()

    def test_get_visible_detections_uses_real_viewport_geometry(
        self, qapp, tmp_path: Path, monkeypatch
    ) -> None:
        main_window_module = _load_main_window_module(monkeypatch)
        monkeypatch.setattr(main_window_module.QMessageBox, "critical", staticmethod(lambda *args, **kwargs: None))
        audio_data = _make_audio_data(duration_s=20.0, n_cols=2000)
        monkeypatch.setattr(
            main_window_module.AudioLoader,
            "load",
            lambda self, _wav_path: audio_data,
        )

        window = main_window_module.MainWindow(default_model_path=None)
        window.resize(900, 700)
        window.show()

        wav_path = tmp_path / "viewport.wav"
        wav_path.write_bytes(b"")
        window._load_wav_file(wav_path)
        qapp.processEvents()

        visible_detection = DetectedUSV(
            start_time_s=5.0,
            end_time_s=5.4,
            start_col=500,
            end_col=540,
            max_probability=0.9,
            mean_probability=0.8,
        )
        offscreen_detection = DetectedUSV(
            start_time_s=15.0,
            end_time_s=15.3,
            start_col=1500,
            end_col=1530,
            max_probability=0.9,
            mean_probability=0.8,
        )

        window.detection_result = DetectionResult(
            usvs=[visible_detection, offscreen_detection],
            probabilities=np.linspace(0.0, 1.0, audio_data.times.size, dtype=float),
            column_indices=np.arange(audio_data.times.size, dtype=int),
            times=audio_data.times,
        )
        window.spectrogram_view.set_detections(window.detection_result.usvs)
        qapp.processEvents()

        scrollbar = window.spectrogram_view.scroll_area.horizontalScrollBar()
        viewport_width = window.spectrogram_view.scroll_area.viewport().width()
        canvas_width = window.spectrogram_view.canvas.width()
        target_center_x = int((visible_detection.start_col + visible_detection.end_col) / 2)
        scrollbar.setValue(max(0, min(target_center_x - viewport_width // 2, canvas_width - viewport_width)))
        qapp.processEvents()

        visible = window._get_visible_detections()

        assert visible_detection in visible
        assert offscreen_detection not in visible

        window.close()

    def test_canvas_click_delete_removes_current_detection_only(
        self, qapp, tmp_path: Path, monkeypatch
    ) -> None:
        main_window_module = _load_main_window_module(monkeypatch)
        monkeypatch.setattr(main_window_module.QMessageBox, "critical", staticmethod(lambda *args, **kwargs: None))
        monkeypatch.setattr(
            main_window_module.QMessageBox,
            "question",
            staticmethod(lambda *_args, **_kwargs: main_window_module.QMessageBox.StandardButton.Yes),
        )
        monkeypatch.setattr(main_window_module.QMessageBox, "information", staticmethod(lambda *args, **kwargs: None))
        monkeypatch.setattr(main_window_module.QMessageBox, "warning", staticmethod(lambda *args, **kwargs: None))

        audio_data = _make_audio_data(duration_s=8.0, n_cols=800)
        monkeypatch.setattr(
            main_window_module.AudioLoader,
            "load",
            lambda self, _wav_path: audio_data,
        )

        window = main_window_module.MainWindow(default_model_path=None)
        window.resize(1000, 700)
        window.show()

        wav_path = tmp_path / "delete.wav"
        wav_path.write_bytes(b"")
        window._load_wav_file(wav_path)
        qapp.processEvents()

        current_detection = DetectedUSV(
            start_time_s=2.0,
            end_time_s=2.3,
            start_col=200,
            end_col=230,
            max_probability=0.9,
            mean_probability=0.8,
        )
        ghost_detection = DetectedUSV(
            start_time_s=5.0,
            end_time_s=5.2,
            start_col=500,
            end_col=520,
            max_probability=0.0,
            mean_probability=0.0,
            save_state="saved_previous",
        )
        window.detection_result = DetectionResult(
            usvs=[current_detection],
            probabilities=np.linspace(0.0, 1.0, audio_data.times.size, dtype=float),
            column_indices=np.arange(audio_data.times.size, dtype=int),
            times=audio_data.times,
        )
        window.inference_result = object()
        window.saved_tracker = main_window_module.SavedDetectionTracker(wav_path.stem, tmp_path / "exports")
        window.detection_exporter = main_window_module.DetectionExporter(tmp_path / "exports", context_ms=20.0)
        window.spectrogram_view.set_detections([current_detection, ghost_detection])
        qapp.processEvents()

        start_x = window.spectrogram_view.canvas._col_to_pixel(current_detection.start_col)
        QTest.mouseClick(
            window.spectrogram_view.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPoint(start_x + 1, 20),
        )
        qapp.processEvents()

        window._remove_detection()
        qapp.processEvents()

        assert window.detection_result.usvs == []
        assert window.spectrogram_view.canvas._selected_detection_idx is None

        window.close()

    def test_load_labels_then_adjust_and_save_current_view(
        self, qapp, tmp_path: Path, monkeypatch
    ) -> None:
        main_window_module = _load_main_window_module(monkeypatch)
        monkeypatch.setattr(main_window_module.QMessageBox, "critical", staticmethod(lambda *args, **kwargs: None))
        monkeypatch.setattr(main_window_module.QMessageBox, "information", staticmethod(lambda *args, **kwargs: None))
        monkeypatch.setattr(main_window_module.QMessageBox, "warning", staticmethod(lambda *args, **kwargs: None))
        monkeypatch.setattr(
            main_window_module.QMessageBox,
            "question",
            staticmethod(lambda *_args, **_kwargs: main_window_module.QMessageBox.StandardButton.Yes),
        )

        audio_data = _make_audio_data(duration_s=10.0, n_cols=1000)
        monkeypatch.setattr(
            main_window_module.AudioLoader,
            "load",
            lambda self, _wav_path: audio_data,
        )

        window = main_window_module.MainWindow(default_model_path=None)
        window.resize(1000, 700)
        window.show()

        wav_path = tmp_path / "adjust.wav"
        wav_path.write_bytes(b"")
        window._load_wav_file(wav_path)
        qapp.processEvents()

        labels_path = tmp_path / "adjust.json"
        from usv_spectrogram.app.core.label_storage import LabelStorage

        loaded_detection = DetectedUSV(
            start_time_s=4.0,
            end_time_s=4.3,
            start_col=400,
            end_col=430,
            max_probability=0.9,
            mean_probability=0.8,
            save_state="saved_current",
        )
        LabelStorage.save(
            output_path=labels_path,
            audio_data=audio_data,
            detection_result=DetectionResult(
                usvs=[loaded_detection],
                probabilities=np.linspace(0.0, 1.0, audio_data.times.size, dtype=float),
                column_indices=np.arange(audio_data.times.size, dtype=int),
                times=audio_data.times,
            ),
            wav_path=wav_path,
            model_path=None,
            high_threshold=0.4,
            low_threshold=0.3,
        )
        monkeypatch.setattr(
            main_window_module.QFileDialog,
            "getOpenFileName",
            staticmethod(lambda *_args, **_kwargs: (str(labels_path), "JSON")),
        )

        window.output_dir = tmp_path / "exports"
        window._load_labels()
        qapp.processEvents()

        detection = window.detection_result.usvs[0]
        start_x = window.spectrogram_view.canvas._col_to_pixel(detection.start_col)
        moved_x = start_x + 20
        QTest.mousePress(
            window.spectrogram_view.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPoint(start_x + 1, 20),
        )
        QTest.mouseMove(window.spectrogram_view.canvas, QPoint(moved_x, 20))
        QTest.mouseRelease(
            window.spectrogram_view.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPoint(moved_x, 20),
        )
        qapp.processEvents()

        adjusted = window.detection_result.usvs[0]
        assert adjusted.user_adjusted is True
        assert adjusted.start_time_s > loaded_detection.start_time_s
        assert adjusted.save_state == "unsaved"

        window._save_current_view()
        qapp.processEvents()

        assert window.detection_result.usvs[0].save_state == "saved_current"
        exported = sorted((window.output_dir / wav_path.stem).glob("detection_*.json"))
        assert len(exported) == 1

        window.detection_result = None
        window.close()

    def test_save_current_view_ignores_visible_ghosts_under_real_scroll(
        self, qapp, tmp_path: Path, monkeypatch
    ) -> None:
        main_window_module = _load_main_window_module(monkeypatch)
        monkeypatch.setattr(main_window_module.QMessageBox, "critical", staticmethod(lambda *args, **kwargs: None))
        monkeypatch.setattr(main_window_module.QMessageBox, "information", staticmethod(lambda *args, **kwargs: None))
        monkeypatch.setattr(main_window_module.QMessageBox, "warning", staticmethod(lambda *args, **kwargs: None))
        monkeypatch.setattr(
            main_window_module.QMessageBox,
            "question",
            staticmethod(lambda *_args, **_kwargs: main_window_module.QMessageBox.StandardButton.Yes),
        )

        audio_data = _make_audio_data(duration_s=20.0, n_cols=2000)
        monkeypatch.setattr(
            main_window_module.AudioLoader,
            "load",
            lambda self, _wav_path: audio_data,
        )

        window = main_window_module.MainWindow(default_model_path=None)
        window.resize(900, 700)
        window.show()

        wav_path = tmp_path / "ghost-save.wav"
        wav_path.write_bytes(b"")
        window.output_dir = tmp_path / "exports"
        window._load_wav_file(wav_path)
        qapp.processEvents()

        current_detection = DetectedUSV(
            start_time_s=5.0,
            end_time_s=5.3,
            start_col=500,
            end_col=530,
            max_probability=0.9,
            mean_probability=0.8,
        )
        prior_saved = DetectedUSV(
            start_time_s=5.6,
            end_time_s=5.8,
            start_col=560,
            end_col=580,
            max_probability=0.9,
            mean_probability=0.8,
            save_state="saved_current",
        )
        window.detection_result = DetectionResult(
            usvs=[current_detection],
            probabilities=np.linspace(0.0, 1.0, audio_data.times.size, dtype=float),
            column_indices=np.arange(audio_data.times.size, dtype=int),
            times=audio_data.times,
        )
        window.inference_result = object()
        window.saved_tracker = main_window_module.SavedDetectionTracker(wav_path.stem, window.output_dir)
        window.detection_exporter = main_window_module.DetectionExporter(window.output_dir, context_ms=20.0)
        exported_png, _, summary_path = window.detection_exporter.export_detection(
            detection=prior_saved,
            audio_data=audio_data,
            wav_filename=wav_path.stem,
            detection_index=1,
            session_id=window.session_id,
            threshold_preset=window.current_preset,
            threshold_high=window.high_threshold,
            threshold_low=window.low_threshold,
        )
        window.saved_tracker.mark_saved(prior_saved, str(exported_png))

        window._refresh_detection_views()
        qapp.processEvents()

        scrollbar = window.spectrogram_view.scroll_area.horizontalScrollBar()
        viewport_width = window.spectrogram_view.scroll_area.viewport().width()
        canvas_width = window.spectrogram_view.canvas.width()
        target_center_x = int((current_detection.start_col + current_detection.end_col) / 2)
        scrollbar.setValue(max(0, min(target_center_x - viewport_width // 2, canvas_width - viewport_width)))
        qapp.processEvents()

        window._save_current_view()
        qapp.processEvents()

        exported_json = sorted((window.output_dir / wav_path.stem).glob("detection_*.json"))
        assert len(exported_json) == 2
        assert window.detection_result.usvs[0].save_state == "saved_current"

        with summary_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 2
        assert [row["start_time_s"] for row in rows] == ["5.000000", "5.600000"]

        window.detection_result = None
        window.close()
