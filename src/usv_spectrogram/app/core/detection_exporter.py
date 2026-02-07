"""Export individual detections as annotated images with metadata."""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for rendering
import matplotlib.pyplot as plt
import numpy as np

from usv_spectrogram.app.core.audio_loader import AudioData
from usv_spectrogram.app.core.detection_logic import DetectedUSV


class DetectionExporter:
    """Export individual detections as annotated images with metadata."""

    def __init__(self, output_dir: Path, context_ms: float = 20.0):
        """Initialize exporter.

        Args:
            output_dir: Base output directory for all saved detections
            context_ms: Amount of context to include before/after detection (milliseconds)
        """
        self.output_dir = output_dir
        self.context_sec = context_ms / 1000.0

    def export_detection(
        self,
        detection: DetectedUSV,
        audio_data: AudioData,
        wav_filename: str,
        detection_index: int,
        session_id: str | None = None,
        threshold_preset: str | None = None,
        threshold_high: float | None = None,
        threshold_low: float | None = None
    ) -> Tuple[Path, Path, Path]:
        """Export detection as PNG, JSON, and CSV entry.

        Args:
            detection: Detection to export
            audio_data: Full audio data (spectrogram, times, frequencies)
            wav_filename: Name of WAV file (without extension)
            detection_index: Index of this detection in the full detection list
            session_id: Session UUID for tracking
            threshold_preset: Name of threshold preset used
            threshold_high: High threshold value
            threshold_low: Low threshold value

        Returns:
            Tuple of (png_path, json_path, csv_path)
        """
        # 1. Create output directory for this WAV file
        output_subdir = self.output_dir / wav_filename
        output_subdir.mkdir(parents=True, exist_ok=True)

        # 2. Determine time range (detection + context)
        start_time_with_context = max(0, detection.start_time_s - self.context_sec)
        end_time_with_context = min(
            audio_data.duration_s,
            detection.end_time_s + self.context_sec
        )

        # 3. Extract spectrogram region
        start_idx = np.searchsorted(audio_data.times, start_time_with_context)
        end_idx = np.searchsorted(audio_data.times, end_time_with_context)

        sub_spec = audio_data.spectrogram_db[:, start_idx:end_idx+1]
        sub_times = audio_data.times[start_idx:end_idx+1]
        sub_freqs = audio_data.frequencies

        # 4. Generate filename (use core detection time, not context)
        base_name = f"detection_{detection_index:03d}_{detection.start_time_s:.3f}s-{detection.end_time_s:.3f}s"

        # 5. Render annotated PNG
        png_path = output_subdir / f"{base_name}.png"
        self._render_annotated_png(
            sub_spec, sub_times, sub_freqs,
            detection, start_time_with_context, end_time_with_context,
            png_path
        )

        # 6. Save JSON metadata
        json_path = output_subdir / f"{base_name}.json"
        self._save_json_metadata(
            detection, detection_index,
            start_time_with_context, end_time_with_context,
            json_path,
            session_id=session_id,
            threshold_preset=threshold_preset,
            threshold_high=threshold_high,
            threshold_low=threshold_low
        )

        # 7. Append to CSV summary
        csv_path = output_subdir / "detections_summary.csv"
        self._append_to_csv(
            detection, detection_index, wav_filename,
            csv_path
        )

        return png_path, json_path, csv_path

    def _render_annotated_png(
        self,
        spec: np.ndarray,
        times: np.ndarray,
        freqs: np.ndarray,
        detection: DetectedUSV,
        t_start: float,
        t_end: float,
        output_path: Path
    ):
        """Render spectrogram with time/frequency axes and detection bounds.

        Args:
            spec: Spectrogram array (freqs x times)
            times: Time values for each column
            freqs: Frequency bins in Hz
            detection: Detection object
            t_start: Start time with context
            t_end: End time with context
            output_path: Where to save PNG
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Convert frequencies to kHz for display
        freqs_khz = freqs / 1000.0

        # Plot spectrogram
        im = ax.imshow(
            spec,
            aspect='auto',
            origin='lower',
            extent=[times[0], times[-1], freqs_khz[0], freqs_khz[-1]],
            cmap='magma',
            interpolation='nearest'
        )

        # Draw detection boundaries as vertical lines
        ax.axvline(detection.start_time_s, color='cyan', linestyle='--',
                   linewidth=2, label='Detection start')
        ax.axvline(detection.end_time_s, color='lime', linestyle='--',
                   linewidth=2, label='Detection end')

        # Labels and formatting
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Frequency (kHz)', fontsize=12)

        duration_ms = (detection.end_time_s - detection.start_time_s) * 1000
        ax.set_title(
            f'Detection {detection.start_time_s:.3f}s - {detection.end_time_s:.3f}s '
            f'({duration_ms:.1f} ms)\n'
            f'Max prob: {detection.max_probability:.3f}, '
            f'Mean prob: {detection.mean_probability:.3f}',
            fontsize=11
        )

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, label='Power (dB)')

        # Grid
        ax.grid(True, alpha=0.3, linestyle=':')

        # Legend
        ax.legend(loc='upper right', fontsize=9)

        # Save
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _save_json_metadata(
        self,
        detection: DetectedUSV,
        index: int,
        t_start_ctx: float,
        t_end_ctx: float,
        output_path: Path,
        session_id: str | None = None,
        threshold_preset: str | None = None,
        threshold_high: float | None = None,
        threshold_low: float | None = None
    ):
        """Save detection metadata as JSON.

        Args:
            detection: Detection object
            index: Detection index
            t_start_ctx: Start time with context
            t_end_ctx: End time with context
            output_path: Where to save JSON
            session_id: Session UUID for tracking
            threshold_preset: Name of threshold preset used
            threshold_high: High threshold value
            threshold_low: Low threshold value
        """
        metadata = {
            "detection_index": index,
            "core_time": {
                "start_s": detection.start_time_s,
                "end_s": detection.end_time_s,
                "duration_ms": (detection.end_time_s - detection.start_time_s) * 1000
            },
            "saved_region": {
                "start_s": t_start_ctx,
                "end_s": t_end_ctx,
                "context_ms": self.context_sec * 1000
            },
            "probabilities": {
                "max": detection.max_probability,
                "mean": detection.mean_probability,
                "original_cnn_probability": detection.original_cnn_probability
            },
            "spectrogram_columns": {
                "start_col": int(detection.start_col),
                "end_col": int(detection.end_col)
            },
            "user_action": detection.user_action,
            "session": {
                "session_id": session_id,
                "threshold_preset": threshold_preset,
                "threshold_high": threshold_high,
                "threshold_low": threshold_low
            },
            "timestamp": datetime.now().isoformat()
        }

        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _append_to_csv(
        self,
        detection: DetectedUSV,
        index: int,
        wav_filename: str,
        csv_path: Path
    ):
        """Append detection info to CSV summary.

        Args:
            detection: Detection object
            index: Detection index
            wav_filename: Name of WAV file
            csv_path: Path to CSV file
        """
        # Check if CSV exists to determine if we need header
        file_exists = csv_path.exists()

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)

            # Write header if new file
            if not file_exists:
                writer.writerow([
                    'wav_file',
                    'detection_index',
                    'start_time_s',
                    'end_time_s',
                    'duration_ms',
                    'max_prob',
                    'mean_prob',
                    'user_action',
                    'timestamp'
                ])

            # Write detection row
            duration_ms = (detection.end_time_s - detection.start_time_s) * 1000
            writer.writerow([
                wav_filename,
                index,
                f'{detection.start_time_s:.6f}',
                f'{detection.end_time_s:.6f}',
                f'{duration_ms:.2f}',
                f'{detection.max_probability:.6f}',
                f'{detection.mean_probability:.6f}',
                detection.user_action or '',  # Empty string if None
                datetime.now().isoformat()
            ])
