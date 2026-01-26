"""JSON storage for USV detection labels and metadata.

Handles saving/loading detection results with full metadata including
probability curves, detection parameters, and export functionality.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from .audio_loader import AudioData
from .detection_logic import DetectionResult, DetectedUSV


class LabelStorage:
    """Manages JSON storage and export for detection results."""

    @staticmethod
    def save(
        output_path: str | Path,
        audio_data: AudioData,
        detection_result: DetectionResult,
        wav_path: str | Path,
        model_path: str | Path,
        high_threshold: float,
        low_threshold: float
    ) -> None:
        """Save detection results to JSON file.

        Args:
            output_path: Path to output JSON file
            audio_data: Audio and spectrogram data
            detection_result: Detection results
            wav_path: Path to source WAV file
            model_path: Path to CNN model checkpoint
            high_threshold: High threshold used for detection
            low_threshold: Low threshold used for detection
        """
        output_path = Path(output_path)

        # Build JSON structure
        data = {
            "metadata": {
                "wav_file": str(Path(wav_path).absolute()),
                "model_file": str(Path(model_path).absolute()),
                "created_at": datetime.now().isoformat(),
                "duration_s": audio_data.duration_s,
                "sample_rate": audio_data.sample_rate,
                "n_detections": len(detection_result.usvs)
            },
            "detection_params": {
                "high_threshold": high_threshold,
                "low_threshold": low_threshold
            },
            "detections": [
                {
                    "start_time_s": usv.start_time_s,
                    "end_time_s": usv.end_time_s,
                    "duration_s": usv.end_time_s - usv.start_time_s,
                    "start_col": int(usv.start_col),
                    "end_col": int(usv.end_col),
                    "max_probability": usv.max_probability,
                    "mean_probability": usv.mean_probability
                }
                for usv in detection_result.usvs
            ],
            "probability_curve": {
                "times": detection_result.times.tolist(),
                "probabilities": detection_result.probabilities.tolist(),
                "column_indices": detection_result.column_indices.tolist()
            }
        }

        # Write JSON with pretty formatting
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(input_path: str | Path) -> Dict[str, Any]:
        """Load detection results from JSON file.

        Args:
            input_path: Path to JSON file

        Returns:
            Dictionary with detection data

        Raises:
            FileNotFoundError: If JSON file doesn't exist
            json.JSONDecodeError: If JSON is malformed
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"JSON file not found: {input_path}")

        with open(input_path, 'r') as f:
            data = json.load(f)

        return data

    @staticmethod
    def export_annotated_image(
        output_path: str | Path,
        audio_data: AudioData,
        detection_result: DetectionResult,
        high_threshold: float,
        low_threshold: float,
        dpi: int = 100,
        figsize: tuple[float, float] = (16, 6)
    ) -> None:
        """Export annotated spectrogram with detection overlays.

        Creates a figure with:
        - Top panel: Spectrogram with USV boundary lines
        - Bottom panel: Probability curve with thresholds

        Args:
            output_path: Path to output PNG file
            audio_data: Audio and spectrogram data
            detection_result: Detection results
            high_threshold: High threshold value
            low_threshold: Low threshold value
            dpi: Image resolution (default: 100)
            figsize: Figure size in inches (default: (16, 6))
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create figure with two subplots
        fig, (ax_spec, ax_prob) = plt.subplots(
            2, 1,
            figsize=figsize,
            height_ratios=[2, 1],
            sharex=True
        )

        # Plot spectrogram
        extent = [
            audio_data.times[0],
            audio_data.times[-1],
            audio_data.frequencies[0] / 1000,  # Convert to kHz
            audio_data.frequencies[-1] / 1000
        ]

        ax_spec.imshow(
            audio_data.spectrogram_db,
            aspect='auto',
            origin='lower',
            extent=extent,
            cmap='magma',
            interpolation='nearest'
        )

        # Add USV boundary lines
        for usv in detection_result.usvs:
            # Start line (green)
            ax_spec.axvline(
                usv.start_time_s,
                color='lime',
                linewidth=1.5,
                linestyle='-',
                alpha=0.8
            )
            # End line (red)
            ax_spec.axvline(
                usv.end_time_s,
                color='red',
                linewidth=1.5,
                linestyle='-',
                alpha=0.8
            )

        ax_spec.set_ylabel('Frequency (kHz)')
        ax_spec.set_title('Spectrogram with Detected USVs')
        ax_spec.grid(True, alpha=0.3)

        # Plot probability curve
        ax_prob.plot(
            detection_result.times,
            detection_result.probabilities,
            color='blue',
            linewidth=1.5,
            label='CNN Probability'
        )

        # Add threshold lines
        ax_prob.axhline(
            high_threshold,
            color='red',
            linestyle='--',
            linewidth=1.5,
            label=f'High Threshold ({high_threshold:.2f})'
        )
        ax_prob.axhline(
            low_threshold,
            color='orange',
            linestyle='--',
            linewidth=1.5,
            label=f'Low Threshold ({low_threshold:.2f})'
        )

        # Shade detected regions
        for usv in detection_result.usvs:
            ax_prob.axvspan(
                usv.start_time_s,
                usv.end_time_s,
                alpha=0.2,
                color='green'
            )

        ax_prob.set_xlabel('Time (s)')
        ax_prob.set_ylabel('Probability')
        ax_prob.set_ylim([0, 1])
        ax_prob.legend(loc='upper right', fontsize=8)
        ax_prob.grid(True, alpha=0.3)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
