"""Spectrogram extraction for USV candidate segments.

Converts detected candidate segments into standardized spectrogram PNG images
for human labeling and CNN training.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from ..io_wav import load_wav_segment_mono
from .candidate import Candidate
from .extraction_config import ExtractionConfig


class SpectrogramExtractor:
    """Extract spectrogram images from candidate USV segments."""

    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()

    def extract_single(
        self,
        candidate: Candidate,
        wav_dir: Path,
        output_dir: Path,
        render_mode: Optional[str] = None,
    ) -> Optional[Path]:
        """Extract spectrogram for a single candidate.

        Parameters
        ----------
        candidate:
            Candidate to extract.
        wav_dir:
            Directory containing WAV files.
        output_dir:
            Directory to save spectrogram PNG.
        render_mode:
            "review" or "training". Defaults to config.default_render_mode.

        Returns
        -------
        Path to saved PNG, or None if extraction failed.
        """
        cfg = self.config
        render_mode = render_mode or cfg.default_render_mode

        # Find source WAV file
        wav_path = wav_dir / candidate.source_file.name
        if not wav_path.exists():
            wav_path = wav_dir / candidate.source_file
            if not wav_path.exists():
                return None

        # Load audio segment with context
        start_s = candidate.context_start_ms / 1000.0
        duration_s = (candidate.context_end_ms - candidate.context_start_ms) / 1000.0

        try:
            samples, sample_rate = load_wav_segment_mono(wav_path, start_s, duration_s)
        except Exception:
            return None

        # Validate sample rate matches config
        if sample_rate != cfg.sample_rate:
            raise ValueError(
                f"Expected sample rate {cfg.sample_rate} Hz, got {sample_rate} Hz. "
                f"Set config.sample_rate to match your recordings."
            )

        if len(samples) < cfg.n_fft:
            return None

        # Compute spectrogram
        spec_db, freqs_hz, times_s = self._compute_spectrogram(samples, sample_rate)
        if spec_db.size == 0:
            return None

        # Render image
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{candidate.candidate_id}.png"

        if render_mode == "review":
            self._render_review(
                spec_db, freqs_hz, times_s, output_path, candidate
            )
        else:
            self._render_training(spec_db, output_path)

        return output_path

    def extract_batch(
        self,
        candidates_csv: Path,
        wav_dir: Path,
        output_dir: Path,
        render_mode: Optional[str] = None,
    ) -> Iterator[tuple[Candidate, Optional[Path]]]:
        """Extract spectrograms for all candidates in a CSV.

        Yields (candidate, output_path) tuples.
        output_path is None if extraction failed.
        """
        candidates_csv = Path(candidates_csv)
        wav_dir = Path(wav_dir)
        output_dir = Path(output_dir)

        with open(candidates_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                candidate = Candidate.from_dict(row, base_path=wav_dir)
                output_path = self.extract_single(
                    candidate, wav_dir, output_dir, render_mode
                )
                yield candidate, output_path

    def save_updated_csv(
        self,
        candidates: list[tuple[Candidate, Optional[Path]]],
        output_csv: Path,
    ) -> None:
        """Save candidates with updated spectrogram_path column."""
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        if not candidates:
            return

        # Update spectrogram_path in candidates
        updated = []
        for candidate, spec_path in candidates:
            if spec_path is not None:
                candidate.spectrogram_path = spec_path
            updated.append(candidate)

        fieldnames = list(updated[0].to_dict().keys())
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for candidate in updated:
                writer.writerow(candidate.to_dict())

    def _compute_dynamic_range(self, spec_db: np.ndarray) -> tuple[float, float]:
        """Compute vmin/vmax for color scaling based on config method.

        Returns:
            (vmin, vmax) tuple for color scaling.
        """
        cfg = self.config
        method = cfg.dynamic_range_method

        if method == "fixed":
            return cfg.db_floor, cfg.db_ceiling

        elif method == "percentile":
            vmax = np.percentile(spec_db, 99)
            vmin = np.percentile(spec_db, 5) - 10
            return vmin, vmax

        elif method == "std":
            mean = spec_db.mean()
            std = spec_db.std()
            vmin = mean - 2 * std
            vmax = mean + 3 * std
            return vmin, vmax

        elif method == "mad":
            median = np.median(spec_db)
            mad = np.median(np.abs(spec_db - median))
            vmin = median - cfg.mad_vmin_scale * mad
            vmax = median + cfg.mad_vmax_scale * mad
            return vmin, vmax

        else:
            # Fallback to fixed
            return cfg.db_floor, cfg.db_ceiling

    def _build_continuity_kernel(self) -> np.ndarray:
        """Build the continuity filter kernel with diagonal emphasis."""
        cfg = self.config
        size = cfg.continuity_kernel_size
        radius = size // 2
        kernel = np.zeros((size, size), dtype=float)

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    weight = cfg.continuity_weight_center
                elif dx == 0 or dy == 0:
                    weight = cfg.continuity_weight_axis / (abs(dx) + abs(dy))
                elif abs(dx) == abs(dy):
                    weight = cfg.continuity_weight_diag / max(abs(dx), 1)
                else:
                    weight = 0.0

                kernel[dy + radius, dx + radius] = weight

        kernel_sum = kernel.sum()
        if kernel_sum > 0:
            kernel /= kernel_sum

        return kernel

    def _apply_continuity_filter(self, spec_db: np.ndarray) -> np.ndarray:
        """Apply continuity filter to reduce fragmented tracks in review images."""
        kernel = self._build_continuity_kernel()
        return signal.convolve2d(spec_db, kernel, mode="same", boundary="symm")

    def _compute_spectrogram(
        self, samples: np.ndarray, sample_rate: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute spectrogram limited to configured frequency band.

        Returns:
            spec_db: Spectrogram in dB, shape (n_freq_bins, n_frames)
            freqs_hz: Frequency bins in Hz
            times_s: Time bins in seconds
        """
        cfg = self.config

        # Create window
        window = signal.get_window(cfg.window, cfg.n_fft, fftbins=True)

        # Extract frames
        n_frames = 1 + (len(samples) - cfg.n_fft) // cfg.hop_length
        if n_frames <= 0:
            return np.empty((0, 0)), np.empty(0), np.empty(0)

        frame_starts = np.arange(n_frames) * cfg.hop_length
        frames = np.stack(
            [samples[start : start + cfg.n_fft] for start in frame_starts],
            axis=0,
        )

        # Apply window and compute FFT
        windowed = frames * window
        stft = np.fft.rfft(windowed, n=cfg.n_fft, axis=1)
        magnitude = np.abs(stft)

        # Get frequency bins and band mask
        freqs_hz = np.fft.rfftfreq(cfg.n_fft, d=1.0 / sample_rate)
        band_mask = (freqs_hz >= cfg.freq_min_hz) & (freqs_hz <= cfg.freq_max_hz)

        # Convert to dB (normalized to max so loudest = 0 dB)
        eps = 1e-12
        magnitude_normalized = magnitude / (np.max(magnitude) + eps)
        spec_db = 20.0 * np.log10(magnitude_normalized + eps)
        spec_db = spec_db[:, band_mask].T  # Shape: (n_freq_bins, n_frames)

        # Time array
        times_s = frame_starts / sample_rate

        return spec_db, freqs_hz[band_mask], times_s

    def _render_review(
        self,
        spec_db: np.ndarray,
        freqs_hz: np.ndarray,
        times_s: np.ndarray,
        output_path: Path,
        candidate: Candidate,
    ) -> None:
        """Render spectrogram with axes and labels for human review."""
        cfg = self.config

        if cfg.continuity_filter_enabled:
            spec_db = self._apply_continuity_filter(spec_db)

        # Dynamic color scaling based on spectrogram statistics
        vmin, vmax = self._compute_dynamic_range(spec_db)

        display_db = np.clip(spec_db, vmin, vmax)

        # Calculate figure size based on duration
        duration_ms = (times_s[-1] - times_s[0]) * 1000 if len(times_s) > 1 else 100
        width_px = cfg.width_px_for_duration_ms(duration_ms)
        height_px = cfg.image_height_px

        # Convert to inches (at 100 dpi for reasonable file size)
        dpi = 100
        fig_width = width_px / dpi
        fig_height = height_px / dpi

        fig, ax = plt.subplots(figsize=(fig_width + 1, fig_height + 0.5), dpi=dpi)

        mesh = ax.pcolormesh(
            times_s * 1000,  # Convert to ms
            freqs_hz / 1000,  # Convert to kHz
            display_db,
            shading="auto",
            cmap=cfg.colormap,
            vmin=vmin,
            vmax=vmax,
        )

        # Mark detected region
        ax.axvline(
            candidate.start_ms - candidate.context_start_ms,
            color="lime",
            linewidth=1,
            linestyle="--",
            alpha=0.7,
        )
        ax.axvline(
            candidate.end_ms - candidate.context_start_ms,
            color="lime",
            linewidth=1,
            linestyle="--",
            alpha=0.7,
        )

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency (kHz)")
        ax.set_title(
            f"{candidate.candidate_id} | {candidate.peak_freq_hz/1000:.1f} kHz | "
            f"{candidate.duration_ms:.0f} ms"
        )

        fig.colorbar(mesh, ax=ax, label="dB")
        fig.tight_layout()
        fig.savefig(str(output_path))
        plt.close(fig)

    def _render_training(
        self,
        spec_db: np.ndarray,
        output_path: Path,
    ) -> None:
        """Render raw spectrogram image for CNN training (no axes)."""
        cfg = self.config

        # Apply color scaling and normalize to 0-1
        vmin, vmax = self._compute_dynamic_range(spec_db)
        display_db = np.clip(spec_db, vmin, vmax)
        normalized = (display_db - vmin) / (vmax - vmin + 1e-12)

        # Get colormap
        cmap = plt.get_cmap(cfg.colormap)
        rgb = cmap(normalized)[:, :, :3]  # Drop alpha channel

        # Resize to target dimensions
        from PIL import Image

        # Flip vertically so low frequencies are at bottom
        rgb_flipped = np.flipud(rgb)

        # Convert to uint8
        rgb_uint8 = (rgb_flipped * 255).astype(np.uint8)

        # Create PIL image and resize
        img = Image.fromarray(rgb_uint8)

        # Calculate target width based on aspect ratio
        n_frames = spec_db.shape[1]
        duration_ms = n_frames * cfg.hop_ms()
        target_width = cfg.width_px_for_duration_ms(duration_ms)
        target_height = cfg.image_height_px

        img_resized = img.resize((target_width, target_height), Image.LANCZOS)
        img_resized.save(str(output_path))
