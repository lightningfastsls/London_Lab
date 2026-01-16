"""Energy-based USV candidate detector.

This detector is intentionally simple and optimized for HIGH RECALL.
Precision is handled downstream by human labeling.

Design principles (see usv_signal_processing_reference.md):
- Optimize for RECALL over precision
- Use conservative (low) energy threshold
- Add duration filters to reject obvious artifacts
- Output candidates with full metadata for traceability
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from scipy import signal

from ..io_wav import load_wav_mono
from .config import DetectionConfig
from .candidate import Candidate


class EnergyDetector:
    """Detect candidate USV segments using energy thresholding.

    This is intentionally a simple detector optimized for recall.
    Precision is handled downstream by human labeling.
    """

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()

    def detect(self, wav_path: Path) -> list[Candidate]:
        """Detect all candidate USVs in a WAV file.

        Returns candidates sorted by start time.

        Implementation steps:
        1. Load audio
        2. Compute spectrogram (STFT)
        3. Sum energy in USV frequency band per frame
        4. Threshold to get candidate frames
        5. Group adjacent frames into segments
        6. Merge segments separated by < merge_gap_ms
        7. Apply duration filters
        8. Extract peak frequency for each candidate
        9. Create Candidate objects with full metadata
        """
        wav_path = Path(wav_path)
        cfg = self.config

        # 1. Load audio
        samples, sample_rate = load_wav_mono(wav_path)
        if sample_rate != cfg.sample_rate:
            raise ValueError(
                f"Expected sample rate {cfg.sample_rate} Hz, got {sample_rate} Hz. "
                f"Set config.sample_rate to match your recordings."
            )

        if len(samples) < cfg.n_fft:
            return []  # Audio too short

        # 2. Compute spectrogram
        spec_db, freqs_hz = self._compute_band_spectrogram(samples)
        if spec_db.shape[1] == 0:
            return []  # No frames

        # 3. Sum energy in USV band per frame (convert from dB first)
        # Use logsumexp-like approach: mean of dB values per frame
        band_energy_db = np.mean(spec_db, axis=0)

        # 4. Threshold to get candidate frames
        max_energy = np.max(band_energy_db)
        threshold_db = max_energy + cfg.energy_threshold_db
        active_frames = band_energy_db >= threshold_db

        if not np.any(active_frames):
            return []  # No detections

        # 5. Group adjacent frames into segments
        segments = self._frames_to_segments(active_frames)

        # 6. Merge nearby segments
        segments = self._merge_segments(segments)

        # 7. Apply duration filters
        segments = self._filter_by_duration(segments)

        if not segments:
            return []

        # 8-9. Create Candidate objects with peak frequency and energy
        candidates = []
        for start_frame, end_frame in segments:
            candidate = self._create_candidate(
                wav_path, spec_db, freqs_hz, start_frame, end_frame
            )
            if candidate is not None:
                candidates.append(candidate)

        # Sort by start time
        candidates.sort(key=lambda c: c.start_ms)
        return candidates

    def detect_batch(
        self,
        wav_dir: Path,
        pattern: str = "*.wav",
    ) -> Iterator[Candidate]:
        """Detect candidates across all WAV files in a directory.

        Yields candidates one at a time for memory efficiency.
        """
        wav_dir = Path(wav_dir)
        wav_files = sorted(wav_dir.glob(pattern))

        for wav_path in wav_files:
            try:
                candidates = self.detect(wav_path)
                for candidate in candidates:
                    yield candidate
            except Exception as e:
                print(f"Warning: Failed to process {wav_path.name}: {e}")
                continue

    def save_candidates_csv(
        self,
        candidates: list[Candidate],
        output_path: Path,
    ) -> None:
        """Save candidates to CSV file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not candidates:
            return

        fieldnames = list(candidates[0].to_dict().keys())
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for candidate in candidates:
                writer.writerow(candidate.to_dict())

    def _compute_band_spectrogram(
        self, samples: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute spectrogram limited to USV frequency band.

        Returns:
            spec_db: Spectrogram in dB, shape (n_freq_bins, n_frames)
            freqs_hz: Frequency bins in Hz
        """
        cfg = self.config

        # Create window
        window = signal.get_window("hann", cfg.n_fft, fftbins=True)

        # Extract frames
        n_frames = 1 + (len(samples) - cfg.n_fft) // cfg.hop_length
        if n_frames <= 0:
            freqs_hz = np.fft.rfftfreq(cfg.n_fft, d=1.0 / cfg.sample_rate)
            band_mask = (freqs_hz >= cfg.freq_min_hz) & (freqs_hz <= cfg.freq_max_hz)
            return np.empty((band_mask.sum(), 0)), freqs_hz[band_mask]

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
        freqs_hz = np.fft.rfftfreq(cfg.n_fft, d=1.0 / cfg.sample_rate)
        band_mask = (freqs_hz >= cfg.freq_min_hz) & (freqs_hz <= cfg.freq_max_hz)

        # Convert to dB and select band
        eps = 1e-12
        spec_db = 20.0 * np.log10(magnitude + eps)
        spec_db = spec_db[:, band_mask].T  # Shape: (n_freq_bins, n_frames)

        return spec_db, freqs_hz[band_mask]

    def _frames_to_segments(
        self, active_frames: np.ndarray
    ) -> list[tuple[int, int]]:
        """Convert boolean frame mask to list of (start, end) segment tuples."""
        segments = []
        in_segment = False
        start_frame = 0

        for i, active in enumerate(active_frames):
            if active and not in_segment:
                start_frame = i
                in_segment = True
            elif not active and in_segment:
                segments.append((start_frame, i))
                in_segment = False

        # Handle segment that extends to end
        if in_segment:
            segments.append((start_frame, len(active_frames)))

        return segments

    def _merge_segments(
        self, segments: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Merge segments that are closer than merge_gap_ms."""
        if not segments:
            return []

        cfg = self.config
        gap_frames = int(cfg.merge_gap_ms / 1000.0 * cfg.sample_rate / cfg.hop_length)

        merged = [segments[0]]
        for start, end in segments[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end <= gap_frames:
                # Merge with previous segment
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))

        return merged

    def _filter_by_duration(
        self, segments: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Filter segments by duration constraints."""
        cfg = self.config
        ms_per_frame = cfg.hop_length / cfg.sample_rate * 1000.0

        filtered = []
        for start_frame, end_frame in segments:
            duration_ms = (end_frame - start_frame) * ms_per_frame
            if cfg.min_duration_ms <= duration_ms <= cfg.max_duration_ms:
                filtered.append((start_frame, end_frame))

        return filtered

    def _create_candidate(
        self,
        wav_path: Path,
        spec_db: np.ndarray,
        freqs_hz: np.ndarray,
        start_frame: int,
        end_frame: int,
    ) -> Optional[Candidate]:
        """Create a Candidate object from a segment."""
        cfg = self.config
        ms_per_frame = cfg.hop_length / cfg.sample_rate * 1000.0

        # Convert frames to milliseconds
        start_ms = start_frame * ms_per_frame
        end_ms = end_frame * ms_per_frame

        # Extract segment spectrogram
        segment_spec = spec_db[:, start_frame:end_frame]
        if segment_spec.size == 0:
            return None

        # Find peak frequency and energy
        max_idx = np.unravel_index(np.argmax(segment_spec), segment_spec.shape)
        peak_freq_hz = float(freqs_hz[max_idx[0]])
        peak_energy_db = float(segment_spec[max_idx])

        # Check for interference
        interference_flag = self._is_likely_interference(peak_freq_hz)

        return Candidate.create(
            source_file=wav_path,
            start_ms=start_ms,
            end_ms=end_ms,
            peak_freq_hz=peak_freq_hz,
            peak_energy_db=peak_energy_db,
            context_before_ms=cfg.context_before_ms,
            context_after_ms=cfg.context_after_ms,
            interference_flag=interference_flag,
        )

    def _is_likely_interference(self, freq_hz: float) -> bool:
        """Check if frequency is near known interference frequencies.

        See Section 2.3 of reference: 60 kHz and harmonics are often
        electrical interference. Flag for review but don't auto-reject.
        """
        cfg = self.config
        for interference_freq in cfg.interference_freqs_hz:
            if abs(freq_hz - interference_freq) <= cfg.interference_tolerance_hz:
                return True
        return False


def analyze_threshold_sensitivity(
    wav_path: Path,
    config: Optional[DetectionConfig] = None,
    threshold_range: tuple[float, float] = (-60.0, -20.0),
    threshold_step: float = 5.0,
) -> dict[float, int]:
    """Analyze how detection count changes with threshold.

    Returns dict mapping threshold_db -> candidate_count.

    Use this to find the "knee" where lowering threshold
    rapidly increases candidates (likely adding noise).

    Start BELOW the knee for high recall.
    """
    config = config or DetectionConfig()
    results = {}

    thresholds = np.arange(threshold_range[0], threshold_range[1] + threshold_step, threshold_step)

    for threshold in thresholds:
        # Create config with this threshold
        test_config = DetectionConfig(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            freq_min_hz=config.freq_min_hz,
            freq_max_hz=config.freq_max_hz,
            energy_threshold_db=threshold,
            min_duration_ms=config.min_duration_ms,
            max_duration_ms=config.max_duration_ms,
            merge_gap_ms=config.merge_gap_ms,
            context_before_ms=config.context_before_ms,
            context_after_ms=config.context_after_ms,
        )
        detector = EnergyDetector(test_config)
        candidates = detector.detect(wav_path)
        results[float(threshold)] = len(candidates)

    return results


def verify_detection_coverage(
    wav_path: Path,
    candidates: list[Candidate],
    manual_usv_times_ms: list[float],
    tolerance_ms: float = 50.0,
) -> dict:
    """Check that known USVs were detected.

    Parameters
    ----------
    wav_path: Path to the WAV file
    candidates: List of detected candidates
    manual_usv_times_ms: List of manually identified USV timestamps (in ms)
    tolerance_ms: How close a candidate must be to count as "detected"

    Returns
    -------
    Dict with:
        - detected: List of manual USV times that were detected
        - missed: List of manual USV times that were NOT detected
        - coverage_rate: Fraction of manual USVs that were detected

    If manual USVs are NOT in candidates, threshold may be too high
    or there's a bug in detection logic.

    See Section 3.2: this prevents training data bias.
    """
    detected = []
    missed = []

    for manual_time in manual_usv_times_ms:
        found = False
        for candidate in candidates:
            # Check if manual time falls within candidate bounds (with tolerance)
            if (candidate.start_ms - tolerance_ms <= manual_time <= candidate.end_ms + tolerance_ms):
                found = True
                break
        if found:
            detected.append(manual_time)
        else:
            missed.append(manual_time)

    coverage_rate = len(detected) / len(manual_usv_times_ms) if manual_usv_times_ms else 1.0

    return {
        "detected": detected,
        "missed": missed,
        "coverage_rate": coverage_rate,
        "total_manual": len(manual_usv_times_ms),
        "total_candidates": len(candidates),
    }
