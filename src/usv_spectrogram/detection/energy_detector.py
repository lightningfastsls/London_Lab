"""Energy-based USV candidate detector.

This detector is intentionally simple and optimized for HIGH RECALL.
Precision is handled downstream by human labeling.

Design principles (see usv_signal_processing_reference.md):
- Optimize for RECALL over precision
- Use conservative (low) energy threshold
- Add duration filters to reject obvious artifacts
- Output candidates with full metadata for traceability
"""

# VAULT: 512-point FFT at 300 kHz gives 1.7 ms temporal resolution with 586 Hz frequency bins,
#        visualization STFT uses different parameters than detection STFT by design

from __future__ import annotations

import csv
from dataclasses import replace
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
        3. Compute energy in USV frequency band per frame (peak or mean mode)
        4. Threshold to get candidate frames
        5. Group adjacent frames into segments
        6. Merge segments separated by < merge_gap_ms
        7. Optionally extend/merge segments by continuity
        8. Apply duration filters
        9. Apply bandwidth filter to reject broadband noise
        10. Extract peak frequency for each candidate
        11. Create Candidate objects with full metadata
        """
        wav_path = Path(wav_path)
        original_cfg = self.config
        cfg = original_cfg

        # 1. Load audio
        samples, sample_rate = load_wav_mono(wav_path)
        if cfg.auto_sample_rate:
            cfg = replace(cfg, sample_rate=sample_rate)
            self.config = cfg
        elif sample_rate != cfg.sample_rate:
            raise ValueError(
                f"Expected sample rate {cfg.sample_rate} Hz, got {sample_rate} Hz. "
                f"Set config.sample_rate to match your recordings."
            )

        try:
            if len(samples) < cfg.n_fft:
                return []  # Audio too short

            # 2. Compute spectrogram
            spec_db, freqs_hz = self._compute_band_spectrogram(samples)
            if spec_db.shape[1] == 0:
                return []  # No frames

            # 3. Compute energy per frame in USV band
            # "peak" mode uses max energy - better for narrow-band USVs
            # "mean" mode uses mean energy - original behavior
            if cfg.energy_mode == "peak":
                band_energy_db = np.max(spec_db, axis=0)
            else:
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

            # 7. Optionally extend/merge segments by continuity
            if cfg.segment_continuity_enabled:
                segments = self._extend_segments_by_continuity(segments, spec_db, freqs_hz)

            # 8. Apply duration filters
            segments = self._filter_by_duration(segments)

            if not segments:
                return []

            # 9-10. Create Candidate objects with peak frequency and energy
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
        finally:
            if self.config is not original_cfg:
                self.config = original_cfg

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

    def _compute_frame_peaks(
        self, spec_db: np.ndarray, freqs_hz: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute peak frequency and energy for each frame."""
        peak_indices = np.argmax(spec_db, axis=0)
        frame_indices = np.arange(spec_db.shape[1])
        peak_freqs = freqs_hz[peak_indices]
        peak_energy_db = spec_db[peak_indices, frame_indices]
        return peak_freqs, peak_energy_db

    def _compute_band_energy_db(
        self,
        spec_db: np.ndarray,
        freqs_hz: np.ndarray,
        center_freq_hz: float,
        bandwidth_hz: float,
        frame_indices: np.ndarray,
    ) -> np.ndarray:
        """Compute mean band energy (dB) per frame around a center frequency."""
        if frame_indices.size == 0 or spec_db.size == 0:
            return np.array([], dtype=float)

        if bandwidth_hz <= 0:
            nearest_idx = int(np.argmin(np.abs(freqs_hz - center_freq_hz)))
            return spec_db[nearest_idx, frame_indices]

        band_mask = np.abs(freqs_hz - center_freq_hz) <= bandwidth_hz
        if not np.any(band_mask):
            nearest_idx = int(np.argmin(np.abs(freqs_hz - center_freq_hz)))
            return spec_db[nearest_idx, frame_indices]

        band_spec = spec_db[band_mask][:, frame_indices]
        return np.mean(band_spec, axis=0)

    def _build_continuity_kernel(self) -> np.ndarray:
        """Build the continuity kernel for smoothing during detection."""
        cfg = self.config
        size = cfg.segment_continuity_kernel_size
        radius = size // 2
        kernel = np.zeros((size, size), dtype=float)

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    weight = cfg.segment_continuity_weight_center
                elif dy == 0:
                    weight = cfg.segment_continuity_weight_time / abs(dx)
                elif dx == 0:
                    weight = cfg.segment_continuity_weight_freq / abs(dy)
                elif abs(dx) == abs(dy):
                    weight = cfg.segment_continuity_weight_diag / max(abs(dx), 1)
                else:
                    weight = 0.0

                kernel[dy + radius, dx + radius] = weight

        kernel_sum = kernel.sum()
        if kernel_sum > 0:
            kernel /= kernel_sum

        return kernel

    def _apply_continuity_smoothing(self, spec_db: np.ndarray) -> np.ndarray:
        """Apply continuity smoothing to stabilize peak tracking across gaps."""
        kernel = self._build_continuity_kernel()
        return signal.convolve2d(spec_db, kernel, mode="same", boundary="symm")

    def _segment_reference(
        self,
        peak_freqs: np.ndarray,
        peak_energy_db: np.ndarray,
        start_frame: int,
        end_frame: int,
    ) -> Optional[tuple[float, float]]:
        """Return median peak frequency/energy for a segment."""
        if end_frame <= start_frame:
            return None
        segment_freqs = peak_freqs[start_frame:end_frame]
        segment_energy = peak_energy_db[start_frame:end_frame]
        if segment_freqs.size == 0:
            return None
        return float(np.median(segment_freqs)), float(np.median(segment_energy))

    def _segment_band_reference(
        self,
        spec_db: np.ndarray,
        freqs_hz: np.ndarray,
        ref_freq_hz: float,
        start_frame: int,
        end_frame: int,
    ) -> Optional[float]:
        """Return median band energy (dB) for a segment around a reference frequency."""
        if end_frame <= start_frame:
            return None
        frame_indices = np.arange(start_frame, end_frame)
        cfg = self.config
        band_energy = self._compute_band_energy_db(
            spec_db,
            freqs_hz,
            ref_freq_hz,
            cfg.segment_continuity_bandwidth_hz,
            frame_indices,
        )
        if band_energy.size == 0:
            return None
        return float(np.median(band_energy))

    def _frame_matches_continuity(
        self,
        frame_idx: int,
        peak_freqs: np.ndarray,
        peak_energy_db: np.ndarray,
        ref_freq_hz: float,
        ref_energy_db: float,
    ) -> bool:
        """Check if a frame matches continuity thresholds."""
        cfg = self.config
        freq_ok = abs(peak_freqs[frame_idx] - ref_freq_hz) <= cfg.segment_continuity_freq_tolerance_hz
        energy_ok = peak_energy_db[frame_idx] >= (ref_energy_db - cfg.segment_continuity_energy_tolerance_db)
        return freq_ok and energy_ok

    def _gap_matches_reference_band(
        self,
        gap_start: int,
        gap_end: int,
        spec_db: np.ndarray,
        freqs_hz: np.ndarray,
        ref_freq_hz: float,
        ref_band_energy_db: float,
    ) -> bool:
        """Check if enough gap frames match the reference band energy."""
        if gap_end <= gap_start:
            return True
        cfg = self.config
        gap_indices = np.arange(gap_start, gap_end)
        band_energy = self._compute_band_energy_db(
            spec_db,
            freqs_hz,
            ref_freq_hz,
            cfg.segment_continuity_bandwidth_hz,
            gap_indices,
        )
        if band_energy.size == 0:
            return False
        energy_ok = band_energy >= (ref_band_energy_db - cfg.segment_continuity_band_energy_tolerance_db)
        match_fraction = float(np.mean(energy_ok)) if gap_indices.size else 1.0
        return match_fraction >= cfg.segment_continuity_band_match_fraction

    def _gap_matches_reference(
        self,
        gap_start: int,
        gap_end: int,
        peak_freqs: np.ndarray,
        peak_energy_db: np.ndarray,
        ref_freq_hz: float,
        ref_energy_db: float,
    ) -> bool:
        """Check if enough frames in a gap match a reference."""
        if gap_end <= gap_start:
            return True
        cfg = self.config
        gap_indices = np.arange(gap_start, gap_end)
        freq_ok = np.abs(peak_freqs[gap_indices] - ref_freq_hz) <= cfg.segment_continuity_freq_tolerance_hz
        energy_ok = peak_energy_db[gap_indices] >= (ref_energy_db - cfg.segment_continuity_energy_tolerance_db)
        match_fraction = float(np.mean(freq_ok & energy_ok)) if gap_indices.size else 1.0
        return match_fraction >= cfg.segment_continuity_gap_match_fraction

    def _segments_continuous(
        self,
        prev_ref: tuple[float, float],
        next_ref: tuple[float, float],
        peak_freqs: np.ndarray,
        peak_energy_db: np.ndarray,
        spec_db: np.ndarray,
        freqs_hz: np.ndarray,
        gap_start: int,
        gap_end: int,
        prev_start: int,
        prev_end: int,
        next_start: int,
        next_end: int,
    ) -> bool:
        """Check if two segments should be bridged based on continuity."""
        cfg = self.config
        prev_freq, prev_energy = prev_ref
        next_freq, next_energy = next_ref

        band_continuity = False
        prev_band_ref = self._segment_band_reference(
            spec_db, freqs_hz, prev_freq, prev_start, prev_end
        )
        if prev_band_ref is not None:
            band_continuity = self._gap_matches_reference_band(
                gap_start, gap_end, spec_db, freqs_hz, prev_freq, prev_band_ref
            )
        next_band_ref = self._segment_band_reference(
            spec_db, freqs_hz, next_freq, next_start, next_end
        )
        if next_band_ref is not None:
            band_continuity = band_continuity or self._gap_matches_reference_band(
                gap_start, gap_end, spec_db, freqs_hz, next_freq, next_band_ref
            )

        peak_continuity = False
        freq_close = abs(prev_freq - next_freq) <= cfg.segment_continuity_freq_tolerance_hz
        energy_close = abs(prev_energy - next_energy) <= cfg.segment_continuity_energy_tolerance_db
        if freq_close and energy_close:
            peak_continuity = self._gap_matches_reference(
                gap_start, gap_end, peak_freqs, peak_energy_db, prev_freq, prev_energy
            ) or self._gap_matches_reference(
                gap_start, gap_end, peak_freqs, peak_energy_db, next_freq, next_energy
            )

        return band_continuity or peak_continuity

    def _extend_segments_by_continuity(
        self,
        segments: list[tuple[int, int]],
        spec_db: np.ndarray,
        freqs_hz: np.ndarray,
    ) -> list[tuple[int, int]]:
        """Extend/merge segments using peak and band-energy continuity."""
        if not segments:
            return []

        cfg = self.config
        hop_ms = cfg.hop_ms()
        if hop_ms <= 0:
            return segments

        max_gap_frames = int(round(cfg.segment_continuity_max_gap_ms / hop_ms))
        if max_gap_frames <= 0:
            return segments

        n_frames = spec_db.shape[1]
        if cfg.segment_continuity_enabled:
            spec_for_peaks = self._apply_continuity_smoothing(spec_db)
        else:
            spec_for_peaks = spec_db
        peak_freqs, peak_energy_db = self._compute_frame_peaks(spec_for_peaks, freqs_hz)

        # Extend each segment into nearby frames when continuity holds
        extended = []
        for start_frame, end_frame in segments:
            ref = self._segment_reference(peak_freqs, peak_energy_db, start_frame, end_frame)
            if ref is None:
                extended.append((start_frame, end_frame))
                continue
            ref_freq, ref_energy = ref

            new_start = start_frame
            for frame_idx in range(start_frame - 1, max(-1, start_frame - max_gap_frames - 1), -1):
                if self._frame_matches_continuity(
                    frame_idx, peak_freqs, peak_energy_db, ref_freq, ref_energy
                ):
                    new_start = frame_idx
                else:
                    break

            new_end = end_frame
            for frame_idx in range(end_frame, min(n_frames, end_frame + max_gap_frames)):
                if self._frame_matches_continuity(
                    frame_idx, peak_freqs, peak_energy_db, ref_freq, ref_energy
                ):
                    new_end = frame_idx + 1
                else:
                    break

            extended.append((new_start, new_end))

        # Merge overlapping or near segments when continuity holds
        merged = []
        for start_frame, end_frame in extended:
            if not merged:
                merged.append((start_frame, end_frame))
                continue

            prev_start, prev_end = merged[-1]
            if start_frame <= prev_end:
                merged[-1] = (prev_start, max(prev_end, end_frame))
                continue

            gap_frames = start_frame - prev_end
            if gap_frames <= max_gap_frames:
                prev_ref = self._segment_reference(peak_freqs, peak_energy_db, prev_start, prev_end)
                next_ref = self._segment_reference(peak_freqs, peak_energy_db, start_frame, end_frame)
                if prev_ref and next_ref and self._segments_continuous(
                    prev_ref,
                    next_ref,
                    peak_freqs,
                    peak_energy_db,
                    spec_db,
                    freqs_hz,
                    prev_end,
                    start_frame,
                    prev_start,
                    prev_end,
                    start_frame,
                    end_frame,
                ):
                    merged[-1] = (prev_start, end_frame)
                else:
                    merged.append((start_frame, end_frame))
            else:
                merged.append((start_frame, end_frame))

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

        # Bandwidth filter - reject broadband noise
        # USVs have energy concentrated in a narrow frequency band
        # Check bandwidth only at the peak frame (not across entire segment)
        if cfg.max_bandwidth_hz > 0:
            peak_frame = max_idx[1]  # Frame index of the peak
            frame_spectrum = segment_spec[:, peak_frame]
            # Find frequency range where energy is within 10 dB of peak
            threshold_for_bandwidth = peak_energy_db - 10.0
            active_freqs = frame_spectrum >= threshold_for_bandwidth
            if np.any(active_freqs):
                active_indices = np.where(active_freqs)[0]
                min_freq = freqs_hz[active_indices[0]]
                max_freq = freqs_hz[active_indices[-1]]
                bandwidth = max_freq - min_freq
                if bandwidth > cfg.max_bandwidth_hz:
                    return None  # Reject broadband noise

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
            auto_sample_rate=config.auto_sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            freq_min_hz=config.freq_min_hz,
            freq_max_hz=config.freq_max_hz,
            energy_threshold_db=threshold,
            energy_mode=config.energy_mode,
            max_bandwidth_hz=config.max_bandwidth_hz,
            min_duration_ms=config.min_duration_ms,
            max_duration_ms=config.max_duration_ms,
            merge_gap_ms=config.merge_gap_ms,
            segment_continuity_enabled=config.segment_continuity_enabled,
            segment_continuity_max_gap_ms=config.segment_continuity_max_gap_ms,
            segment_continuity_freq_tolerance_hz=config.segment_continuity_freq_tolerance_hz,
            segment_continuity_energy_tolerance_db=config.segment_continuity_energy_tolerance_db,
            segment_continuity_gap_match_fraction=config.segment_continuity_gap_match_fraction,
            segment_continuity_bandwidth_hz=config.segment_continuity_bandwidth_hz,
            segment_continuity_band_energy_tolerance_db=config.segment_continuity_band_energy_tolerance_db,
            segment_continuity_band_match_fraction=config.segment_continuity_band_match_fraction,
            segment_continuity_kernel_size=config.segment_continuity_kernel_size,
            segment_continuity_weight_center=config.segment_continuity_weight_center,
            segment_continuity_weight_time=config.segment_continuity_weight_time,
            segment_continuity_weight_freq=config.segment_continuity_weight_freq,
            segment_continuity_weight_diag=config.segment_continuity_weight_diag,
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
