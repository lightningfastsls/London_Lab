"""Configuration for energy-based USV detection.

Design principles (see usv_signal_processing_reference.md):
- Optimize for RECALL over precision at candidate generation stage
- Use conservative (low) energy threshold to avoid missing USV types
- Add duration filters to reject obvious artifacts
- Full metadata for traceability through the pipeline
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DetectionConfig:
    """Configuration for energy-based USV candidate detection.

    Parameters are tuned for 250 kHz recordings of mouse USVs.
    See usv_signal_processing_reference.md Section 1.1 for parameter trade-offs.
    """

    # STFT parameters - matched to existing SpectrogramConfig defaults
    sample_rate: int = 250_000  # Must be >= 2 * max_freq (Nyquist)
    n_fft: int = 512  # ~488 Hz freq resolution, ~2 ms time resolution at 250 kHz
    hop_length: int = 128  # 75% overlap for smooth temporal coverage

    # Frequency band for USV detection
    freq_min_hz: int = 25_000  # High-pass: remove sub-ultrasonic noise
    freq_max_hz: int = 110_000  # Upper bound of mouse USV range

    # Energy threshold - deliberately LOW for high recall
    # See Section 3.2: threshold bias creates systematic blind spots
    # Relative to max energy in the frequency band
    energy_threshold_db: float = -50.0  # Tune based on your recordings

    # Energy detection mode
    # "peak" = use max energy in band per frame (better for narrow-band USVs)
    # "mean" = use mean energy in band per frame (original behavior)
    energy_mode: str = "peak"

    # Bandwidth filter - reject broadband noise
    # USVs typically have energy concentrated in 5-15 kHz bandwidth
    # Set to 0 to disable bandwidth filtering
    max_bandwidth_hz: int = 20_000  # Reject candidates with bandwidth > this

    # Duration filters - reject obvious non-USVs
    # See Section 2.4: minimum duration filtering
    min_duration_ms: float = 10.0  # USVs are >= 10 ms
    max_duration_ms: float = 500.0  # USVs are <= 300 ms, allow margin for safety

    # Merging nearby detections into single candidates
    merge_gap_ms: float = 5.0  # If two detections are < 5 ms apart, merge them

    # Context window for spectrogram extraction
    # Include context around the detection for visual review
    context_before_ms: float = 50.0  # Include 50 ms before detection
    context_after_ms: float = 50.0  # Include 50 ms after detection

    # Known interference frequencies to flag (Hz harmonics of AC power)
    interference_freqs_hz: tuple[int, ...] = (50000, 60000, 100000, 120000)
    interference_tolerance_hz: int = 500  # Flag if peak within +/- this of interference

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.n_fft <= 0:
            raise ValueError("n_fft must be positive")
        if self.hop_length <= 0 or self.hop_length >= self.n_fft:
            raise ValueError("hop_length must be positive and < n_fft")
        if self.freq_min_hz >= self.freq_max_hz:
            raise ValueError("freq_min_hz must be < freq_max_hz")
        if self.freq_max_hz > self.sample_rate // 2:
            raise ValueError("freq_max_hz exceeds Nyquist frequency")
        if self.energy_mode not in ("peak", "mean"):
            raise ValueError("energy_mode must be 'peak' or 'mean'")
        if self.max_bandwidth_hz < 0:
            raise ValueError("max_bandwidth_hz must be >= 0")
        if self.min_duration_ms <= 0:
            raise ValueError("min_duration_ms must be positive")
        if self.max_duration_ms <= self.min_duration_ms:
            raise ValueError("max_duration_ms must be > min_duration_ms")
        if self.context_before_ms < 0 or self.context_after_ms < 0:
            raise ValueError("context_before_ms and context_after_ms must be >= 0")

    def hop_ms(self) -> float:
        """Return hop length in milliseconds."""
        return (self.hop_length / self.sample_rate) * 1000.0

    def frame_duration_ms(self) -> float:
        """Return STFT frame duration in milliseconds."""
        return (self.n_fft / self.sample_rate) * 1000.0

    def freq_resolution_hz(self) -> float:
        """Return frequency resolution in Hz."""
        return self.sample_rate / self.n_fft
