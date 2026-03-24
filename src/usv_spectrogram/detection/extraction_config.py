"""Configuration for spectrogram extraction from detected candidates.

This module defines parameters for converting candidate segments into
standardized spectrogram PNG images for human labeling and CNN training.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractionConfig:
    """Configuration for spectrogram image extraction.

    STFT parameters match the detection pipeline for consistency.
    Image parameters are tuned for human review and CNN training.
    """

    # STFT parameters - match detection for consistency
    sample_rate: int = 300_000  # Actual recording sample rate
    n_fft: int = 512  # ~586 Hz freq resolution at 300 kHz
    hop_length: int = 128  # 75% overlap
    window: str = "hann"  # Window function

    # Frequency range - wider than detection for visual context
    freq_min_hz: int = 20_000  # Below USV band for context
    freq_max_hz: int = 120_000  # Above USV band for context

    # Image dimensions
    image_height_px: int = 256  # Fixed height for CNN input
    pixels_per_ms: float = 2.0  # Temporal resolution
    min_width_px: int = 128  # Minimum width (pad short candidates)
    max_width_px: int = 512  # Maximum width (truncate long candidates)

    # Color scale (dB)
    db_floor: float = -80.0  # Black level (used when dynamic_range_method="fixed")
    db_ceiling: float = 0.0  # White level (used when dynamic_range_method="fixed")
    colormap: str = "magma"  # Dark background, bright signals (like Audacity)

    # Dynamic range method for adaptive color scaling
    # "fixed" - Use db_floor/db_ceiling
    # "percentile" - Use percentiles (99th for vmax, 5th for vmin)
    # "std" - Use mean +/- std deviations
    # "mad" - Use median +/- MAD (most robust to outliers)
    dynamic_range_method: str = "mad"
    mad_vmin_scale: float = 2.0  # Scale factor for vmin (median - scale*MAD)
    mad_vmax_scale: float = 4.0  # Scale factor for vmax (median + scale*MAD) - extra headroom for bright USVs

    # Continuity filter (review-only)
    # Applies a kernel-weighted smoothing with diagonal emphasis to reduce fragmented tracks.
    continuity_filter_enabled: bool = False
    continuity_kernel_size: int = 3  # Odd kernel size (e.g., 3 or 5)
    continuity_weight_center: float = 1.0
    continuity_weight_axis: float = 0.5
    continuity_weight_diag: float = 0.8

    # Render modes
    # "review" - Matplotlib with axes/labels for human labeling
    # "training" - Raw images for CNN (no axes, exact dimensions)
    default_render_mode: str = "review"

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
        if self.window not in ("hann", "hamming", "blackman", "boxcar"):
            raise ValueError("window must be hann, hamming, blackman, or boxcar")
        if self.image_height_px <= 0:
            raise ValueError("image_height_px must be positive")
        if self.pixels_per_ms <= 0:
            raise ValueError("pixels_per_ms must be positive")
        if self.min_width_px <= 0:
            raise ValueError("min_width_px must be positive")
        if self.max_width_px < self.min_width_px:
            raise ValueError("max_width_px must be >= min_width_px")
        if self.db_floor >= self.db_ceiling:
            raise ValueError("db_floor must be < db_ceiling")
        if self.default_render_mode not in ("review", "training"):
            raise ValueError("default_render_mode must be 'review' or 'training'")
        if self.dynamic_range_method not in ("fixed", "percentile", "std", "mad"):
            raise ValueError("dynamic_range_method must be 'fixed', 'percentile', 'std', or 'mad'")
        if self.mad_vmin_scale <= 0:
            raise ValueError("mad_vmin_scale must be positive")
        if self.mad_vmax_scale <= 0:
            raise ValueError("mad_vmax_scale must be positive")
        if self.continuity_kernel_size < 3 or self.continuity_kernel_size % 2 == 0:
            raise ValueError("continuity_kernel_size must be an odd integer >= 3")
        if self.continuity_weight_center < 0:
            raise ValueError("continuity_weight_center must be >= 0")
        if self.continuity_weight_axis < 0:
            raise ValueError("continuity_weight_axis must be >= 0")
        if self.continuity_weight_diag < 0:
            raise ValueError("continuity_weight_diag must be >= 0")
        if (
            self.continuity_filter_enabled
            and (self.continuity_weight_center + self.continuity_weight_axis + self.continuity_weight_diag)
            <= 0
        ):
            raise ValueError("continuity filter weights must sum to > 0 when enabled")

    def width_px_for_duration_ms(self, duration_ms: float) -> int:
        """Calculate image width in pixels for a given duration.

        Returns clamped value between min_width_px and max_width_px.
        """
        raw_width = int(duration_ms * self.pixels_per_ms)
        return max(self.min_width_px, min(raw_width, self.max_width_px))

    def freq_resolution_hz(self) -> float:
        """Return frequency resolution in Hz."""
        return self.sample_rate / self.n_fft

    def hop_ms(self) -> float:
        """Return hop length in milliseconds."""
        return (self.hop_length / self.sample_rate) * 1000.0
