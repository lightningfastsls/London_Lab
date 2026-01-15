"""Configuration defaults for USV spectrogram generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpectrogramConfig:
    """Configures STFT and rendering defaults for USV spectrograms.

    Parameters are tuned for 250 kHz WAVs and USV band visualization.
    """

    # Expected sample rate for USV recordings; set enforce_sample_rate to False to allow others.
    expected_sample_rate_hz: int = 250_000
    enforce_sample_rate: bool = True

    # STFT window length in samples; 2048 at 250 kHz is ~8.2 ms.
    window_length: int = 2048
    # Zero-padding factor for the FFT length; 2 => n_fft=4096 for smoother spectra.
    zero_padding_factor: int = 2
    # Hop size in milliseconds; 0.5 ms at 250 kHz is 125 samples.
    hop_ms: float = 0.5
    # Window function name passed to scipy.signal.get_window.
    window: str = "hann"

    # Frequency band of interest (USV band) in Hz.
    f_min_hz: float = 30_000.0
    f_max_hz: float = 125_000.0

    # Display gain and range for dB-scaled spectrograms.
    gain_db: float = 20.0
    range_db: float = 40.0

    # Numerical floor to avoid log(0) when computing dB.
    eps: float = 1e-12

    # Streaming block size in samples; 250k ~= 1 second at 250 kHz.
    stream_block_size_samples: int = 250_000

    # Zarr chunk size along time axis in frames for incremental writes.
    zarr_time_chunk_frames: int = 512

    # Tiled page rendering settings for long spectrograms.
    tile_duration_s: float = 5.0
    tiles_per_row: int = 4
    tiles_per_col: int = 3
    tile_width_in: float = 3.0
    tile_height_in: float = 2.2
    page_dpi: int = 150

    def n_fft(self) -> int:
        """Return FFT length derived from window length and padding factor."""
        return int(self.window_length * self.zero_padding_factor)

    def hop_length(self, sample_rate_hz: int) -> int:
        """Return hop length in samples derived from hop_ms and sample rate."""
        return max(1, int(round(sample_rate_hz * (self.hop_ms / 1000.0))))

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.window_length <= 0:
            raise ValueError("window_length must be positive")
        if self.zero_padding_factor < 1:
            raise ValueError("zero_padding_factor must be >= 1")
        if self.f_min_hz >= self.f_max_hz:
            raise ValueError("f_min_hz must be < f_max_hz")
        if self.hop_ms <= 0:
            raise ValueError("hop_ms must be positive")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.stream_block_size_samples <= 0:
            raise ValueError("stream_block_size_samples must be positive")
        if self.zarr_time_chunk_frames <= 0:
            raise ValueError("zarr_time_chunk_frames must be positive")
