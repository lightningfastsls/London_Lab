"""Parameter explanations for the USV Parameter Lab."""

from __future__ import annotations


def parameter_explanations() -> dict[str, str]:
    """Return markdown explanations for common spectrogram parameters."""
    return {
        "window_length": "STFT window length in samples; larger windows improve frequency resolution but smear time detail.",
        "hop_ms": "Time step between successive windows in milliseconds; smaller hops give finer time detail at higher cost.",
        "zero_padding_factor": "FFT zero-padding factor; larger values interpolate the spectrum for smoother bins.",
        "window": "Windowing function applied before the FFT to reduce spectral leakage.",
        "band_limits": "Frequency band (Hz) cropped for display and metrics; keep focused on the USV range.",
        "gain_range": "Display gain and range (dB) used for the colormap; does not change the raw dB values.",
        "heuristic_threshold": "Threshold above the noise floor (dB) used by the simple candidate detector.",
    }
