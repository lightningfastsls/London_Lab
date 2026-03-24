"""Reusable UI components for parameter controls."""

from __future__ import annotations

import streamlit as st

from usv_spectrogram.config import SpectrogramConfig

WINDOW_OPTIONS = ("hann", "hamming", "blackman", "boxcar")


def render_parameter_controls(
    prefix: str,
    default_cfg: SpectrogramConfig,
    max_freq_hz: float,
    f_min_default: float,
    f_max_default: float,
    disabled: bool = False,
) -> dict[str, object]:
    """Render a reusable parameter control block.

    Returns a dict with keys: window_length, zero_padding_factor, hop_ms,
    window, f_min_hz, f_max_hz.
    """
    window_length = st.number_input(
        "Window length (samples)",
        min_value=256,
        max_value=8192,
        value=int(default_cfg.window_length),
        step=128,
        disabled=disabled,
        key=f"{prefix}.window_length",
    )
    zero_padding_factor = st.number_input(
        "Zero padding factor",
        min_value=1,
        max_value=8,
        value=int(default_cfg.zero_padding_factor),
        step=1,
        disabled=disabled,
        key=f"{prefix}.zero_padding_factor",
    )
    hop_ms = st.number_input(
        "Hop (ms)",
        min_value=0.1,
        max_value=5.0,
        value=float(default_cfg.hop_ms),
        step=0.1,
        disabled=disabled,
        key=f"{prefix}.hop_ms",
    )
    window = st.selectbox(
        "Window",
        options=list(WINDOW_OPTIONS),
        index=WINDOW_OPTIONS.index(default_cfg.window),
        disabled=disabled,
        key=f"{prefix}.window",
    )
    f_min_hz = st.number_input(
        "F min (Hz)",
        min_value=0.0,
        max_value=max_freq_hz,
        value=f_min_default,
        step=1000.0,
        disabled=disabled,
        key=f"{prefix}.f_min_hz",
    )
    f_max_hz = st.number_input(
        "F max (Hz)",
        min_value=0.0,
        max_value=max_freq_hz,
        value=f_max_default,
        step=1000.0,
        disabled=disabled,
        key=f"{prefix}.f_max_hz",
    )

    return {
        "window_length": int(window_length),
        "zero_padding_factor": int(zero_padding_factor),
        "hop_ms": float(hop_ms),
        "window": str(window),
        "f_min_hz": float(f_min_hz),
        "f_max_hz": float(f_max_hz),
    }
