"""Session state management and caching for Parameter Lab."""

from __future__ import annotations

from dataclasses import asdict, replace
import time

import soundfile as sf
import streamlit as st

from usv_spectrogram.config import SpectrogramConfig
from usv_spectrogram.io_wav import load_wav_segment_mono
from usv_spectrogram.spectrogram import compute_spectrogram_db


@st.cache_data(show_spinner=False)
def read_wav_info(path_str: str) -> tuple[int, int, float]:
    """Return sample rate, frames, and duration for a WAV file."""
    info = sf.info(path_str)
    duration_s = info.frames / info.samplerate if info.samplerate else 0.0
    return int(info.samplerate), int(info.frames), float(duration_s)


@st.cache_data(show_spinner=False)
def compute_segment_spectrogram(
    path_str: str,
    start_s: float,
    duration_s: float,
    cfg_key: tuple[object, ...],
    cfg_dict: dict[str, object],
) -> dict[str, object]:
    """Load a segment and compute its dB spectrogram with timing."""
    _ = cfg_key
    cfg = SpectrogramConfig(**cfg_dict)
    t0 = time.perf_counter()
    samples, sample_rate_hz = load_wav_segment_mono(path_str, start_s, duration_s)
    spec_db, freqs_hz, times_s = compute_spectrogram_db(samples, sample_rate_hz, cfg)
    elapsed_s = time.perf_counter() - t0
    return {
        "spec_db": spec_db,
        "freqs_hz": freqs_hz,
        "times_s": times_s,
        "sample_rate_hz": sample_rate_hz,
        "elapsed_s": elapsed_s,
    }


def config_from_controls(
    base_cfg: SpectrogramConfig,
    window_length: int,
    zero_padding_factor: int,
    hop_ms: float,
    window: str,
    f_min_hz: float,
    f_max_hz: float,
    gain_db: float,
    range_db: float,
) -> SpectrogramConfig:
    """Build a config from individual parameter values."""
    return replace(
        base_cfg,
        window_length=window_length,
        zero_padding_factor=zero_padding_factor,
        hop_ms=hop_ms,
        window=window,
        f_min_hz=f_min_hz,
        f_max_hz=f_max_hz,
        gain_db=gain_db,
        range_db=range_db,
    )


def stft_cache_key(cfg: SpectrogramConfig) -> tuple[object, ...]:
    """Return a cache key for spectrogram computation inputs."""
    return (
        cfg.window_length,
        cfg.zero_padding_factor,
        cfg.hop_ms,
        cfg.window,
        cfg.f_min_hz,
        cfg.f_max_hz,
        cfg.eps,
        cfg.enforce_sample_rate,
        cfg.expected_sample_rate_hz,
    )


def parse_sweep_values(param_name: str, value_text: str) -> list[object]:
    """Parse comma-separated sweep values for the given parameter."""
    items = [item.strip() for item in value_text.split(",") if item.strip()]
    if not items:
        return []

    if param_name in {"window_length", "zero_padding_factor"}:
        return [int(item) for item in items]
    if param_name in {"hop_ms", "f_min_hz", "f_max_hz"}:
        return [float(item) for item in items]
    return items
