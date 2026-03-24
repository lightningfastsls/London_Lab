"""Sidebar UI for file selection and parameter configuration."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from usv_spectrogram.config import SpectrogramConfig
from usv_spectrogram.param_lab.state import read_wav_info
from usv_spectrogram.param_lab.ui.components import render_parameter_controls


def render_sidebar(
    default_cfg: SpectrogramConfig,
    default_wav_dir: Path,
) -> dict[str, object]:
    """Render the complete sidebar and return all user selections.

    Returns a dict with keys:
    - wav_path: Path
    - sample_rate_hz: int
    - total_frames: int
    - duration_s: float
    - enforce_sample_rate: bool
    - expected_sample_rate: int
    - max_freq_hz: float
    - f_min_default: float
    - f_max_default: float
    - segment_start_s: float
    - segment_duration_s: float
    - baseline_display_gain_db: float
    - baseline_display_range_db: float
    - variant_display_gain_db: float
    - variant_display_range_db: float
    - lock_baseline: bool
    - baseline_params: dict
    - variant_params: dict
    - show_overlay: bool
    - overlay_target: str
    - overlay_threshold: float
    - overlay_min_area: int
    - overlay_min_frames: int
    - overlay_min_bins: int
    - show_diff: bool
    """
    default_dir = default_wav_dir if default_wav_dir.exists() else Path.cwd()
    wav_choices = sorted(default_dir.glob("*.wav")) if default_dir.exists() else []

    st.header("Input")
    if wav_choices:
        choice = st.selectbox(
            "WAV file",
            ["Custom path"] + [p.name for p in wav_choices],
        )
        if choice == "Custom path":
            wav_path_text = st.text_input(
                "WAV path", value=str(wav_choices[0])
            ).strip()
        else:
            wav_path_text = str(default_dir / choice)
    else:
        wav_path_text = st.text_input("WAV path", value="").strip()

    if not wav_path_text:
        st.error("WAV path is required.")
        st.stop()

    wav_path = Path(wav_path_text)
    if not wav_path.is_file():
        st.error("WAV path not found.")
        st.stop()
    if wav_path.suffix.lower() != ".wav":
        st.error("WAV path must point to a .wav file.")
        st.stop()

    sample_rate_hz, total_frames, duration_s = read_wav_info(str(wav_path))
    st.caption(
        f"Sample rate: {sample_rate_hz} Hz | Duration: {duration_s:.2f} s | Frames: {total_frames}"
    )

    enforce_sample_rate = not st.checkbox("Auto sample rate", value=False)
    expected_sample_rate = default_cfg.expected_sample_rate_hz
    if not enforce_sample_rate:
        expected_sample_rate = sample_rate_hz

    max_freq_hz = float(sample_rate_hz / 2.0) if sample_rate_hz > 0 else float(
        default_cfg.f_max_hz
    )
    f_min_default = min(float(default_cfg.f_min_hz), max_freq_hz)
    f_max_default = min(float(default_cfg.f_max_hz), max_freq_hz)
    if f_max_default <= f_min_default:
        f_min_default = 0.0
        f_max_default = max_freq_hz

    st.subheader("Segment")
    segment_duration_s = st.number_input(
        "Segment duration (s)",
        min_value=0.01,
        max_value=float(duration_s) if duration_s > 0 else 0.01,
        value=min(1.0, float(duration_s)) if duration_s > 0 else 0.01,
        step=0.05,
    )
    max_start = max(0.0, float(duration_s) - float(segment_duration_s))
    segment_start_s = st.number_input(
        "Segment start (s)",
        min_value=0.0,
        max_value=max_start,
        value=0.0,
        step=0.05,
    )

    st.subheader("Display")
    st.caption("Set gain/range independently for baseline and variant views.")
    baseline_display_gain_db = st.number_input(
        "Baseline gain (dB)",
        value=float(default_cfg.gain_db),
        step=1.0,
    )
    baseline_display_range_db = st.number_input(
        "Baseline range (dB)",
        value=float(default_cfg.range_db),
        step=1.0,
    )
    variant_display_gain_db = st.number_input(
        "Variant gain (dB)",
        value=float(default_cfg.gain_db),
        step=1.0,
    )
    variant_display_range_db = st.number_input(
        "Variant range (dB)",
        value=float(default_cfg.range_db),
        step=1.0,
    )

    lock_baseline = st.checkbox("Lock baseline settings", value=False)

    st.subheader("Baseline")
    with st.expander("Baseline parameters", expanded=True):
        baseline_params = render_parameter_controls(
            "baseline",
            default_cfg,
            max_freq_hz,
            f_min_default,
            f_max_default,
            disabled=lock_baseline,
        )

    st.subheader("Variant")
    with st.expander("Variant parameters", expanded=True):
        variant_params = render_parameter_controls(
            "variant",
            default_cfg,
            max_freq_hz,
            f_min_default,
            f_max_default,
            disabled=False,
        )

    st.subheader("Heuristic overlay")
    show_overlay = st.checkbox("Show overlay", value=False)
    overlay_target = st.selectbox("Overlay target", ["Baseline", "Variant"])
    overlay_threshold = st.number_input(
        "Threshold above noise floor (dB)",
        min_value=1.0,
        max_value=30.0,
        value=6.0,
        step=1.0,
    )
    overlay_min_area = st.number_input(
        "Min area (bins)",
        min_value=1,
        max_value=500,
        value=24,
        step=1,
    )
    overlay_min_frames = st.number_input(
        "Min time bins",
        min_value=1,
        max_value=100,
        value=3,
        step=1,
    )
    overlay_min_bins = st.number_input(
        "Min freq bins",
        min_value=1,
        max_value=100,
        value=3,
        step=1,
    )

    st.subheader("Views")
    show_diff = st.checkbox("Show difference view", value=False)

    return {
        "wav_path": wav_path,
        "sample_rate_hz": sample_rate_hz,
        "total_frames": total_frames,
        "duration_s": duration_s,
        "enforce_sample_rate": enforce_sample_rate,
        "expected_sample_rate": expected_sample_rate,
        "max_freq_hz": max_freq_hz,
        "f_min_default": f_min_default,
        "f_max_default": f_max_default,
        "segment_start_s": float(segment_start_s),
        "segment_duration_s": float(segment_duration_s),
        "baseline_display_gain_db": float(baseline_display_gain_db),
        "baseline_display_range_db": float(baseline_display_range_db),
        "variant_display_gain_db": float(variant_display_gain_db),
        "variant_display_range_db": float(variant_display_range_db),
        "lock_baseline": lock_baseline,
        "baseline_params": baseline_params,
        "variant_params": variant_params,
        "show_overlay": show_overlay,
        "overlay_target": str(overlay_target),
        "overlay_threshold": float(overlay_threshold),
        "overlay_min_area": int(overlay_min_area),
        "overlay_min_frames": int(overlay_min_frames),
        "overlay_min_bins": int(overlay_min_bins),
        "show_diff": show_diff,
    }
