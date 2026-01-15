"""Streamlit UI for exploring USV spectrogram parameters."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
from matplotlib import patches

from usv_spectrogram.config import SpectrogramConfig
from usv_spectrogram.io_wav import get_default_wav_dir, load_wav_segment_mono
from usv_spectrogram.spectrogram import compute_spectrogram_db
from usv_spectrogram.param_lab.explain import parameter_explanations
from usv_spectrogram.param_lab.heuristic_detect import (
    HeuristicConfig,
    detect_candidates,
    summarize_candidates,
)
from usv_spectrogram.param_lab.metrics import compute_metrics
from usv_spectrogram.param_lab.sweep import run_sweep

DEFAULT_WAV_DIR = get_default_wav_dir()
WINDOW_OPTIONS = ("hann", "hamming", "blackman", "boxcar")


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


def plot_spectrogram(
    spec_db: np.ndarray,
    freqs_hz: np.ndarray,
    times_s: np.ndarray,
    title: str,
    gain_db: float,
    range_db: float,
    start_offset_s: float,
    xlim: tuple[float, float] | None,
    ylim_khz: tuple[float, float] | None,
    overlay_boxes: list[dict[str, float]] | None,
) -> plt.Figure:
    """Render a spectrogram with shared scaling and optional overlays."""
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=150)
    if spec_db.size == 0 or freqs_hz.size == 0 or times_s.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return fig

    display_db = spec_db + gain_db
    vmin = gain_db - range_db
    vmax = gain_db

    mesh = ax.pcolormesh(
        times_s + start_offset_s,
        freqs_hz / 1000.0,
        display_db,
        shading="auto",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )

    if overlay_boxes:
        for box in overlay_boxes:
            rect = patches.Rectangle(
                (start_offset_s + box["t_start_s"], box["f_start_hz"] / 1000.0),
                box["duration_s"],
                box["bandwidth_hz"] / 1000.0,
                linewidth=1.0,
                edgecolor="cyan",
                facecolor="none",
            )
            ax.add_patch(rect)

    if xlim:
        ax.set_xlim(*xlim)
    if ylim_khz:
        ax.set_ylim(*ylim_khz)

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    fig.colorbar(mesh, ax=ax, label="dB")
    fig.tight_layout()
    return fig


def plot_difference(
    diff_db: np.ndarray,
    freqs_hz: np.ndarray,
    times_s: np.ndarray,
    start_offset_s: float,
    xlim: tuple[float, float] | None,
    ylim_khz: tuple[float, float] | None,
    range_db: float,
) -> plt.Figure:
    """Render a difference plot between variant and baseline."""
    fig, ax = plt.subplots(figsize=(6.5, 3.5), dpi=150)
    if diff_db.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return fig

    vmax = float(np.max(np.abs(diff_db))) if diff_db.size else 1.0
    display_limit = max(1e-6, float(range_db) / 2.0)
    vmax = max(vmax, display_limit)

    mesh = ax.pcolormesh(
        times_s + start_offset_s,
        freqs_hz / 1000.0,
        diff_db,
        shading="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )

    if xlim:
        ax.set_xlim(*xlim)
    if ylim_khz:
        ax.set_ylim(*ylim_khz)

    ax.set_title("Variant - Baseline (dB)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    fig.colorbar(mesh, ax=ax, label="dB")
    fig.tight_layout()
    return fig


def _parse_sweep_values(param_name: str, value_text: str) -> list[object]:
    items = [item.strip() for item in value_text.split(",") if item.strip()]
    if not items:
        return []

    if param_name in {"window_length", "zero_padding_factor"}:
        return [int(item) for item in items]
    if param_name in {"hop_ms", "f_min_hz", "f_max_hz"}:
        return [float(item) for item in items]
    return items


def _config_from_controls(
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


def _stft_cache_key(cfg: SpectrogramConfig) -> tuple[object, ...]:
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


def run() -> None:
    """Run the Streamlit app for USV parameter exploration."""
    st.set_page_config(page_title="USV Parameter Lab", layout="wide")
    st.title("USV Parameter Lab")
    st.write(
        "Explore STFT parameter effects on a short WAV segment and compare baseline vs variant settings."
    )

    default_cfg = SpectrogramConfig()
    default_dir = DEFAULT_WAV_DIR if DEFAULT_WAV_DIR.exists() else Path.cwd()
    wav_choices = sorted(default_dir.glob("*.wav")) if default_dir.exists() else []

    with st.sidebar:
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
            base_window_length = st.number_input(
                "Window length (samples)",
                min_value=256,
                max_value=8192,
                value=int(default_cfg.window_length),
                step=128,
                disabled=lock_baseline,
                key="base_window_length",
            )
            base_zero_padding = st.number_input(
                "Zero padding factor",
                min_value=1,
                max_value=8,
                value=int(default_cfg.zero_padding_factor),
                step=1,
                disabled=lock_baseline,
                key="base_zero_padding",
            )
            base_hop_ms = st.number_input(
                "Hop (ms)",
                min_value=0.1,
                max_value=5.0,
                value=float(default_cfg.hop_ms),
                step=0.1,
                disabled=lock_baseline,
                key="base_hop_ms",
            )
            base_window = st.selectbox(
                "Window",
                options=list(WINDOW_OPTIONS),
                index=WINDOW_OPTIONS.index(default_cfg.window),
                disabled=lock_baseline,
                key="base_window",
            )
            base_f_min_hz = st.number_input(
                "F min (Hz)",
                min_value=0.0,
                max_value=max_freq_hz,
                value=f_min_default,
                step=1000.0,
                disabled=lock_baseline,
                key="base_f_min",
            )
            base_f_max_hz = st.number_input(
                "F max (Hz)",
                min_value=0.0,
                max_value=max_freq_hz,
                value=f_max_default,
                step=1000.0,
                disabled=lock_baseline,
                key="base_f_max",
            )

        st.subheader("Variant")
        with st.expander("Variant parameters", expanded=True):
            variant_window_length = st.number_input(
                "Window length (samples)",
                min_value=256,
                max_value=8192,
                value=int(default_cfg.window_length),
                step=128,
                key="variant_window_length",
            )
            variant_zero_padding = st.number_input(
                "Zero padding factor",
                min_value=1,
                max_value=8,
                value=int(default_cfg.zero_padding_factor),
                step=1,
                key="variant_zero_padding",
            )
            variant_hop_ms = st.number_input(
                "Hop (ms)",
                min_value=0.1,
                max_value=5.0,
                value=float(default_cfg.hop_ms),
                step=0.1,
                key="variant_hop_ms",
            )
            variant_window = st.selectbox(
                "Window",
                options=list(WINDOW_OPTIONS),
                index=WINDOW_OPTIONS.index(default_cfg.window),
                key="variant_window",
            )
            variant_f_min_hz = st.number_input(
                "F min (Hz)",
                min_value=0.0,
                max_value=max_freq_hz,
                value=f_min_default,
                step=1000.0,
                key="variant_f_min",
            )
            variant_f_max_hz = st.number_input(
                "F max (Hz)",
                min_value=0.0,
                max_value=max_freq_hz,
                value=f_max_default,
                step=1000.0,
                key="variant_f_max",
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

    if base_f_min_hz >= base_f_max_hz or variant_f_min_hz >= variant_f_max_hz:
        st.error("F min must be smaller than F max.")
        st.stop()

    base_cfg = SpectrogramConfig(
        expected_sample_rate_hz=int(expected_sample_rate),
        enforce_sample_rate=enforce_sample_rate,
    )
    baseline_cfg = _config_from_controls(
        base_cfg,
        int(base_window_length),
        int(base_zero_padding),
        float(base_hop_ms),
        str(base_window),
        float(base_f_min_hz),
        float(base_f_max_hz),
        float(baseline_display_gain_db),
        float(baseline_display_range_db),
    )
    variant_cfg = _config_from_controls(
        base_cfg,
        int(variant_window_length),
        int(variant_zero_padding),
        float(variant_hop_ms),
        str(variant_window),
        float(variant_f_min_hz),
        float(variant_f_max_hz),
        float(variant_display_gain_db),
        float(variant_display_range_db),
    )

    try:
        baseline_result = compute_segment_spectrogram(
            str(wav_path),
            float(segment_start_s),
            float(segment_duration_s),
            _stft_cache_key(baseline_cfg),
            asdict(baseline_cfg),
        )
        variant_result = compute_segment_spectrogram(
            str(wav_path),
            float(segment_start_s),
            float(segment_duration_s),
            _stft_cache_key(variant_cfg),
            asdict(variant_cfg),
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    base_spec = baseline_result["spec_db"]
    base_freqs = baseline_result["freqs_hz"]
    base_times = baseline_result["times_s"]
    variant_spec = variant_result["spec_db"]
    variant_freqs = variant_result["freqs_hz"]
    variant_times = variant_result["times_s"]

    xlim = (
        float(segment_start_s),
        float(segment_start_s) + float(segment_duration_s),
    )
    freq_min = None
    freq_max = None
    if base_freqs.size:
        freq_min = float(base_freqs[0])
        freq_max = float(base_freqs[-1])
    if variant_freqs.size:
        if freq_min is None or float(variant_freqs[0]) < freq_min:
            freq_min = float(variant_freqs[0])
        if freq_max is None or float(variant_freqs[-1]) > freq_max:
            freq_max = float(variant_freqs[-1])
    ylim_khz = None
    if freq_min is not None and freq_max is not None:
        ylim_khz = (freq_min / 1000.0, freq_max / 1000.0)

    overlay_boxes = None
    overlay_summary = None
    if show_overlay:
        overlay_cfg = HeuristicConfig(
            threshold_db=float(overlay_threshold),
            min_area_bins=int(overlay_min_area),
            min_frames=int(overlay_min_frames),
            min_bins=int(overlay_min_bins),
        )
        if overlay_target == "Baseline":
            overlay_boxes = detect_candidates(base_spec, base_freqs, base_times, overlay_cfg)
            overlay_summary = summarize_candidates(overlay_boxes)
        else:
            overlay_boxes = detect_candidates(
                variant_spec, variant_freqs, variant_times, overlay_cfg
            )
            overlay_summary = summarize_candidates(overlay_boxes)

    st.subheader("Baseline")
    fig = plot_spectrogram(
        base_spec,
        base_freqs,
        base_times,
        "Baseline",
        float(baseline_display_gain_db),
        float(baseline_display_range_db),
        float(segment_start_s),
        xlim,
        ylim_khz,
        overlay_boxes if show_overlay and overlay_target == "Baseline" else None,
    )
    st.pyplot(fig)
    base_metrics = compute_metrics(base_spec, base_freqs, base_times)
    st.caption(f"Baseline compute time: {baseline_result['elapsed_s']:.3f} s")
    with st.expander("Baseline metrics", expanded=False):
        st.write(base_metrics)
        if show_overlay and overlay_target == "Baseline" and overlay_summary:
            st.write({"candidates": overlay_summary})

    st.subheader("Variant")
    fig = plot_spectrogram(
        variant_spec,
        variant_freqs,
        variant_times,
        "Variant",
        float(variant_display_gain_db),
        float(variant_display_range_db),
        float(segment_start_s),
        xlim,
        ylim_khz,
        overlay_boxes if show_overlay and overlay_target == "Variant" else None,
    )
    st.pyplot(fig)
    variant_metrics = compute_metrics(variant_spec, variant_freqs, variant_times)
    st.caption(f"Variant compute time: {variant_result['elapsed_s']:.3f} s")
    with st.expander("Variant metrics", expanded=False):
        st.write(variant_metrics)
        if show_overlay and overlay_target == "Variant" and overlay_summary:
            st.write({"candidates": overlay_summary})

    if show_diff:
        can_diff = (
            base_spec.shape == variant_spec.shape
            and np.allclose(base_freqs, variant_freqs)
            and np.allclose(base_times, variant_times)
        )
        if can_diff:
            st.subheader("Difference view")
            diff_fig = plot_difference(
                variant_spec - base_spec,
                base_freqs,
                base_times,
                float(segment_start_s),
                xlim,
                ylim_khz,
                float(variant_display_range_db),
            )
            st.pyplot(diff_fig)
        else:
            st.info("Difference view requires identical time and frequency bins.")

    with st.expander("Parameter explanations", expanded=False):
        for name, text in parameter_explanations().items():
            st.markdown(f"**{name}**: {text}")

    with st.expander("Sweep export", expanded=False):
        st.write("Run a parameter sweep on the current segment and export images and a report.")
        sweep_param = st.selectbox(
            "Sweep parameter",
            ["window_length", "hop_ms", "zero_padding_factor", "window", "f_min_hz", "f_max_hz"],
        )
        sweep_values_text = st.text_input("Values (comma-separated)")
        sweep_output_dir = st.text_input(
            "Output directory",
            value=str(Path.cwd() / "sweep_output"),
        )
        if st.button("Run sweep"):
            try:
                sweep_values = _parse_sweep_values(sweep_param, sweep_values_text)
            except ValueError as exc:
                st.error(f"Invalid values: {exc}")
                st.stop()
            if not sweep_values:
                st.error("Provide at least one sweep value.")
                st.stop()

            cfgs = []
            for value in sweep_values:
                cfg = replace(baseline_cfg, **{sweep_param: value})
                label = f"{sweep_param}={value}"
                cfgs.append((label, cfg))

            with st.spinner("Running sweep..."):
                summary = run_sweep(
                    wav_path,
                    float(segment_start_s),
                    float(segment_duration_s),
                    cfgs,
                    float(baseline_display_gain_db),
                    float(baseline_display_range_db),
                    sweep_output_dir,
                )
            st.success(f"Sweep complete: {summary['count']} configs")
            st.write(summary)
