"""Main content area rendering for spectrograms and analysis."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import streamlit as st

from usv_spectrogram.config import SpectrogramConfig
from usv_spectrogram.param_lab.explain import parameter_explanations
from usv_spectrogram.param_lab.heuristic_detect import (
    HeuristicConfig,
    detect_candidates,
    summarize_candidates,
)
from usv_spectrogram.param_lab.metrics import compute_metrics
from usv_spectrogram.param_lab.plotting import plot_difference, plot_spectrogram
from usv_spectrogram.param_lab.state import (
    compute_segment_spectrogram,
    config_from_controls,
    parse_sweep_values,
    stft_cache_key,
)
from usv_spectrogram.param_lab.sweep import run_sweep


def render_spectrogram_panel(
    title: str,
    spec_db: np.ndarray,
    freqs_hz: np.ndarray,
    times_s: np.ndarray,
    display_gain_db: float,
    display_range_db: float,
    segment_start_s: float,
    xlim: tuple[float, float] | None,
    ylim_khz: tuple[float, float] | None,
    elapsed_s: float,
    overlay_boxes: list[dict[str, float]] | None,
    overlay_summary: dict[str, object] | None,
) -> None:
    """Render a single spectrogram panel with metrics."""
    st.subheader(title)
    fig = plot_spectrogram(
        spec_db,
        freqs_hz,
        times_s,
        title,
        display_gain_db,
        display_range_db,
        segment_start_s,
        xlim,
        ylim_khz,
        overlay_boxes,
    )
    st.pyplot(fig)
    metrics = compute_metrics(spec_db, freqs_hz, times_s)
    st.caption(f"{title} compute time: {elapsed_s:.3f} s")
    with st.expander(f"{title} metrics", expanded=False):
        st.write(metrics)
        if overlay_boxes and overlay_summary:
            st.write({"candidates": overlay_summary})


def render_diff_view(
    base_spec: np.ndarray,
    variant_spec: np.ndarray,
    base_freqs: np.ndarray,
    variant_freqs: np.ndarray,
    base_times: np.ndarray,
    variant_times: np.ndarray,
    segment_start_s: float,
    xlim: tuple[float, float] | None,
    ylim_khz: tuple[float, float] | None,
    variant_display_range_db: float,
) -> None:
    """Render the difference view if compatible."""
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
            segment_start_s,
            xlim,
            ylim_khz,
            variant_display_range_db,
        )
        st.pyplot(diff_fig)
    else:
        st.info("Difference view requires identical time and frequency bins.")


def render_parameter_explanations() -> None:
    """Render expandable parameter documentation."""
    with st.expander("Parameter explanations", expanded=False):
        for name, text in parameter_explanations().items():
            st.markdown(f"**{name}**: {text}")


def render_sweep_export(
    baseline_cfg: SpectrogramConfig,
    wav_path: Path,
    segment_start_s: float,
    segment_duration_s: float,
    baseline_display_gain_db: float,
    baseline_display_range_db: float,
) -> None:
    """Render the sweep export interface."""
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
                sweep_values = parse_sweep_values(sweep_param, sweep_values_text)
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
                    segment_start_s,
                    segment_duration_s,
                    cfgs,
                    baseline_display_gain_db,
                    baseline_display_range_db,
                    sweep_output_dir,
                )
            st.success(f"Sweep complete: {summary['count']} configs")
            st.write(summary)


def render_main_content(
    sidebar_state: dict[str, object],
    base_cfg: SpectrogramConfig,
) -> None:
    """Render the main content area with spectrograms and controls."""
    baseline_params = sidebar_state["baseline_params"]
    variant_params = sidebar_state["variant_params"]

    if (
        baseline_params["f_min_hz"] >= baseline_params["f_max_hz"]
        or variant_params["f_min_hz"] >= variant_params["f_max_hz"]
    ):
        st.error("F min must be smaller than F max.")
        st.stop()

    baseline_cfg = config_from_controls(
        base_cfg,
        baseline_params["window_length"],
        baseline_params["zero_padding_factor"],
        baseline_params["hop_ms"],
        baseline_params["window"],
        baseline_params["f_min_hz"],
        baseline_params["f_max_hz"],
        sidebar_state["baseline_display_gain_db"],
        sidebar_state["baseline_display_range_db"],
    )
    variant_cfg = config_from_controls(
        base_cfg,
        variant_params["window_length"],
        variant_params["zero_padding_factor"],
        variant_params["hop_ms"],
        variant_params["window"],
        variant_params["f_min_hz"],
        variant_params["f_max_hz"],
        sidebar_state["variant_display_gain_db"],
        sidebar_state["variant_display_range_db"],
    )

    wav_path = sidebar_state["wav_path"]
    segment_start_s = sidebar_state["segment_start_s"]
    segment_duration_s = sidebar_state["segment_duration_s"]

    with st.spinner("Computing spectrograms..."):
        try:
            baseline_result = compute_segment_spectrogram(
                str(wav_path),
                segment_start_s,
                segment_duration_s,
                stft_cache_key(baseline_cfg),
                asdict(baseline_cfg),
            )
            variant_result = compute_segment_spectrogram(
                str(wav_path),
                segment_start_s,
                segment_duration_s,
                stft_cache_key(variant_cfg),
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

    xlim = (segment_start_s, segment_start_s + segment_duration_s)
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

    overlay_boxes_baseline = None
    overlay_boxes_variant = None
    overlay_summary = None
    if sidebar_state["show_overlay"]:
        overlay_cfg = HeuristicConfig(
            threshold_db=sidebar_state["overlay_threshold"],
            min_area_bins=sidebar_state["overlay_min_area"],
            min_frames=sidebar_state["overlay_min_frames"],
            min_bins=sidebar_state["overlay_min_bins"],
        )
        if sidebar_state["overlay_target"] == "Baseline":
            overlay_boxes_baseline = detect_candidates(base_spec, base_freqs, base_times, overlay_cfg)
            overlay_summary = summarize_candidates(overlay_boxes_baseline)
        else:
            overlay_boxes_variant = detect_candidates(
                variant_spec, variant_freqs, variant_times, overlay_cfg
            )
            overlay_summary = summarize_candidates(overlay_boxes_variant)

    render_spectrogram_panel(
        "Baseline",
        base_spec,
        base_freqs,
        base_times,
        sidebar_state["baseline_display_gain_db"],
        sidebar_state["baseline_display_range_db"],
        segment_start_s,
        xlim,
        ylim_khz,
        baseline_result["elapsed_s"],
        overlay_boxes_baseline,
        overlay_summary if sidebar_state["overlay_target"] == "Baseline" else None,
    )

    render_spectrogram_panel(
        "Variant",
        variant_spec,
        variant_freqs,
        variant_times,
        sidebar_state["variant_display_gain_db"],
        sidebar_state["variant_display_range_db"],
        segment_start_s,
        xlim,
        ylim_khz,
        variant_result["elapsed_s"],
        overlay_boxes_variant,
        overlay_summary if sidebar_state["overlay_target"] == "Variant" else None,
    )

    if sidebar_state["show_diff"]:
        render_diff_view(
            base_spec,
            variant_spec,
            base_freqs,
            variant_freqs,
            base_times,
            variant_times,
            segment_start_s,
            xlim,
            ylim_khz,
            sidebar_state["variant_display_range_db"],
        )

    render_parameter_explanations()
    render_sweep_export(
        baseline_cfg,
        wav_path,
        segment_start_s,
        segment_duration_s,
        sidebar_state["baseline_display_gain_db"],
        sidebar_state["baseline_display_range_db"],
    )
