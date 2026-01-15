"""Plotting functions for spectrogram visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches


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
