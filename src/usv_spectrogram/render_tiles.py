"""PNG rendering helpers for spectrograms."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .config import SpectrogramConfig
from .utils import ensure_dir


def render_png(
    spec_db: np.ndarray,
    freqs_hz: np.ndarray,
    times_s: np.ndarray,
    output_path: str | Path,
    cfg: SpectrogramConfig,
    title: Optional[str] = None,
) -> None:
    """Render a single-page spectrogram PNG."""
    display_db = spec_db + cfg.gain_db
    vmin = cfg.gain_db - cfg.range_db
    vmax = cfg.gain_db

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    mesh = ax.pcolormesh(
        times_s,
        freqs_hz / 1000.0,
        display_db,
        shading="auto",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    if title:
        ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label="dB")
    fig.tight_layout()
    fig.savefig(str(output_path))
    plt.close(fig)


def render_tiled_pages(
    spec_db: np.ndarray,
    freqs_hz: np.ndarray,
    times_s: np.ndarray,
    output_dir: str | Path,
    base_name: str,
    cfg: SpectrogramConfig,
    title: Optional[str] = None,
) -> list[Path]:
    """Render tiled spectrogram pages for long recordings."""
    output_dir = ensure_dir(output_dir)
    if times_s.size == 0:
        raise ValueError("times_s is empty; cannot render tiled pages.")

    display_db = spec_db + cfg.gain_db
    vmin = cfg.gain_db - cfg.range_db
    vmax = cfg.gain_db

    times_rel = times_s - times_s[0]
    tile_duration = cfg.tile_duration_s
    if tile_duration <= 0:
        raise ValueError("tile_duration_s must be positive.")

    tiles_per_page = cfg.tiles_per_row * cfg.tiles_per_col
    if tiles_per_page <= 0:
        raise ValueError("tiles_per_row and tiles_per_col must be positive.")

    total_tiles = int(np.ceil(times_rel[-1] / tile_duration))
    total_tiles = max(1, total_tiles)
    page_count = int(np.ceil(total_tiles / tiles_per_page))
    output_paths: list[Path] = []

    for page_index in range(page_count):
        fig_width = cfg.tiles_per_row * cfg.tile_width_in
        fig_height = cfg.tiles_per_col * cfg.tile_height_in
        fig, axes = plt.subplots(
            cfg.tiles_per_col,
            cfg.tiles_per_row,
            figsize=(fig_width, fig_height),
            dpi=cfg.page_dpi,
            squeeze=False,
            constrained_layout=True,
        )

        page_start = page_index * tiles_per_page
        page_end = min(page_start + tiles_per_page, total_tiles)
        last_mesh = None

        for tile_index in range(page_start, page_end):
            tile_offset = tile_index - page_start
            row = tile_offset // cfg.tiles_per_row
            col = tile_offset % cfg.tiles_per_row
            ax = axes[row][col]

            tile_start = tile_index * tile_duration
            tile_end = tile_start + tile_duration
            mask = (times_rel >= tile_start) & (times_rel < tile_end)
            if not np.any(mask):
                ax.axis("off")
                continue

            tile_times = times_rel[mask] - tile_start
            tile_spec = display_db[:, mask]

            last_mesh = ax.pcolormesh(
                tile_times,
                freqs_hz / 1000.0,
                tile_spec,
                shading="auto",
                cmap="magma",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xlim(0, tile_duration)
            ax.set_ylim(freqs_hz[0] / 1000.0, freqs_hz[-1] / 1000.0)
            ax.set_title(f"{tile_start:.1f}-{tile_end:.1f}s", fontsize=8)

            if row == cfg.tiles_per_col - 1:
                ax.set_xlabel("Time (s)")
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel("Frequency (kHz)")
            else:
                ax.set_yticklabels([])

        for tile_index in range(page_end, page_start + tiles_per_page):
            tile_offset = tile_index - page_start
            row = tile_offset // cfg.tiles_per_row
            col = tile_offset % cfg.tiles_per_row
            axes[row][col].axis("off")

        if title:
            fig.suptitle(title, fontsize=10)

        if last_mesh is not None:
            fig.colorbar(last_mesh, ax=axes, label="dB", shrink=0.75)

        output_path = output_dir / f"{base_name}_page{page_index + 1:03d}.png"
        fig.savefig(str(output_path))
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths
