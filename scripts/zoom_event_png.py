"""Render a zoomed spectrogram PNG of a single detection event.

Used during interactive labeling when the chunk-wide audit PNG is too small
to read the detection-window content. Zooms the time axis to the event ±margin.

Usage:
    .venv/bin/python scripts/zoom_event_png.py \
        --chunk-stem 131208_1000_m2fm2_chunk_173 \
        --event-idx 0 \
        --output data/lab_finetune_v1/labeling_queue/typ05_zoom.png \
        --margin-ms 80
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from usv_spectrogram.corpus import (
    SAMPLE_RATE_HZ,
    USV_FREQ_MIN_HZ,
    USV_FREQ_MAX_HZ,
    STFT_N_FFT,
    STFT_HOP,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--chunk-stem", required=True)
    p.add_argument("--event-idx", type=int, required=True)
    p.add_argument(
        "--parquet",
        default="results/batch_lab_131204_full/merged_events_with_filter.parquet",
    )
    p.add_argument("--chunks-dir", default="USV_lab_131204_chunked_2s_full")
    p.add_argument("--output", required=True)
    p.add_argument("--margin-ms", type=float, default=80.0,
                   help="Time padding either side of event (ms)")
    p.add_argument("--center-only-ms", type=float, default=None,
                   help="If set, render only the central N ms of the event "
                        "(replaces event-bounds + margin logic). Use for long events "
                        "to inspect the slice the trainer will actually sample.")
    p.add_argument("--title-id", default=None,
                   help="Stable ID like 'typ05' to put in title")
    args = p.parse_args()

    df = pd.read_parquet(args.parquet)
    sel = df[
        (df["chunk_stem"] == args.chunk_stem)
        & (df["chunk_detection_idx"] == args.event_idx)
    ]
    if len(sel) == 0:
        print(f"ERROR: event not found ({args.chunk_stem} ev{args.event_idx})", file=sys.stderr)
        return 1
    row = sel.iloc[0]

    wav_path = Path(args.chunks_dir) / f"{row['chunk_stem']}.wav"
    sr, samples = wavfile.read(wav_path)
    if sr != SAMPLE_RATE_HZ:
        print(f"ERROR: expected {SAMPLE_RATE_HZ} Hz, got {sr}", file=sys.stderr)
        return 1
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples = samples.astype(np.float32)

    f, t, S = scipy_signal.stft(
        samples, fs=sr, window="hann", nperseg=STFT_N_FFT,
        noverlap=STFT_N_FFT - STFT_HOP, boundary=None, padded=False,
    )
    spec_db = 20 * np.log10(np.abs(S) + 1e-12)

    band = (f >= USV_FREQ_MIN_HZ) & (f <= USV_FREQ_MAX_HZ)
    spec_band = spec_db[band]
    f_band = f[band]

    event_t0 = row["original_begin_time_s"] - row["start_s_in_original"]
    event_t1 = row["original_end_time_s"] - row["start_s_in_original"]
    if args.center_only_ms is not None:
        center = (event_t0 + event_t1) / 2.0
        half_window = (args.center_only_ms / 1000.0) / 2.0
        zoom_lo = max(0.0, center - half_window)
        zoom_hi = min(t[-1], center + half_window)
    else:
        margin = args.margin_ms / 1000.0
        zoom_lo = max(0.0, event_t0 - margin)
        zoom_hi = min(t[-1], event_t1 + margin)
    time_mask = (t >= zoom_lo) & (t <= zoom_hi)
    spec_zoom = spec_band[:, time_mask]
    t_zoom = t[time_mask]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    extent = [t_zoom[0] * 1000, t_zoom[-1] * 1000, f_band[0] / 1000, f_band[-1] / 1000]
    vmin = np.percentile(spec_zoom, 5)
    vmax = np.percentile(spec_zoom, 99)
    ax.imshow(spec_zoom, aspect="auto", origin="lower", extent=extent,
              cmap="magma", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.axvline(event_t0 * 1000, color="cyan", linestyle="--", alpha=0.8, linewidth=1.5)
    ax.axvline(event_t1 * 1000, color="cyan", linestyle="--", alpha=0.8, linewidth=1.5)
    ax.set_xlabel("Time in chunk (ms)")
    ax.set_ylabel("Frequency (kHz)")
    title_id = f"[{args.title_id}] " if args.title_id else ""
    ax.set_title(
        f"{title_id}ZOOM | {row['chunk_stem']} ev{row['chunk_detection_idx']} | "
        f"event {event_t0*1000:.0f}-{event_t1*1000:.0f}ms (dur={row['duration_s']*1000:.0f}ms) | "
        f"max_prob={row['max_probability']:.3f} | SEF={row['stationary_energy_fraction']:.3f}"
    )
    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output)
    print(f"Wrote {args.output}")
    print(f"  Event window in chunk: {event_t0*1000:.1f}–{event_t1*1000:.1f} ms")
    if args.center_only_ms is not None:
        print(f"  Zoom window:           {zoom_lo*1000:.1f}–{zoom_hi*1000:.1f} ms (center-only, {args.center_only_ms:.0f}ms wide)")
    else:
        print(f"  Zoom window:           {zoom_lo*1000:.1f}–{zoom_hi*1000:.1f} ms (±{args.margin_ms:.0f}ms margin)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
