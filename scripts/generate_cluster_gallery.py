#!/usr/bin/env python3
"""Generate a visual gallery of USV cluster examples.

For each of the 27 DeepSqueak clusters, picks N random examples and renders
spectrogram PNGs showing the call in context (with padding).

Output: results/cluster_gallery/Cluster_XX/*.png
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# USV spectrogram parameters (300 kHz recordings, per ADR-001/ADR-002)
SAMPLE_RATE = 300_000
FREQ_MIN_HZ = 20_000
FREQ_MAX_HZ = 125_000
NFFT = 1024
HOP = 128
PADDING_S = 0.05  # 50ms context on each side


def build_wav_lookup(search_dirs: list[Path]) -> dict[str, Path]:
    """Build stem -> path dict. Prefers 5970_reviewed paths over duplicates."""
    lookup: dict[str, Path] = {}
    for d in search_dirs:
        if not d.exists():
            continue
        for p in d.rglob("*.wav"):
            stem = p.stem
            # Prefer 5970_reviewed paths (reviewed data)
            if stem not in lookup or "reviewed" in str(p):
                lookup[stem] = p
    return lookup


def render_call_spectrogram(
    wav_path: Path,
    begin_s: float,
    end_s: float,
    output_path: Path,
    title: str,
    freq_info: dict,
):
    """Load WAV snippet around a call and render spectrogram PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import signal

    sr, data = wavfile.read(wav_path)
    if sr != SAMPLE_RATE:
        log.warning(f"Unexpected sample rate {sr} for {wav_path.name}, expected {SAMPLE_RATE}")

    # Convert to float
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)

    # If stereo, take first channel
    if data.ndim > 1:
        data = data[:, 0]

    total_duration = len(data) / sr

    # Extract window with padding
    t_start = max(0.0, begin_s - PADDING_S)
    t_end = min(total_duration, end_s + PADDING_S)
    i_start = int(t_start * sr)
    i_end = int(t_end * sr)
    snippet = data[i_start:i_end]

    if len(snippet) < NFFT:
        log.warning(f"Snippet too short for {wav_path.name} @ {begin_s:.3f}s, skipping")
        return False

    # Compute spectrogram
    freqs, times, Sxx = signal.spectrogram(
        snippet, fs=sr, nperseg=NFFT, noverlap=NFFT - HOP,
        nfft=NFFT, window="hann",
    )

    # Band-limit to USV range
    freq_mask = (freqs >= FREQ_MIN_HZ) & (freqs <= FREQ_MAX_HZ)
    Sxx_band = Sxx[freq_mask, :]
    freqs_band = freqs[freq_mask]

    # dB scale
    Sxx_db = 10 * np.log10(Sxx_band + 1e-12)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
    extent = [
        t_start,
        t_start + times[-1],
        freqs_band[0] / 1000,
        freqs_band[-1] / 1000,
    ]
    ax.imshow(
        Sxx_db, aspect="auto", origin="lower", cmap="magma",
        extent=extent,
        vmin=np.percentile(Sxx_db, 5),
        vmax=np.percentile(Sxx_db, 99),
    )

    # Mark the call boundaries
    ax.axvline(begin_s, color="cyan", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axvline(end_s, color="cyan", linewidth=0.8, linestyle="--", alpha=0.7)

    # Frequency annotation
    if freq_info.get("principal_freq_hz"):
        pf = freq_info["principal_freq_hz"]
        # DeepSqueak freqs might be in kHz already if < 1000
        if pf < 1000:
            pf_khz = pf
        else:
            pf_khz = pf / 1000
        ax.axhline(pf_khz, color="lime", linewidth=0.5, linestyle=":", alpha=0.5)

    ax.set_ylabel("Freq (kHz)")
    ax.set_xlabel("Time (s)")
    dur_ms = (end_s - begin_s) * 1000
    ax.set_title(title + f"  [{dur_ms:.0f}ms]", fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate cluster gallery PNGs")
    parser.add_argument(
        "--csv", default=str(REPO_ROOT / "classified_detections_full.csv"),
        help="Path to classified_detections_full.csv",
    )
    parser.add_argument(
        "--output-dir", default=str(REPO_ROOT / "results" / "cluster_gallery"),
        help="Output directory for gallery",
    )
    parser.add_argument("--n-per-cluster", type=int, default=5, help="Examples per cluster")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    output_dir = Path(args.output_dir)
    rng = np.random.default_rng(args.seed)

    # Build WAV lookup
    search_dirs = [
        REPO_ROOT / "5970_reviewed",
        REPO_ROOT / "5970 USV",
    ]
    log.info("Building WAV lookup...")
    wav_lookup = build_wav_lookup(search_dirs)
    log.info(f"Found {len(wav_lookup)} unique WAV stems on disk")

    # Drop rows with missing labels
    df = df.dropna(subset=["label"])
    clusters = sorted(df["label"].unique(), key=lambda x: int(x.split("_")[1]))
    log.info(f"Processing {len(clusters)} clusters, {args.n_per_cluster} examples each")

    total_generated = 0
    total_skipped = 0

    for cluster in clusters:
        cluster_df = df[df["label"] == cluster]
        # Filter to rows where we have a WAV file
        available = cluster_df[cluster_df["wav_stem"].isin(wav_lookup)]

        if len(available) == 0:
            log.warning(f"{cluster}: no WAV files available, skipping")
            total_skipped += 1
            continue

        n_sample = min(args.n_per_cluster, len(available))
        sample = available.sample(n=n_sample, random_state=rng.integers(0, 2**31))

        cluster_dir = output_dir / cluster
        generated = 0

        for idx, (_, row) in enumerate(sample.iterrows()):
            wav_path = wav_lookup[row["wav_stem"]]
            begin_s = row["begin_time_s"]
            end_s = row["end_time_s"]
            freq_info = {
                "principal_freq_hz": row.get("principal_freq_hz"),
                "low_freq_hz": row.get("low_freq_hz"),
                "high_freq_hz": row.get("high_freq_hz"),
            }

            fname = f"{idx+1:02d}_{row['wav_stem']}_{begin_s:.3f}s.png"
            out_path = cluster_dir / fname

            title = f"{cluster} | {row['wav_stem']} @ {begin_s:.3f}s"
            ok = render_call_spectrogram(wav_path, begin_s, end_s, out_path, title, freq_info)
            if ok:
                generated += 1

        total_generated += generated
        log.info(f"{cluster}: {generated}/{n_sample} PNGs (from {len(cluster_df)} total calls)")

    log.info(f"Done! {total_generated} PNGs in {output_dir}")
    if total_skipped:
        log.warning(f"{total_skipped} clusters had no available WAV files")


if __name__ == "__main__":
    main()
