#!/usr/bin/env python3
"""Extract hard negative training PNGs from manual review labels.

Takes the labeled detections in data/manual_review_labels.csv, extracts
spectrogram windows matching the EXACT training pipeline (global MAD
normalization, 100-column windows, magma colormap, 256px height), and
generates jittered augmented versions.

Output goes to data/training/hard_negatives/ with the same PNG format
as the assembler produces, ready to be added to training CSVs.

Usage:
    python scripts/extract_hard_negatives.py [--jitter-n 3]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib.pyplot as plt
from scipy import signal as scipy_signal

from usv_spectrogram.io_wav import load_wav_mono
from usv_spectrogram._stft_core import compute_stft_frames_db, extract_frames

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Must match assembler exactly
SAMPLE_RATE = 300_000
N_FFT = 512
HOP_LENGTH = 128
WINDOW_COLUMNS = 100
IMAGE_HEIGHT = 256
COLORMAP = "magma"
MAD_VMIN_SCALE = 2.0
MAD_VMAX_SCALE = 4.0
JITTER_MIN_OVERLAP = 0.5

WAV_SEARCH_DIRS = [
    REPO_ROOT / "5970",
    REPO_ROOT / "5970_reviewed",
    REPO_ROOT / "5970_manual_review",
    REPO_ROOT / "5970_manual_review_reviewed",
]


def find_wav(stem: str) -> Path | None:
    for d in WAV_SEARCH_DIRS:
        if not d.exists():
            continue
        m = list(d.rglob(f"{stem}.wav"))
        if m:
            return m[0]
    return None


def load_global_spectrogram(wav_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load WAV and compute globally MAD-normalized spectrogram.

    Matches DatasetAssembler._load_global_spectrogram exactly.
    """
    samples, sr = load_wav_mono(wav_path)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Sample rate mismatch: {sr}")

    frames = extract_frames(samples, N_FFT, HOP_LENGTH)
    window = scipy_signal.get_window("hann", N_FFT, fftbins=True)

    freqs_hz = np.fft.rfftfreq(N_FFT, d=1.0 / SAMPLE_RATE)
    band_mask = (freqs_hz >= 20_000) & (freqs_hz <= 120_000)

    spec_db = compute_stft_frames_db(
        frames, window, N_FFT, band_mask, eps=1e-10, normalize_magnitude=True
    )

    # Global MAD normalization
    median = np.median(spec_db)
    mad = np.median(np.abs(spec_db - median))
    vmin = median - MAD_VMIN_SCALE * mad
    vmax = median + MAD_VMAX_SCALE * mad

    spec_clipped = np.clip(spec_db, vmin, vmax)
    if vmax > vmin:
        spec_norm = (spec_clipped - vmin) / (vmax - vmin + 1e-12)
    else:
        spec_norm = np.zeros_like(spec_db)

    times_s = (
        (np.arange(frames.shape[0]) * HOP_LENGTH) + N_FFT / 2.0
    ) / SAMPLE_RATE

    return spec_norm, times_s


def render_window_png(window: np.ndarray, output_path: Path) -> None:
    """Render spectrogram window to PNG matching training pipeline exactly.

    Matches DatasetAssembler._render_window_png.
    """
    cmap = plt.get_cmap(COLORMAP)
    rgb = cmap(window)[:, :, :3]
    rgb = np.flipud(rgb)
    rgb_uint8 = (rgb * 255).astype(np.uint8)

    img = Image.fromarray(rgb_uint8)
    current_w, current_h = img.size
    if current_h != IMAGE_HEIGHT:
        img = img.resize((current_w, IMAGE_HEIGHT), Image.LANCZOS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path))


def time_to_col(time_s: float) -> int:
    """Convert time to STFT column index. Matches assembler._time_to_col."""
    sample_idx = time_s * SAMPLE_RATE
    return max(0, int(round((sample_idx - N_FFT / 2.0) / HOP_LENGTH)))


def jitter_windows(
    center_col: int, detection_span: int, n_cols: int, n_jitter: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    """Generate centered + jittered windows for a detection.

    Mirrors assembler._jitter_windows_for_short_usv logic.
    """
    wc = WINDOW_COLUMNS

    # Centered window
    centered_start = max(0, center_col - wc // 2)
    centered_start = min(centered_start, max(0, n_cols - wc))
    windows = [(centered_start, centered_start + wc)]

    # Jitter range
    min_overlap = max(1, int(detection_span * JITTER_MIN_OVERLAP))
    start_col = center_col - detection_span // 2
    end_col = start_col + detection_span

    earliest = max(0, end_col - wc)
    latest = min(n_cols - wc, start_col + detection_span - min_overlap)

    if latest > earliest and n_jitter > 0:
        for _ in range(n_jitter):
            jit_start = int(rng.integers(earliest, latest + 1))
            windows.append((jit_start, jit_start + wc))

    return windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jitter-n", type=int, default=3,
                        help="Number of jittered versions per detection")
    parser.add_argument("--label-filter", default="noise",
                        help="Which label to extract (noise, usv, or all)")
    args = parser.parse_args()

    labels_path = REPO_ROOT / "data" / "manual_review_labels.csv"
    df = pd.read_csv(labels_path)

    if args.label_filter != "all":
        df = df[df["label"] == args.label_filter]

    log.info("Extracting %d %s detections with %d jitter each",
             len(df), args.label_filter, args.jitter_n)

    output_dir = REPO_ROOT / "data" / "training" / f"hard_{args.label_filter}s"
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    # Group by recording to load spectrogram once per file
    csv_rows = []
    spec_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    total_extracted = 0
    for idx, (_, row) in enumerate(df.iterrows(), 1):
        stem = row["stem"]

        # Load spectrogram (cached per recording)
        if stem not in spec_cache:
            wav_path = find_wav(stem)
            if wav_path is None:
                log.warning("WAV not found: %s", stem)
                continue
            try:
                spec_cache[stem] = load_global_spectrogram(wav_path)
            except Exception as e:
                log.error("Failed to load %s: %s", stem, e)
                continue

        spec_norm, times_s = spec_cache[stem]
        n_cols = spec_norm.shape[1]

        # Convert detection time to columns
        start_col = time_to_col(row["start_time_s"])
        end_col = time_to_col(row["end_time_s"])
        det_span = max(1, end_col - start_col)
        center_col = (start_col + end_col) // 2

        # Generate windows (centered + jittered)
        windows = jitter_windows(center_col, det_span, n_cols, args.jitter_n, rng)

        for i, (win_start, win_end) in enumerate(windows):
            if win_end > n_cols:
                continue
            window_data = spec_norm[:, win_start:win_end]
            if window_data.shape[1] != WINDOW_COLUMNS:
                continue

            suffix = "" if i == 0 else f"_jit{i:02d}"
            sample_id = f"{stem}_{start_col:06d}{suffix}"

            png_path = output_dir / "spectrograms" / f"{sample_id}.png"
            render_window_png(window_data, png_path)

            csv_rows.append({
                "candidate_id": sample_id,
                "source_file": f"{stem}.wav",
                "label": row["label"].upper(),  # Match assembler convention: "NOISE" / "USV"
                "spectrogram_path": str(png_path),
                "original_detection_idx": row["detection_idx"],
                "jitter_idx": i,
            })
            total_extracted += 1

        if idx % 20 == 0:
            log.info("[%d/%d] %d windows extracted so far", idx, len(df), total_extracted)

    # Write CSV
    out_df = pd.DataFrame(csv_rows)
    csv_path = output_dir / f"hard_{args.label_filter}s.csv"
    out_df.to_csv(csv_path, index=False)

    n_original = len(out_df[out_df["jitter_idx"] == 0])
    n_jittered = len(out_df[out_df["jitter_idx"] > 0])

    log.info("\n=== DONE ===")
    log.info("Extracted: %d total (%d original + %d jittered)", total_extracted, n_original, n_jittered)
    log.info("PNGs: %s/spectrograms/", output_dir)
    log.info("CSV:  %s", csv_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
