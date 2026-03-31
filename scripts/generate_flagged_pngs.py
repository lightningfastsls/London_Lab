#!/usr/bin/env python3
"""Generate spectrogram PNGs for files flagged by spectral flatness.

Reads the sweep results and generates annotated spectrograms for all
auto-accept files with file_mean_flatness >= a given threshold.

Usage:
    python scripts/generate_flagged_pngs.py [--threshold 0.53]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.app.core.audio_loader import AudioLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

BATCH_DIR = REPO_ROOT / "results" / "batch_5970"
WAV_SEARCH_DIRS = [
    REPO_ROOT / "5970",
    REPO_ROOT / "5970_reviewed",
    REPO_ROOT / "5970_manual_review",
    REPO_ROOT / "5970_manual_review_reviewed",
]

SPEC_FREQ_MIN = 20_000
SPEC_FREQ_MAX = 120_000
USV_BAND_MIN_HZ = 35_000
USV_BAND_MAX_HZ = 110_000


def find_wav(stem: str):
    for d in WAV_SEARCH_DIRS:
        if not d.exists():
            continue
        m = list(d.rglob(f"{stem}.wav"))
        if m:
            return m[0]
    return None


def save_png(spec_db, stem, events, mean_flat, max_consec, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 4))
    extent = [0, spec_db.shape[1], SPEC_FREQ_MIN / 1000, SPEC_FREQ_MAX / 1000]
    ax.imshow(
        spec_db, aspect="auto", origin="lower", cmap="magma",
        extent=extent,
        vmin=np.percentile(spec_db, 5),
        vmax=np.percentile(spec_db, 99),
    )
    for ev in events:
        ax.axvspan(ev["start_col"], ev["end_col"], alpha=0.25, color="cyan", linewidth=0)

    ax.axhline(y=USV_BAND_MIN_HZ / 1000, color="white", linestyle="--", alpha=0.4, linewidth=0.7)
    ax.axhline(y=USV_BAND_MAX_HZ / 1000, color="white", linestyle="--", alpha=0.4, linewidth=0.7)

    n_ev = len(events)
    max_prob = max((e.get("max_probability", 0) for e in events), default=0)
    ax.set_title(
        f"{stem}  |  {n_ev} events  |  max_prob={max_prob:.3f}  |  "
        f"mean_flat={mean_flat:.3f}  |  max_consec_tonal={max_consec}",
        fontsize=9,
    )
    ax.set_xlabel("Column")
    ax.set_ylabel("Frequency (kHz)")
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.53)
    args = parser.parse_args()

    sweep_csv = BATCH_DIR / "spectral_flatness_sweep" / "sweep_file_summary.csv"
    df = pd.read_csv(sweep_csv)

    flagged = df[(df["is_problem"] == 0) & (df["file_mean_flatness"] >= args.threshold)]
    flagged = flagged.sort_values("file_mean_flatness", ascending=False)

    output_dir = BATCH_DIR / f"flatness_flagged_{str(args.threshold).replace('.', '')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Generating PNGs for %d files with mean_flatness >= %.2f", len(flagged), args.threshold)

    loader = AudioLoader()
    detections_dir = BATCH_DIR / "detections"

    for idx, (_, row) in enumerate(flagged.iterrows(), 1):
        stem = row["stem"]
        log.info("[%d/%d] %s (flat=%.3f, consec=%d)",
                 idx, len(flagged), stem, row["file_mean_flatness"],
                 int(row["file_max_consec_tonal"]))

        wav_path = find_wav(stem)
        if wav_path is None:
            log.warning("WAV not found: %s", stem)
            continue

        det_path = detections_dir / f"{stem}.json"
        if not det_path.exists():
            continue
        with open(det_path) as f:
            events = json.load(f)

        try:
            audio_data = loader.load(wav_path)
            save_png(audio_data.spectrogram_db, stem, events,
                     row["file_mean_flatness"], int(row["file_max_consec_tonal"]),
                     output_dir)
        except Exception as e:
            log.error("Failed %s: %s", stem, e)

    log.info("Done. PNGs saved to %s", output_dir)


if __name__ == "__main__":
    main()
