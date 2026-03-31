#!/usr/bin/env python3
"""Generate spectrogram PNGs for the highest-confidence auto-accept detections.

These are the CNN's most assured detections — any FPs among them are the
most valuable hard negatives for retraining.

Outputs PNGs sorted by max_confidence (descending) to:
  results/batch_5970/high_confidence_review/

Usage:
    python scripts/generate_high_confidence_pngs.py [--n 100]
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

# Exclude files we already know are noise (original 10 + 8 new)
KNOWN_NOISE_SUFFIXES = {
    "0001960", "0002431", "0002522", "0003502", "0003503",
    "0003781", "0003794", "0005107", "0005656", "0006086",
    "0000570", "0000716", "0000717", "0003579",
    "0003825", "0004706", "0005108", "0005647",
}


def find_wav(stem: str):
    for d in WAV_SEARCH_DIRS:
        if not d.exists():
            continue
        m = list(d.rglob(f"{stem}.wav"))
        if m:
            return m[0]
    return None


def save_png(spec_db, stem, events, output_dir):
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

    n_ev = len(events)
    max_prob = max((e.get("max_probability", 0) for e in events), default=0)
    ax.set_title(f"{stem}  |  {n_ev} events  |  max_prob={max_prob:.4f}", fontsize=10)
    ax.set_xlabel("Column")
    ax.set_ylabel("Frequency (kHz)")
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100,
                        help="Number of highest-confidence files to generate")
    args = parser.parse_args()

    summary_path = BATCH_DIR / "summary_full.parquet"
    df = pd.read_parquet(summary_path)

    # Get auto_accept files, exclude known noise, sort by confidence
    aa = df[df["tier"] == "auto_accept"].copy()
    aa["short"] = aa["stem"].str[-7:]
    aa = aa[~aa["short"].isin(KNOWN_NOISE_SUFFIXES)]
    aa = aa.sort_values("max_confidence", ascending=False)

    # Take top N, prioritizing files with fewer events (more likely FP)
    # Among files with same confidence, fewer events = more suspicious
    top = aa.head(args.n * 2)  # oversample then pick
    # Sort by (confidence desc, n_events asc) to prioritize single-event high-conf
    top = top.sort_values(["max_confidence", "n_events"], ascending=[False, True])
    top = top.head(args.n)

    output_dir = BATCH_DIR / "high_confidence_review"
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Generating PNGs for top %d highest-confidence files", len(top))

    loader = AudioLoader()
    detections_dir = BATCH_DIR / "detections"

    for idx, (_, row) in enumerate(top.iterrows(), 1):
        stem = row["stem"]
        if idx % 20 == 0:
            log.info("[%d/%d] %s (conf=%.4f, events=%d)",
                     idx, len(top), stem, row["max_confidence"], row["n_events"])

        wav_path = find_wav(stem)
        if wav_path is None:
            continue

        det_path = detections_dir / f"{stem}.json"
        if not det_path.exists():
            continue
        with open(det_path) as f:
            events = json.load(f)

        try:
            audio_data = loader.load(wav_path)
            save_png(audio_data.spectrogram_db, stem, events, output_dir)
        except Exception as e:
            log.error("Failed %s: %s", stem, e)

    log.info("Done. PNGs saved to %s", output_dir)


if __name__ == "__main__":
    main()
