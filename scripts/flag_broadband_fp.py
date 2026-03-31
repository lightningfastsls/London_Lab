#!/usr/bin/env python3
"""Flag auto-accept detections with the "1960 pattern".

Checks each detection window for a broadband column (>=55% of freq bins active)
FOLLOWED BY a low-frequency streak below 30 kHz (>=10 consecutive columns).

Outputs:
  results/batch_5970/broadband_flags.csv
  results/batch_5970/broadband_flagged/*.png

Usage:
    python scripts/flag_broadband_fp.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.app.core.audio_loader import AudioLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BATCH_DIR = REPO_ROOT / "results" / "batch_5970"
WAV_SEARCH_DIRS = [
    REPO_ROOT / "5970",
    REPO_ROOT / "5970_reviewed",
    REPO_ROOT / "5970_manual_review",
    REPO_ROOT / "5970_manual_review_reviewed",
]

FREQ_MIN = 20_000
FREQ_MAX = 120_000
SR = 300_000
N_FFT = 512
FREQ_RES = SR / N_FFT  # ~585.9 Hz per bin

# Low band: 20-30 kHz (row 0 = 20 kHz in spectrogram)
LOW_BAND_MAX_HZ = 30_000
LOW_BAND_ROWS = int((LOW_BAND_MAX_HZ - FREQ_MIN) / FREQ_RES)  # ~17 rows

# Thresholds
# The 1960 broadband is PARTIAL (<55% of bins) — full broadband (>55%) is
# real noise with USVs in it, which the CNN handles correctly.
BROADBAND_FRAC_MIN = 0.20        # minimum to count as broadband at all
BROADBAND_FRAC_MAX = 0.55        # above this is full-range noise, not 1960
BROADBAND_DB_ABOVE_MEDIAN = 6.0  # dB above file median to count as "active"
LOW_STREAK_MIN_COLS = 10          # consecutive low-band columns after broadband
LOW_STREAK_DB_ABOVE_MEDIAN = 6.0  # dB above median for low-freq band

# Context before/after detection window — keep tight to avoid
# picking up unrelated cage noise outside the detection
CONTEXT_COLS_BEFORE = 2
CONTEXT_COLS_AFTER = 2


def find_wav(stem: str) -> Optional[Path]:
    for search_dir in WAV_SEARCH_DIRS:
        matches = list(search_dir.rglob(f"{stem}.wav"))
        if matches:
            return matches[0]
    return None


def check_broadband_then_streak(
    spec: np.ndarray,
    start_col: int,
    end_col: int,
    file_median: float,
) -> dict:
    """Check detection window for broadband column followed by low-freq streak."""
    n_freq, n_time = spec.shape

    col_start = max(0, start_col - CONTEXT_COLS_BEFORE)
    col_end = min(n_time, end_col + CONTEXT_COLS_AFTER)
    window = spec[:, col_start:col_end]

    # Step 1: find broadband columns (>=55% of bins above median+6dB)
    active_mask = window > (file_median + BROADBAND_DB_ABOVE_MEDIAN)
    frac_active = active_mask.mean(axis=0)
    broadband_mask = (frac_active >= BROADBAND_FRAC_MIN) & (frac_active <= BROADBAND_FRAC_MAX)

    max_broadband_frac = float(frac_active.max()) if len(frac_active) > 0 else 0.0

    if not broadband_mask.any():
        return {"flagged": False, "max_broadband_frac": max_broadband_frac, "streak": 0}

    # Step 2: for each broadband column, count consecutive low-band active cols after it
    low_band = window[:LOW_BAND_ROWS, :]
    low_band_active = (low_band > (file_median + LOW_STREAK_DB_ABOVE_MEDIAN)).any(axis=0)

    best_streak = 0
    for bb_col in np.where(broadband_mask)[0]:
        streak = 0
        for j in range(bb_col + 1, len(low_band_active)):
            if low_band_active[j]:
                streak += 1
            else:
                break
        best_streak = max(best_streak, streak)

    return {
        "flagged": best_streak >= LOW_STREAK_MIN_COLS,
        "max_broadband_frac": max_broadband_frac,
        "streak": best_streak,
    }


def save_flagged_png(spec: np.ndarray, stem: str, events: list, output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 4))
    extent = [0, spec.shape[1], FREQ_MIN / 1000, FREQ_MAX / 1000]
    ax.imshow(
        spec, aspect="auto", origin="lower", cmap="magma",
        extent=extent,
        vmin=np.percentile(spec, 5),
        vmax=np.percentile(spec, 99),
    )
    for ev in events:
        color = "red" if ev.get("flagged", False) else "cyan"
        ax.axvspan(ev["start_col"], ev["end_col"], alpha=0.25, color=color, linewidth=0)
    ax.axhline(y=30, color="white", linestyle="--", alpha=0.5, linewidth=0.8)

    n_events = len(events)
    flagged_events = sum(1 for e in events if e.get("flagged", False))
    ax.set_title(f"{stem}  |  {n_events} events, {flagged_events} flagged (red)", fontsize=10)
    ax.set_xlabel("Column")
    ax.set_ylabel("Frequency (kHz)")
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=100)
    plt.close(fig)


def main():
    summary_path = BATCH_DIR / "summary_full.parquet"
    detections_dir = BATCH_DIR / "detections"

    df = pd.read_parquet(summary_path)
    auto_accept = df[df["tier"] == "auto_accept"].copy()
    log.info("Auto-accept recordings: %d", len(auto_accept))

    output_dir = BATCH_DIR / "broadband_flagged"
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = AudioLoader()
    results = []
    n_total = len(auto_accept)

    for idx, (_, row) in enumerate(auto_accept.iterrows(), 1):
        stem = row["stem"]
        if idx % 100 == 0:
            log.info("  %d/%d done", idx, n_total)

        det_path = detections_dir / f"{stem}.json"
        if not det_path.exists():
            continue
        with open(det_path) as f:
            events = json.load(f)
        if not events:
            continue

        wav_path = find_wav(stem)
        if wav_path is None:
            log.warning("WAV not found: %s", stem)
            continue

        try:
            audio_data = loader.load(wav_path)
            spec = audio_data.spectrogram_db
        except Exception as e:
            log.error("Spectrogram failed for %s: %s", stem, e)
            continue

        file_median = float(np.median(spec))

        any_flagged = False
        flagged_event_count = 0
        file_max_broadband = 0.0
        file_max_streak = 0

        for ev in events:
            result = check_broadband_then_streak(
                spec, ev["start_col"], ev["end_col"], file_median
            )
            ev["flagged"] = result["flagged"]
            if result["flagged"]:
                any_flagged = True
                flagged_event_count += 1
            file_max_broadband = max(file_max_broadband, result["max_broadband_frac"])
            file_max_streak = max(file_max_streak, result["streak"])

        if any_flagged:
            results.append({
                "stem": stem,
                "n_events": len(events),
                "flagged_events": flagged_event_count,
                "max_confidence": row["max_confidence"],
                "max_broadband_frac": file_max_broadband,
                "max_streak": file_max_streak,
            })
            save_flagged_png(spec, stem, events, output_dir)

    log.info("\n=== RESULTS ===")
    log.info("Scanned: %d auto-accept recordings", n_total)
    log.info("Flagged: %d recordings", len(results))

    if results:
        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values("max_broadband_frac", ascending=False)
        csv_path = BATCH_DIR / "broadband_flags.csv"
        result_df.to_csv(csv_path, index=False)
        log.info("Saved flags to %s", csv_path)
        log.info("Saved %d PNGs to %s", len(results), output_dir)

        log.info("\n=== FLAGGED FILES ===")
        for _, r in result_df.iterrows():
            log.info(
                "  %-45s events=%2d  flagged=%2d  broadband=%.2f  streak=%d  conf=%.3f",
                r["stem"], r["n_events"], r["flagged_events"],
                r["max_broadband_frac"], r["max_streak"], r["max_confidence"],
            )
    else:
        log.info("No recordings flagged!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
