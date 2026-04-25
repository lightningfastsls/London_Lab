"""Render sanity-check spectrograms for 9 representative 9252 WAVs.

Produces one PNG per WAV with CNN-detected events overlaid, so a human
can eyeball (a) whether the events are real USVs and (b) whether the
spectrogram looks "healthy" (not broadband-garbled, not noise-dominated).

Files are picked to span the three tier categories:
    * 3 auto_accept (summary tier)
    * 3 manual_review (summary tier)
    * 3 no-summary (JSON events but no tier assignment — USV1-3, or
      uncovered USV4)

Run (from repo root):
    PYTHONPATH=src:. .venv/bin/python scripts/render_sanity_check_9252.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sig
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from usv_spectrogram.corpus import (
    SAMPLE_RATE_HZ,
    USV_FREQ_MIN_HZ,
    USV_FREQ_MAX_HZ,
    STFT_N_FFT,
    STFT_HOP,
)

OUTPUT_DIR = REPO_ROOT / "results/rate_anomaly_9252/sanity_check"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Three files per category, hand-picked from the merged CSV by
# cross-referencing summary.parquet tiers. Paths are absolute under
# USV_9252/ (filesystem walk confirmed they exist).
SANITY_FILES: list[tuple[str, str, Path]] = [
    ("auto_accept", "2024-10-06_14-58-31_0000121",
     REPO_ROOT / "USV_9252/USV6/experiment 9252 USVs/usv_lmt_036/2024-10-06_14-58-31_0000121.wav"),
    ("auto_accept", "2024-10-06_15-00-22_0000127",
     REPO_ROOT / "USV_9252/USV6/experiment 9252 USVs/usv_lmt_036/2024-10-06_15-00-22_0000127.wav"),
    ("auto_accept", "2024-10-06_15-19-18_0000215",
     REPO_ROOT / "USV_9252/USV6/experiment 9252 USVs/usv_lmt_036/2024-10-06_15-19-18_0000215.wav"),
    ("manual_review", "2024-10-06_15-03-22_0000144",
     REPO_ROOT / "USV_9252/USV6/experiment 9252 USVs/usv_lmt_036/2024-10-06_15-03-22_0000144.wav"),
    ("manual_review", "2024-10-06_15-07-36_0000160",
     REPO_ROOT / "USV_9252/USV6/experiment 9252 USVs/usv_lmt_036/2024-10-06_15-07-36_0000160.wav"),
    ("manual_review", "2024-10-06_15-07-48_0000162",
     REPO_ROOT / "USV_9252/USV6/experiment 9252 USVs/usv_lmt_036/2024-10-06_15-07-48_0000162.wav"),
    ("no_summary", "2024-10-06_15-54-05_0000377",
     REPO_ROOT / "USV_9252/USV1/experiment 9252 USVs/usv_lmt_036/2024-10-06_15-54-05_0000377.wav"),
    ("no_summary", "2024-10-07_09-13-48_0003015",
     REPO_ROOT / "USV_9252/USV4/experiment 9252 USVs/usv_lmt_036/2024-10-07_09-13-48_0003015.wav"),
    ("no_summary", "2024-10-07_09-28-12_0003024",
     REPO_ROOT / "USV_9252/USV4/experiment 9252 USVs/usv_lmt_036/2024-10-07_09-28-12_0003024.wav"),
]


def render(stem: str, tier: str, wav_path: Path, events: pd.DataFrame) -> Path:
    audio, sr = sf.read(wav_path)
    if sr != SAMPLE_RATE_HZ:
        print(f"[warn] {stem}: sample rate {sr} != {SAMPLE_RATE_HZ}")

    f, t, Sxx = sig.spectrogram(
        audio,
        fs=sr,
        nperseg=STFT_N_FFT,
        noverlap=STFT_N_FFT - STFT_HOP,
        scaling="spectrum",
        mode="magnitude",
    )
    # Crop to USV band
    band_mask = (f >= USV_FREQ_MIN_HZ) & (f <= USV_FREQ_MAX_HZ)
    f_crop = f[band_mask]
    Sxx_crop = Sxx[band_mask, :]
    Sxx_db = 20 * np.log10(Sxx_crop + 1e-10)

    fig, ax = plt.subplots(figsize=(12, 4))
    vmin = np.percentile(Sxx_db, 50)
    vmax = np.percentile(Sxx_db, 99.5)
    ax.imshow(
        Sxx_db,
        aspect="auto",
        origin="lower",
        extent=(t[0], t[-1], f_crop[0] / 1000, f_crop[-1] / 1000),
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    # Overlay detected events as red boxes
    for _, ev in events.iterrows():
        ax.axvspan(ev["start_time_s"], ev["end_time_s"], color="red", alpha=0.15)
        ax.plot(
            [ev["start_time_s"], ev["end_time_s"]],
            [USV_FREQ_MAX_HZ / 1000 - 2, USV_FREQ_MAX_HZ / 1000 - 2],
            color="red",
            lw=2,
        )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("freq (kHz)")
    duration = len(audio) / sr
    ax.set_title(
        f"[{tier}] {stem}  |  {len(events)} events  |  "
        f"dur={duration:.2f}s  |  sr={sr}Hz"
    )
    fig.tight_layout()
    out = OUTPUT_DIR / f"{tier}__{stem}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> int:
    ev_all = pd.read_csv(REPO_ROOT / "results/batch_9252/all_detections.csv")

    print(f"{'tier':<15} {'stem':<40} {'events':>6}  → file")
    print("-" * 90)
    for tier, stem, wav_path in SANITY_FILES:
        if not wav_path.exists():
            print(f"[skip] {stem}: WAV missing at {wav_path}")
            continue
        events = ev_all[ev_all["stem"] == stem]
        out = render(stem, tier, wav_path, events)
        print(f"{tier:<15} {stem:<40} {len(events):>6}  → {out.relative_to(REPO_ROOT)}")

    print(f"\n[ok] 9 sanity-check PNGs in {OUTPUT_DIR.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
