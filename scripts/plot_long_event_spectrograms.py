"""Render spectrograms of the longest-duration detections from the lab batch.

Each panel shows:
  - dB-scaled spectrogram of the chunk, USV band only
  - Detection time-window as a red dashed rectangle
  - Unmatched-tonal centers from the audit table as cyan horizontal lines

Goal: visually confirm whether long-duration detections are real USVs or
equipment tonals being misclassified by the CNN.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import stft

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from usv_spectrogram.corpus import (
    SAMPLE_RATE_HZ,
    STFT_HOP,
    STFT_N_FFT,
    USV_FREQ_MAX_HZ,
    USV_FREQ_MIN_HZ,
)

BATCH = Path("results/batch_lab_full_softnotch_20260513_1538")
WAV_DIR = Path("USV_lab_131204_chunked_2s_full")
OUT_DIR = BATCH / "long_event_inspection"
OUT_DIR.mkdir(exist_ok=True)

# Top 5 longest events from the duration scan (>500ms, max_prob >0.99).
TARGETS = [
    ("131208_1000_m1fm1_chunk_302", 593.07, 1.000),
    ("131209_1000_m4fm4_chunk_041", 588.80, 0.998),
    ("131209_1000_m3fm3_chunk_281", 588.80, 0.994),
    ("131209_1000_m6fm6_chunk_005", 588.80, 1.000),
    ("131209_1000_m3fm3_chunk_189", 584.53, 1.000),
]


def main() -> None:
    events = pd.read_parquet(BATCH / "all_events_with_unmatched_flag.parquet")

    for stem, expected_ms, expected_prob in TARGETS:
        wav_path = WAV_DIR / f"{stem}.wav"
        if not wav_path.exists():
            print(f"MISSING: {wav_path}")
            continue
        ev = events[events["stem"] == stem]
        ev = ev[abs(ev["duration_ms"] - expected_ms) < 1].iloc[0]
        out_path = OUT_DIR / f"{stem}_{int(expected_ms)}ms.png"
        _render(wav_path, ev, out_path)
        print(f"Wrote {out_path}")


def _render(wav_path: Path, ev: pd.Series, out_path: Path) -> None:
    sr, audio = wavfile.read(wav_path)
    assert sr == SAMPLE_RATE_HZ, f"Expected {SAMPLE_RATE_HZ} Hz, got {sr}"
    audio = audio.astype(np.float64)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    f, t, Z = stft(audio, fs=sr, nperseg=STFT_N_FFT, noverlap=STFT_N_FFT - STFT_HOP)
    psd_db = 20 * np.log10(np.abs(Z) + 1e-12)

    band = (f >= USV_FREQ_MIN_HZ) & (f <= USV_FREQ_MAX_HZ)
    f_band = f[band] / 1000  # kHz
    psd_band = psd_db[band, :]

    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(
        psd_band,
        origin="lower",
        aspect="auto",
        extent=(t[0], t[-1], f_band[0], f_band[-1]),
        cmap="magma",
        vmin=np.percentile(psd_band, 10),
        vmax=np.percentile(psd_band, 99.5),
    )

    rect = plt.Rectangle(
        (ev["start_s"], USV_FREQ_MIN_HZ / 1000),
        ev["end_s"] - ev["start_s"],
        (USV_FREQ_MAX_HZ - USV_FREQ_MIN_HZ) / 1000,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(rect)
    ax.text(
        ev["start_s"], USV_FREQ_MAX_HZ / 1000 - 3,
        f"CNN detection  {ev['duration_ms']:.0f} ms  max_p={ev['max_prob']:.3f}",
        color="red", fontsize=10, va="top",
    )

    for c_khz in ev["unmatched_centers_khz"]:
        ax.axhline(c_khz, color="cyan", lw=1.2, alpha=0.85, linestyle=":")
        ax.text(
            t[-1] - 0.05, c_khz, f"{c_khz:.1f} kHz",
            ha="right", va="center", color="cyan", fontsize=8,
            bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1),
        )

    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(USV_FREQ_MIN_HZ / 1000, USV_FREQ_MAX_HZ / 1000)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title(
        f"{out_path.stem}\n"
        f"Red dashed = CNN detection window. Cyan dotted = unmatched tonal centers (audit)."
    )
    plt.colorbar(im, ax=ax, label="PSD (dB)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
