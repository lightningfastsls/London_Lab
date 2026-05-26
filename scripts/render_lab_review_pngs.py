"""Render per-event spectrogram PNGs from the lab full-corpus merge.

Two output sets, both stratified-sampled to keep the PNG count manageable:

1. ``borderline/``  — events with ``max_probability < auto_accept_min_peak``
   (the actual calls that pushed their chunks into manual_review).
2. ``drag_along/`` — events with ``max_probability >= auto_accept_min_peak``
   that nonetheless live in ``tier='manual_review'`` chunks (CNN-confident
   on their own merit, but flagged because a sibling event in the same chunk
   was borderline).

Each PNG shows the parent 2 s chunk's spectrogram (300 kHz, 20-120 kHz)
with the event highlighted as a translucent cyan band. Title carries the
original-file timing and CNN confidence so the figure is self-describing.

Inputs:
    results/batch_lab_131204_full/merged_events_full.parquet
    USV_lab_131204_chunked_2s_full/<chunk_stem>.wav

Outputs:
    results/batch_lab_131204_full/review_pngs/borderline/*.png
    results/batch_lab_131204_full/review_pngs/drag_along/*.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import spectrogram

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from usv_spectrogram.corpus import (  # noqa: E402
    SAMPLE_RATE_HZ,
    STFT_HOP,
    STFT_N_FFT,
    USV_FREQ_MAX_HZ,
    USV_FREQ_MIN_HZ,
)
from usv_spectrogram.postprocessing.triage import TriageConfig  # noqa: E402

_TIER = TriageConfig()


def compute_spectrogram_db(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (freqs_hz, times_s, spec_db) using canonical STFT params."""
    f, t, S = spectrogram(
        audio.astype(np.float32),
        fs=SAMPLE_RATE_HZ,
        nperseg=STFT_N_FFT,
        noverlap=STFT_N_FFT - STFT_HOP,
        nfft=STFT_N_FFT,
        scaling="spectrum",
        mode="magnitude",
    )
    eps = 1e-12
    spec_db = 20.0 * np.log10(S + eps)
    band = (f >= USV_FREQ_MIN_HZ) & (f <= USV_FREQ_MAX_HZ)
    return f[band], t, spec_db[band]


def render_event_png(
    chunk_wav: Path,
    event: pd.Series,
    out_path: Path,
    role: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sr, audio = wavfile.read(chunk_wav)
    if sr != SAMPLE_RATE_HZ:
        raise ValueError(
            f"{chunk_wav}: sample rate {sr} != canonical {SAMPLE_RATE_HZ}"
        )
    if audio.ndim > 1:
        audio = audio[:, 0]

    freqs_hz, times_s, spec_db = compute_spectrogram_db(audio)

    fig, ax = plt.subplots(figsize=(11, 4), dpi=140)
    ax.imshow(
        spec_db,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[times_s[0], times_s[-1], freqs_hz[0] / 1000.0, freqs_hz[-1] / 1000.0],
        vmin=np.percentile(spec_db, 5),
        vmax=np.percentile(spec_db, 99),
    )

    ax.axvspan(
        event["start_time_s_in_chunk"],
        event["end_time_s_in_chunk"],
        alpha=0.30,
        color="cyan",
        linewidth=0,
    )
    ax.axvline(event["start_time_s_in_chunk"], color="cyan", linewidth=0.5, alpha=0.8)
    ax.axvline(event["end_time_s_in_chunk"], color="cyan", linewidth=0.5, alpha=0.8)

    title = (
        f"[{role}]  {event['original_filename']}  "
        f"orig_t={event['original_begin_time_s']:.3f}s  "
        f"dur={event['duration_s'] * 1000:.1f}ms  "
        f"max_p={event['max_probability']:.3f}  mean_p={event['mean_probability']:.3f}\n"
        f"chunk={event['chunk_stem']}  det_idx={event['chunk_detection_idx']}  "
        f"tier={event['tier']}"
    )
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Time in chunk (s)")
    ax.set_ylabel("Freq (kHz)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def stratified_borderline_sample(events: pd.DataFrame, n_total: int) -> pd.DataFrame:
    bord = events[events["max_probability"] < _TIER.auto_accept_min_peak].sort_values(
        "max_probability"
    )
    if len(bord) <= n_total:
        return bord.copy()

    # Three strata: lowest (most ambiguous), middle band (around 0.85), top (just-under-0.90)
    n_low = max(1, n_total // 3)
    n_mid = max(1, n_total // 3)
    n_high = n_total - n_low - n_mid

    low = bord.head(n_low)
    high = bord.tail(n_high)
    mid_pool = bord.iloc[n_low : len(bord) - n_high]
    mid = mid_pool.sample(n=min(n_mid, len(mid_pool)), random_state=0)

    return pd.concat([low, mid, high]).drop_duplicates("chunk_stem")


def drag_along_sample(events: pd.DataFrame, n_total: int) -> pd.DataFrame:
    drag = events[
        (events["tier"] == "manual_review")
        & (events["max_probability"] >= _TIER.auto_accept_min_peak)
    ]
    if len(drag) <= n_total:
        return drag.copy()
    return drag.sample(n=n_total, random_state=0)


def add_chunk_local_times(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    events["start_time_s_in_chunk"] = (
        events["original_begin_time_s"] - events["start_s_in_original"]
    )
    events["end_time_s_in_chunk"] = (
        events["original_end_time_s"] - events["start_s_in_original"]
    )
    return events


def render_set(
    sampled: pd.DataFrame,
    out_dir: Path,
    chunks_dir: Path,
    role: str,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_done = 0
    for _, ev in sampled.iterrows():
        chunk_wav = chunks_dir / f"{ev['chunk_stem']}.wav"
        if not chunk_wav.exists():
            print(f"[skip] missing chunk WAV: {chunk_wav}")
            continue
        out_name = (
            f"p{ev['max_probability']:.3f}__{ev['chunk_stem']}"
            f"__det{int(ev['chunk_detection_idx']):02d}.png"
        )
        render_event_png(chunk_wav, ev, out_dir / out_name, role)
        n_done += 1
    return n_done


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--events",
        type=Path,
        default=REPO_ROOT / "results/batch_lab_131204_full/merged_events_full.parquet",
    )
    ap.add_argument(
        "--chunks-dir",
        type=Path,
        default=REPO_ROOT / "USV_lab_131204_chunked_2s_full",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=REPO_ROOT / "results/batch_lab_131204_full/review_pngs",
    )
    ap.add_argument("--n-borderline", type=int, default=24)
    ap.add_argument("--n-dragalong", type=int, default=24)
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    events = pd.read_parquet(args.events)
    print(f"[load] events: {len(events)} rows")
    print(f"[params] auto_accept_min_peak (canonical): {_TIER.auto_accept_min_peak}")

    events = add_chunk_local_times(events)

    bord = stratified_borderline_sample(events, args.n_borderline)
    drag = drag_along_sample(events, args.n_dragalong)

    print(f"[sample] borderline: {len(bord)} events")
    print(
        f"  max_prob range: {bord['max_probability'].min():.3f}"
        f" .. {bord['max_probability'].max():.3f}"
    )
    print(f"[sample] drag-along: {len(drag)} events")
    print(
        f"  max_prob range: {drag['max_probability'].min():.3f}"
        f" .. {drag['max_probability'].max():.3f}"
    )

    n_b = render_set(bord, args.out_root / "borderline", args.chunks_dir, "BORDERLINE")
    n_d = render_set(drag, args.out_root / "drag_along", args.chunks_dir, "DRAG-ALONG")

    print(f"\n[done] {n_b} borderline PNGs in {args.out_root / 'borderline'}")
    print(f"[done] {n_d} drag-along PNGs in {args.out_root / 'drag_along'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
