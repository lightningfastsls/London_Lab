"""Render a CNN-pipeline progression slide: one spectrogram, stacked panels,
each overlaying a detection stage's event windows.

The story this slide tells is *false-positive / broadband-noise suppression*
across pipeline stages — NOT "catching more USVs". Recall intentionally dropped
~4.6% going to production; the win is precision +3.35% and FP -32%
(see docs/handoffs/v2-full-pipeline-results.md).

Detection events are TIME windows (full-band, energy-based), so each is drawn as
a translucent vertical span over the shared spectrogram, labelled with its
max CNN probability.

Canonical STFT params are imported from corpus.py — never redeclared (ADR-002).

Usage (vetting, 2 panels):
    python scripts/make_cnn_progression_slide.py \
        --wav 5970/USV5/usv_lmt_034/2024-09-30_11-22-17_0000053.wav \
        --stage "Original (Feb-2)=results/batch_5970/detections/2024-09-30_11-22-17_0000053.json" \
        --stage "Production=results/batch_5970_v2_full/detections/2024-09-30_11-22-17_0000053.json" \
        --out /tmp/draft_0000053.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from usv_spectrogram.corpus import (  # noqa: E402
    SAMPLE_RATE_HZ,
    STFT_N_FFT,
    USV_FREQ_MIN_HZ,
    USV_FREQ_MAX_HZ,
)

# corpus exposes STFT_HOP as a module constant; import defensively
try:
    from usv_spectrogram.corpus import STFT_HOP  # noqa: E402
except ImportError:  # pragma: no cover
    STFT_HOP = 128


def load_events(path: Path) -> list[dict]:
    d = json.loads(Path(path).read_text())
    ev = d if isinstance(d, list) else (d.get("detections") or d.get("events") or [])
    return sorted(ev, key=lambda e: e["start_time_s"])


def compute_spectrogram(wav_path: Path):
    """Return (S_db, freqs_khz, times_s) cropped to the USV band, sr asserted."""
    y, sr = sf.read(str(wav_path))
    if y.ndim > 1:
        y = y[:, 0]
    assert sr == SAMPLE_RATE_HZ, f"WAV sr={sr}, expected {SAMPLE_RATE_HZ}"
    y = y.astype(np.float32)
    S = np.abs(librosa.stft(y, n_fft=STFT_N_FFT, hop_length=STFT_HOP))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=STFT_N_FFT)
    band = (freqs >= USV_FREQ_MIN_HZ) & (freqs <= USV_FREQ_MAX_HZ)
    S_db = S_db[band, :]
    freqs_khz = freqs[band] / 1000.0
    times = np.arange(S_db.shape[1]) * STFT_HOP / sr
    return S_db, freqs_khz, times


def _overlaps(e, ref_events, tol=0.06):
    """True if event e time-overlaps any event in ref_events (within tol)."""
    for r in ref_events:
        if e["start_time_s"] <= r["end_time_s"] + tol and \
           e["end_time_s"] >= r["start_time_s"] - tol:
            return True
    return False


# colour-blind-safe: teal = retained by production, orange = removed by production
KEPT_C = "#26c6da"
DROP_C = "#ff7043"


def render(wav_path, stages, out_path, caption=None, title=None, kept_ref=None):
    """kept_ref: path to the production-stage JSON. Events that survive to it are
    drawn teal (retained real USVs); events absent from it are drawn orange
    (rejected by the production pipeline). If None, all events are teal."""
    S_db, freqs_khz, times = compute_spectrogram(wav_path)
    extent = [times[0], times[-1], freqs_khz[0], freqs_khz[-1]]
    n = len(stages)
    ref_events = load_events(kept_ref) if kept_ref else None

    fig, axes = plt.subplots(
        n, 1, figsize=(11, 2.5 * n + 1.4), sharex=True, constrained_layout=True
    )
    if n == 1:
        axes = [axes]

    vmin = np.percentile(S_db, 5)
    for ax, (label, ev_path) in zip(axes, stages):
        ax.imshow(
            S_db, origin="lower", aspect="auto", extent=extent,
            cmap="magma", vmin=vmin, vmax=0,
        )
        events = load_events(ev_path)
        n_drop = 0
        for e in events:
            p = e.get("max_probability", e.get("mean_probability", 0.0))
            kept = True if ref_events is None else _overlaps(e, ref_events)
            color = KEPT_C if kept else DROP_C
            if not kept:
                n_drop += 1
            ax.add_patch(Rectangle(
                (e["start_time_s"], freqs_khz[0]),
                e["end_time_s"] - e["start_time_s"],
                freqs_khz[-1] - freqs_khz[0],
                fill=False, edgecolor=color, linewidth=2.0,
            ))
            ax.text(
                (e["start_time_s"] + e["end_time_s"]) / 2, freqs_khz[-1] * 0.96,
                f"{p:.2f}", color=color, fontsize=7, ha="center", va="top",
                fontweight="bold",
            )
        ax.set_ylabel("kHz", fontsize=9)
        subtitle = f"{len(events)} detection{'s' if len(events) != 1 else ''}"
        if ref_events is not None and n_drop:
            subtitle += f"  ({n_drop} rejected by production)"
        ax.set_title(f"{label}   —   {subtitle}", fontsize=11, loc="left")
        ax.set_ylim(freqs_khz[0], freqs_khz[-1])
    axes[-1].set_xlabel("Time (s)", fontsize=9)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    if caption:
        fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=9, color="0.2")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}  ({n} panels, {wav_path.name})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=Path, required=True)
    ap.add_argument(
        "--stage", action="append", required=True,
        help='Repeatable. Format "Label=path/to/detections.json"',
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--caption", default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument(
        "--kept-ref", type=Path, default=None,
        help="Production-stage JSON; events absent from it are drawn as rejected.",
    )
    args = ap.parse_args()

    stages = []
    for s in args.stage:
        label, _, path = s.partition("=")
        stages.append((label.strip(), Path(path.strip())))
    render(args.wav, stages, args.out, caption=args.caption, title=args.title,
           kept_ref=args.kept_ref)


if __name__ == "__main__":
    main()
