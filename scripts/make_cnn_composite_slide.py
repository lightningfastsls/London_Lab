"""2x2 composite slide: 'the CNN pipeline learned to reject noise while keeping
real USVs', built ENTIRELY from human-confirmed labels (no eyeball judgment).

Layout:
                Noise-only recording        USV recording
                (human label: all_noise)    (human label: all_usv)
  Old model     boxes on broadband noise    boxes on real USVs
  Production    0 boxes (noise rejected)     same boxes (USVs kept)

Column files come from scripts/build_manual_review_labels.py ground truth:
  0005656 -> ("all_noise")  |  0004783 -> ("all_usv")

Noise-file detections are drawn orange (false positives); USV-file detections
teal (true positives). Spectrogram + event loading reuse the sibling module so
the canonical STFT params (corpus.py) are shared and never redeclared.
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from make_cnn_progression_slide import (
    load_events, STFT_N_FFT, STFT_HOP, USV_FREQ_MIN_HZ, USV_FREQ_MAX_HZ,
    SAMPLE_RATE_HZ,
)

FP_C = "#ff7043"   # orange: false positive (noise)
TP_C = "#26c6da"   # teal: true positive (USV)


def spectrogram_median_subtracted(wav):
    """Per-frequency-bin median-subtracted dB spectrogram — makes tonal USVs pop
    above the noise floor (the ref=np.max display drowns them under loud
    broadband transients). Returns (B, freqs_khz, times)."""
    y, sr = sf.read(str(wav))
    if y.ndim > 1:
        y = y[:, 0]
    assert sr == SAMPLE_RATE_HZ, f"WAV sr={sr}, expected {SAMPLE_RATE_HZ}"
    S = np.abs(librosa.stft(y.astype(np.float32), n_fft=STFT_N_FFT, hop_length=STFT_HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=STFT_N_FFT)
    band = (freqs >= USV_FREQ_MIN_HZ) & (freqs <= USV_FREQ_MAX_HZ)
    Sdb = librosa.amplitude_to_db(S[band], ref=1.0)
    B = Sdb - np.median(Sdb, axis=1, keepdims=True)
    return B, freqs[band] / 1000.0, np.arange(S.shape[1]) * STFT_HOP / sr


def _draw(ax, wav, ev_path, color, fk0, fk1):
    B, freqs_khz, times = spectrogram_median_subtracted(wav)
    ax.imshow(
        B, origin="lower", aspect="auto",
        extent=[times[0], times[-1], freqs_khz[0], freqs_khz[-1]],
        cmap="magma", vmin=0, vmax=np.percentile(B, 99.5),
    )
    events = load_events(ev_path)
    for e in events:
        ax.add_patch(Rectangle(
            (e["start_time_s"], freqs_khz[0]),
            e["end_time_s"] - e["start_time_s"], freqs_khz[-1] - freqs_khz[0],
            fill=False, edgecolor=color, linewidth=2.0,
        ))
    ax.set_ylim(freqs_khz[0], freqs_khz[-1])
    ax.tick_params(labelsize=8)
    return len(events)


def render(noise, usv, out_path, title=None, caption=None):
    """noise/usv: dict with keys wav, old, prod."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.4), constrained_layout=True)
    cols = [
        ("Noise-only recording\n(human label: all_noise)", noise, FP_C),
        ("USV recording\n(human label: all_usv)", usv, TP_C),
    ]
    rows = [("Old model\n(matched_windows)", "old"),
            ("Production\n(hard-neg retrain + noise filter)", "prod")]

    # per-column zoom window: union of that file's old-stage events (present in
    # both rows), padded — so USVs/noise fill the panel instead of being tiny.
    xlims = []
    for _, spec, _ in cols:
        ev = load_events(spec["old"])
        if ev:
            xlims.append((max(0, min(e["start_time_s"] for e in ev) - 0.15),
                          max(e["end_time_s"] for e in ev) + 0.15))
        else:
            xlims.append(None)

    for r, (row_label, key) in enumerate(rows):
        for c, (col_label, spec, color) in enumerate(cols):
            ax = axes[r][c]
            n = _draw(ax, spec["wav"], spec[key], color, None, None)
            if xlims[c]:
                ax.set_xlim(*xlims[c])
            verdict = ("false alarm" if color == FP_C else "real USV")
            tag = f"{n} detection{'s' if n != 1 else ''}"
            if n:
                tag += f"  ({verdict}{'s' if n != 1 else ''})"
            ax.set_title(tag, fontsize=11, color=color if n else "0.4",
                         fontweight="bold", loc="left")
            if r == 0:
                ax.annotate(col_label, xy=(0.5, 1.28), xycoords="axes fraction",
                            ha="center", va="bottom", fontsize=12, fontweight="bold")
            if c == 0:
                ax.set_ylabel(row_label + "\n\nkHz", fontsize=10, fontweight="bold")
            else:
                ax.set_ylabel("kHz", fontsize=9)
            if r == 1:
                ax.set_xlabel("Time (s)", fontsize=9)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    if caption:
        fig.text(0.5, -0.015, caption, ha="center", va="top", fontsize=9.5,
                 color="0.2")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise-wav", type=Path, required=True)
    ap.add_argument("--noise-old", type=Path, required=True)
    ap.add_argument("--noise-prod", type=Path, required=True)
    ap.add_argument("--usv-wav", type=Path, required=True)
    ap.add_argument("--usv-old", type=Path, required=True)
    ap.add_argument("--usv-prod", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--caption", default=None)
    a = ap.parse_args()
    render(
        {"wav": a.noise_wav, "old": a.noise_old, "prod": a.noise_prod},
        {"wav": a.usv_wav, "old": a.usv_old, "prod": a.usv_prod},
        a.out, title=a.title, caption=a.caption,
    )


if __name__ == "__main__":
    main()
