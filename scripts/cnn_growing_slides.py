"""Six presentation slides for the CNN-iteration story.

  slides 1-3 : ONE per CNN — that CNN's detections on the WAV (standalone,
               same recording + same axes, only the boxes change).
  slides 4-6 : the whole-file ROC, GROWING — 1 curve, then 2, then all 3.

All 6 slides share one 16:9 widescreen canvas so they drop into a deck uniformly.

NOTHING is re-inferred. Sources, all already on disk:
  - ROC curves     : presentation/figures/cnn_improvement/_predictions/
                     wholefile_window_scores_nativefirst.csv  (cached window scores)
  - detection boxes: detection_review/detection_review_verdicts.csv
  - background     : the DEFAULT magma/20-120 spectrogram of the WAV, recomputed
                     once via AudioLoader (exactly what cnn_detection_review.py
                     paints behind every panel; only the boxes differ per model).
                     No CNN, no ExtractionConfig override.

Unified palette: each model's ROC curve color == its detection-box color, so the
audience reads "this curve == this panel" across the two slide groups.

Output: presentation/figures/cnn_improvement/growing_slides/
  slide_1_detections_first.png   slide_4_roc_first.png
  slide_2_detections_matched.png slide_5_roc_first+matched.png
  slide_3_detections_hardneg.png slide_6_roc_all3.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import roc_curve, roc_auc_score

REPO = Path("/home/shachar/projects/mickey_london_lab")
sys.path.insert(0, str(REPO / "src"))
from usv_spectrogram.app.core.audio_loader import AudioLoader  # noqa: E402
from usv_spectrogram.corpus import SAMPLE_RATE_HZ              # noqa: E402

SR = SAMPLE_RATE_HZ
STEM = "2024-10-01_08-58-52_0004954"
FIG_DIR = REPO / "presentation/figures/cnn_improvement"
SCORES_CSV = FIG_DIR / "_predictions/wholefile_window_scores_nativefirst.csv"
VERDICTS_CSV = FIG_DIR / "detection_review/detection_review_verdicts.csv"
OUTDIR = FIG_DIR / "growing_slides"
FIGSIZE = (13.33, 7.5)  # standard 16:9 widescreen

# Unified palette: each model's ROC curve == its detection-box color.
# (score_model, verdict_model, label, color, detect_tag)
MODELS = [
    ("production",       "first_cnn", "first CNN  (native inferno/25-110, 150px)", "#d62728"),
    ("matched_windows",  "matched",   "matched_windows  (app default)",           "#5c8fb3"),
    ("hard_neg_retrain", "hard_neg",  "hard_neg / production  (== the app)",       "#1a7d1a"),
]
DET_SLIDE = ["detections_first", "detections_matched", "detections_hardneg"]
ROC_SLIDE = ["roc_first", "roc_first+matched", "roc_all3"]
SHORT = ["First CNN", "Matched", "Hard-neg"]  # lay-friendly short names


def resolve_wav(stem):
    p = REPO / "5970_manual_review_reviewed" / f"{stem}.wav"
    if p.exists():
        return p
    hits = list((REPO / "5970").rglob(f"{stem}.wav"))
    return hits[0] if hits else None


def compute_roc():
    df = pd.read_csv(SCORES_CSV)
    out = {}
    for score_model, _, _, _ in MODELS:
        d = df[df.model == score_model]
        y, s = d.y_true.values, d.score.values
        fpr, tpr, _ = roc_curve(y, s)
        out[score_model] = (fpr, tpr, roc_auc_score(y, s),
                            int(y.sum()), int((y == 0).sum()))
    return out


def compute_spectrogram():
    wav = resolve_wav(STEM)
    if wav is None:
        raise FileNotFoundError(f"WAV for {STEM} not found")
    loader = AudioLoader()  # default magma / 20-120 kHz
    audio = loader.load(str(wav))
    samples, _ = librosa.load(str(wav), sr=SR, mono=True)
    bd_db, bd_freqs, _ = loader._compute_spectrogram(samples, SR)
    dur_s = float(audio.times[-1]) if len(audio.times) else len(samples) / SR
    return bd_db, bd_freqs[0] / 1e3, bd_freqs[-1] / 1e3, dur_s


def load_boxes():
    v = pd.read_csv(VERDICTS_CSV)
    v = v[v.stem == STEM]
    return {m: v[v.model == m] for _, m, _, _ in MODELS}


# ----------------------------------------------------------------------------- #
# Detection slides (one per CNN)                                                 #
# ----------------------------------------------------------------------------- #
def detection_slide(i, bd_db, fmin, fmax, dur_s, boxes):
    _, vmodel, _, color = MODELS[i]
    bx = boxes[vmodel]
    n = len(bx)
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes([0.07, 0.17, 0.89, 0.57])
    ax.imshow(bd_db, aspect="auto", origin="lower", cmap="magma",
              extent=[0, dur_s, fmin, fmax],
              vmin=np.percentile(bd_db, 5), vmax=np.percentile(bd_db, 99))
    for _, e in bx.iterrows():
        ax.add_patch(Rectangle((e.start_s, fmin), e.end_s - e.start_s, fmax - fmin,
                               fill=False, edgecolor=color, lw=2.4))
    ax.set_xlabel("time (seconds)", fontsize=12)
    ax.set_ylabel("frequency (kHz)", fontsize=12)
    ax.tick_params(labelsize=10)

    fig.text(0.07, 0.885, SHORT[i], fontsize=26, fontweight="bold", color=color)
    fig.text(0.07, 0.825,
             f"detector version {i+1} of 3    ·    {n} detection{'s' if n != 1 else ''}",
             fontsize=14, color="0.35")
    fig.text(0.07, 0.05,
             "Each box marks a sound the model flagged as a mouse call.    "
             "Same recording shown in all three slides.",
             fontsize=10.5, color="0.45")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"slide_{i+1}_{DET_SLIDE[i]}.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"[fig] detection slide {i+1} ({SHORT[i]}: {n} det) -> {out}")
    return out


# ----------------------------------------------------------------------------- #
# ROC growth slides                                                              #
# ----------------------------------------------------------------------------- #
def roc_slide(n, roc):
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes([0.08, 0.14, 0.45, 0.72])
    for i, (score_model, _, _, color) in enumerate(MODELS):
        if i >= n:
            continue
        fpr, tpr, auc, _, _ = roc[score_model]
        lw = 3.2 if i == 2 else 2.5
        ax.plot(fpr, tpr, color=color, lw=lw, label=f"{SHORT[i]}  (score {auc:.2f})")
    ax.plot([0, 1], [0, 1], ls=":", color="0.6", lw=1.2, label="random guessing")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.001)
    ax.set_xlabel("false alarms  →", fontsize=12)
    ax.set_ylabel("real calls caught  →", fontsize=12)
    ax.legend(loc="lower right", fontsize=11, frameon=True)
    ax.grid(alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.annotate("↖ better", xy=(0.03, 0.93), fontsize=12, color="0.4",
                fontstyle="italic", fontweight="bold")

    # right-hand minimal narration of the growth
    fig.text(0.60, 0.85, "Telling real calls\nfrom noise", fontsize=24,
             fontweight="bold", color="0.15")
    fig.text(0.60, 0.73, f"step {n} of 3", fontsize=13, color="0.45")
    y = 0.57
    for i, (score_model, _, _, color) in enumerate(MODELS):
        auc = roc[score_model][2]
        if i < n:
            fig.text(0.60, y, f"● {SHORT[i]}", fontsize=16, color=color,
                     fontweight="bold")
            fig.text(0.628, y - 0.046, f"score {auc:.2f}", fontsize=13, color="0.35")
        else:
            fig.text(0.60, y, f"○ {SHORT[i]}", fontsize=16, color="#c8c8c8",
                     fontweight="bold", fontstyle="italic")
            fig.text(0.628, y - 0.046, "—", fontsize=13, color="#c8c8c8")
        y -= 0.15
    fig.text(0.60, 0.08, "higher score = better   (1.00 = perfect)", fontsize=11,
             color="0.4", fontweight="bold")

    fig.text(0.08, 0.045, "Illustrative comparison on hand-labeled recordings.",
             fontsize=9, color="0.5")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"slide_{n+3}_{ROC_SLIDE[n-1]}.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"[fig] ROC slide {n+3} (step {n}/3) -> {out}")
    return out


def main():
    print("=" * 78)
    print("CNN SLIDES — 3 detection (one per CNN) + 3 ROC growth")
    print("=" * 78)
    roc = compute_roc()
    print("[roc] per-model AUC (cached window scores):")
    for sm, _, _, _ in MODELS:
        print(f"   {sm:18s} AUC={roc[sm][2]:.4f}")
    bd_db, fmin, fmax, dur_s = compute_spectrogram()
    print(f"[spec] {STEM}: {bd_db.shape} band {fmin:.0f}-{fmax:.0f} kHz dur {dur_s:.2f}s")
    boxes = load_boxes()
    print("[boxes] per-model counts:", {m: len(b) for m, b in boxes.items()})

    # remove superseded combined slides from the prior version
    for old in ("slide_1_first.png", "slide_2_first+matched.png", "slide_3_all3.png"):
        p = OUTDIR / old
        if p.exists():
            p.unlink(); print(f"[clean] removed superseded {old}")

    outs = [detection_slide(i, bd_db, fmin, fmax, dur_s, boxes) for i in range(3)]
    outs += [roc_slide(n, roc) for n in (1, 2, 3)]
    print(f"[done] {len(outs)} slides in {OUTDIR}")


if __name__ == "__main__":
    main()
