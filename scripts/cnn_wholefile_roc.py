"""Whole-file ROC: run each CNN iteration over hand-labeled WAVs with the REAL
sliding-inference pipeline, score against human ground truth.

WHY this differs from scripts/make_cnn_metrics_slide.py: that one scored a frozen
1,829-window candidate set — same windows for every model — so it structurally
cannot show an old model firing on noise that isn't in the candidate set (AUC
plateaued at 0.989). This script sweeps each model across WHOLE recordings, so
noise the old models light up across the file becomes false positives and pulls
their AUC down. This is the real-world number.

Ground truth (scripts/build_manual_review_labels.py — Shachar's per-detection
labels on results/batch_5970/manual_review_all_detections.csv):
  - positives = sliding windows overlapping a hand-labeled USV event
  - negatives = EVERY sliding window from the 19 `all_noise` files (100% noise,
    zero ambiguity) — this is where "old models add noise detections" shows up

Inference matches production exactly: SlidingInference defaults (global MAD
normalization once, per-window norm OFF — see sliding_inference.py:147 and the
feedback_cnn_inference_global_mad lesson).

Outputs (presentation/figures/cnn_improvement/):
  cnn_wholefile_roc.png         — 3-model ROC overlay + FP@90%-recall bars
  _predictions/wholefile_window_scores.csv  — per-window (file,time,y,score,model)
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score

REPO = Path("/home/shachar/projects/mickey_london_lab")
sys.path.insert(0, str(REPO / "src"))
from usv_spectrogram.app.core.sliding_inference import SlidingInference  # noqa: E402
from usv_spectrogram.app.core.audio_loader import AudioLoader            # noqa: E402

MODELS = [
    ("production",       "Feb-2 original",               "#9e9e9e"),
    ("matched_windows",  "matched_windows (old)",        "#5c8fb3"),
    ("hard_neg_retrain", "hard-neg retrain (production)", "#26c6da"),
]
WAV_DIRS = [REPO / "5970_manual_review_reviewed", REPO / "5970"]  # 2nd searched recursively
TOL_S = 0.02  # window-center tolerance when testing overlap with a labeled USV event


def load_ground_truth():
    """Return (usv_intervals, noise_stems): per-stem list of (start_s,end_s) USV
    events, and the set of stems that are 100% noise."""
    spec = importlib.util.spec_from_file_location(
        "blab", REPO / "scripts" / "build_manual_review_labels.py")
    blab = importlib.util.module_from_spec(spec); spec.loader.exec_module(blab)
    df = pd.read_csv(REPO / "results" / "batch_5970" / "manual_review_all_detections.csv")
    df["suffix"] = df["stem"].str[-7:]

    usv_intervals, noise_stems, all_stems = {}, set(), {}
    for suf, (rule, data) in blab.ANNOTATIONS.items():
        if rule == "skip":
            continue
        sub = df[df["suffix"] == suf].sort_values("detection_idx")
        if sub.empty:
            continue
        stem = sub["stem"].iloc[0]
        all_stems[stem] = rule
        labels = blab.apply_labels(len(sub), rule, data)  # idx -> 'usv'/'noise'
        ivs = [(r["start_time_s"], r["end_time_s"])
               for i, (_, r) in enumerate(sub.iterrows()) if labels.get(i) == "usv"]
        if rule == "all_noise":
            noise_stems.add(stem)
        if ivs:
            usv_intervals[stem] = ivs
    return usv_intervals, noise_stems, all_stems


def resolve_wav(stem):
    p = WAV_DIRS[0] / f"{stem}.wav"
    if p.exists():
        return p
    hits = list((WAV_DIRS[1]).rglob(f"{stem}.wav"))
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=REPO / "presentation/figures/cnn_improvement/cnn_wholefile_roc.png")
    ap.add_argument("--scores-csv", type=Path,
                    default=REPO / "presentation/figures/cnn_improvement/_predictions/wholefile_window_scores.csv")
    a = ap.parse_args()

    usv_intervals, noise_stems, all_stems = load_ground_truth()
    print(f"[gt] {len(all_stems)} labeled files: {len(noise_stems)} all_noise, "
          f"{len(usv_intervals)} with USV events "
          f"({sum(len(v) for v in usv_intervals.values())} USV events total)")

    # resolve wavs once
    wavs = {stem: resolve_wav(stem) for stem in all_stems}
    missing = [s for s, p in wavs.items() if p is None]
    if missing:
        print(f"[warn] {len(missing)} WAVs unresolved: {missing}")

    # load each model once, compute the spectrogram once per file, score with all
    loaders = {tag: SlidingInference(str(REPO / "models" / tag / "best_model.pt"))
               for tag, _, _ in MODELS}
    audio_loader = AudioLoader()

    rows = []  # (stem, time_s, y_true, score, model)
    for stem, rule in sorted(all_stems.items()):
        wav = wavs[stem]
        if wav is None:
            continue
        audio = audio_loader.load(str(wav))
        ivs = usv_intervals.get(stem, [])
        for tag, _, _ in MODELS:
            res = loaders[tag].infer(audio.spectrogram_db, audio.times)
            for t, p in zip(res.times, res.probabilities):
                if stem in noise_stems:
                    y = 0                                   # whole-file noise -> negative
                elif any(s - TOL_S <= t <= e + TOL_S for s, e in ivs):
                    y = 1                                   # overlaps a hand-labeled USV
                else:
                    continue                                # ambiguous background -> skip
                rows.append((stem, float(t), y, float(p), tag))

    scores = pd.DataFrame(rows, columns=["stem", "time_s", "y_true", "score", "model"])
    a.scores_csv.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(a.scores_csv, index=False)
    print(f"[scores] wrote {len(scores)} rows -> {a.scores_csv}")

    # ---- metrics + figure ----
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    summary = []
    for tag, label, color in MODELS:
        d = scores[scores.model == tag]
        y, s = d.y_true.values, d.score.values
        npos, nneg = int(y.sum()), int((y == 0).sum())
        auc = roc_auc_score(y, s); pra = average_precision_score(y, s)
        fpr, tpr, thr = roc_curve(y, s)
        # FP rate on noise at the threshold giving >=90% recall
        idx = np.argmax(tpr >= 0.90)
        fpr_at90 = fpr[idx]
        lw = 2.6 if tag == "hard_neg_retrain" else 2.0
        axL.plot(fpr, tpr, color=color, lw=lw,
                 label=f"{label}  (AUC {auc:.3f})")
        summary.append((label, color, auc, pra, fpr_at90, npos, nneg))
        print(f"[{tag:16s}] AUC={auc:.4f} PR-AUC={pra:.4f} "
              f"FPR@90%recall={fpr_at90:.3f}  pos={npos} neg={nneg}")

    axL.plot([0, 1], [0, 1], ls=":", color="0.6", lw=1.2, label="chance")
    axL.set_xlim(0, 1); axL.set_ylim(0, 1.001)
    axL.set_xlabel("False-positive rate (noise windows flagged)", fontsize=10)
    axL.set_ylabel("True-positive rate (USV windows found)", fontsize=10)
    axL.set_title("A.  Whole-file ROC — real sliding pipeline on hand-labeled WAVs",
                  fontsize=12, fontweight="bold", loc="left")
    axL.legend(loc="lower right", fontsize=9, frameon=True)
    axL.grid(alpha=0.25)

    # Panel B: FP-on-noise at 90% recall (lower = better noise rejection)
    labels = [s[0] for s in summary]
    fp90 = [s[4] * 100 for s in summary]
    colors = [s[1] for s in summary]
    x = np.arange(len(labels))
    axR.bar(x, fp90, color=colors, width=0.6)
    for xi, v in zip(x, fp90):
        axR.annotate(f"{v:.1f}%", (xi, v), ha="center", va="bottom",
                     fontsize=10, fontweight="bold")
    axR.set_xticks(x)
    axR.set_xticklabels([l.replace(" (", "\n(") for l in labels], fontsize=9)
    axR.set_ylabel("% of noise windows flagged\n(at 90% USV recall)", fontsize=10)
    axR.set_title("B.  Noise lit up at matched recall — lower is better",
                  fontsize=12, fontweight="bold", loc="left")
    axR.grid(axis="y", alpha=0.25)

    npos = summary[0][5]; nneg = summary[0][6]
    fig.suptitle("CNN noise rejection on whole recordings (human-labeled ground truth)",
                 fontsize=15, fontweight="bold")
    fig.text(0.5, -0.02,
             f"Sliding inference over {len(all_stems)} hand-labeled WAVs "
             f"({len(noise_stems)} all-noise). Positives = {npos} windows on labeled "
             f"USV events; negatives = {nneg} windows from all-noise files. "
             "Unlike the frozen-candidate AUC, whole-file scoring exposes noise the "
             "old models flag — so their curves separate.",
             ha="center", va="top", fontsize=9, color="0.3")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
