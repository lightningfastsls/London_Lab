"""Whole-file ROC — 3 CNN iterations, with the FIRST CNN scored under its
NATIVE inferno/25-110 extraction (the "resurrection recipe") instead of the
silent default magma/20-120.

WHY this exists: scripts/cnn_wholefile_roc.py fed ALL three models through the
default pipeline (magma/20-120). The first CNN (models/production, trained on
inferno/25-110) goes SILENT there — its per-window scores collapse to ~0, so the
0.863 ROC-AUC it shows in cnn_wholefile_roc.png is a FEEDING ARTIFACT, not the
model's true discrimination. This script feeds the first CNN its native
extraction (so its probs are faithful) while keeping matched_windows + hard_neg
on default magma/20-120. Same 33 stems, same ground-truth scheme, SAME
SlidingInference window geometry for all three -> apples-to-apples overlay.

ILLUSTRATIVE figure (no scientific weight): the first CNN's native footing is a
PARTIAL reconstruction (colormap + band fixed; its 4 other historic-pipeline
differences are not). ROC is rank-based, so this is a fair discrimination
estimate, but it is not a re-creation of the model's original operating point.

Authorized per docs/handoffs/2026-05-27_cnn-iteration-comparison-followup.md:
legacy inferno/25-110 passed EXPLICITLY for the first CNN only; ExtractionConfig
defaults and corpus.py are UNTOUCHED (CNN-FREEZE).

Reuses the validated ground-truth + WAV-resolution logic from
scripts/cnn_wholefile_roc.py (imported, not duplicated).

Outputs (presentation/figures/cnn_improvement/):
  cnn_wholefile_roc_nativefirst.png
  _predictions/wholefile_window_scores_nativefirst.csv
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score

REPO = Path("/home/shachar/projects/mickey_london_lab")
sys.path.insert(0, str(REPO / "src"))
from usv_spectrogram.app.core.sliding_inference import SlidingInference  # noqa: E402
from usv_spectrogram.app.core.audio_loader import AudioLoader            # noqa: E402
from usv_spectrogram.detection.extraction_config import ExtractionConfig  # noqa: E402
from usv_spectrogram.corpus import SAMPLE_RATE_HZ  # noqa: E402  (canonical sr; do not redeclare)

# Reuse the validated GT + WAV-resolution from the original whole-file ROC script.
_spec = importlib.util.spec_from_file_location(
    "wfroc", REPO / "scripts" / "cnn_wholefile_roc.py")
wfroc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wfroc)

SR = SAMPLE_RATE_HZ  # canonical 300 kHz from corpus.py (not redeclared)
TOL_S = wfroc.TOL_S
# Legacy first-CNN extraction (the only place we deviate from defaults).
LEG = ExtractionConfig(freq_min_hz=25_000, freq_max_hz=110_000, colormap="inferno")

# (tag, label, color, native_extraction?)
MODELS = [
    ("production",       "first CNN — native inferno/25-110", "#9e9e9e", True),
    ("matched_windows",  "matched_windows — magma/20-120",    "#5c8fb3", False),
    ("hard_neg_retrain", "hard-neg retrain (production)",      "#26c6da", False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=REPO / "presentation/figures/cnn_improvement/cnn_wholefile_roc_nativefirst.png")
    ap.add_argument("--scores-csv", type=Path,
                    default=REPO / "presentation/figures/cnn_improvement/_predictions/wholefile_window_scores_nativefirst.csv")
    a = ap.parse_args()

    print("=" * 78)
    print("WHOLE-FILE ROC — first CNN on NATIVE extraction")
    print("=" * 78)
    print(f"[params] SlidingInference DEFAULTS for all 3 models: "
          f"window_width_px=100 hop_px=10 batch_size=32 energy_threshold=0.1 "
          f"per_window_norm=OFF (global MAD once)")
    print(f"[params] first-CNN extraction: inferno colormap, "
          f"band {LEG.freq_min_hz}-{LEG.freq_max_hz} Hz (passed EXPLICITLY; defaults untouched)")
    print(f"[params] matched/hard_neg extraction: magma, 20000-120000 Hz (defaults)")
    print(f"[params] GT overlap tolerance TOL_S={TOL_S}s; ROC onset/threshold = none (rank-based AUC)")

    usv_intervals, noise_stems, all_stems = wfroc.load_ground_truth()
    print(f"[gt] {len(all_stems)} labeled files: {len(noise_stems)} all_noise, "
          f"{len(usv_intervals)} with USV events "
          f"({sum(len(v) for v in usv_intervals.values())} USV events total)")

    wavs = {stem: wfroc.resolve_wav(stem) for stem in all_stems}
    missing = [s for s, p in wavs.items() if p is None]
    if missing:
        print(f"[warn] {len(missing)} WAVs unresolved: {missing}")

    loaders = {tag: SlidingInference(str(REPO / "models" / tag / "best_model.pt"))
               for tag, _, _, _ in MODELS}
    default_loader = AudioLoader()         # magma / 20-120 (defaults)
    native_loader = AudioLoader(config=LEG)  # inferno / 25-110

    rows = []  # (stem, time_s, y_true, score, model)
    n_files_scored = 0
    for stem, rule in sorted(all_stems.items()):
        wav = wavs[stem]
        if wav is None:
            continue
        n_files_scored += 1
        ivs = usv_intervals.get(stem, [])

        # default-extraction spectrogram (matched_windows + hard_neg)
        audio = default_loader.load(str(wav))

        # native-extraction spectrogram (first CNN only) — same n_fft/hop, so the
        # time grid matches; only the frequency band differs.
        samples, _ = librosa.load(str(wav), sr=SR, mono=True)
        sd_native, _, t_native = native_loader._compute_spectrogram(samples, SR)

        def label_window(t):
            if stem in noise_stems:
                return 0
            if any(s - TOL_S <= t <= e + TOL_S for s, e in ivs):
                return 1
            return None  # ambiguous background -> skip

        for tag, _, _, native in MODELS:
            if native:
                orig_get_cmap = plt.get_cmap
                plt.get_cmap = lambda *args, **kw: orig_get_cmap("inferno")
                try:
                    res = loaders[tag].infer(sd_native, t_native)
                finally:
                    plt.get_cmap = orig_get_cmap
            else:
                res = loaders[tag].infer(audio.spectrogram_db, audio.times)

            for t, p in zip(res.times, res.probabilities):
                y = label_window(float(t))
                if y is None:
                    continue
                rows.append((stem, float(t), y, float(p), tag))

    scores = pd.DataFrame(rows, columns=["stem", "time_s", "y_true", "score", "model"])
    a.scores_csv.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(a.scores_csv, index=False)
    print(f"[scores] {n_files_scored} files scored; wrote {len(scores)} rows -> {a.scores_csv}")

    # ---- metrics + figure ----
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    summary = []
    print("\n[metrics] per-model whole-file discrimination:")
    for tag, label, color, native in MODELS:
        d = scores[scores.model == tag]
        y, s = d.y_true.values, d.score.values
        npos, nneg = int(y.sum()), int((y == 0).sum())
        auc = roc_auc_score(y, s)
        pra = average_precision_score(y, s)
        fpr, tpr, thr = roc_curve(y, s)
        idx = int(np.argmax(tpr >= 0.90))
        fpr_at90 = fpr[idx]
        lw = 2.6 if tag == "hard_neg_retrain" else 2.0
        axL.plot(fpr, tpr, color=color, lw=lw, label=f"{label}  (AUC {auc:.3f})")
        summary.append((label, color, auc, pra, fpr_at90, npos, nneg))
        smin, smax, smean = float(s.min()), float(s.max()), float(s.mean())
        print(f"  [{tag:16s}] AUC={auc:.4f} PR-AUC={pra:.4f} FPR@90%recall={fpr_at90:.3f} "
              f"pos={npos} neg={nneg}  score[min={smin:.4f} max={smax:.4f} mean={smean:.4f}]")

    axL.plot([0, 1], [0, 1], ls=":", color="0.6", lw=1.2, label="chance")
    axL.set_xlim(0, 1)
    axL.set_ylim(0, 1.001)
    axL.set_xlabel("False-positive rate (noise windows flagged)", fontsize=10)
    axL.set_ylabel("True-positive rate (USV windows found)", fontsize=10)
    axL.set_title("A.  Whole-file ROC — real sliding pipeline on hand-labeled WAVs",
                  fontsize=12, fontweight="bold", loc="left")
    axL.legend(loc="lower right", fontsize=9, frameon=True)
    axL.grid(alpha=0.25)

    labels = [s[0] for s in summary]
    fp90 = [s[4] * 100 for s in summary]
    colors = [s[1] for s in summary]
    x = np.arange(len(labels))
    axR.bar(x, fp90, color=colors, width=0.6)
    for xi, v in zip(x, fp90):
        axR.annotate(f"{v:.1f}%", (xi, v), ha="center", va="bottom",
                     fontsize=10, fontweight="bold")
    axR.set_xticks(x)
    axR.set_xticklabels([l.replace(" — ", "\n").replace(" (", "\n(") for l in labels], fontsize=8)
    axR.set_ylabel("% of noise windows flagged\n(at 90% USV recall)", fontsize=10)
    axR.set_title("B.  Noise lit up at matched recall — lower is better",
                  fontsize=12, fontweight="bold", loc="left")
    axR.grid(axis="y", alpha=0.25)

    npos = summary[0][5]
    nneg = summary[0][6]
    fig.suptitle("CNN iterations got better — whole-file noise rejection (illustrative)",
                 fontsize=15, fontweight="bold")
    fig.text(0.5, -0.04,
             f"ILLUSTRATIVE (no scientific weight). Sliding inference over {n_files_scored} "
             f"hand-labeled WAVs ({len(noise_stems)} all-noise). Positives = {npos} windows on "
             f"labeled USV events; negatives = {nneg} windows from all-noise files. The first CNN "
             "is scored on its NATIVE inferno/25-110 extraction (a partial reconstruction: colormap "
             "+ band only) so it is not silenced by the default magma feeding; matched_windows and "
             "hard-neg use the production magma/20-120 pipeline.",
             ha="center", va="top", fontsize=8.5, color="0.3", wrap=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[fig] wrote {a.out}")


if __name__ == "__main__":
    main()
