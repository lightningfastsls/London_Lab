"""ACTUALLY calculate the first CNN's whole-file ROC — faithful ablation.

Finding (git-verified): the first CNN (models/production, 101K) was trained on
inferno/25-110 with PER-CANDIDATE MAD (median-2*MAD .. median+4*MAD) -> colormap
-> grayscale -> per-image min/max (SpectrogramExtractor training render +
USVDataset). It had NO faithful in-app inference pipeline (both historical app
versions mismatched training: 3cb00a97 fed raw dB w/ no colormap; d3cee3f9 added
global-MAD + magma). So the only faithful reference is the TRAINING render.

Two legitimate normalization framings for a whole-file sliding ROC:
  F1 per-window MAD  -> matches per-candidate training (each window self-normalized).
                        Expected to flood on noise (per-candidate design has no
                        whole-file noise notion) -> honest, likely lower AUC.
  F2 global MAD once -> the modern inference scheme matched/hard_neg use; only the
                        first CNN's colormap/band differ. Generous to the 1st CNN.

Ablation (first CNN, inferno/25-110 fixed): {per_window, global} MAD x {100,150}px.
matched_windows + hard_neg reference = their production inference (global MAD,
100px, magma/20-120) from wholefile_window_scores_nativefirst.csv.

Authorized legacy override (first CNN only): ExtractionConfig(25_000,110_000,
colormap='inferno'). Defaults / corpus.py untouched (CNN-FREEZE).
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score

REPO = Path("/home/shachar/projects/mickey_london_lab")
sys.path.insert(0, str(REPO / "src"))
from usv_spectrogram.app.core.sliding_inference import SlidingInference  # noqa: E402
from usv_spectrogram.app.core.audio_loader import AudioLoader            # noqa: E402
from usv_spectrogram.detection.extraction_config import ExtractionConfig  # noqa: E402
from usv_spectrogram.corpus import SAMPLE_RATE_HZ                         # noqa: E402

_spec = importlib.util.spec_from_file_location("wfroc", REPO / "scripts" / "cnn_wholefile_roc.py")
wfroc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wfroc)

SR = SAMPLE_RATE_HZ
HOP = 10
ENERGY_THRESH = 0.1
LEG = ExtractionConfig(freq_min_hz=25_000, freq_max_hz=110_000, colormap="inferno")
# MAD scales consumed from the config (not redeclared) — the training-render formula.
MAD_VMIN_SCALE = LEG.mad_vmin_scale
MAD_VMAX_SCALE = LEG.mad_vmax_scale
EXISTING = REPO / "presentation/figures/cnn_improvement/_predictions/wholefile_window_scores_nativefirst.csv"
OUT_FIG = REPO / "presentation/figures/cnn_improvement/cnn_first_roc_faithful.png"
OUT_CSV = REPO / "presentation/figures/cnn_improvement/_predictions/first_roc_faithful_ablation.csv"


def per_window_mad(crop):
    """Training-render MAD on a single window's dB crop -> [0,1]."""
    med = np.median(crop)
    mad = np.median(np.abs(crop - med))
    vmin = med - MAD_VMIN_SCALE * mad
    vmax = med + MAD_VMAX_SCALE * mad
    w = np.clip(crop, vmin, vmax)
    return (w - vmin) / (vmax - vmin + 1e-12)


def score_first_cnn(si, spec_db_native, times, width, norm_mode):
    """Score the first CNN over one file's native (25-110) spectrogram.
    norm_mode: 'per_window' (faithful-to-training) or 'global' (modern)."""
    n_freqs, n_times = spec_db_native.shape
    if n_times < width:
        return [], []
    global_norm = si._apply_mad_normalization(spec_db_native) if norm_mode == "global" else None
    half = width // 2
    centers = list(range(half, n_times - half, HOP))
    win_times, win_probs = [], []
    orig_get_cmap = plt.get_cmap
    plt.get_cmap = lambda *a, **k: orig_get_cmap("inferno")
    try:
        for c in centers:
            sl = slice(c - half, c - half + width)
            if norm_mode == "per_window":
                w = per_window_mad(spec_db_native[:, sl])
            else:
                w = global_norm[:, sl]
            win_times.append(float(times[c]))
            if float(np.max(w)) < ENERGY_THRESH:      # same energy gate as infer()
                win_probs.append(0.0)
                continue
            t = si._prepare_batch([w])
            with torch.no_grad():
                p = float(si.model.predict_proba(t.to(si.device)).cpu().numpy().flatten()[0])
            win_probs.append(p)
    finally:
        plt.get_cmap = orig_get_cmap
    return win_times, win_probs


def main():
    print("=" * 78)
    print("FIRST CNN whole-file ROC — faithful ablation")
    print(f"[params] inferno/25000-110000 fixed; MAD scales vmin={MAD_VMIN_SCALE} vmax={MAD_VMAX_SCALE}; "
          f"hop={HOP}; energy_gate={ENERGY_THRESH}; SR={SR}")
    print(f"[params] variants: norm in (per_window[F1-faithful], global[F2-fair]) x width in (100,150)")

    usv_intervals, noise_stems, all_stems = wfroc.load_ground_truth()
    TOL = wfroc.TOL_S
    wavs = {s: wfroc.resolve_wav(s) for s in all_stems}
    print(f"[gt] {len(all_stems)} files, {len(noise_stems)} all-noise, "
          f"{sum(len(v) for v in usv_intervals.values())} USV events; TOL={TOL}s")

    si_first = SlidingInference(str(REPO / "models/production/best_model.pt"))
    native_loader = AudioLoader(config=LEG)

    variants = [("per_window", 100), ("per_window", 150), ("global", 100), ("global", 150)]
    rows = []
    # cache native spectrograms per file
    native_cache = {}
    for stem in sorted(all_stems):
        wav = wavs[stem]
        if wav is None:
            continue
        samples, _ = librosa.load(str(wav), sr=SR, mono=True)
        sd, _, t = native_loader._compute_spectrogram(samples, SR)
        native_cache[stem] = (sd, t)

    def y_for(stem, tt):
        if stem in noise_stems:
            return 0
        ivs = usv_intervals.get(stem, [])
        return 1 if any(s - TOL <= tt <= e + TOL for s, e in ivs) else None

    for norm_mode, width in variants:
        for stem, (sd, t) in native_cache.items():
            wt, wp = score_first_cnn(si_first, sd, t, width, norm_mode)
            for tt, pp in zip(wt, wp):
                y = y_for(stem, tt)
                if y is None:
                    continue
                rows.append((norm_mode, width, stem, tt, y, pp))
        print(f"  [scored] norm={norm_mode:10s} width={width}  done")

    df = pd.DataFrame(rows, columns=["norm", "width", "stem", "time_s", "y_true", "score"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    # reference: matched / hard_neg from the existing native-first run
    ref = pd.read_csv(EXISTING)
    ref_auc = {}
    for m in ("matched_windows", "hard_neg_retrain"):
        d = ref[ref.model == m]
        ref_auc[m] = roc_auc_score(d.y_true, d.score)

    print("\n[results] first-CNN whole-file ROC-AUC by variant:")
    print(f"  {'norm':12s} {'width':>5s}  {'AUC':>7s} {'PR-AUC':>7s} {'FPR@90':>7s}  "
          f"{'pos':>5s} {'neg':>6s}  score[max/mean]")
    summary = []
    for norm_mode, width in variants:
        d = df[(df.norm == norm_mode) & (df.width == width)]
        y, s = d.y_true.values, d.score.values
        auc = roc_auc_score(y, s)
        pra = average_precision_score(y, s)
        fpr, tpr, _ = roc_curve(y, s)
        fpr90 = fpr[int(np.argmax(tpr >= 0.90))]
        summary.append((norm_mode, width, auc, pra, fpr90))
        print(f"  {norm_mode:12s} {width:5d}  {auc:7.4f} {pra:7.4f} {fpr90:7.3f}  "
              f"{int(y.sum()):5d} {int((y==0).sum()):6d}  {s.max():.3f}/{s.mean():.4f}")
    print(f"\n  reference (production inference, global MAD/100px/magma):")
    print(f"    matched_windows AUC={ref_auc['matched_windows']:.4f}   "
          f"hard_neg_retrain AUC={ref_auc['hard_neg_retrain']:.4f}")

    # figure: ROC overlay — both first-CNN framings (150px) vs the two later models
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    for norm_mode, width, color, ls in [("per_window", 150, "#d62728", "-"),
                                        ("global", 150, "#9e9e9e", "--")]:
        d = df[(df.norm == norm_mode) & (df.width == width)]
        fpr, tpr, _ = roc_curve(d.y_true, d.score)
        auc = roc_auc_score(d.y_true, d.score)
        lbl = ("first CNN — FAITHFUL (per-window MAD)" if norm_mode == "per_window"
               else "first CNN — fair (global MAD)")
        ax.plot(fpr, tpr, color=color, lw=2.2, ls=ls, label=f"{lbl}  (AUC {auc:.3f})")
    for m, color, lbl in [("matched_windows", "#5c8fb3", "matched_windows"),
                          ("hard_neg_retrain", "#26c6da", "hard-neg (production)")]:
        d = ref[ref.model == m]
        fpr, tpr, _ = roc_curve(d.y_true, d.score)
        ax.plot(fpr, tpr, color=color, lw=2.4, label=f"{lbl}  (AUC {ref_auc[m]:.3f})")
    ax.plot([0, 1], [0, 1], ls=":", color="0.6", lw=1.2, label="chance")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.001)
    ax.set_xlabel("False-positive rate (noise windows flagged)")
    ax.set_ylabel("True-positive rate (USV windows found)")
    ax.set_title("First CNN's whole-file ROC, computed faithfully\n"
                 "(per-window MAD = its true training normalization)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.text(0.5, -0.02,
             "First CNN trained on per-candidate MAD (inferno/25-110). FAITHFUL = per-window MAD "
             "(self-normalizes each window -> floods on noise). 'fair' = global MAD, the scheme "
             "matched/hard_neg use. 33 hand-labeled WAVs, 150px window.",
             ha="center", va="top", fontsize=8.5, color="0.3", wrap=True)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[fig] wrote {OUT_FIG}")
    print(f"[csv] wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
