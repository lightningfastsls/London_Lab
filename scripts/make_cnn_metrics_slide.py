"""Quantitative CNN-improvement slide — two panels that tell ONE honest story:

  Panel A  ROC overlay (all 3 model iterations on the SAME 1,829-row test set):
           the CNN's raw discrimination jumps Feb-2 -> matched_windows, then
           PLATEAUS matched_windows -> hard_neg (production). AUC can't see the
           final win because it isn't a discrimination change.

  Panel B  Operating-point improvement (matched_windows -> production): precision,
           specificity and false-positive count improve via a deliberate
           recall trade + downstream FP-filter — exactly what the AUC plateau
           hides. This is where "better and better" actually lives.

ALL numbers are RE-SCORED on one common test set so the curves are comparable
(scripts/evaluate_model.py on data/training/matched_windows/test.csv). The
Feb-2 model is genuinely out-of-distribution on this set (it was trained on
differently-extracted windows): its probability *ranking* survives (AUC ~0.96,
consistent with its 0.957 in-domain AUC) but its *calibration* collapses, which
is why its fixed-threshold operating point is degenerate here — and why a retrain
was needed. We therefore plot Feb-2 in the ROC panel (ranking is fair) but NOT in
the operating-point panel (its threshold is meaningless on this data).

Reproduce the inputs:
    for tag in production matched_windows hard_neg_retrain; do
      .venv/bin/python scripts/evaluate_model.py \
        --model models/$tag/best_model.pt \
        --test-csv data/training/matched_windows/test.csv \
        --output-dir /tmp/$tag --save-predictions <dest>/$tag.csv
    done
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score

# progression palette: faded -> hero (teal = current production, matches the
# companion composite slide where teal = the good/kept detections)
C_FEB2 = "#9e9e9e"   # gray  — original baseline
C_MW   = "#5c8fb3"   # steel — matched_windows (old)
C_HN   = "#26c6da"   # teal  — hard_neg_retrain (production hero)


def _auc_from_csv(path):
    df = pd.read_csv(path)
    y = (df.true_label == "USV").astype(int).values
    s = df.confidence.values
    return y, s, roc_auc_score(y, s), average_precision_score(y, s)


def panel_roc(ax, preds_dir, native_auc):
    """3-curve ROC overlay on the common test set."""
    order = [
        ("production",      "Feb-2 original",                 C_FEB2),
        ("matched_windows", "matched_windows (old)",          C_MW),
        ("hard_neg_retrain","hard-neg retrain (production)",   C_HN),
    ]
    for tag, label, color in order:
        y, s, auc, _ = _auc_from_csv(preds_dir / f"{tag}.csv")
        fpr, tpr, _ = roc_curve(y, s)
        lw = 2.6 if tag == "hard_neg_retrain" else 2.0
        ax.plot(fpr, tpr, color=color, lw=lw, label=f"{label}  (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], ls=":", color="0.6", lw=1.2, label="chance")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.001)
    ax.set_xlabel("False-positive rate", fontsize=10)
    ax.set_ylabel("True-positive rate (recall)", fontsize=10)
    ax.set_title("A.  CNN discrimination — same 1,829-window test set",
                 fontsize=12, fontweight="bold", loc="left")
    ax.legend(loc="lower right", fontsize=8.5, frameon=True)
    ax.grid(alpha=0.25)
    # honesty annotation for the OOD Feb-2 curve
    ax.annotate(
        f"Feb-2 is out-of-distribution here\n(diff. window extraction): ranking\nholds — AUC {native_auc:.3f} in-domain — but\nits calibration collapses, so its fixed\nthreshold is unusable (see panel B).",
        xy=(0.50, 0.32), xycoords="axes fraction", fontsize=7.3, color="0.35",
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="0.8", lw=0.8),
    )
    # plateau callout between the two retrains
    ax.annotate("retrain 1->2: AUC plateaus\n(0.989 vs 0.989)",
                xy=(0.06, 0.965), xytext=(0.18, 0.66), fontsize=7.6, color=C_HN,
                ha="left", va="center", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_HN, lw=1.2))


def panel_operating(ax, mw_metrics, hn_metrics):
    """Grouped bars: matched_windows -> production operating point."""
    keys = [("precision", "Precision"), ("recall", "Recall"),
            ("specificity", "Specificity"), ("f1", "F1")]
    old = [mw_metrics[k] * 100 for k, _ in keys]
    new = [hn_metrics[k] * 100 for k, _ in keys]
    x = np.arange(len(keys)); w = 0.38
    ax.bar(x - w/2, old, w, label="matched_windows (old)", color=C_MW)
    ax.bar(x + w/2, new, w, label="hard-neg retrain (production)", color=C_HN)
    for xi, (o, n) in enumerate(zip(old, new)):
        d = n - o
        ax.annotate(f"{o:.1f}", (xi - w/2, o + 0.4), ha="center", fontsize=7.5, color="0.3")
        col = "#2e7d32" if d >= 0 else "#c62828"
        ax.annotate(f"{n:.1f}\n({d:+.1f})", (xi + w/2, n + 0.4), ha="center",
                    fontsize=7.5, color=col, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([lbl for _, lbl in keys], fontsize=9.5)
    ax.set_ylim(80, 100)
    ax.set_ylabel("percent", fontsize=10)
    ax.set_title("B.  Production operating point — precision up, noise down",
                 fontsize=12, fontweight="bold", loc="left")
    ax.legend(loc="lower left", fontsize=8.5, frameon=True)
    ax.grid(axis="y", alpha=0.25)
    # FP / FN count callout — the headline the AUC plateau hides
    fp_o, fp_n = mw_metrics["false_positives"], hn_metrics["false_positives"]
    fn_o, fn_n = mw_metrics["false_negatives"], hn_metrics["false_negatives"]
    txt = (f"False positives:  {fp_o} -> {fp_n}  ({(fp_n-fp_o)/fp_o*100:+.0f}%)\n"
           f"False negatives:  {fn_o} -> {fn_n}  (recall traded by design)")
    ax.annotate(txt, xy=(0.5, 0.04), xycoords="axes fraction", ha="center",
                va="bottom", fontsize=8.2, color="0.2",
                bbox=dict(boxstyle="round,pad=0.5", fc="#fff8e1", ec="#e0c060", lw=0.9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir", type=Path,
                    default=Path("presentation/figures/cnn_improvement/_predictions"))
    ap.add_argument("--out", type=Path,
                    default=Path("presentation/figures/cnn_improvement/cnn_metrics_slide.png"))
    a = ap.parse_args()

    mw = json.loads((a.preds_dir / "matched_windows_metrics.json").read_text())
    hn = json.loads((a.preds_dir / "hard_neg_retrain_metrics.json").read_text())
    _, _, native_auc, _ = _auc_from_csv(a.preds_dir / "production_native.csv")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    panel_roc(axL, a.preds_dir, native_auc)
    panel_operating(axR, mw, hn)
    fig.suptitle("How the USV-detection CNN improved across iterations",
                 fontsize=15, fontweight="bold")
    fig.text(0.5, -0.02,
             "Left: raw classifier discrimination saturates after the first retrain.  "
             "Right: the production win is a precision / false-positive gain at the "
             "operating point — a deliberate recall trade + downstream noise filter — "
             "which the AUC plateau cannot show.",
             ha="center", va="top", fontsize=9, color="0.3")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
