#!/usr/bin/env python3
"""Full-population spectral flatness sweep with ROC/PR curves.

Runs spectral flatness analysis against ALL auto-accept files (n≈1344)
plus the 10 known 1960-problem files. Produces:
  - Per-detection CSV with flatness metrics
  - ROC curves for mean_flatness, tonal_fraction, max_consecutive_tonal
  - Precision-recall curves for the same metrics
  - Combined 2D threshold analysis (mean_flatness × max_consecutive_tonal)

Usage:
    python scripts/sweep_spectral_flatness.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.app.core.audio_loader import AudioLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config (same as profile script)
# ---------------------------------------------------------------------------
BATCH_DIR = REPO_ROOT / "results" / "batch_5970"
WAV_SEARCH_DIRS = [
    REPO_ROOT / "5970",
    REPO_ROOT / "5970_reviewed",
    REPO_ROOT / "5970_manual_review",
    REPO_ROOT / "5970_manual_review_reviewed",
]

SR = 300_000
N_FFT = 512
FREQ_RES = SR / N_FFT
SPEC_FREQ_MIN = 20_000
SPEC_FREQ_MAX = 120_000

USV_BAND_MIN_HZ = 35_000
USV_BAND_MAX_HZ = 110_000
USV_BAND_ROW_START = round((USV_BAND_MIN_HZ - SPEC_FREQ_MIN) / FREQ_RES)
USV_BAND_ROW_END = round((USV_BAND_MAX_HZ - SPEC_FREQ_MIN) / FREQ_RES)

BROADBAND_DB_ABOVE_MEDIAN = 6.0
BROADBAND_FRAC_THRESHOLD = 0.60
TONAL_FLATNESS_THRESHOLD = 0.5

PROBLEM_STEMS = set([
    "2024-09-30_17-45-49_0001960",
    "2024-09-30_19-00-57_0002431",
    "2024-09-30_19-20-03_0002522",
    "2024-09-30_22-36-23_0003502",
    "2024-09-30_22-36-29_0003503",
    "2024-09-30_23-37-29_0003781",
    "2024-09-30_23-39-35_0003794",
    "2024-10-01_12-20-03_0005107",
    "2024-10-01_17-18-59_0005656",
    "2024-10-01_18-29-16_0006086",
])


def find_wav(stem: str) -> Optional[Path]:
    for search_dir in WAV_SEARCH_DIRS:
        if not search_dir.exists():
            continue
        matches = list(search_dir.rglob(f"{stem}.wav"))
        if matches:
            return matches[0]
    return None


def spectral_flatness(power_linear: np.ndarray) -> float:
    eps = 1e-10
    log_mean = np.mean(np.log(power_linear + eps))
    geo_mean = np.exp(log_mean)
    arith_mean = np.mean(power_linear)
    if arith_mean < eps:
        return 1.0
    return float(geo_mean / arith_mean)


def _max_consecutive_true(mask: np.ndarray) -> int:
    if len(mask) == 0:
        return 0
    best = 0
    current = 0
    for val in mask:
        if val:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def analyze_detection(
    spec_db: np.ndarray,
    start_col: int,
    end_col: int,
    file_median_db: float,
) -> dict:
    n_freq, n_time = spec_db.shape
    col_start = max(0, start_col)
    col_end = min(n_time, end_col)
    n_cols = col_end - col_start

    if n_cols <= 0:
        return {
            "n_total_columns": 0, "n_broadband_columns": 0,
            "n_tonal_columns": 0, "tonal_fraction": 0.0,
            "max_consecutive_tonal": 0, "mean_flatness": 1.0,
            "min_flatness": 1.0, "median_flatness": 1.0, "std_flatness": 0.0,
        }

    window = spec_db[:, col_start:col_end]
    usv_band = window[USV_BAND_ROW_START:USV_BAND_ROW_END, :]

    active_mask = window > (file_median_db + BROADBAND_DB_ABOVE_MEDIAN)
    frac_active_per_col = active_mask.mean(axis=0)
    broadband_mask = frac_active_per_col > BROADBAND_FRAC_THRESHOLD

    n_broadband = int(broadband_mask.sum())
    n_non_broadband = n_cols - n_broadband

    flatness_values = []
    tonal_mask = np.zeros(n_cols, dtype=bool)

    for i in range(n_cols):
        if broadband_mask[i]:
            continue
        col_db = usv_band[:, i]
        col_linear = 10.0 ** (col_db / 10.0)
        sf = spectral_flatness(col_linear)
        flatness_values.append(sf)
        if sf < TONAL_FLATNESS_THRESHOLD:
            tonal_mask[i] = True

    flatness_arr = np.array(flatness_values) if flatness_values else np.array([])
    max_consec_tonal = _max_consecutive_true(tonal_mask & ~broadband_mask)
    n_tonal = int(tonal_mask.sum())
    tonal_fraction = n_tonal / n_non_broadband if n_non_broadband > 0 else 0.0

    return {
        "n_total_columns": n_cols,
        "n_broadband_columns": n_broadband,
        "n_tonal_columns": n_tonal,
        "tonal_fraction": tonal_fraction,
        "max_consecutive_tonal": max_consec_tonal,
        "mean_flatness": float(flatness_arr.mean()) if len(flatness_arr) > 0 else 1.0,
        "min_flatness": float(flatness_arr.min()) if len(flatness_arr) > 0 else 1.0,
        "median_flatness": float(np.median(flatness_arr)) if len(flatness_arr) > 0 else 1.0,
        "std_flatness": float(flatness_arr.std()) if len(flatness_arr) > 0 else 0.0,
    }


def process_file(stem: str, label: str, loader: AudioLoader, detections_dir: Path) -> list[dict]:
    det_path = detections_dir / f"{stem}.json"
    if not det_path.exists():
        return []

    with open(det_path) as f:
        events = json.load(f)
    if not events:
        return []

    wav_path = find_wav(stem)
    if wav_path is None:
        return []

    try:
        audio_data = loader.load(wav_path)
        spec = audio_data.spectrogram_db
    except Exception as e:
        log.error("Spectrogram failed for %s: %s", stem, e)
        return []

    file_median = float(np.median(spec))

    rows = []
    for ev_idx, ev in enumerate(events):
        metrics = analyze_detection(spec, ev["start_col"], ev["end_col"], file_median)
        rows.append({
            "stem": stem,
            "label": label,
            "is_problem": 1 if label == "1960_problem" else 0,
            "event_idx": ev_idx,
            "start_col": ev["start_col"],
            "end_col": ev["end_col"],
            "max_probability": ev.get("max_probability", 0),
            "mean_probability": ev.get("mean_probability", 0),
            **{k: v for k, v in metrics.items()},
        })

    return rows


def compute_roc_pr(y_true: np.ndarray, scores: np.ndarray, higher_is_positive: bool):
    """Compute ROC and PR curves by threshold sweep.

    Args:
        y_true: Binary labels (1 = 1960_problem, 0 = good)
        scores: Metric values
        higher_is_positive: If True, higher score → more likely problem.
                           If False, lower score → more likely problem.

    Returns:
        dict with thresholds, tpr, fpr, precision, recall, f1, f2
    """
    if not higher_is_positive:
        scores = -scores

    sorted_scores = np.sort(np.unique(scores))
    # Add endpoints
    thresholds = np.concatenate([
        [sorted_scores[0] - 0.01],
        sorted_scores,
        [sorted_scores[-1] + 0.01],
    ])

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    tpr_list, fpr_list, prec_list, recall_list = [], [], [], []

    for t in thresholds:
        predicted_pos = scores >= t
        tp = (predicted_pos & (y_true == 1)).sum()
        fp = (predicted_pos & (y_true == 0)).sum()
        fn = (~predicted_pos & (y_true == 1)).sum()

        tpr = tp / n_pos if n_pos > 0 else 0
        fpr = fp / n_neg if n_neg > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tpr

        tpr_list.append(tpr)
        fpr_list.append(fpr)
        prec_list.append(prec)
        recall_list.append(recall)

    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)
    prec_arr = np.array(prec_list)
    recall_arr = np.array(recall_list)

    # F1 and F2 scores
    denom_f1 = prec_arr + recall_arr
    f1_arr = np.where(denom_f1 > 0, 2 * prec_arr * recall_arr / denom_f1, 0.0)
    beta2 = 4.0  # F2: beta^2 = 4
    denom_f2 = beta2 * prec_arr + recall_arr
    f2_arr = np.where(denom_f2 > 0, (1 + beta2) * prec_arr * recall_arr / denom_f2, 0.0)

    # Restore original thresholds for display
    if not higher_is_positive:
        thresholds = -thresholds

    # AUC (trapezoidal)
    # Sort by FPR for proper integration
    sort_idx = np.argsort(fpr_arr)
    auc = float(np.trapezoid(tpr_arr[sort_idx], fpr_arr[sort_idx]))

    return {
        "thresholds": thresholds,
        "tpr": tpr_arr,
        "fpr": fpr_arr,
        "precision": prec_arr,
        "recall": recall_arr,
        "f1": f1_arr,
        "f2": f2_arr,
        "auc": auc,
    }


def plot_curves(results: dict, metric_name: str, output_dir: Path,
                higher_is_positive: bool):
    """Plot ROC and PR curves side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ROC curve
    ax = axes[0]
    ax.plot(results["fpr"], results["tpr"], "b-", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(f"ROC Curve — {metric_name}\nAUC = {results['auc']:.4f}")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    # PR curve
    ax = axes[1]
    ax.plot(results["recall"], results["precision"], "r-", linewidth=2)
    ax.set_xlabel("Recall (TPR)")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {metric_name}")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    # F1/F2 vs threshold
    ax = axes[2]
    thresholds = results["thresholds"]
    ax.plot(thresholds, results["f1"], "g-", linewidth=2, label="F1")
    ax.plot(thresholds, results["f2"], "m-", linewidth=2, label="F2")

    # Mark best F1 and F2
    best_f1_idx = np.argmax(results["f1"])
    best_f2_idx = np.argmax(results["f2"])
    ax.axvline(thresholds[best_f1_idx], color="g", linestyle=":", alpha=0.5)
    ax.axvline(thresholds[best_f2_idx], color="m", linestyle=":", alpha=0.5)
    ax.set_xlabel(f"Threshold ({metric_name})")
    ax.set_ylabel("Score")
    ax.set_title(
        f"F1/F2 vs Threshold\n"
        f"Best F1={results['f1'][best_f1_idx]:.3f} @ {thresholds[best_f1_idx]:.4f}  |  "
        f"Best F2={results['f2'][best_f2_idx]:.3f} @ {thresholds[best_f2_idx]:.4f}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    safe_name = metric_name.replace(" ", "_").lower()
    fig.savefig(output_dir / f"curves_{safe_name}.png", dpi=120)
    plt.close(fig)


def plot_2d_threshold(df: pd.DataFrame, output_dir: Path):
    """Scatter plot of mean_flatness vs max_consecutive_tonal, colored by label."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7))

    good = df[df["is_problem"] == 0]
    prob = df[df["is_problem"] == 1]

    ax.scatter(good["mean_flatness"], good["max_consecutive_tonal"],
               c="steelblue", alpha=0.3, s=15, label=f"Good (n={len(good)})", zorder=2)
    ax.scatter(prob["mean_flatness"], prob["max_consecutive_tonal"],
               c="red", alpha=0.9, s=40, marker="X", label=f"1960 problem (n={len(prob)})", zorder=3)

    # Draw candidate threshold box
    ax.axvline(x=0.50, color="orange", linestyle="--", alpha=0.7, label="mean_flatness = 0.50")
    ax.axhline(y=15, color="green", linestyle="--", alpha=0.7, label="max_consec_tonal = 15")

    # Shade the "flag" quadrant (high flatness, low tonal)
    ax.axvspan(0.50, ax.get_xlim()[1] if ax.get_xlim()[1] > 0.50 else 0.70,
               ymin=0, ymax=15 / max(ax.get_ylim()[1], 20),
               alpha=0.1, color="red")

    ax.set_xlabel("Mean Spectral Flatness (higher = flatter / noisier)")
    ax.set_ylabel("Max Consecutive Tonal Columns (higher = more USV-like)")
    ax.set_title("2D Threshold Analysis: mean_flatness × max_consecutive_tonal\n"
                 "(per-detection level)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "scatter_2d_threshold.png", dpi=120)
    plt.close(fig)


def plot_distributions(df: pd.DataFrame, output_dir: Path):
    """Histogram distributions of key metrics, problem vs good."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("mean_flatness", "Mean Spectral Flatness", True),
        ("min_flatness", "Min Spectral Flatness", True),
        ("tonal_fraction", "Tonal Fraction", False),
        ("max_consecutive_tonal", "Max Consecutive Tonal Cols", False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, (col, title, higher_is_problem) in zip(axes.flat, metrics):
        good = df[df["is_problem"] == 0][col].values
        prob = df[df["is_problem"] == 1][col].values

        all_vals = np.concatenate([good, prob])
        if len(all_vals) == 0:
            continue
        bins = np.linspace(all_vals.min(), all_vals.max(), 50)

        ax.hist(good, bins=bins, alpha=0.6, color="steelblue", label=f"Good (n={len(good)})",
                density=True)
        ax.hist(prob, bins=bins, alpha=0.7, color="red", label=f"Problem (n={len(prob)})",
                density=True)
        ax.set_xlabel(title)
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Per-Detection Metric Distributions: 1960 Problem vs Good", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "distributions.png", dpi=120)
    plt.close(fig)


def print_operating_points(df: pd.DataFrame):
    """Print key operating points for decision-making."""
    y_true = df["is_problem"].values
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    log.info("\n" + "=" * 70)
    log.info("OPERATING POINT ANALYSIS (per-detection)")
    log.info("=" * 70)
    log.info("Total detections: %d (problem=%d, good=%d)", len(df), n_pos, n_neg)

    # Test specific thresholds for mean_flatness
    log.info("\n--- mean_flatness threshold sweep (flag if >= threshold) ---")
    for thresh in [0.45, 0.47, 0.49, 0.50, 0.51, 0.52, 0.53, 0.55]:
        flagged = df["mean_flatness"] >= thresh
        tp = (flagged & (y_true == 1)).sum()
        fp = (flagged & (y_true == 0)).sum()
        fn = (~flagged & (y_true == 1)).sum()
        recall = tp / n_pos if n_pos > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        fpr = fp / n_neg if n_neg > 0 else 0
        log.info("  thresh=%.2f  TPR=%.1f%% (%d/%d)  FPR=%.2f%% (%d/%d)  prec=%.1f%%",
                 thresh, recall * 100, tp, n_pos, fpr * 100, fp, n_neg, prec * 100)

    # Test specific thresholds for max_consecutive_tonal
    log.info("\n--- max_consecutive_tonal threshold sweep (flag if <= threshold) ---")
    for thresh in [5, 8, 10, 12, 15, 20, 25, 30]:
        flagged = df["max_consecutive_tonal"] <= thresh
        tp = (flagged & (y_true == 1)).sum()
        fp = (flagged & (y_true == 0)).sum()
        fn = (~flagged & (y_true == 1)).sum()
        recall = tp / n_pos if n_pos > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        fpr = fp / n_neg if n_neg > 0 else 0
        log.info("  thresh=%2d  TPR=%.1f%% (%d/%d)  FPR=%.2f%% (%d/%d)  prec=%.1f%%",
                 thresh, recall * 100, tp, n_pos, fpr * 100, fp, n_neg, prec * 100)

    # Test combo thresholds
    log.info("\n--- COMBO: mean_flatness >= X AND max_consecutive_tonal <= Y ---")
    for mf_thresh in [0.48, 0.49, 0.50, 0.51, 0.52]:
        for mc_thresh in [10, 12, 15, 20]:
            flagged = (df["mean_flatness"] >= mf_thresh) & (df["max_consecutive_tonal"] <= mc_thresh)
            tp = (flagged & (y_true == 1)).sum()
            fp = (flagged & (y_true == 0)).sum()
            recall = tp / n_pos if n_pos > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            fpr = fp / n_neg if n_neg > 0 else 0
            log.info("  mf>=%.2f & mc<=%2d  TPR=%.1f%% (%d/%d)  FPR=%.2f%% (%d/%d)  prec=%.1f%%",
                     mf_thresh, mc_thresh, recall * 100, tp, n_pos,
                     fpr * 100, fp, n_neg, prec * 100)


def main():
    detections_dir = BATCH_DIR / "detections"
    output_dir = BATCH_DIR / "spectral_flatness_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = AudioLoader()

    # -----------------------------------------------------------------------
    # Collect ALL auto-accept + problem files
    # -----------------------------------------------------------------------
    summary_path = BATCH_DIR / "summary_full.parquet"
    df_summary = pd.read_parquet(summary_path)
    auto_accept = df_summary[df_summary["tier"] == "auto_accept"]

    # Include all auto_accept files + problem files (which may be in manual_review tier)
    all_stems = []
    seen = set()
    # Add problem files first (regardless of tier)
    for stem in PROBLEM_STEMS:
        all_stems.append((stem, "1960_problem"))
        seen.add(stem)
    # Add all auto_accept files (excluding any already added as problem)
    for _, row in auto_accept.iterrows():
        stem = row["stem"]
        if stem not in seen:
            all_stems.append((stem, "auto_accept"))
            seen.add(stem)

    log.info("Total files to process: %d (%d problem, %d auto_accept)",
             len(all_stems),
             sum(1 for _, l in all_stems if l == "1960_problem"),
             sum(1 for _, l in all_stems if l == "auto_accept"))

    # -----------------------------------------------------------------------
    # Process all files
    # -----------------------------------------------------------------------
    all_rows = []
    t0 = time.time()
    n_skipped = 0

    for idx, (stem, label) in enumerate(all_stems, 1):
        if idx % 100 == 0 or idx == 1:
            elapsed = time.time() - t0
            rate = idx / elapsed if elapsed > 0 else 0
            eta = (len(all_stems) - idx) / rate if rate > 0 else 0
            log.info("[%d/%d] (%.1f files/s, ETA %.0fs) %s",
                     idx, len(all_stems), rate, eta, stem)

        rows = process_file(stem, label, loader, detections_dir)
        if rows:
            all_rows.extend(rows)
        else:
            n_skipped += 1

    elapsed = time.time() - t0
    log.info("Processed %d files in %.1fs (%.1f files/s), %d skipped",
             len(all_stems), elapsed, len(all_stems) / elapsed, n_skipped)

    if not all_rows:
        log.error("No results!")
        return 1

    result_df = pd.DataFrame(all_rows)
    csv_path = output_dir / "sweep_all_detections.csv"
    result_df.to_csv(csv_path, index=False)
    log.info("Saved %d detection rows to %s", len(result_df), csv_path)

    n_prob_det = result_df["is_problem"].sum()
    n_good_det = len(result_df) - n_prob_det
    log.info("Detections: %d problem, %d good", n_prob_det, n_good_det)

    # -----------------------------------------------------------------------
    # ROC / PR curves for individual metrics
    # -----------------------------------------------------------------------
    metrics_to_sweep = [
        ("mean_flatness", "Mean Flatness", True),      # higher = more problem-like
        ("min_flatness", "Min Flatness", True),         # higher = more problem-like
        ("tonal_fraction", "Tonal Fraction", False),    # lower = more problem-like
        ("max_consecutive_tonal", "Max Consecutive Tonal", False),  # lower = more problem-like
        ("median_flatness", "Median Flatness", True),
    ]

    y_true = result_df["is_problem"].values

    log.info("\n" + "=" * 70)
    log.info("ROC / PR CURVE ANALYSIS")
    log.info("=" * 70)

    for col, name, higher_is_pos in metrics_to_sweep:
        scores = result_df[col].values
        curves = compute_roc_pr(y_true, scores, higher_is_pos)
        plot_curves(curves, name, output_dir, higher_is_pos)

        best_f1_idx = np.argmax(curves["f1"])
        best_f2_idx = np.argmax(curves["f2"])
        log.info("  %-25s  AUC=%.4f  Best F1=%.3f @ %.4f  Best F2=%.3f @ %.4f",
                 name, curves["auc"],
                 curves["f1"][best_f1_idx], curves["thresholds"][best_f1_idx],
                 curves["f2"][best_f2_idx], curves["thresholds"][best_f2_idx])

    # -----------------------------------------------------------------------
    # Distribution plots
    # -----------------------------------------------------------------------
    plot_distributions(result_df, output_dir)

    # -----------------------------------------------------------------------
    # 2D scatter plot
    # -----------------------------------------------------------------------
    plot_2d_threshold(result_df, output_dir)

    # -----------------------------------------------------------------------
    # Operating point table
    # -----------------------------------------------------------------------
    print_operating_points(result_df)

    # -----------------------------------------------------------------------
    # File-level analysis (for catching all-events-in-file)
    # -----------------------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("FILE-LEVEL ANALYSIS")
    log.info("=" * 70)

    file_agg = result_df.groupby(["stem", "label", "is_problem"]).agg(
        n_events=("event_idx", "count"),
        file_min_flatness=("min_flatness", "min"),
        file_mean_flatness=("mean_flatness", "mean"),
        file_max_tonal_fraction=("tonal_fraction", "max"),
        file_max_consec_tonal=("max_consecutive_tonal", "max"),
    ).reset_index()

    file_agg_csv = output_dir / "sweep_file_summary.csv"
    file_agg.to_csv(file_agg_csv, index=False)

    # File-level operating points
    y_file = file_agg["is_problem"].values
    n_file_pos = y_file.sum()
    n_file_neg = len(y_file) - n_file_pos
    log.info("Files with detections: %d (problem=%d, good=%d)",
             len(file_agg), n_file_pos, n_file_neg)

    log.info("\n--- File-level: mean_flatness >= X AND max_consec_tonal <= Y ---")
    for mf_thresh in [0.48, 0.49, 0.50, 0.51, 0.52]:
        for mc_thresh in [10, 12, 15, 20]:
            flagged = ((file_agg["file_mean_flatness"] >= mf_thresh) &
                       (file_agg["file_max_consec_tonal"] <= mc_thresh))
            tp = (flagged & (y_file == 1)).sum()
            fp = (flagged & (y_file == 0)).sum()
            recall = tp / n_file_pos if n_file_pos > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            fpr = fp / n_file_neg if n_file_neg > 0 else 0
            log.info("  mf>=%.2f & mc<=%2d  TPR=%.1f%% (%d/%d)  FPR=%.2f%% (%d/%d)  prec=%.1f%%",
                     mf_thresh, mc_thresh, recall * 100, tp, n_file_pos,
                     fpr * 100, fp, n_file_neg, prec * 100)

    log.info("\nAll outputs saved to %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
