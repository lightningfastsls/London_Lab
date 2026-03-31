#!/usr/bin/env python3
"""Test whether extending detection windows improves spectral flatness separation.

Re-runs spectral flatness analysis on the 18 known-noise files + 48 confirmed-USV
files with various window extensions past end_col. Tests whether the detection
windows are systematically ending too early (before the USV tonal content).

Usage:
    python scripts/test_window_extension.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.app.core.audio_loader import AudioLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

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
USV_BAND_MIN_HZ = 35_000
USV_BAND_MAX_HZ = 110_000
USV_BAND_ROW_START = round((USV_BAND_MIN_HZ - SPEC_FREQ_MIN) / FREQ_RES)
USV_BAND_ROW_END = round((USV_BAND_MAX_HZ - SPEC_FREQ_MIN) / FREQ_RES)

BROADBAND_DB_ABOVE_MEDIAN = 6.0
BROADBAND_FRAC_THRESHOLD = 0.60
TONAL_FLATNESS_THRESHOLD = 0.5

# Extensions to test (in spectrogram columns)
# hop=128, sr=300000 → 1 column ≈ 0.427 ms
# 25 cols ≈ 10.7 ms, 50 cols ≈ 21.3 ms, 100 cols ≈ 42.7 ms, 200 cols ≈ 85.3 ms
EXTENSIONS = [0, 25, 50, 100, 150, 200]

# Original 10 problem files
ORIGINAL_NOISE = [
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
]

# 8 newly identified noise files
NEW_NOISE_SUFFIXES = ["0000570", "0000716", "0000717", "0003579",
                      "0003825", "0004706", "0005108", "0005647"]

# 48 confirmed USV files (flagged at 0.53 but confirmed real)
# We'll derive these from the sweep results


def find_wav(stem: str):
    for d in WAV_SEARCH_DIRS:
        if not d.exists():
            continue
        m = list(d.rglob(f"{stem}.wav"))
        if m:
            return m[0]
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
    best = current = 0
    for val in mask:
        if val:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def analyze_detection_extended(
    spec_db: np.ndarray, start_col: int, end_col: int,
    file_median_db: float, extension: int,
) -> dict:
    """Analyze detection with window extended by `extension` columns past end_col."""
    n_freq, n_time = spec_db.shape
    col_start = max(0, start_col)
    col_end = min(n_time, end_col + extension)
    n_cols = col_end - col_start

    if n_cols <= 0:
        return {"mean_flatness": 1.0, "min_flatness": 1.0,
                "tonal_fraction": 0.0, "max_consecutive_tonal": 0,
                "n_total_columns": 0, "n_broadband_columns": 0}

    window = spec_db[:, col_start:col_end]
    usv_band = window[USV_BAND_ROW_START:USV_BAND_ROW_END, :]

    active_mask = window > (file_median_db + BROADBAND_DB_ABOVE_MEDIAN)
    frac_active = active_mask.mean(axis=0)
    broadband_mask = frac_active > BROADBAND_FRAC_THRESHOLD

    n_broadband = int(broadband_mask.sum())
    n_non_broadband = n_cols - n_broadband

    flatness_values = []
    tonal_mask = np.zeros(n_cols, dtype=bool)

    for i in range(n_cols):
        if broadband_mask[i]:
            continue
        col_linear = 10.0 ** (usv_band[:, i] / 10.0)
        sf = spectral_flatness(col_linear)
        flatness_values.append(sf)
        if sf < TONAL_FLATNESS_THRESHOLD:
            tonal_mask[i] = True

    flatness_arr = np.array(flatness_values) if flatness_values else np.array([])
    n_tonal = int(tonal_mask.sum())

    return {
        "mean_flatness": float(flatness_arr.mean()) if len(flatness_arr) > 0 else 1.0,
        "min_flatness": float(flatness_arr.min()) if len(flatness_arr) > 0 else 1.0,
        "tonal_fraction": n_tonal / n_non_broadband if n_non_broadband > 0 else 0.0,
        "max_consecutive_tonal": _max_consecutive_true(tonal_mask & ~broadband_mask),
        "n_total_columns": n_cols,
        "n_broadband_columns": n_broadband,
    }


def main():
    detections_dir = BATCH_DIR / "detections"
    output_dir = BATCH_DIR / "window_extension_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build file list with labels
    sweep_df = pd.read_csv(BATCH_DIR / "spectral_flatness_sweep" / "sweep_file_summary.csv")
    sweep_df["short"] = sweep_df["stem"].str[-7:]

    all_noise_stems = set(ORIGINAL_NOISE)
    for _, row in sweep_df.iterrows():
        if row["short"] in NEW_NOISE_SUFFIXES and row["is_problem"] == 0:
            all_noise_stems.add(row["stem"])

    # Confirmed USVs: flagged at 0.53 but not noise
    flagged_usv_stems = set()
    for _, row in sweep_df.iterrows():
        if (row["is_problem"] == 0
            and row["file_mean_flatness"] >= 0.53
            and row["stem"] not in all_noise_stems):
            flagged_usv_stems.add(row["stem"])

    files = []
    for s in all_noise_stems:
        files.append((s, "noise"))
    for s in flagged_usv_stems:
        files.append((s, "real_usv"))

    log.info("Files: %d noise, %d real_usv", len(all_noise_stems), len(flagged_usv_stems))

    loader = AudioLoader()
    all_rows = []

    for idx, (stem, label) in enumerate(files, 1):
        if idx % 10 == 0:
            log.info("[%d/%d] %s", idx, len(files), stem)

        det_path = detections_dir / f"{stem}.json"
        if not det_path.exists():
            continue
        with open(det_path) as f:
            events = json.load(f)
        if not events:
            continue

        wav_path = find_wav(stem)
        if wav_path is None:
            log.warning("WAV not found: %s", stem)
            continue

        try:
            audio_data = loader.load(wav_path)
            spec = audio_data.spectrogram_db
        except Exception as e:
            log.error("Failed %s: %s", stem, e)
            continue

        file_median = float(np.median(spec))

        for ext in EXTENSIONS:
            # Per-file: aggregate across all detections
            file_metrics = []
            for ev in events:
                m = analyze_detection_extended(
                    spec, ev["start_col"], ev["end_col"], file_median, ext)
                file_metrics.append(m)

            # File-level aggregation
            mean_flats = [m["mean_flatness"] for m in file_metrics]
            min_flats = [m["min_flatness"] for m in file_metrics]
            tonal_fracs = [m["tonal_fraction"] for m in file_metrics]
            consec_tonals = [m["max_consecutive_tonal"] for m in file_metrics]

            all_rows.append({
                "stem": stem,
                "label": label,
                "extension": ext,
                "ext_ms": ext * 128 / 300000 * 1000,
                "file_mean_flatness": np.mean(mean_flats),
                "file_min_flatness": min(min_flats),
                "file_max_tonal_fraction": max(tonal_fracs),
                "file_max_consec_tonal": max(consec_tonals),
                "n_events": len(events),
            })

    result_df = pd.DataFrame(all_rows)
    csv_path = output_dir / "window_extension_results.csv"
    result_df.to_csv(csv_path, index=False)

    # -----------------------------------------------------------------------
    # Print comparison table
    # -----------------------------------------------------------------------
    log.info("\n" + "=" * 80)
    log.info("WINDOW EXTENSION COMPARISON")
    log.info("=" * 80)

    for ext in EXTENSIONS:
        ext_ms = ext * 128 / 300000 * 1000
        sub = result_df[result_df["extension"] == ext]
        noise = sub[sub["label"] == "noise"]
        usv = sub[sub["label"] == "real_usv"]

        log.info("\n--- Extension: +%d cols (%.1f ms) ---", ext, ext_ms)
        log.info("  mean_flatness   noise=[%.4f, %.4f]  usv=[%.4f, %.4f]",
                 noise["file_mean_flatness"].min(), noise["file_mean_flatness"].max(),
                 usv["file_mean_flatness"].min(), usv["file_mean_flatness"].max())
        log.info("  max_consec_ton  noise=[%d, %d]  usv=[%d, %d]",
                 noise["file_max_consec_tonal"].min(), noise["file_max_consec_tonal"].max(),
                 usv["file_max_consec_tonal"].min(), usv["file_max_consec_tonal"].max())

        # Test combo thresholds
        for mf_t in [0.50, 0.53]:
            for mc_t in [6, 10, 15]:
                noise_caught = ((noise["file_mean_flatness"] >= mf_t) &
                                (noise["file_max_consec_tonal"] <= mc_t)).sum()
                usv_flagged = ((usv["file_mean_flatness"] >= mf_t) &
                               (usv["file_max_consec_tonal"] <= mc_t)).sum()
                log.info("    mf>=%.2f & mc<=%2d: catches %2d/%d noise, flags %2d/%d USVs",
                         mf_t, mc_t, noise_caught, len(noise), usv_flagged, len(usv))

    # -----------------------------------------------------------------------
    # Plot: mean_flatness and max_consec_tonal vs extension
    # -----------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for label, color, marker in [("noise", "red", "X"), ("real_usv", "steelblue", "o")]:
        sub = result_df[result_df["label"] == label]
        for stem in sub["stem"].unique():
            stem_data = sub[sub["stem"] == stem].sort_values("extension")
            alpha = 0.8 if label == "noise" else 0.3
            lw = 1.5 if label == "noise" else 0.7
            axes[0].plot(stem_data["ext_ms"], stem_data["file_mean_flatness"],
                        color=color, alpha=alpha, linewidth=lw)
            axes[1].plot(stem_data["ext_ms"], stem_data["file_max_consec_tonal"],
                        color=color, alpha=alpha, linewidth=lw)

    # Legend with dummy lines
    for ax in axes:
        ax.plot([], [], color="red", linewidth=2, label="Noise (n=18)")
        ax.plot([], [], color="steelblue", linewidth=2, label="Real USV (n=48)")
        ax.legend()
        ax.set_xlabel("Window Extension (ms)")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("File Mean Flatness")
    axes[0].set_title("Mean Spectral Flatness vs Window Extension")
    axes[1].set_ylabel("File Max Consecutive Tonal Cols")
    axes[1].set_title("Max Consecutive Tonal vs Window Extension")

    fig.suptitle("Effect of Extending Detection Windows Past end_col", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "extension_comparison.png", dpi=120)
    plt.close(fig)

    log.info("\nPlot saved to %s", output_dir / "extension_comparison.png")
    log.info("CSV saved to %s", csv_path)


if __name__ == "__main__":
    main()
