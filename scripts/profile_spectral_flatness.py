#!/usr/bin/env python3
"""Profile spectral flatness to detect the "1960 pattern" false positives.

Phase 1: Discovery/profiling script. Computes per-column spectral flatness
inside detection windows for known-problem files vs. known-good auto-accept
files. Outputs a CSV + diagnostic PNGs to evaluate whether spectral flatness
separates the two groups.

Key idea: Real USVs have narrow-band ridges (low spectral flatness in 35-110 kHz).
The 1960 pattern has diffuse broadband energy (high spectral flatness).
Spectral flatness = geometric_mean / arithmetic_mean of power spectrum.
  - 0 = all energy in one bin (pure tone)
  - 1 = energy uniformly spread (flat noise)

Usage:
    python scripts/profile_spectral_flatness.py
"""

from __future__ import annotations

import json
import logging
import sys
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
# Config
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
FREQ_RES = SR / N_FFT  # ~585.9 Hz/bin
SPEC_FREQ_MIN = 20_000  # Spectrogram lower bound
SPEC_FREQ_MAX = 120_000  # Spectrogram upper bound

# USV analysis band for spectral flatness (35-110 kHz)
USV_BAND_MIN_HZ = 35_000
USV_BAND_MAX_HZ = 110_000
# Row indices in the spectrogram (which starts at 20 kHz)
USV_BAND_ROW_START = round((USV_BAND_MIN_HZ - SPEC_FREQ_MIN) / FREQ_RES)  # ~26
USV_BAND_ROW_END = round((USV_BAND_MAX_HZ - SPEC_FREQ_MIN) / FREQ_RES)    # ~154

# Broadband column detection
BROADBAND_DB_ABOVE_MEDIAN = 6.0  # dB above file median to count as "active"
BROADBAND_FRAC_THRESHOLD = 0.60  # >60% of ALL bins active = broadband column

# Spectral flatness threshold for "tonal" classification
TONAL_FLATNESS_THRESHOLD = 0.5

# Number of random auto-accept files to sample as known-good
N_GOOD_SAMPLE = 50
RANDOM_SEED = 42

# Known 1960-problem files
PROBLEM_STEMS = [
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

# Known-good reference files
GOOD_REFERENCE_STEMS = [
    "2024-09-30_11-22-17_0000053",
    "2024-09-30_11-22-19_0000054",
]


def find_wav(stem: str) -> Optional[Path]:
    """Find WAV file across search directories."""
    for search_dir in WAV_SEARCH_DIRS:
        if not search_dir.exists():
            continue
        matches = list(search_dir.rglob(f"{stem}.wav"))
        if matches:
            return matches[0]
    return None


def spectral_flatness(power_linear: np.ndarray) -> float:
    """Compute spectral flatness (Wiener entropy) of a power spectrum.

    Args:
        power_linear: Linear power values (not dB). Shape: (n_bins,)

    Returns:
        Spectral flatness in [0, 1]. 0 = tonal, 1 = flat noise.
    """
    eps = 1e-10
    # Geometric mean via log domain
    log_mean = np.mean(np.log(power_linear + eps))
    geo_mean = np.exp(log_mean)
    arith_mean = np.mean(power_linear)
    if arith_mean < eps:
        return 1.0  # silence → treat as flat
    return float(geo_mean / arith_mean)


def analyze_detection(
    spec_db: np.ndarray,
    start_col: int,
    end_col: int,
    file_median_db: float,
) -> dict:
    """Compute per-column spectral flatness metrics for one detection window.

    Args:
        spec_db: Full spectrogram in dB, shape (n_freq, n_time)
        start_col: Detection window start column
        end_col: Detection window end column
        file_median_db: Median dB value of the full spectrogram

    Returns:
        Dict with aggregated metrics for this detection.
    """
    n_freq, n_time = spec_db.shape
    col_start = max(0, start_col)
    col_end = min(n_time, end_col)
    n_cols = col_end - col_start

    if n_cols <= 0:
        return _empty_metrics()

    window = spec_db[:, col_start:col_end]
    usv_band = window[USV_BAND_ROW_START:USV_BAND_ROW_END, :]

    # Identify broadband columns: >60% of ALL frequency bins above median + 6 dB
    active_mask = window > (file_median_db + BROADBAND_DB_ABOVE_MEDIAN)
    frac_active_per_col = active_mask.mean(axis=0)
    broadband_mask = frac_active_per_col > BROADBAND_FRAC_THRESHOLD

    n_broadband = int(broadband_mask.sum())
    n_non_broadband = n_cols - n_broadband

    # Compute spectral flatness for each non-broadband column in the USV band
    flatness_values = []
    tonal_mask = np.zeros(n_cols, dtype=bool)

    for i in range(n_cols):
        if broadband_mask[i]:
            continue
        # Convert dB to linear power
        col_db = usv_band[:, i]
        col_linear = 10.0 ** (col_db / 10.0)
        sf = spectral_flatness(col_linear)
        flatness_values.append(sf)
        if sf < TONAL_FLATNESS_THRESHOLD:
            tonal_mask[i] = True

    flatness_arr = np.array(flatness_values) if flatness_values else np.array([])

    # Max consecutive tonal columns
    max_consec_tonal = _max_consecutive_true(tonal_mask & ~broadband_mask)

    # Tonal fraction among non-broadband columns
    n_tonal = int(tonal_mask.sum())
    tonal_fraction = n_tonal / n_non_broadband if n_non_broadband > 0 else 0.0

    return {
        "n_total_columns": n_cols,
        "n_broadband_columns": n_broadband,
        "n_non_broadband_columns": n_non_broadband,
        "n_tonal_columns": n_tonal,
        "tonal_fraction": tonal_fraction,
        "max_consecutive_tonal": max_consec_tonal,
        "mean_flatness": float(flatness_arr.mean()) if len(flatness_arr) > 0 else 1.0,
        "min_flatness": float(flatness_arr.min()) if len(flatness_arr) > 0 else 1.0,
        "median_flatness": float(np.median(flatness_arr)) if len(flatness_arr) > 0 else 1.0,
        "std_flatness": float(flatness_arr.std()) if len(flatness_arr) > 0 else 0.0,
        # Raw arrays for diagnostic plots
        "_flatness_per_col": flatness_values,
        "_broadband_mask": broadband_mask,
        "_tonal_mask": tonal_mask,
    }


def _max_consecutive_true(mask: np.ndarray) -> int:
    """Find the longest run of True values in a boolean array."""
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


def _empty_metrics() -> dict:
    return {
        "n_total_columns": 0,
        "n_broadband_columns": 0,
        "n_non_broadband_columns": 0,
        "n_tonal_columns": 0,
        "tonal_fraction": 0.0,
        "max_consecutive_tonal": 0,
        "mean_flatness": 1.0,
        "min_flatness": 1.0,
        "median_flatness": 1.0,
        "std_flatness": 0.0,
        "_flatness_per_col": [],
        "_broadband_mask": np.array([]),
        "_tonal_mask": np.array([]),
    }


def save_diagnostic_png(
    spec_db: np.ndarray,
    stem: str,
    label: str,
    events: list,
    event_metrics: list[dict],
    output_dir: Path,
):
    """Save diagnostic PNG with spectrogram + per-column flatness overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n_events = len(events)
    fig, axes = plt.subplots(2, 1, figsize=(16, 6), height_ratios=[3, 1],
                             sharex=True, gridspec_kw={"hspace": 0.05})
    ax_spec, ax_flat = axes

    # Plot spectrogram
    extent = [0, spec_db.shape[1], SPEC_FREQ_MIN / 1000, SPEC_FREQ_MAX / 1000]
    ax_spec.imshow(
        spec_db, aspect="auto", origin="lower", cmap="magma",
        extent=extent,
        vmin=np.percentile(spec_db, 5),
        vmax=np.percentile(spec_db, 99),
    )

    # Draw detection windows and overlay flatness
    for ev, metrics in zip(events, event_metrics):
        sc, ec = ev["start_col"], ev["end_col"]
        # Detection window box
        ax_spec.axvspan(sc, ec, alpha=0.15, color="cyan", linewidth=0)

        bb_mask = metrics["_broadband_mask"]
        tonal_mask = metrics["_tonal_mask"]
        n_cols = len(bb_mask)
        if n_cols == 0:
            continue

        col_indices = np.arange(sc, sc + n_cols)

        # Plot flatness values on bottom axis
        flat_idx = 0
        for i in range(n_cols):
            col_x = col_indices[i]
            if bb_mask[i]:
                # Broadband column: red marker
                ax_flat.axvspan(col_x, col_x + 1, alpha=0.3, color="red", linewidth=0)
            elif flat_idx < len(metrics["_flatness_per_col"]):
                fv = metrics["_flatness_per_col"][flat_idx]
                color = "green" if fv < TONAL_FLATNESS_THRESHOLD else "orange"
                ax_flat.bar(col_x, fv, width=1, color=color, alpha=0.7)
                flat_idx += 1

    # Horizontal lines for USV band on spectrogram
    ax_spec.axhline(y=USV_BAND_MIN_HZ / 1000, color="white", linestyle="--",
                     alpha=0.4, linewidth=0.7)
    ax_spec.axhline(y=USV_BAND_MAX_HZ / 1000, color="white", linestyle="--",
                     alpha=0.4, linewidth=0.7)

    ax_spec.set_ylabel("Frequency (kHz)")
    ax_spec.set_title(
        f"{stem}  [{label}]  |  {n_events} events  |  "
        f"green=tonal, orange=flat, red=broadband",
        fontsize=10,
    )

    ax_flat.set_ylabel("Spectral\nFlatness")
    ax_flat.set_xlabel("Column")
    ax_flat.set_ylim(0, 1.05)
    ax_flat.axhline(y=TONAL_FLATNESS_THRESHOLD, color="gray", linestyle=":",
                     alpha=0.6, linewidth=0.8)

    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}_{label}.png", dpi=120)
    plt.close(fig)


def process_file(
    stem: str,
    label: str,
    loader: AudioLoader,
    detections_dir: Path,
    output_dir: Path,
    save_png: bool = False,
) -> list[dict]:
    """Process one recording: compute spectral flatness for all detections.

    Returns list of per-detection result rows.
    """
    det_path = detections_dir / f"{stem}.json"
    if not det_path.exists():
        log.warning("No detection JSON for %s", stem)
        return []

    with open(det_path) as f:
        events = json.load(f)
    if not events:
        log.info("  %s: no events", stem)
        return []

    wav_path = find_wav(stem)
    if wav_path is None:
        log.warning("WAV not found: %s", stem)
        return []

    try:
        audio_data = loader.load(wav_path)
        spec = audio_data.spectrogram_db
    except Exception as e:
        log.error("Spectrogram failed for %s: %s", stem, e)
        return []

    file_median = float(np.median(spec))

    rows = []
    event_metrics = []
    for ev_idx, ev in enumerate(events):
        metrics = analyze_detection(spec, ev["start_col"], ev["end_col"], file_median)
        event_metrics.append(metrics)

        rows.append({
            "stem": stem,
            "label": label,
            "event_idx": ev_idx,
            "start_col": ev["start_col"],
            "end_col": ev["end_col"],
            "max_probability": ev.get("max_probability", 0),
            "mean_probability": ev.get("mean_probability", 0),
            "n_total_columns": metrics["n_total_columns"],
            "n_broadband_columns": metrics["n_broadband_columns"],
            "n_tonal_columns": metrics["n_tonal_columns"],
            "tonal_fraction": metrics["tonal_fraction"],
            "max_consecutive_tonal": metrics["max_consecutive_tonal"],
            "mean_flatness": metrics["mean_flatness"],
            "min_flatness": metrics["min_flatness"],
            "median_flatness": metrics["median_flatness"],
            "std_flatness": metrics["std_flatness"],
        })

    if save_png:
        save_diagnostic_png(spec, stem, label, events, event_metrics, output_dir)

    return rows


def main():
    detections_dir = BATCH_DIR / "detections"
    output_dir = BATCH_DIR / "spectral_flatness_profile"
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = AudioLoader()

    # -----------------------------------------------------------------------
    # Collect files to analyze
    # -----------------------------------------------------------------------
    # 1) Known problem files
    problem_files = [(s, "1960_problem") for s in PROBLEM_STEMS]

    # 2) Known-good reference files
    good_ref_files = [(s, "good_reference") for s in GOOD_REFERENCE_STEMS]

    # 3) Random sample of auto-accept files (excluding problem & reference)
    summary_path = BATCH_DIR / "summary_full.parquet"
    df = pd.read_parquet(summary_path)
    auto_accept = df[df["tier"] == "auto_accept"]
    exclude = set(PROBLEM_STEMS + GOOD_REFERENCE_STEMS)
    candidates = auto_accept[~auto_accept["stem"].isin(exclude)]
    rng = np.random.RandomState(RANDOM_SEED)
    sampled = candidates.sample(n=min(N_GOOD_SAMPLE, len(candidates)), random_state=rng)
    good_sample_files = [(row["stem"], "auto_accept") for _, row in sampled.iterrows()]

    all_files = problem_files + good_ref_files + good_sample_files
    log.info("Files to analyze: %d problem, %d good_ref, %d auto_accept_sample",
             len(problem_files), len(good_ref_files), len(good_sample_files))

    # -----------------------------------------------------------------------
    # Process all files
    # -----------------------------------------------------------------------
    all_rows = []
    for idx, (stem, label) in enumerate(all_files, 1):
        # Save PNGs for all problem files + good references + 5 random good
        save_png = label in ("1960_problem", "good_reference") or idx <= len(problem_files) + len(good_ref_files) + 5
        log.info("[%d/%d] %s (%s)", idx, len(all_files), stem, label)
        rows = process_file(stem, label, loader, detections_dir, output_dir, save_png=save_png)
        all_rows.extend(rows)

    if not all_rows:
        log.error("No results! Check WAV paths and detection JSONs.")
        return 1

    # -----------------------------------------------------------------------
    # Output CSV
    # -----------------------------------------------------------------------
    result_df = pd.DataFrame(all_rows)
    csv_path = output_dir / "spectral_flatness_profile.csv"
    result_df.to_csv(csv_path, index=False)
    log.info("Saved %d detection rows to %s", len(result_df), csv_path)

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    # Aggregate to per-file level (take the "best" detection per file)
    file_agg = result_df.groupby(["stem", "label"]).agg(
        n_events=("event_idx", "count"),
        file_min_flatness=("min_flatness", "min"),
        file_mean_flatness=("mean_flatness", "mean"),
        file_max_tonal_fraction=("tonal_fraction", "max"),
        file_max_consec_tonal=("max_consecutive_tonal", "max"),
    ).reset_index()

    log.info("\n" + "=" * 70)
    log.info("SUMMARY: Per-file aggregated metrics")
    log.info("=" * 70)

    for label in ["1960_problem", "good_reference", "auto_accept"]:
        subset = file_agg[file_agg["label"] == label]
        if subset.empty:
            continue
        log.info("\n--- %s (n=%d files) ---", label, len(subset))
        for col in ["file_min_flatness", "file_mean_flatness",
                     "file_max_tonal_fraction", "file_max_consec_tonal"]:
            vals = subset[col]
            log.info("  %-28s  mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
                     col, vals.mean(), vals.std(), vals.min(), vals.max())

    # -----------------------------------------------------------------------
    # Separation analysis
    # -----------------------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("SEPARATION ANALYSIS")
    log.info("=" * 70)

    problem = file_agg[file_agg["label"] == "1960_problem"]
    good = file_agg[file_agg["label"].isin(["good_reference", "auto_accept"])]

    for metric in ["file_min_flatness", "file_mean_flatness",
                    "file_max_tonal_fraction", "file_max_consec_tonal"]:
        p_vals = problem[metric].values
        g_vals = good[metric].values
        p_range = f"[{p_vals.min():.4f}, {p_vals.max():.4f}]"
        g_range = f"[{g_vals.min():.4f}, {g_vals.max():.4f}]"

        # Check separation
        if metric in ("file_min_flatness", "file_mean_flatness"):
            # Problem files should have HIGHER flatness
            separates = p_vals.min() > g_vals.max()
            overlap_pct = (g_vals >= p_vals.min()).mean() * 100 if not separates else 0
        else:
            # Problem files should have LOWER tonal fraction / consecutive tonal
            separates = p_vals.max() < g_vals.min()
            overlap_pct = (g_vals <= p_vals.max()).mean() * 100 if not separates else 0

        sep_str = "CLEAN SEPARATION" if separates else f"OVERLAP ({overlap_pct:.0f}% of good files in problem range)"
        log.info("  %-28s  problem=%s  good=%s  → %s", metric, p_range, g_range, sep_str)

    # Save aggregated file
    agg_csv = output_dir / "spectral_flatness_file_summary.csv"
    file_agg.to_csv(agg_csv, index=False)
    log.info("\nSaved file-level summary to %s", agg_csv)
    log.info("Saved diagnostic PNGs to %s", output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
