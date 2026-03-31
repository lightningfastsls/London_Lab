#!/usr/bin/env python3
"""Generate 2-panel overview PNGs for manual_review recordings.

For each manual_review file, produces a figure with:
  - Top panel:  spectrogram (magma) with event overlay bands
  - Bottom panel: CNN probability curve with onset/sustain threshold lines

Band colors:
  - cyan:  hysteresis-passed events (onset >= 0.6 triggered)
  - gray:  sub-threshold regions (prob > sustain but never reached onset)

Also exports a CSV with ALL detections visible in the PNGs (both accepted
and sub-threshold), not just the ones that survived the full pipeline.

Usage:
    python scripts/generate_manual_review_pngs.py
    python scripts/generate_manual_review_pngs.py --csv-only
    python scripts/generate_manual_review_pngs.py --png-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.app.core.audio_loader import AudioLoader
from usv_spectrogram.app.core.sliding_inference import SlidingInference
from usv_spectrogram.postprocessing.calibration import TemperatureScaler
from usv_spectrogram.postprocessing.hysteresis import (
    HysteresisConfig,
    USVEvent,
    hysteresis_detect,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

BATCH_DIR = REPO_ROOT / "results" / "batch_5970"
MODEL_PATH = REPO_ROOT / "models" / "matched_windows" / "best_model.pt"
TEMP_PATH = REPO_ROOT / "models" / "matched_windows" / "temperature.json"
HYSTERESIS_PATH = REPO_ROOT / "results" / "hysteresis_optimization.json"

WAV_SEARCH_DIRS = [
    REPO_ROOT / "5970",
    REPO_ROOT / "5970_reviewed",
    REPO_ROOT / "5970_manual_review",
    REPO_ROOT / "5970_manual_review_reviewed",
]

SPEC_FREQ_MIN = 20_000
SPEC_FREQ_MAX = 120_000

# Thresholds drawn on the probability panel
ONSET_THRESHOLD = 0.6
SUSTAIN_THRESHOLD = 0.45

# Low threshold for "candidate" regions — any CNN activity visible in the plot.
# These appear as gray bands; hysteresis-passed events overlay them in cyan.
CANDIDATE_THRESHOLD = 0.1


def find_wav(stem: str) -> Path | None:
    for d in WAV_SEARCH_DIRS:
        if not d.exists():
            continue
        m = list(d.rglob(f"{stem}.wav"))
        if m:
            return m[0]
    return None


def find_candidate_regions(
    probabilities: np.ndarray,
    times: np.ndarray,
    accepted_events: List[USVEvent],
    threshold: float = CANDIDATE_THRESHOLD,
) -> List[dict]:
    """Find contiguous regions where prob > threshold that are NOT hysteresis-accepted.

    These are ALL regions where the CNN showed elevated activity — the gray
    bands in the overview PNGs.  Regions that overlap with an accepted
    hysteresis event are excluded (those get cyan instead).
    """
    above = probabilities > threshold
    regions: List[Tuple[int, int]] = []
    in_region = False
    start_idx = 0

    for i, val in enumerate(above):
        if val and not in_region:
            in_region = True
            start_idx = i
        elif not val and in_region:
            in_region = False
            regions.append((start_idx, i - 1))
    if in_region:
        regions.append((start_idx, len(above) - 1))

    # Build set of windows covered by accepted events
    accepted_ranges = set()
    for ev in accepted_events:
        for w in range(ev.start_window, ev.end_window + 1):
            accepted_ranges.add(w)

    candidates = []
    for s, e in regions:
        region_windows = set(range(s, e + 1))
        # Remove windows that belong to accepted events
        remaining = sorted(region_windows - accepted_ranges)
        if not remaining:
            continue
        # Split remaining into contiguous runs
        runs: List[Tuple[int, int]] = []
        run_start = remaining[0]
        prev = remaining[0]
        for w in remaining[1:]:
            if w != prev + 1:
                runs.append((run_start, prev))
                run_start = w
            prev = w
        runs.append((run_start, prev))

        for rs, re in runs:
            seg_probs = probabilities[rs : re + 1]
            candidates.append({
                "start_window": int(rs),
                "end_window": int(re),
                "start_time_s": float(times[rs]),
                "end_time_s": float(times[re]),
                "duration_s": float(times[re] - times[rs]),
                "max_probability": float(seg_probs.max()),
                "mean_probability": float(seg_probs.mean()),
                "window_count": re - rs + 1,
                "status": "sub_threshold",
            })

    return candidates


def save_png(
    spec_db: np.ndarray,
    times: np.ndarray,
    probabilities: np.ndarray,
    prob_times: np.ndarray,
    stem: str,
    accepted_events: List[USVEvent],
    sub_threshold_regions: List[dict],
    output_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_spec, ax_prob) = plt.subplots(
        2, 1, figsize=(14, 6), dpi=150,
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=False,
    )

    # --- Top panel: spectrogram ---
    duration_s = times[-1] if len(times) > 0 else spec_db.shape[1] * 0.001
    extent = [0, duration_s, SPEC_FREQ_MIN / 1000, SPEC_FREQ_MAX / 1000]
    ax_spec.imshow(
        spec_db, aspect="auto", origin="lower", cmap="magma",
        extent=extent,
        vmin=np.percentile(spec_db, 5),
        vmax=np.percentile(spec_db, 99),
    )

    # Gray bands: sub-threshold regions
    for region in sub_threshold_regions:
        ax_spec.axvspan(
            region["start_time_s"], region["end_time_s"],
            alpha=0.3, color="gray", linewidth=0,
        )

    # Cyan bands: accepted events
    for ev in accepted_events:
        ax_spec.axvspan(
            ev.start_time_s, ev.end_time_s,
            alpha=0.3, color="cyan", linewidth=0,
        )

    ax_spec.set_ylabel("Freq (kHz)")
    ax_spec.set_xlim(0, duration_s)

    # Title
    n_events = len(accepted_events)
    max_conf = max((ev.peak_probability for ev in accepted_events), default=0)
    ax_spec.set_title(
        f"{stem} \u2014 MANUAL REVIEW \u2014 {n_events} events, max_conf={max_conf:.3f}",
        fontsize=10,
    )

    # --- Bottom panel: probability curve ---
    ax_prob.plot(prob_times, probabilities, color="steelblue", linewidth=0.7)
    ax_prob.axhline(
        y=ONSET_THRESHOLD, color="red", linestyle="--", linewidth=0.8,
        label=f"Onset ({ONSET_THRESHOLD})",
    )
    ax_prob.axhline(
        y=SUSTAIN_THRESHOLD, color="orange", linestyle="--", linewidth=0.8,
        label=f"Sustain ({SUSTAIN_THRESHOLD})",
    )

    # Gray bands on prob panel too
    for region in sub_threshold_regions:
        ax_prob.axvspan(
            region["start_time_s"], region["end_time_s"],
            alpha=0.15, color="gray", linewidth=0,
        )

    # Cyan bands on prob panel
    for ev in accepted_events:
        ax_prob.axvspan(
            ev.start_time_s, ev.end_time_s,
            alpha=0.25, color="cyan", linewidth=0,
        )

    ax_prob.set_xlabel("Time (s)")
    ax_prob.set_ylabel("Probability")
    ax_prob.set_xlim(0, duration_s)
    ax_prob.set_ylim(0, 1.05)
    ax_prob.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png")
    plt.close(fig)


def process_one(
    stem: str,
    loader: AudioLoader,
    inference: SlidingInference,
    hysteresis_config: HysteresisConfig,
    temp_scaler: TemperatureScaler | None,
    saved_events: List[dict],
    output_dir: Path | None,
    generate_png: bool = True,
) -> List[dict]:
    """Process one recording. Returns rows for ALL hysteresis detections.

    Each row is marked as 'pipeline_accepted' (survived FP filter into saved
    JSONs) or 'fp_filtered' (hysteresis accepted but FP filter removed).
    """
    wav_path = find_wav(stem)
    if wav_path is None:
        log.warning("WAV not found: %s", stem)
        return []

    audio_data = loader.load(wav_path)
    result = inference.infer(
        audio_data.spectrogram_db,
        audio_data.times,
        return_logits=temp_scaler is not None,
    )

    probabilities = result.probabilities
    if temp_scaler is not None and result.logits is not None:
        probabilities = temp_scaler.calibrate(result.logits)

    # Hysteresis detection → all events the CNN + hysteresis found
    events = hysteresis_detect(probabilities, result.times, hysteresis_config)

    # Candidate regions for gray bands in the PNG
    sub_regions = find_candidate_regions(
        probabilities, result.times, events,
    )

    if generate_png and output_dir is not None:
        save_png(
            audio_data.spectrogram_db,
            audio_data.times,
            probabilities,
            result.times,
            stem,
            events,
            sub_regions,
            output_dir,
        )

    # Match hysteresis events against saved JSON events to determine status.
    # An event is "pipeline_accepted" if a saved event overlaps it in time.
    saved_intervals = [
        (s["start_time_s"], s["end_time_s"]) for s in saved_events
    ]

    rows = []
    for i, ev in enumerate(events):
        # Check overlap with any saved event
        status = "fp_filtered"
        for s_start, s_end in saved_intervals:
            if ev.start_time_s <= s_end and ev.end_time_s >= s_start:
                status = "pipeline_accepted"
                break

        rows.append({
            "stem": stem,
            "detection_idx": i,
            "status": status,
            "start_time_s": ev.start_time_s,
            "end_time_s": ev.end_time_s,
            "duration_s": ev.duration_ms / 1000.0,
            "max_probability": ev.peak_probability,
            "mean_probability": ev.mean_probability,
            "window_count": ev.window_count,
            "start_window": ev.start_window,
            "end_window": ev.end_window,
        })

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-only", action="store_true", help="Skip PNG generation")
    parser.add_argument("--png-only", action="store_true", help="Skip CSV generation")
    args = parser.parse_args()

    generate_png = not args.csv_only
    generate_csv = not args.png_only

    # Load summary to find manual_review files
    summary_path = BATCH_DIR / "summary_full.parquet"
    df = pd.read_parquet(summary_path)
    mr = df[df["tier"] == "manual_review"].sort_values("stem")
    stems = mr["stem"].tolist()
    log.info("Found %d manual_review recordings", len(stems))

    output_dir = BATCH_DIR / "manual_review"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline components
    loader = AudioLoader()
    inference = SlidingInference(model_path=MODEL_PATH)

    temp_scaler = None
    if TEMP_PATH.exists():
        temp_scaler = TemperatureScaler.load(TEMP_PATH)
        log.info("Temperature scaling loaded (T=%.4f)", temp_scaler.temperature)

    with open(HYSTERESIS_PATH) as f:
        hparams = json.load(f)["best_params"]
    hysteresis_config = HysteresisConfig(
        onset_threshold=hparams["onset_threshold"],
        sustain_threshold=hparams["sustain_threshold"],
        gap_fill_windows=hparams["gap_fill_windows"],
        min_duration_windows=hparams["min_duration_windows"],
    )
    log.info(
        "Hysteresis config: onset=%.2f, sustain=%.2f, gap=%d, min_dur=%d",
        hysteresis_config.onset_threshold,
        hysteresis_config.sustain_threshold,
        hysteresis_config.gap_fill_windows,
        hysteresis_config.min_duration_windows,
    )

    # Load saved detection JSONs (post-pipeline) to flag FP-filtered events
    detections_dir = BATCH_DIR / "detections"

    all_rows = []

    for idx, stem in enumerate(stems, 1):
        log.info("[%d/%d] %s", idx, len(stems), stem)

        # Load saved events for this file
        det_path = detections_dir / f"{stem}.json"
        saved_events = []
        if det_path.exists():
            with open(det_path) as f:
                saved_events = json.load(f)

        try:
            rows = process_one(
                stem, loader, inference, hysteresis_config, temp_scaler,
                saved_events, output_dir, generate_png=generate_png,
            )
            all_rows.extend(rows)
        except Exception as e:
            log.error("Failed %s: %s", stem, e)

    n_accepted = sum(1 for r in all_rows if r["status"] == "pipeline_accepted")
    n_filtered = sum(1 for r in all_rows if r["status"] == "fp_filtered")
    log.info(
        "Done. %d total detections (%d pipeline_accepted, %d fp_filtered) across %d files",
        len(all_rows), n_accepted, n_filtered, len(stems),
    )

    if generate_png:
        log.info("PNGs saved to %s", output_dir)

    if generate_csv:
        csv_df = pd.DataFrame(all_rows)
        csv_df = csv_df.sort_values(["stem", "start_time_s"]).reset_index(drop=True)
        csv_path = BATCH_DIR / "manual_review_all_detections.csv"
        csv_df.to_csv(csv_path, index=False)
        log.info(
            "CSV saved to %s (%d rows: %d pipeline_accepted, %d fp_filtered)",
            csv_path, len(csv_df), n_accepted, n_filtered,
        )


if __name__ == "__main__":
    main()
