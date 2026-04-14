#!/usr/bin/env python3
"""Generate overview PNGs for 3452 batch detections (auto_accept + manual_review).

For each file, produces a 2-panel figure:
  - Top: spectrogram (magma) with event overlay bands (cyan)
  - Bottom: CNN probability curve with onset/sustain thresholds

Output directories:
  results/batch_3452_sample/review_pngs/
  results/batch_3452_reviewed/review_pngs/
  + symlinked WAV folders for easy app access
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.app.core.audio_loader import AudioLoader
from usv_spectrogram.app.core.sliding_inference import SlidingInference
from usv_spectrogram.postprocessing.calibration import TemperatureScaler
from usv_spectrogram.postprocessing.hysteresis import HysteresisConfig, hysteresis_detect

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Production pipeline
MODEL_PATH = REPO_ROOT / "models" / "hard_neg_retrain" / "best_model.pt"
TEMP_PATH = REPO_ROOT / "models" / "hard_neg_retrain" / "temperature.json"
HYSTERESIS_PATH = REPO_ROOT / "models" / "hard_neg_retrain" / "hysteresis_optimization_v2.json"

SPEC_FREQ_MIN = 20_000
SPEC_FREQ_MAX = 120_000
ONSET_THRESHOLD = 0.6
SUSTAIN_THRESHOLD = 0.4


def save_png(
    spec_db, times, probabilities, prob_times, title_stem, tier,
    events, output_path,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_spec, ax_prob) = plt.subplots(
        2, 1, figsize=(14, 6), dpi=150,
        gridspec_kw={"height_ratios": [3, 1]}, sharex=False,
    )

    duration_s = times[-1] if len(times) > 0 else spec_db.shape[1] * 0.001
    extent = [0, duration_s, SPEC_FREQ_MIN / 1000, SPEC_FREQ_MAX / 1000]
    ax_spec.imshow(
        spec_db, aspect="auto", origin="lower", cmap="magma", extent=extent,
        vmin=np.percentile(spec_db, 5), vmax=np.percentile(spec_db, 99),
    )

    for ev in events:
        ax_spec.axvspan(ev.start_time_s, ev.end_time_s, alpha=0.3, color="cyan", linewidth=0)

    ax_spec.set_ylabel("Freq (kHz)")
    ax_spec.set_xlim(0, duration_s)

    n_events = len(events)
    max_conf = max((ev.peak_probability for ev in events), default=0)
    ax_spec.set_title(
        f"{title_stem} — {tier.upper()} — {n_events} events, max_conf={max_conf:.3f}",
        fontsize=10,
    )

    ax_prob.plot(prob_times, probabilities, color="steelblue", linewidth=0.7)
    ax_prob.axhline(y=ONSET_THRESHOLD, color="red", linestyle="--", linewidth=0.8, label=f"Onset ({ONSET_THRESHOLD})")
    ax_prob.axhline(y=SUSTAIN_THRESHOLD, color="orange", linestyle="--", linewidth=0.8, label=f"Sustain ({SUSTAIN_THRESHOLD})")

    for ev in events:
        ax_prob.axvspan(ev.start_time_s, ev.end_time_s, alpha=0.25, color="cyan", linewidth=0)

    ax_prob.set_xlabel("Time (s)")
    ax_prob.set_ylabel("Probability")
    ax_prob.set_xlim(0, duration_s)
    ax_prob.set_ylim(0, 1.05)
    ax_prob.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def process_batch(batch_dir: Path, wav_search_dirs: list[Path]):
    summary_path = batch_dir / "summary.parquet"
    df = pd.read_parquet(summary_path)

    detected = df[df["tier"].isin(["auto_accept", "manual_review"])].copy()
    if detected.empty:
        log.info("No detections in %s", batch_dir)
        return

    log.info("Processing %d files from %s (auto_accept=%d, manual_review=%d)",
             len(detected), batch_dir.name,
             (detected["tier"] == "auto_accept").sum(),
             (detected["tier"] == "manual_review").sum())

    png_dir = batch_dir / "review_pngs"
    wav_dir = batch_dir / "review_wavs"
    png_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    loader = AudioLoader()
    inference = SlidingInference(model_path=MODEL_PATH)

    temp_scaler = None
    if TEMP_PATH.exists():
        temp_scaler = TemperatureScaler.load(TEMP_PATH)
        log.info("Temperature scaling: T=%.4f", temp_scaler.temperature)

    with open(HYSTERESIS_PATH) as f:
        hparams = json.load(f)["best_params"]
    hyst_config = HysteresisConfig(
        onset_threshold=hparams["onset_threshold"],
        sustain_threshold=hparams["sustain_threshold"],
        gap_fill_windows=hparams["gap_fill_windows"],
        min_duration_windows=hparams["min_duration_windows"],
    )

    detections_dir = batch_dir / "detections"

    for idx, (_, row) in enumerate(detected.iterrows(), 1):
        wav_path = Path(row["filepath"])
        if not wav_path.is_absolute():
            wav_path = REPO_ROOT / wav_path

        stem = wav_path.stem
        tier = row["tier"]

        if not wav_path.exists():
            # Try search dirs
            found = False
            for d in wav_search_dirs:
                candidates = list(d.rglob(f"{stem}.wav"))
                if candidates:
                    wav_path = candidates[0]
                    found = True
                    break
            if not found:
                log.warning("[%d/%d] WAV not found: %s", idx, len(detected), stem)
                continue

        log.info("[%d/%d] %s (%s)", idx, len(detected), stem, tier)

        try:
            audio_data = loader.load(wav_path)
            result = inference.infer(
                audio_data.spectrogram_db, audio_data.times,
                return_logits=temp_scaler is not None,
            )

            probabilities = result.probabilities
            if temp_scaler is not None and result.logits is not None:
                probabilities = temp_scaler.calibrate(result.logits)

            events = hysteresis_detect(probabilities, result.times, hyst_config)

            # Load saved events from detection JSON to get pipeline-accepted events
            det_path = detections_dir / f"{stem}.json"
            saved_events = []
            if det_path.exists():
                with open(det_path) as f:
                    saved_events = json.load(f)

            save_png(
                audio_data.spectrogram_db, audio_data.times,
                probabilities, result.times,
                stem, tier, events,
                png_dir / f"{stem}_{tier}.png",
            )

            # Symlink WAV
            link_path = wav_dir / wav_path.name
            if not link_path.exists():
                os.symlink(wav_path.resolve(), link_path)

        except Exception as e:
            log.error("Failed %s: %s", stem, e)

    log.info("PNGs: %s", png_dir)
    log.info("WAVs: %s", wav_dir)


def main():
    sample_dir = REPO_ROOT / "results" / "batch_3452_sample"
    reviewed_dir = REPO_ROOT / "results" / "batch_3452_reviewed"

    sample_wav_dirs = [
        REPO_ROOT / "USV_3452_sample",
    ]
    reviewed_wav_dirs = [
        REPO_ROOT / "USV_3452_sample_reviewed",
    ]

    if sample_dir.exists():
        process_batch(sample_dir, sample_wav_dirs)

    if reviewed_dir.exists():
        process_batch(reviewed_dir, reviewed_wav_dirs)


if __name__ == "__main__":
    main()
