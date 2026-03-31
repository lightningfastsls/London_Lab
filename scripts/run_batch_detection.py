#!/usr/bin/env python3
"""Run the full USV detection pipeline on a directory of WAV files.

Orchestrates the complete chain:
  AudioLoader → SlidingInference → [TemperatureScaling] → [Normalization]
    → HysteresisDetection → [EventFeatures → FPFilter] → Triage → Output

Outputs:
  - summary.parquet  — one row per recording with tier + QC metrics
  - detections/*.json — per-recording events in ADR-010 format

Usage:
    python scripts/run_batch_detection.py \
        --wav-dir /path/to/wavs \
        --model models/matched_windows/best_model.pt \
        --output-dir results/batch_001/

    # With optional post-processing stages:
    python scripts/run_batch_detection.py \
        --wav-dir /path/to/wavs \
        --model models/matched_windows/best_model.pt \
        --output-dir results/batch_001/ \
        --temperature models/matched_windows/temperature.json \
        --fp-filter models/matched_windows/fp_filter.pkl \
        --hysteresis-config results/hysteresis_optimization.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap: ensure src/ is importable when running as script
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.app.core.audio_loader import AudioLoader  # noqa: E402
from usv_spectrogram.app.core.sliding_inference import SlidingInference  # noqa: E402
from usv_spectrogram.postprocessing.batch_output import write_batch_results  # noqa: E402
from usv_spectrogram.postprocessing.hysteresis import (  # noqa: E402
    HysteresisConfig,
    USVEvent,
    hysteresis_detect,
)
from usv_spectrogram.postprocessing.triage import (  # noqa: E402
    RecordingResult,
    TriageConfig,
    triage_recording,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional pipeline stages (loaded on demand)
# ---------------------------------------------------------------------------

def _load_temperature_scaler(path: Path):
    """Load a fitted TemperatureScaler from JSON."""
    from usv_spectrogram.postprocessing.calibration import TemperatureScaler
    return TemperatureScaler.load(path)


def _load_fp_filter(path: Path):
    """Load a fitted FalsePositiveFilter from pickle."""
    from usv_spectrogram.postprocessing.fp_filter import FalsePositiveFilter
    return FalsePositiveFilter.load(path)


def _load_hysteresis_config(path: Path) -> HysteresisConfig:
    """Load HysteresisConfig from a JSON optimization result."""
    with open(path) as f:
        data = json.load(f)
    params = data.get("best_params", data)
    return HysteresisConfig(
        onset_threshold=params["onset_threshold"],
        sustain_threshold=params["sustain_threshold"],
        gap_fill_windows=params["gap_fill_windows"],
        min_duration_windows=params["min_duration_windows"],
    )


# ---------------------------------------------------------------------------
# Per-recording pipeline
# ---------------------------------------------------------------------------

def process_one_recording(
    wav_path: Path,
    loader: AudioLoader,
    inference: SlidingInference,
    hysteresis_config: HysteresisConfig,
    temperature_scaler=None,
    normalize_fn=None,
    fp_filter=None,
    spectrogram_for_features: bool = False,
) -> tuple[List[USVEvent], np.ndarray]:
    """Run the detection pipeline on a single WAV file.

    Returns (events, probabilities).
    """
    # 1. Load audio + compute spectrogram
    audio_data = loader.load(wav_path)

    # 2. Sliding window CNN inference
    result = inference.infer(
        audio_data.spectrogram_db,
        audio_data.times,
        return_logits=temperature_scaler is not None,
    )

    probabilities = result.probabilities

    # 3. Optional: temperature scaling (operates on logits)
    if temperature_scaler is not None and result.logits is not None:
        probabilities = temperature_scaler.calibrate(result.logits)

    # 4. Optional: per-recording normalization
    if normalize_fn is not None:
        probabilities = normalize_fn(probabilities)

    # 5. Hysteresis detection
    events = hysteresis_detect(probabilities, result.times, hysteresis_config)

    # 6. Optional: event features + FP filter
    if fp_filter is not None:
        from usv_spectrogram.postprocessing.event_features import extract_event_features

        features = [
            extract_event_features(e, audio_data.spectrogram_db)
            for e in events
        ]
        keep = fp_filter.predict(features)
        events = [e for e, k in zip(events, keep) if k]

    return events, probabilities


# ---------------------------------------------------------------------------
# Worker: processes one file and writes JSON immediately
# ---------------------------------------------------------------------------

def _process_and_save_one(
    wav_path: Path,
    detections_dir: Path,
    loader: AudioLoader,
    inference: SlidingInference,
    hysteresis_config: HysteresisConfig,
    triage_config: TriageConfig,
    temperature_scaler,
    fp_filter,
) -> Optional[RecordingResult]:
    """Process one recording, write its JSON, return result for summary."""
    events, probabilities = process_one_recording(
        wav_path,
        loader,
        inference,
        hysteresis_config,
        temperature_scaler=temperature_scaler,
        fp_filter=fp_filter,
    )
    result = triage_recording(
        filepath=str(wav_path),
        events=events,
        probabilities=probabilities,
        config=triage_config,
    )
    # Write per-recording JSON immediately (crash-safe)
    from usv_spectrogram.postprocessing.batch_output import _event_to_adr010_dict
    stem = wav_path.stem
    detections = [_event_to_adr010_dict(e) for e in result.events]
    json_path = detections_dir / f"{stem}.json"
    with open(json_path, "w") as f:
        json.dump(detections, f, indent=2)
    return result


# ---------------------------------------------------------------------------
# Multiprocessing worker initializer and task function
# ---------------------------------------------------------------------------

# Module-level worker state (initialized once per worker process)
_worker_state = {}


def _worker_init(
    model_path: Path,
    hysteresis_config: HysteresisConfig,
    triage_config: TriageConfig,
    temperature_path: Optional[Path],
    fp_filter_path: Optional[Path],
    detections_dir: Path,
):
    """Initialize per-worker pipeline components (called once per process)."""
    _worker_state["loader"] = AudioLoader()
    _worker_state["inference"] = SlidingInference(model_path=model_path)
    _worker_state["hysteresis_config"] = hysteresis_config
    _worker_state["triage_config"] = triage_config
    _worker_state["detections_dir"] = detections_dir
    _worker_state["temperature_scaler"] = (
        _load_temperature_scaler(temperature_path) if temperature_path else None
    )
    _worker_state["fp_filter"] = (
        _load_fp_filter(fp_filter_path) if fp_filter_path else None
    )


def _worker_process_file(wav_path: Path) -> Optional[dict]:
    """Process a single WAV file in a worker process.

    Returns a serializable dict (not RecordingResult, to avoid pickling issues).
    """
    try:
        result = _process_and_save_one(
            wav_path,
            _worker_state["detections_dir"],
            _worker_state["loader"],
            _worker_state["inference"],
            _worker_state["hysteresis_config"],
            _worker_state["triage_config"],
            _worker_state["temperature_scaler"],
            _worker_state["fp_filter"],
        )
        return {
            "filepath": result.filepath,
            "tier": result.tier,
            "n_events": result.n_events,
            "max_confidence": result.max_confidence,
            "mean_event_confidence": result.mean_event_confidence,
            "total_usv_duration_ms": result.total_usv_duration_ms,
            "noise_floor_p90": result.noise_floor_p90,
            "confidence_score": result.confidence_score,
            "qc_flags": result.qc_flags,
        }
    except Exception as e:
        log.error("Failed to process %s: %s", wav_path.name, e)
        return None


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------

def run_batch(
    wav_dir: Path,
    model_path: Path,
    output_dir: Path,
    temperature_path: Optional[Path] = None,
    fp_filter_path: Optional[Path] = None,
    hysteresis_config_path: Optional[Path] = None,
    triage_config: Optional[TriageConfig] = None,
    n_workers: int = 1,
    resume: bool = True,
) -> List[dict]:
    """Run the full pipeline on all WAV files in a directory.

    Args:
        n_workers: Number of parallel worker processes (default 1 = serial).
        resume: If True, skip files that already have a detection JSON.
    """
    wav_files = sorted(wav_dir.glob("**/*.wav"))
    if not wav_files:
        log.warning("No WAV files found in %s", wav_dir)
        return []

    log.info("Found %d WAV files in %s", len(wav_files), wav_dir)

    # Prepare output directory
    detections_dir = output_dir / "detections"
    detections_dir.mkdir(parents=True, exist_ok=True)

    # Resume: skip files with existing detection JSONs
    if resume:
        existing = {p.stem for p in detections_dir.glob("*.json")}
        before = len(wav_files)
        wav_files = [w for w in wav_files if w.stem not in existing]
        skipped = before - len(wav_files)
        if skipped > 0:
            log.info("Resuming: skipped %d already-processed files", skipped)

    if not wav_files:
        log.info("All files already processed")
        return []

    log.info("Processing %d files with %d worker(s)", len(wav_files), n_workers)

    hysteresis_config = (
        _load_hysteresis_config(hysteresis_config_path)
        if hysteresis_config_path
        else HysteresisConfig()
    )

    if triage_config is None:
        triage_config = TriageConfig()

    # --------------- Parallel execution ---------------
    if n_workers > 1:
        from multiprocessing import Pool

        with Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(
                model_path,
                hysteresis_config,
                triage_config,
                temperature_path,
                fp_filter_path,
                detections_dir,
            ),
        ) as pool:
            raw_results = []
            for i, result_dict in enumerate(
                pool.imap_unordered(_worker_process_file, wav_files), 1
            ):
                if result_dict is not None:
                    raw_results.append(result_dict)
                if i % 50 == 0:
                    elapsed_so_far = time.time() - _batch_t0
                    rate = i / elapsed_so_far
                    eta = (len(wav_files) - i) / rate
                    log.info(
                        "[%d/%d] %.1f files/s, ETA %.0fm",
                        i, len(wav_files), rate, eta / 60,
                    )
    else:
        # --------------- Serial execution ---------------
        loader = AudioLoader()
        inference = SlidingInference(model_path=model_path)
        temperature_scaler = (
            _load_temperature_scaler(temperature_path) if temperature_path else None
        )
        fp_filter = _load_fp_filter(fp_filter_path) if fp_filter_path else None

        raw_results = []
        for i, wav_path in enumerate(wav_files, 1):
            if i % 50 == 0:
                log.info("[%d/%d] Processing %s", i, len(wav_files), wav_path.name)
            try:
                result = _process_and_save_one(
                    wav_path, detections_dir,
                    loader, inference, hysteresis_config, triage_config,
                    temperature_scaler, fp_filter,
                )
                raw_results.append({
                    "filepath": result.filepath,
                    "tier": result.tier,
                    "n_events": result.n_events,
                    "max_confidence": result.max_confidence,
                    "mean_event_confidence": result.mean_event_confidence,
                    "total_usv_duration_ms": result.total_usv_duration_ms,
                    "noise_floor_p90": result.noise_floor_p90,
                    "confidence_score": result.confidence_score,
                    "qc_flags": result.qc_flags,
                })
            except Exception:
                log.exception("Failed to process %s", wav_path)

    # Outlier detection on batch-level stats
    if len(raw_results) >= 2:
        counts = np.array([r["n_events"] for r in raw_results], dtype=float)
        batch_mean = float(np.mean(counts))
        batch_std = float(np.std(counts, ddof=1))
        if batch_std > 0.0:
            for r in raw_results:
                z = (r["n_events"] - batch_mean) / batch_std
                if z > triage_config.outlier_count_zscore:
                    if "outlier_event_count" not in r["qc_flags"]:
                        r["qc_flags"].append("outlier_event_count")
        log.info("Batch stats: mean=%.1f events, std=%.1f", batch_mean, batch_std)

    # Write summary parquet (includes resumed files if any)
    _write_summary_parquet(raw_results, detections_dir, output_dir)

    # Summary
    tiers = {"auto_accept": 0, "auto_reject": 0, "manual_review": 0}
    for r in raw_results:
        tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
    log.info("Triage distribution: %s", tiers)

    return raw_results


def _write_summary_parquet(
    new_results: List[dict],
    detections_dir: Path,
    output_dir: Path,
) -> None:
    """Write summary.parquet from new results."""
    import pandas as pd
    from usv_spectrogram.postprocessing.batch_output import _PARQUET_COLUMNS

    if not new_results:
        return
    df = pd.DataFrame(new_results, columns=_PARQUET_COLUMNS)
    df.to_parquet(output_dir / "summary.parquet", index=False)
    log.info("Wrote summary.parquet (%d rows)", len(df))


# Module-level timestamp for ETA calculation in parallel mode
_batch_t0 = 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    global _batch_t0

    parser = argparse.ArgumentParser(
        description="Run batch USV detection pipeline on a directory of WAV files."
    )
    parser.add_argument(
        "--wav-dir", type=Path, required=True,
        help="Directory containing WAV files (searched recursively).",
    )
    parser.add_argument(
        "--model", type=Path, required=True,
        help="Path to the trained CNN model (.pt file).",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory for output files (summary.parquet + detections/*.json).",
    )
    parser.add_argument(
        "--temperature", type=Path, default=None,
        help="Path to fitted temperature scaling JSON.",
    )
    parser.add_argument(
        "--fp-filter", type=Path, default=None,
        help="Path to fitted FP filter pickle.",
    )
    parser.add_argument(
        "--hysteresis-config", type=Path, default=None,
        help="Path to hysteresis optimization JSON.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel worker processes (default: 1).",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Don't skip already-processed files (reprocess everything).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    _batch_t0 = time.time()
    t0 = time.perf_counter()

    results = run_batch(
        wav_dir=args.wav_dir,
        model_path=args.model,
        output_dir=args.output_dir,
        temperature_path=args.temperature,
        fp_filter_path=args.fp_filter,
        hysteresis_config_path=args.hysteresis_config,
        n_workers=args.workers,
        resume=not args.no_resume,
    )

    elapsed = time.perf_counter() - t0
    log.info(
        "Batch complete: %d recordings in %.1fs (%.2f s/rec)",
        len(results),
        elapsed,
        elapsed / max(len(results), 1),
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
