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
from usv_spectrogram.app.core.notch import (  # noqa: E402
    ReconciliationResult,
    TonalLibrary,
)
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

# Spec parameter: fraction of chunks with unmatched audit detections above
# which we fire the stale-library warning. Tunable; currently fixed at 10%.
_UNMATCHED_RATE_WARNING_THRESHOLD = 0.10

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
        max_duration_ms=params.get("max_duration_ms", 600.0),
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
) -> tuple[List[USVEvent], np.ndarray, ReconciliationResult | None]:
    """Run the detection pipeline on a single WAV file.

    Returns ``(events, probabilities, soft_notch_reconciliation)``. The
    third element is ``None`` unless the loader has ``auto_soft_notch=True``.
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

    return events, probabilities, audio_data.soft_notch_reconciliation


def _build_soft_notch_rows(
    wav_path: Path, recon: ReconciliationResult,
) -> list[dict]:
    """Flatten a ReconciliationResult into one parquet row per applied / audit event.

    Schema (matches spec ``soft_notch_applied.parquet``):
      recording_path, chunk_idx, center_hz, width_hz, peak_db, local_median_db,
      cut_depth_db, source ("library"/"audit"), is_drift, intensity_drift_sigma.
    chunk_idx is always 0 — one filter application per WAV.
    """
    drift_lookup = {id(entry): sigma for entry, sigma in recon.intensity_drifts}
    rows: list[dict] = []
    for entry, det in recon.matched:
        rows.append({
            "recording_path": str(wav_path),
            "chunk_idx": 0,
            "center_hz": float(entry.center_hz),
            "width_hz": float(entry.width_hz),
            "peak_db": float(det.peak_db),
            "local_median_db": float(det.local_median_db),
            "cut_depth_db": float(det.above_median_db),
            "source": "library",
            "is_drift": False,
            "intensity_drift_sigma": float(drift_lookup.get(id(entry), float("nan"))),
        })
    for entry in recon.unmatched_library_entries:
        # Library entry's filter still ran, but the per-chunk PSD measurement
        # did not find a peak above threshold. The actual cut depth applied is
        # unknown to us here (auto_soft_notch internalized it) — record 0 as a
        # safe sentinel; the row exists so downstream joins see every library
        # frequency for every chunk.
        rows.append({
            "recording_path": str(wav_path),
            "chunk_idx": 0,
            "center_hz": float(entry.center_hz),
            "width_hz": float(entry.width_hz),
            "peak_db": float("nan"),
            "local_median_db": float("nan"),
            "cut_depth_db": 0.0,
            "source": "library",
            "is_drift": False,
            "intensity_drift_sigma": float("nan"),
        })
    for det in recon.unmatched_detections:
        rows.append({
            "recording_path": str(wav_path),
            "chunk_idx": 0,
            "center_hz": float(det.center_hz),
            "width_hz": float(det.width_hz),
            "peak_db": float(det.peak_db),
            "local_median_db": float(det.local_median_db),
            "cut_depth_db": 0.0,  # audit-only — library is source of truth
            "source": "audit",
            "is_drift": True,
            "intensity_drift_sigma": float("nan"),
        })
    return rows


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
) -> tuple[Optional[RecordingResult], list[dict]]:
    """Process one recording, write its JSON, return (result, soft_notch_rows)."""
    events, probabilities, recon = process_one_recording(
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

    soft_notch_rows = _build_soft_notch_rows(wav_path, recon) if recon is not None else []
    return result, soft_notch_rows


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
    subtract_baseline: bool = False,
    subtraction_method: str = "percentile",
    soft_notch_enabled: bool = False,
    soft_notch_library_path: Optional[Path] = None,
):
    """Initialize per-worker pipeline components (called once per process)."""
    soft_notch_library = (
        TonalLibrary.load(soft_notch_library_path)
        if soft_notch_enabled and soft_notch_library_path is not None
        else None
    )
    _worker_state["loader"] = AudioLoader(
        subtract_baseline=subtract_baseline,
        subtraction_method=subtraction_method,
        auto_soft_notch=soft_notch_enabled,
        soft_notch_library=soft_notch_library,
    )
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
        result, soft_notch_rows = _process_and_save_one(
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
            "soft_notch_rows": soft_notch_rows,
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
    subtract_baseline: bool = False,
    subtraction_method: str = "percentile",
    soft_notch_enabled: bool = False,
    soft_notch_library_path: Optional[Path] = None,
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
                subtract_baseline,
                subtraction_method,
                soft_notch_enabled,
                soft_notch_library_path,
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
        soft_notch_library_serial = (
            TonalLibrary.load(soft_notch_library_path)
            if soft_notch_enabled and soft_notch_library_path is not None
            else None
        )
        loader = AudioLoader(
            subtract_baseline=subtract_baseline,
            subtraction_method=subtraction_method,
            auto_soft_notch=soft_notch_enabled,
            soft_notch_library=soft_notch_library_serial,
        )
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
                result, soft_notch_rows = _process_and_save_one(
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
                    "soft_notch_rows": soft_notch_rows,
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

    # Soft-notch sidecars (only when --soft-notch is active)
    if soft_notch_enabled:
        _write_soft_notch_sidecars(
            raw_results, output_dir,
            soft_notch_library_path=soft_notch_library_path,
        )

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


def _write_soft_notch_sidecars(
    raw_results: List[dict],
    output_dir: Path,
    *,
    soft_notch_library_path: Optional[Path],
) -> None:
    """Write soft_notch_applied.parquet + soft_notch_summary.json.

    See ``docs/handoffs/2026-05-11_adaptive-soft-notch.md`` §1f for schema.
    """
    import pandas as pd

    all_rows: list[dict] = []
    n_chunks_with_unmatched = 0
    for r in raw_results:
        rows = r.get("soft_notch_rows") or []
        all_rows.extend(rows)
        if any(row.get("is_drift") for row in rows):
            n_chunks_with_unmatched += 1

    batch_n_chunks = len(raw_results)
    columns = [
        "recording_path", "chunk_idx", "center_hz", "width_hz",
        "peak_db", "local_median_db", "cut_depth_db", "source",
        "is_drift", "intensity_drift_sigma",
    ]
    df = pd.DataFrame(all_rows, columns=columns)
    parquet_path = output_dir / "soft_notch_applied.parquet"
    df.to_parquet(parquet_path, index=False)
    log.info("Wrote soft_notch_applied.parquet (%d rows)", len(df))

    # Summary JSON
    unmatched_rate = (
        n_chunks_with_unmatched / batch_n_chunks if batch_n_chunks > 0 else 0.0
    )
    stale_warning_fired = unmatched_rate > _UNMATCHED_RATE_WARNING_THRESHOLD
    stale_reason = (
        f"unmatched_rate {unmatched_rate:.3f} > {_UNMATCHED_RATE_WARNING_THRESHOLD} "
        "— library may be stale; consider recalibrating"
    ) if stale_warning_fired else None

    library_metadata: dict = {}
    if soft_notch_library_path is not None:
        try:
            library = TonalLibrary.load(soft_notch_library_path)
            library_metadata = {
                "library_path": str(soft_notch_library_path),
                "library_rig_id": library.rig_id,
                "library_calibrated_at": library.calibrated_at,
                "library_n_entries": len(library.entries),
            }
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not re-load library for summary: %s", exc)
            library_metadata = {
                "library_path": str(soft_notch_library_path),
                "library_rig_id": None,
                "library_calibrated_at": None,
                "library_n_entries": 0,
            }
    else:
        library_metadata = {
            "library_path": None,
            "library_rig_id": None,
            "library_calibrated_at": None,
            "library_n_entries": 0,
        }

    summary = {
        **library_metadata,
        "batch_n_chunks": batch_n_chunks,
        "n_chunks_with_unmatched": n_chunks_with_unmatched,
        "unmatched_rate": unmatched_rate,
        "stale_library_warning_fired": stale_warning_fired,
        "stale_library_warning_reason": stale_reason,
    }
    summary_path = output_dir / "soft_notch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Wrote soft_notch_summary.json (unmatched_rate=%.3f, stale=%s)",
             unmatched_rate, stale_warning_fired)


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
    parser.add_argument(
        "--subtract-baseline", action="store_true",
        help=(
            "Apply per-frequency-bin temporal-baseline subtraction to the "
            "spectrogram BEFORE the CNN sees it. Lab-only opt-in: targets "
            "stationary equipment harmonics that the wild-trained CNN was "
            "never exposed to. Default off — wild-mouse runs (5970/3452/9252) "
            "must omit this flag for byte-identical results. See "
            "docs/handoffs/2026-05-08_pre-cnn-spectral-subtraction-lab.md."
        ),
    )
    parser.add_argument(
        "--subtraction-method", default="percentile",
        choices=["percentile", "median_envelope"],
        help=(
            "Statistic used for the per-bin baseline (only consulted with "
            "--subtract-baseline). 'percentile' = Boll 1979 floor subtraction "
            "(p10 default). 'median_envelope' = sliding-median per bin "
            "(captures slow band amplitude modulation; ~0.5 s kernel)."
        ),
    )
    parser.add_argument(
        "--soft-notch", type=str, default=None, metavar="PATH|auto",
        help=(
            "Apply the adaptive pre-CNN soft-notch filter to every WAV BEFORE "
            "STFT. Lab-only opt-in: removes rig-specific equipment tonals so "
            "the wild-trained CNN sees a cleaner signal. Pass a TonalLibrary "
            "JSON path (preferred — library mode) or the literal 'auto' "
            "(pure per-chunk auto-detect; no library). Default off — wild-mouse "
            "runs (5970/3452/9252) must omit this flag for byte-identical "
            "results. See docs/handoffs/2026-05-11_adaptive-soft-notch.md."
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    _batch_t0 = time.time()
    t0 = time.perf_counter()

    if args.subtract_baseline:
        log.info(
            "Pre-CNN spectral subtraction ENABLED  method=%s",
            args.subtraction_method,
        )

    soft_notch_enabled = args.soft_notch is not None
    soft_notch_library_path: Optional[Path] = None
    if soft_notch_enabled:
        if args.soft_notch != "auto":
            soft_notch_library_path = Path(args.soft_notch)
            if not soft_notch_library_path.exists():
                log.error("Soft-notch library not found: %s", soft_notch_library_path)
                return 2
            log.info(
                "Adaptive soft-notch ENABLED  library=%s", soft_notch_library_path,
            )
        else:
            log.info("Adaptive soft-notch ENABLED  mode=auto (no library; per-chunk detect)")

    results = run_batch(
        wav_dir=args.wav_dir,
        model_path=args.model,
        output_dir=args.output_dir,
        temperature_path=args.temperature,
        fp_filter_path=args.fp_filter,
        hysteresis_config_path=args.hysteresis_config,
        n_workers=args.workers,
        resume=not args.no_resume,
        subtract_baseline=args.subtract_baseline,
        subtraction_method=args.subtraction_method,
        soft_notch_enabled=soft_notch_enabled,
        soft_notch_library_path=soft_notch_library_path,
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
