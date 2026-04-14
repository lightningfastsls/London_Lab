#!/usr/bin/env python3
"""Train a second-stage false positive filter on event-level features.

Runs CNN inference + hysteresis detection on labeled recordings, extracts
EventFeatures for each detected event, labels them via collar-based matching,
then trains a LogisticRegression filter with cross-validated evaluation.

Usage:
    python scripts/train_fp_filter.py \
        --model models/matched_windows/best_model.pt \
        --labels data/unified_labels.json \
        --hysteresis-config results/hysteresis_optimization.json \
        --output models/matched_windows/fp_filter.pkl

Data flow:
    1. Load labels → group by recording → 229 recordings (126 pos, 103 noise)
    2. Cache SlidingInference results per recording
    3. Hysteresis detect → extract EventFeatures → label via collar matching
    4. 5-fold CV evaluation with F2 scoring
    5. Train final model on all events → save pickle
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap: ensure src/ is importable when running as script
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sklearn.model_selection import StratifiedKFold  # noqa: E402

from usv_spectrogram.app.core.audio_loader import AudioLoader  # noqa: E402
from usv_spectrogram.app.core.sliding_inference import (  # noqa: E402
    InferenceResult,
    SlidingInference,
)
from usv_spectrogram.postprocessing.event_features import (  # noqa: E402
    EventFeatures,
    extract_event_features,
)
from usv_spectrogram.postprocessing.event_scoring import (  # noqa: E402
    compute_f_beta,
    match_events_collar,
)
from usv_spectrogram.postprocessing.fp_filter import FalsePositiveFilter  # noqa: E402
from usv_spectrogram.postprocessing.hysteresis import (  # noqa: E402
    HysteresisConfig,
    USVEvent,
    hysteresis_detect,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading (same pattern as optimize_hysteresis.py)
# ---------------------------------------------------------------------------

@dataclass
class RecordingInfo:
    """A single recording with its ground-truth labels."""
    stem: str
    wav_path: Path
    gt_intervals: List[Tuple[float, float]]  # (start_s, end_s) or empty
    has_usvs: bool


def load_recording_info(
    labels_path: Path,
    wav_base_dir: Path,
) -> List[RecordingInfo]:
    """Load unified labels and group by recording."""
    with open(labels_path) as f:
        data = json.load(f)

    pos_by_stem: Dict[str, List[Tuple[float, float]]] = {}
    wav_paths: Dict[str, str] = {}

    for entry in data["positives"]:
        stem = entry["recording_stem"]
        pos_by_stem.setdefault(stem, []).append((entry["start_s"], entry["end_s"]))
        if stem not in wav_paths:
            wav_paths[stem] = entry["wav_path"]

    recordings: List[RecordingInfo] = []
    skipped = 0

    for stem, intervals in pos_by_stem.items():
        wav_abs = wav_base_dir / wav_paths[stem]
        if not wav_abs.exists():
            logger.warning("Missing WAV (positive): %s", wav_abs)
            skipped += 1
            continue
        recordings.append(RecordingInfo(
            stem=stem, wav_path=wav_abs,
            gt_intervals=sorted(intervals), has_usvs=True,
        ))

    for entry in data.get("noise_recordings", []):
        stem = entry["recording_stem"]
        if entry.get("wav_path") is None:
            skipped += 1
            continue
        wav_abs = wav_base_dir / entry["wav_path"]
        if not wav_abs.exists():
            logger.warning("Missing WAV (noise): %s", wav_abs)
            skipped += 1
            continue
        recordings.append(RecordingInfo(
            stem=stem, wav_path=wav_abs,
            gt_intervals=[], has_usvs=False,
        ))

    if skipped > 0:
        logger.warning("Skipped %d recordings with missing WAV files", skipped)
    if not recordings:
        raise RuntimeError("No recordings found — all WAV files missing?")

    logger.info(
        "Loaded %d recordings (%d positive, %d noise)",
        len(recordings),
        sum(1 for r in recordings if r.has_usvs),
        sum(1 for r in recordings if not r.has_usvs),
    )
    return recordings


# ---------------------------------------------------------------------------
# Inference caching (same pattern as optimize_hysteresis.py)
# ---------------------------------------------------------------------------

def cache_inference_results(
    recordings: List[RecordingInfo],
    inference: SlidingInference,
    audio_loader: AudioLoader,
    cache_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Tuple[Dict[str, InferenceResult], Dict[str, np.ndarray]]:
    """Run SlidingInference on all recordings, caching results and spectrograms.

    Returns:
        (inference_cache, spectrogram_cache) — both keyed by recording stem.
    """
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    inf_results: Dict[str, InferenceResult] = {}
    spec_cache: Dict[str, np.ndarray] = {}
    skipped = 0

    for i, rec in enumerate(recordings):
        if verbose and (i + 1) % 20 == 0:
            logger.info("Inference progress: %d / %d", i + 1, len(recordings))

        # Check npz cache (inference only — spectrograms are too large to cache)
        if cache_dir is not None:
            cache_file = cache_dir / f"{rec.stem}.npz"
            if cache_file.exists():
                cached = np.load(str(cache_file))
                inf_results[rec.stem] = InferenceResult(
                    probabilities=cached["probabilities"],
                    column_indices=cached["column_indices"],
                    times=cached["times"],
                )
                # Still need spectrogram for feature extraction
                try:
                    audio_data = audio_loader.load(rec.wav_path)
                    spec_cache[rec.stem] = audio_data.spectrogram_db
                except Exception as e:
                    logger.warning("Error loading spectrogram for %s: %s", rec.stem, e)
                    del inf_results[rec.stem]
                    skipped += 1
                continue

        try:
            audio_data = audio_loader.load(rec.wav_path)
        except Exception as e:
            logger.warning("Error loading %s: %s", rec.stem, e)
            skipped += 1
            continue

        try:
            result = inference.infer(audio_data.spectrogram_db, audio_data.times)
        except ValueError as e:
            logger.warning("Inference skipped for %s: %s", rec.stem, e)
            skipped += 1
            continue

        inf_results[rec.stem] = result
        spec_cache[rec.stem] = audio_data.spectrogram_db

        if cache_dir is not None:
            cache_file = cache_dir / f"{rec.stem}.npz"
            np.savez_compressed(
                str(cache_file),
                probabilities=result.probabilities,
                column_indices=result.column_indices,
                times=result.times,
            )

    if skipped > 0:
        logger.warning("Skipped %d recordings during inference", skipped)
    if not inf_results:
        raise RuntimeError("No inference results produced")

    logger.info("Inference complete: %d recordings", len(inf_results))
    return inf_results, spec_cache


# ---------------------------------------------------------------------------
# Event extraction + labeling
# ---------------------------------------------------------------------------

@dataclass
class LabeledEvent:
    """An event with its extracted features and ground-truth label."""
    features: EventFeatures
    is_usv: bool
    recording_stem: str


def extract_labeled_events(
    recordings: List[RecordingInfo],
    inference_cache: Dict[str, InferenceResult],
    spectrogram_cache: Dict[str, np.ndarray],
    hysteresis_config: HysteresisConfig,
    hop_px: int = 10,
    collar_s: float = 0.200,
) -> List[LabeledEvent]:
    """Extract events from all recordings and label them via collar matching.

    For each recording:
    1. Run hysteresis detection on CNN probabilities
    2. Extract EventFeatures for each event
    3. Label each event: TP events = True, FP events = False

    Events from noise recordings are all labeled False.
    """
    labeled: List[LabeledEvent] = []
    skipped_features = 0

    for rec in recordings:
        if rec.stem not in inference_cache:
            continue

        result = inference_cache[rec.stem]
        spectrogram = spectrogram_cache.get(rec.stem)
        if spectrogram is None or len(result.probabilities) == 0:
            continue

        events = hysteresis_detect(result.probabilities, result.times, hysteresis_config)
        if not events:
            continue

        # Label events via collar matching
        tp, fp, fn = match_events_collar(events, rec.gt_intervals, collar_s=collar_s)

        # Determine which events are TPs by checking overlap with ground truth.
        # match_events_collar returns counts, so we re-check per event.
        event_labels = _label_individual_events(events, rec.gt_intervals, collar_s)

        for event, is_usv in zip(events, event_labels):
            try:
                feats = extract_event_features(event, spectrogram, hop_px=hop_px)
            except (ValueError, IndexError):
                skipped_features += 1
                continue
            labeled.append(LabeledEvent(
                features=feats, is_usv=is_usv, recording_stem=rec.stem,
            ))

    if skipped_features > 0:
        logger.warning("Skipped %d events due to feature extraction errors", skipped_features)

    n_pos = sum(1 for e in labeled if e.is_usv)
    n_neg = len(labeled) - n_pos
    logger.info("Extracted %d labeled events (%d USV, %d FP)", len(labeled), n_pos, n_neg)
    return labeled


def _label_individual_events(
    events: List[USVEvent],
    gt_intervals: List[Tuple[float, float]],
    collar_s: float,
) -> List[bool]:
    """Label each event as TP (True) or FP (False) using collar matching.

    Mirrors the three-condition match from match_events_collar:
    onset within collar OR offset within collar OR temporal overlap.
    Returns per-event labels (no greedy one-to-one assignment).
    """
    if not gt_intervals:
        return [False] * len(events)

    labels = []
    for event in events:
        matched = any(
            abs(event.start_time_s - gt_start) <= collar_s
            or abs(event.end_time_s - gt_end) <= collar_s
            or (event.end_time_s > gt_start and event.start_time_s < gt_end)
            for gt_start, gt_end in gt_intervals
        )
        labels.append(matched)

    return labels


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_filter(
    labeled_events: List[LabeledEvent],
    n_folds: int = 5,
    seed: int = 42,
    excluded_features: set[str] | None = None,
) -> Dict:
    """Evaluate the FP filter with stratified k-fold CV on recordings.

    Splits by recording stem (not by event) to avoid data leakage — events
    from the same recording share the same CNN inference context.

    Returns dict with per-fold F2 scores and summary statistics.
    """
    # Group events by recording
    stems = sorted(set(e.recording_stem for e in labeled_events))
    if len(stems) < n_folds:
        logger.warning(
            "Only %d recordings but %d folds requested — reducing to %d folds",
            len(stems), n_folds, len(stems),
        )
        n_folds = len(stems)
    events_by_stem: Dict[str, List[LabeledEvent]] = {}
    for e in labeled_events:
        events_by_stem.setdefault(e.recording_stem, []).append(e)

    # Stratify by whether recording has any USV events
    stem_labels = np.array([
        1 if any(e.is_usv for e in events_by_stem[s]) else 0
        for s in stems
    ])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results = []

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(stems, stem_labels)):
        train_stems = {stems[i] for i in train_idx}
        val_stems = {stems[i] for i in val_idx}

        train_events = [e for e in labeled_events if e.recording_stem in train_stems]
        val_events = [e for e in labeled_events if e.recording_stem in val_stems]

        train_features = [e.features for e in train_events]
        train_labels = [e.is_usv for e in train_events]
        val_features = [e.features for e in val_events]
        val_labels = [e.is_usv for e in val_events]

        if len(set(train_labels)) < 2:
            logger.warning("Fold %d: training set has only one class, skipping", fold_i)
            continue

        filt = FalsePositiveFilter(excluded_features=excluded_features)
        filt.fit(train_features, train_labels)
        preds = filt.predict(val_features)

        tp = sum(1 for p, t in zip(preds, val_labels) if p is True and t is True)
        fp = sum(1 for p, t in zip(preds, val_labels) if p is True and t is False)
        fn = sum(1 for p, t in zip(preds, val_labels) if p is False and t is True)
        f2 = compute_f_beta(tp, fp, fn, beta=2.0)

        fold_results.append({
            "fold": fold_i,
            "f2": round(f2, 4),
            "tp": tp, "fp": fp, "fn": fn,
            "n_train": len(train_events),
            "n_val": len(val_events),
            "n_val_recordings": len(val_stems),
        })

    f2_scores = [fr["f2"] for fr in fold_results]
    return {
        "fold_results": fold_results,
        "mean_f2": round(float(np.mean(f2_scores)), 4) if f2_scores else 0.0,
        "std_f2": round(float(np.std(f2_scores)), 4) if f2_scores else 0.0,
        "n_folds_completed": len(fold_results),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a second-stage false positive filter on event-level features."
    )
    parser.add_argument(
        "--model", required=True, type=Path,
        help="Path to trained CNN checkpoint (.pt)",
    )
    parser.add_argument(
        "--labels", required=True, type=Path,
        help="Path to unified_labels.json",
    )
    parser.add_argument(
        "--hysteresis-config", required=True, type=Path,
        help="Path to hysteresis_optimization.json",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Output path for trained filter (.pkl)",
    )
    parser.add_argument(
        "--wav-base-dir", type=Path, default=REPO_ROOT,
        help="Base directory for resolving relative WAV paths (default: REPO_ROOT)",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Number of CV folds (default: 5)",
    )
    parser.add_argument(
        "--collar-s", type=float, default=0.200,
        help="Collar tolerance in seconds (default: 0.200)",
    )
    parser.add_argument(
        "--hop-px", type=int, default=10,
        help="SlidingInference hop in pixels (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Directory to cache inference .npz files",
    )
    parser.add_argument(
        "--exclude-features", nargs="*", default=[],
        help="Feature names to exclude (e.g., --exclude-features duration_windows)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load hysteresis config
    logger.info("Loading hysteresis config from %s", args.hysteresis_config)
    with open(args.hysteresis_config) as f:
        hyst_data = json.load(f)
    hysteresis_config = HysteresisConfig(**hyst_data["best_params"])

    # Load labels
    logger.info("Loading labels from %s", args.labels)
    recordings = load_recording_info(args.labels, args.wav_base_dir)

    # Run inference
    logger.info("Loading model from %s", args.model)
    inference = SlidingInference(model_path=args.model, hop_px=args.hop_px, device="cpu")
    audio_loader = AudioLoader()

    logger.info("Running inference on %d recordings...", len(recordings))
    inference_cache, spectrogram_cache = cache_inference_results(
        recordings, inference, audio_loader,
        cache_dir=args.cache_dir, verbose=args.verbose,
    )

    # Extract labeled events
    logger.info("Extracting events with hysteresis config: %s", hysteresis_config)
    labeled_events = extract_labeled_events(
        recordings, inference_cache, spectrogram_cache,
        hysteresis_config=hysteresis_config,
        hop_px=args.hop_px, collar_s=args.collar_s,
    )

    if len(labeled_events) < 10:
        logger.error("Too few labeled events (%d) — need at least 10", len(labeled_events))
        return 1

    # Feature exclusion
    excluded = set(args.exclude_features) if args.exclude_features else set()
    if excluded:
        logger.info("Excluding features: %s", excluded)

    # Cross-validation
    logger.info("Running %d-fold cross-validation...", args.cv_folds)
    cv_results = cross_validate_filter(
        labeled_events, n_folds=args.cv_folds, seed=args.seed,
        excluded_features=excluded,
    )

    # Train final model on all data
    logger.info("Training final model on all %d events...", len(labeled_events))
    all_features = [e.features for e in labeled_events]
    all_labels = [e.is_usv for e in labeled_events]

    final_filter = FalsePositiveFilter(excluded_features=excluded)
    final_filter.fit(all_features, all_labels)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    final_filter.save(args.output)

    # Feature importance report
    importances = final_filter.feature_importances()
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    # Print summary
    print(f"\n{'=' * 60}")
    print("FALSE POSITIVE FILTER — TRAINING RESULTS")
    print(f"{'=' * 60}")
    print(f"Total events: {len(labeled_events)} "
          f"({sum(1 for e in labeled_events if e.is_usv)} USV, "
          f"{sum(1 for e in labeled_events if not e.is_usv)} FP)")
    print(f"\nCross-validation ({cv_results['n_folds_completed']} folds):")
    print(f"  F2 (mean ± std): {cv_results['mean_f2']:.4f} ± {cv_results['std_f2']:.4f}")
    for fr in cv_results["fold_results"]:
        print(f"  Fold {fr['fold']}: F2={fr['f2']:.4f}  "
              f"TP={fr['tp']} FP={fr['fp']} FN={fr['fn']}  "
              f"(n_val={fr['n_val']})")
    print(f"\nFeature importances (|coefficient|):")
    for name, imp in sorted_imp:
        bar = "█" * int(imp * 20)
        print(f"  {name:25s} {imp:8.4f}  {bar}")
    print(f"\nModel saved to: {args.output}")
    print(f"{'=' * 60}")

    # Save training report alongside model
    report_path = args.output.with_suffix(".json")
    report = {
        "cv_results": cv_results,
        "feature_importances": importances,
        "n_events": len(labeled_events),
        "n_usv": sum(1 for e in labeled_events if e.is_usv),
        "n_fp": sum(1 for e in labeled_events if not e.is_usv),
        "excluded_features": sorted(excluded),
        "hysteresis_config": hyst_data["best_params"],
        "args": {
            "model": str(args.model),
            "labels": str(args.labels),
            "cv_folds": args.cv_folds,
            "collar_s": args.collar_s,
            "hop_px": args.hop_px,
            "seed": args.seed,
        },
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Training report saved to %s", report_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
