#!/usr/bin/env python3
"""Optimize hysteresis detection parameters using cross-validated F2 scoring.

Grid search over onset, sustain, gap_fill, and min_duration parameters with
5-fold stratified cross-validation. Uses collar-based event matching (±200ms)
and micro-averaged F2 scoring.

Usage:
    python scripts/optimize_hysteresis.py \
        --model models/matched_windows/best_model.pt \
        --labels data/unified_labels.json \
        --output results/hysteresis_optimization.json

Data flow:
    1. Load labels → group by recording → 229 recordings (126 pos, 103 noise)
    2. Cache SlidingInference results per recording
    3. StratifiedKFold(5) on has_usvs
    4. Per fold, per param combo: hysteresis_detect → collar match → micro F2
    5. Report best params + 1SD params → save JSON
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
from usv_spectrogram.app.core.sliding_inference import InferenceResult, SlidingInference  # noqa: E402
from usv_spectrogram.postprocessing.event_scoring import compute_f_beta, match_events_collar  # noqa: E402
from usv_spectrogram.postprocessing.hysteresis import HysteresisConfig, hysteresis_detect  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class RecordingInfo:
    """A single recording with its ground-truth labels."""
    stem: str
    wav_path: Path  # Absolute path to WAV
    gt_intervals: List[Tuple[float, float]]  # (start_s, end_s) or empty for noise
    has_usvs: bool


def load_recording_info(
    labels_path: Path,
    wav_base_dir: Path,
) -> List[RecordingInfo]:
    """Load unified labels and group by recording.

    Returns list of RecordingInfo with resolved WAV paths.
    Logs warnings for missing WAV files and skips them.
    """
    with open(labels_path) as f:
        data = json.load(f)

    # Group positives by recording_stem
    pos_by_stem: Dict[str, List[Tuple[float, float]]] = {}
    wav_paths: Dict[str, str] = {}  # stem -> relative wav_path

    for entry in data["positives"]:
        stem = entry["recording_stem"]
        pos_by_stem.setdefault(stem, []).append((entry["start_s"], entry["end_s"]))
        if stem not in wav_paths:
            wav_paths[stem] = entry["wav_path"]

    # Build recording list
    recordings: List[RecordingInfo] = []
    skipped = 0

    # Positive recordings
    for stem, intervals in pos_by_stem.items():
        wav_abs = wav_base_dir / wav_paths[stem]
        if not wav_abs.exists():
            logger.warning("Missing WAV (positive): %s", wav_abs)
            skipped += 1
            continue
        recordings.append(RecordingInfo(
            stem=stem,
            wav_path=wav_abs,
            gt_intervals=sorted(intervals),
            has_usvs=True,
        ))

    # Noise recordings
    for entry in data.get("noise_recordings", []):
        stem = entry["recording_stem"]
        if entry.get("wav_path") is None:
            logger.warning("No wav_path for noise recording: %s", stem)
            skipped += 1
            continue
        wav_abs = wav_base_dir / entry["wav_path"]
        if not wav_abs.exists():
            logger.warning("Missing WAV (noise): %s", wav_abs)
            skipped += 1
            continue
        recordings.append(RecordingInfo(
            stem=stem,
            wav_path=wav_abs,
            gt_intervals=[],
            has_usvs=False,
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
# Inference caching
# ---------------------------------------------------------------------------

def cache_inference_results(
    recordings: List[RecordingInfo],
    inference: SlidingInference,
    audio_loader: AudioLoader,
    cache_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Dict[str, InferenceResult]:
    """Run SlidingInference on all recordings, with optional .npz caching.

    Args:
        recordings: List of RecordingInfo.
        inference: SlidingInference instance.
        audio_loader: AudioLoader instance.
        cache_dir: Optional directory to cache .npz files.
        verbose: Log progress for each recording.

    Returns:
        Dict mapping recording stem -> InferenceResult.
    """
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, InferenceResult] = {}
    skipped = 0

    for i, rec in enumerate(recordings):
        if verbose and (i + 1) % 20 == 0:
            logger.info("Inference progress: %d / %d", i + 1, len(recordings))

        # Check cache
        if cache_dir is not None:
            cache_file = cache_dir / f"{rec.stem}.npz"
            if cache_file.exists():
                cached = np.load(str(cache_file))
                results[rec.stem] = InferenceResult(
                    probabilities=cached["probabilities"],
                    column_indices=cached["column_indices"],
                    times=cached["times"],
                )
                continue

        # Load audio + compute spectrogram
        try:
            audio_data = audio_loader.load(rec.wav_path)
        except Exception as e:
            logger.warning("Error loading %s: %s", rec.stem, e)
            skipped += 1
            continue

        # Run inference
        try:
            result = inference.infer(audio_data.spectrogram_db, audio_data.times)
        except ValueError as e:
            logger.warning("Inference skipped for %s: %s", rec.stem, e)
            skipped += 1
            continue

        results[rec.stem] = result

        # Save cache
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
    if not results:
        raise RuntimeError("No inference results produced — all recordings failed?")

    logger.info("Inference complete: %d recordings cached", len(results))
    return results


# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

def build_grid() -> List[Dict]:
    """Build parameter grid: 8 onset × 7 sustain × 6 gap × 8 min_dur combos.

    Constraint: sustain <= onset (enforced by HysteresisConfig).
    """
    onsets = [round(0.60 + i * 0.05, 2) for i in range(8)]        # 0.60..0.95
    sustains = [round(0.20 + i * 0.05, 2) for i in range(7)]      # 0.20..0.50
    gaps = list(range(0, 6))                                        # 0..5
    min_durs = [3, 4, 5, 6, 7, 8, 9, 10]                           # 8 values

    combos = []
    for onset in onsets:
        for sustain in sustains:
            if sustain > onset:
                continue
            for gap in gaps:
                for min_dur in min_durs:
                    combos.append({
                        "onset_threshold": onset,
                        "sustain_threshold": sustain,
                        "gap_fill_windows": gap,
                        "min_duration_windows": min_dur,
                    })

    return combos


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_params(
    params: Dict,
    recordings: List[RecordingInfo],
    inference_cache: Dict[str, InferenceResult],
    collar_s: float,
) -> Tuple[float, int, int, int]:
    """Evaluate one param combo on a set of recordings.

    Returns (f2, total_tp, total_fp, total_fn) with micro-averaging.
    """
    config = HysteresisConfig(**params)
    total_tp, total_fp, total_fn = 0, 0, 0

    for rec in recordings:
        if rec.stem not in inference_cache:
            continue

        result = inference_cache[rec.stem]
        if len(result.probabilities) == 0:
            continue

        events = hysteresis_detect(result.probabilities, result.times, config)
        tp, fp, fn = match_events_collar(events, rec.gt_intervals, collar_s=collar_s)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    f2 = compute_f_beta(total_tp, total_fp, total_fn, beta=2.0)
    return f2, total_tp, total_fp, total_fn


# ---------------------------------------------------------------------------
# Cross-validated grid search
# ---------------------------------------------------------------------------

def run_grid_search(
    recordings: List[RecordingInfo],
    inference_cache: Dict[str, InferenceResult],
    n_folds: int = 5,
    collar_s: float = 0.200,
    seed: int = 42,
    verbose: bool = False,
) -> Dict:
    """Run stratified k-fold CV grid search.

    Returns dict with best_params, one_se_params, fold_results, metadata,
    and grid_summary (top 10 combos).
    """
    grid = build_grid()
    logger.info("Grid size: %d combos × %d folds = %d evaluations",
                len(grid), n_folds, len(grid) * n_folds)

    # Prepare fold splits
    labels = np.array([1 if r.has_usvs else 0 for r in recordings])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_indices = list(skf.split(np.zeros(len(recordings)), labels))

    # Score each combo: mean F2 across folds
    combo_scores = np.zeros((len(grid), n_folds))

    t0 = time.time()
    for ci, params in enumerate(grid):
        if verbose and (ci + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (ci + 1) / elapsed
            eta = (len(grid) - ci - 1) / rate
            logger.info(
                "Grid progress: %d / %d (%.1f combos/s, ETA %.0fs)",
                ci + 1, len(grid), rate, eta,
            )

        for fi, (train_idx, val_idx) in enumerate(fold_indices):
            val_recs = [recordings[i] for i in val_idx]
            f2, _, _, _ = evaluate_params(params, val_recs, inference_cache, collar_s)
            combo_scores[ci, fi] = f2

    elapsed = time.time() - t0
    logger.info("Grid search complete in %.1fs", elapsed)

    # Compute mean and std across folds
    mean_scores = combo_scores.mean(axis=1)
    std_scores = combo_scores.std(axis=1)

    # Best params
    best_idx = int(np.argmax(mean_scores))
    best_params = grid[best_idx]
    best_mean = float(mean_scores[best_idx])
    best_std = float(std_scores[best_idx])

    # 1SD rule: find most conservative params within 1 raw standard deviation
    # of best. Uses raw std (not SEM = std/sqrt(k)) intentionally — the wider
    # band favors conservative params, reducing overfitting risk for production.
    threshold = best_mean - best_std
    one_se_idx = _find_one_se_params(grid, mean_scores, threshold)
    one_se_params = grid[one_se_idx]

    # Fold-level details for best params
    fold_results = []
    for fi, (train_idx, val_idx) in enumerate(fold_indices):
        val_recs = [recordings[i] for i in val_idx]
        f2, tp, fp, fn = evaluate_params(best_params, val_recs, inference_cache, collar_s)
        fold_results.append({
            "fold": fi,
            "f2": round(f2, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "n_val_recordings": len(val_recs),
            "n_val_positive": sum(1 for r in val_recs if r.has_usvs),
        })

    # Top 10 grid summary
    top_indices = np.argsort(mean_scores)[::-1][:10]
    grid_summary = []
    for idx in top_indices:
        grid_summary.append({
            "params": grid[idx],
            "mean_f2": round(float(mean_scores[idx]), 4),
            "std_f2": round(float(std_scores[idx]), 4),
            "fold_scores": [round(float(s), 4) for s in combo_scores[idx]],
        })

    return {
        "best_params": best_params,
        "best_f2_mean": round(best_mean, 4),
        "best_f2_std": round(best_std, 4),
        "one_se_params": one_se_params,
        "one_se_f2_mean": round(float(mean_scores[one_se_idx]), 4),
        "fold_results": fold_results,
        "metadata": {
            "n_recordings": len(recordings),
            "n_positive": sum(1 for r in recordings if r.has_usvs),
            "n_noise": sum(1 for r in recordings if not r.has_usvs),
            "n_folds": n_folds,
            "collar_s": collar_s,
            "seed": seed,
            "grid_size": len(grid),
            "elapsed_s": round(elapsed, 1),
        },
        "grid_summary": grid_summary,
    }


def _find_one_se_params(
    grid: List[Dict],
    mean_scores: np.ndarray,
    threshold: float,
) -> int:
    """Find most conservative params with mean F2 >= threshold.

    "Conservative" = highest onset, highest sustain, lowest gap, highest min_dur.
    All four directions make the detector stricter (fewer, higher-confidence events),
    reducing overfitting risk for production use.
    """
    candidates = []
    for i, params in enumerate(grid):
        if mean_scores[i] >= threshold:
            # Conservative key: all directions favor stricter detection
            simplicity = (
                params["onset_threshold"],        # higher = harder to seed
                params["sustain_threshold"],       # higher = harder to extend
                -params["gap_fill_windows"],       # lower = less gap merging
                params["min_duration_windows"],    # higher = drops short events
            )
            candidates.append((simplicity, i))

    if not candidates:
        # Fallback: return best
        return int(np.argmax(mean_scores))

    candidates.sort(reverse=True)  # Most "simple" first
    return candidates[0][1]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize hysteresis detection parameters via cross-validated F2 grid search."
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
        "--wav-base-dir", type=Path, default=REPO_ROOT,
        help="Base directory for resolving relative WAV paths (default: REPO_ROOT)",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Output JSON path for results",
    )
    parser.add_argument(
        "--n-folds", type=int, default=5,
        help="Number of CV folds (default: 5)",
    )
    parser.add_argument(
        "--collar-s", type=float, default=0.200,
        help="Collar tolerance in seconds (default: 0.200)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for fold splitting (default: 42)",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Directory to cache .npz inference results",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose progress logging",
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

    # Load labels
    logger.info("Loading labels from %s", args.labels)
    recordings = load_recording_info(args.labels, args.wav_base_dir)

    # Initialize inference
    logger.info("Loading model from %s", args.model)
    inference = SlidingInference(model_path=args.model, device="cpu")
    audio_loader = AudioLoader()

    # Cache inference results
    logger.info("Running inference on %d recordings...", len(recordings))
    inference_cache = cache_inference_results(
        recordings, inference, audio_loader,
        cache_dir=args.cache_dir,
        verbose=args.verbose,
    )

    # Run grid search
    logger.info("Starting grid search...")
    results = run_grid_search(
        recordings, inference_cache,
        n_folds=args.n_folds,
        collar_s=args.collar_s,
        seed=args.seed,
        verbose=args.verbose,
    )

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print("HYSTERESIS OPTIMIZATION RESULTS")
    print(f"{'=' * 60}")
    print(f"Best F2 (mean ± std): {results['best_f2_mean']:.4f} ± {results['best_f2_std']:.4f}")
    print(f"Best params: {results['best_params']}")
    print(f"1SD params:  {results['one_se_params']} (F2={results['one_se_f2_mean']:.4f})")
    print(f"\nFold results (best params):")
    for fr in results["fold_results"]:
        print(f"  Fold {fr['fold']}: F2={fr['f2']:.4f}  TP={fr['tp']} FP={fr['fp']} FN={fr['fn']}")
    print(f"\nSaved to: {args.output}")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
