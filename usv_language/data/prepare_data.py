"""CLI script for end-to-end USV bout data preparation.

Usage:
    python -m usv_language.data.prepare_data \\
        --detection-dir USV_Detections/5970 \\
        --wav-dir "5970 USV" \\
        --output-dir usv_language/prepared_data

Pipeline steps:
    1. Extract bouts from detection results
    2. Compute spectrograms for each bout
    3. Split recordings into train/val/test
    4. Compute normalization stats on training set
    5. Save processed data
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Bootstrap: ensure project root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from usv_language.data.bout_extractor import BoutExtractionConfig, BoutExtractor
from usv_language.data.dataset import TransformerDataConfig, split_recordings
from usv_language.data.normalization import (
    compute_normalization_stats,
    save_normalization_stats,
)
from usv_language.data.spectrogram import BoutSpectrogramConfig, compute_bout_spectrogram

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare USV bout data for transformer training",
    )
    parser.add_argument(
        "--detection-dir",
        type=Path,
        required=True,
        help="Directory containing detection results (subdirs per recording)",
    )
    parser.add_argument(
        "--wav-dir",
        type=Path,
        default=None,
        help="Directory containing WAV files (to resolve stems from CSV)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--bout-gap-ms",
        type=float,
        default=500.0,
        help="Gap threshold for bout grouping (ms)",
    )
    parser.add_argument(
        "--padding-ms",
        type=float,
        default=200.0,
        help="Context padding around bouts (ms)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Maximum sequence length for chunks",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract bouts
    logger.info("Step 1: Extracting bouts from %s", args.detection_dir)
    bout_config = BoutExtractionConfig(
        bout_gap_threshold_ms=args.bout_gap_ms,
        context_padding_ms=args.padding_ms,
    )
    extractor = BoutExtractor(bout_config)
    bouts = extractor.extract_from_detection_dir(
        args.detection_dir, wav_dir=args.wav_dir,
    )
    logger.info("  Found %d bouts from detection results", len(bouts))

    if not bouts:
        logger.warning("No bouts found. Exiting.")
        return

    # Step 2: Compute spectrograms
    logger.info("Step 2: Computing spectrograms")
    spec_config = BoutSpectrogramConfig()
    spectrograms: list[np.ndarray] = []
    recording_ids: list[str] = []
    valid_bouts = []

    for i, bout in enumerate(bouts):
        try:
            spec = compute_bout_spectrogram(
                bout.source_file,
                bout.start_time_s,
                bout.end_time_s,
                spec_config,
            )
            if spec.shape[1] > 0:
                spectrograms.append(spec)
                recording_ids.append(Path(bout.source_file).stem)
                valid_bouts.append(bout)
        except Exception as e:
            logger.warning("  Skipping bout %d: %s", i, e)

    logger.info("  Computed %d valid spectrograms", len(spectrograms))

    if not spectrograms:
        logger.warning("No valid spectrograms. Exiting.")
        return

    # Step 3: Split recordings
    logger.info("Step 3: Splitting recordings into train/val/test")
    dataset_config = TransformerDataConfig(
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    unique_recordings = list(set(recording_ids))
    train_ids, val_ids, test_ids = split_recordings(
        unique_recordings, dataset_config,
    )
    logger.info(
        "  Split: %d train, %d val, %d test recordings",
        len(train_ids), len(val_ids), len(test_ids),
    )

    # Partition spectrograms by split
    train_set = set(train_ids)
    val_set = set(val_ids)
    train_specs = [s for s, r in zip(spectrograms, recording_ids) if r in train_set]
    val_specs = [s for s, r in zip(spectrograms, recording_ids) if r in val_set]
    test_specs = [
        s for s, r in zip(spectrograms, recording_ids)
        if r not in train_set and r not in val_set
    ]

    logger.info(
        "  Spectrograms: %d train, %d val, %d test",
        len(train_specs), len(val_specs), len(test_specs),
    )

    # Step 4: Compute normalization stats (training set only)
    logger.info("Step 4: Computing normalization stats from training set")
    if train_specs:
        stats = compute_normalization_stats(train_specs)
        stats_path = output_dir / "normalization_stats.npz"
        save_normalization_stats(stats, stats_path)
        logger.info(
            "  Stats computed from %d frames, saved to %s",
            stats.count, stats_path,
        )

    # Step 5: Save processed data
    logger.info("Step 5: Saving processed data")
    for split_name, split_specs, split_rec_ids in [
        ("train", train_specs, [r for r, s in zip(recording_ids, spectrograms) if r in train_set]),
        ("val", val_specs, [r for r, s in zip(recording_ids, spectrograms) if r in val_set]),
        ("test", test_specs, [r for r, s in zip(recording_ids, spectrograms) if r not in train_set and r not in val_set]),
    ]:
        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True)
        for i, (spec, rec_id) in enumerate(zip(split_specs, split_rec_ids)):
            np.save(str(split_dir / f"spec_{i:05d}.npy"), spec)

        # Save manifest
        manifest = {
            "n_spectrograms": len(split_specs),
            "recording_ids": split_rec_ids,
        }
        with open(split_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    # Save pipeline config
    config_out = {
        "bout_extraction": {
            "bout_gap_threshold_ms": bout_config.bout_gap_threshold_ms,
            "context_padding_ms": bout_config.context_padding_ms,
            "min_bout_duration_ms": bout_config.min_bout_duration_ms,
            "max_bout_duration_ms": bout_config.max_bout_duration_ms,
        },
        "spectrogram": {
            "sample_rate": spec_config.sample_rate,
            "n_fft": spec_config.n_fft,
            "hop_length": spec_config.hop_length,
            "freq_min_hz": spec_config.freq_min_hz,
            "freq_max_hz": spec_config.freq_max_hz,
            "window": spec_config.window,
            "eps": spec_config.eps,
        },
        "dataset": {
            "max_seq_len": dataset_config.max_seq_len,
            "overlap_ratio": dataset_config.overlap_ratio,
            "train_split": dataset_config.train_split,
            "val_split": dataset_config.val_split,
            "test_split": dataset_config.test_split,
            "batch_size": dataset_config.batch_size,
            "seed": dataset_config.seed,
        },
        "n_bouts": len(valid_bouts),
        "n_spectrograms": len(spectrograms),
        "n_recordings": len(unique_recordings),
    }
    with open(output_dir / "pipeline_config.json", "w") as f:
        json.dump(config_out, f, indent=2)

    logger.info("Done! Output saved to %s", output_dir)


if __name__ == "__main__":
    main()
