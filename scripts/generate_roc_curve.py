#!/usr/bin/env python3
"""Generate ROC curve, PR curve, and threshold table from labeled data.

Two evaluation modes:

  sliding (default) — Runs SlidingInference on whole WAV files, matching the
      app's exact inference pipeline. Probabilities are in the same compressed
      space the app produces, so the resulting threshold applies directly.

  png — Loads pre-extracted spectrogram PNGs and runs CNN inference. Evaluates
      the CNN in isolation on training-format images (wider windows, different
      normalization). Useful for measuring raw model discrimination.

Usage:
    # Sliding mode (matches app pipeline) — recommended for picking a DV
    python scripts/generate_roc_curve.py --split test --wav-dir "5970 USV"

    # PNG mode (isolated CNN evaluation)
    python scripts/generate_roc_curve.py --mode png --split test

    # All splits, custom output
    python scripts/generate_roc_curve.py --split all --wav-dir "5970 USV" --output-dir results/eval/

Output:
  <output-dir>/roc_curve_<mode>_<split>.png
  <output-dir>/pr_curve_<mode>_<split>.png
  <output-dir>/threshold_table_<mode>_<split>.csv
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add src to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from usv_spectrogram.models.cnn_classifier import USVClassifierCNN
from usv_spectrogram.models.data_loader import USVDataset, pad_collate_fn
from usv_spectrogram.models.evaluate import (
    evaluate_model,
    load_model_checkpoint,
    plot_roc_curve_annotated,
    plot_pr_curve_annotated,
    generate_threshold_table,
)
from usv_spectrogram.app.core.audio_loader import AudioLoader
from usv_spectrogram.app.core.sliding_inference import SlidingInference


def build_evaluation_csv(
    split: str,
    splits_dir: Path,
    spectrograms_dir: Path,
    output_csv: Path,
) -> Path:
    """Join split CSV (labels) with extracted CSV (paths) into USVDataset-compatible CSV.

    The split CSV has: candidate_id, final_label, source_file
    The extracted CSV has: candidate_id, ..., spectrogram_path (Windows backslashes)

    We join on candidate_id, fix paths to point at local PNGs, and rename
    final_label -> label for USVDataset compatibility.
    """
    split_csv = splits_dir / f"{split}.csv"
    extracted_csv = spectrograms_dir / split / f"{split}_extracted.csv"

    if not split_csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv}")
    if not extracted_csv.exists():
        raise FileNotFoundError(f"Extracted CSV not found: {extracted_csv}")

    df_labels = pd.read_csv(split_csv)
    df_extracted = pd.read_csv(extracted_csv)

    # Join on candidate_id
    df = df_labels.merge(df_extracted[['candidate_id']], on='candidate_id', how='inner')

    # Build spectrogram_path pointing to local PNGs
    png_dir = spectrograms_dir / split
    df['spectrogram_path'] = df['candidate_id'].apply(
        lambda cid: str(png_dir / f"{cid}.png")
    )

    # Rename final_label -> label for USVDataset, and remap 'noise' -> 'Not USV'
    df = df.rename(columns={'final_label': 'label'})
    df['label'] = df['label'].replace({'noise': 'Not USV'})

    # Verify PNGs exist
    missing = df[~df['spectrogram_path'].apply(lambda p: Path(p).exists())]
    if len(missing) > 0:
        print(f"Warning: {len(missing)} PNGs missing, dropping from evaluation")
        df = df[df['spectrogram_path'].apply(lambda p: Path(p).exists())]

    df.to_csv(output_csv, index=False)
    print(f"Built evaluation CSV: {output_csv} ({len(df)} samples)")
    return output_csv


def evaluate_sliding_inference(
    splits: list[str],
    splits_dir: Path,
    candidates_csv: Path,
    wav_dir: Path,
    model_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate CNN using the app's SlidingInference pipeline on whole WAV files.

    For each labeled candidate, runs the full app pipeline (AudioLoader →
    SlidingInference) on the source WAV and reads the max probability in
    the candidate's time window. This produces probabilities in the same
    space as the app, so thresholds from the ROC curve apply directly.

    Returns:
        (y_true, y_proba) arrays aligned by candidate
    """
    # Load candidates metadata (has start_ms, end_ms, source_file)
    df_candidates = pd.read_csv(candidates_csv)

    # Load and merge all requested splits
    split_dfs = []
    for split in splits:
        split_csv = splits_dir / f"{split}.csv"
        if not split_csv.exists():
            raise FileNotFoundError(f"Split CSV not found: {split_csv}")
        df_split = pd.read_csv(split_csv)
        df_split['split'] = split
        split_dfs.append(df_split)

    df_labels = pd.concat(split_dfs, ignore_index=True)

    # Remap noise -> Not USV
    df_labels['final_label'] = df_labels['final_label'].replace({'noise': 'Not USV'})

    # Filter to valid labels
    df_labels = df_labels[df_labels['final_label'].isin(['USV', 'Not USV'])].copy()

    # Join with candidates to get timing info
    df = df_labels.merge(
        df_candidates[['candidate_id', 'start_ms', 'end_ms']],
        on='candidate_id',
        how='inner',
    )

    if len(df) < len(df_labels):
        print(f"Warning: {len(df_labels) - len(df)} candidates not found in {candidates_csv}")

    print(f"Evaluating {len(df)} labeled candidates from {df['source_file'].nunique()} WAV files")

    # Group by source WAV file
    grouped = df.groupby('source_file')

    # Initialize SlidingInference (uses same defaults as app)
    inference = SlidingInference(
        model_path=model_path,
        window_width_px=100,
        hop_px=10,
        batch_size=32,
        energy_threshold=0.1,
        enable_per_window_norm=True,
    )
    audio_loader = AudioLoader()

    all_probabilities = []
    all_labels = []
    total_start = time.time()

    for wav_idx, (wav_name, group) in enumerate(grouped, 1):
        wav_path = wav_dir / wav_name
        if not wav_path.exists():
            # Try without extension
            wav_path = wav_dir / f"{wav_name}"
            if not wav_path.exists():
                print(f"  [{wav_idx}/{len(grouped)}] {wav_name} — WAV not found, skipping {len(group)} candidates")
                continue

        print(f"  [{wav_idx}/{len(grouped)}] {wav_name} ({len(group)} candidates)...", end=" ", flush=True)

        # Load WAV and compute spectrogram (same as app)
        audio_data = audio_loader.load(wav_path)

        # Run SlidingInference (same as app)
        result = inference.infer(audio_data.spectrogram_db, audio_data.times)

        # For each labeled candidate, find max probability in its time window
        for _, row in group.iterrows():
            start_s = row['start_ms'] / 1000.0
            end_s = row['end_ms'] / 1000.0

            # Find inference windows overlapping this candidate
            mask = (result.times >= start_s) & (result.times <= end_s)

            if mask.any():
                max_prob = float(result.probabilities[mask].max())
            else:
                # Candidate might be very short — find nearest window
                center_s = (start_s + end_s) / 2.0
                nearest_idx = np.argmin(np.abs(result.times - center_s))
                max_prob = float(result.probabilities[nearest_idx])

            label = 1.0 if row['final_label'] == 'USV' else 0.0
            all_probabilities.append(max_prob)
            all_labels.append(label)

        print(f"done")

    elapsed = time.time() - total_start
    print(f"\nSliding inference complete: {len(all_labels)} candidates, {elapsed:.1f}s")

    return np.array(all_labels), np.array(all_probabilities)


def main():
    parser = argparse.ArgumentParser(description="Generate ROC curve and threshold analysis")
    parser.add_argument(
        "--mode", default="sliding", choices=["sliding", "png"],
        help="Evaluation mode: 'sliding' runs SlidingInference on WAVs (matches app), "
             "'png' evaluates on pre-extracted spectrogram PNGs (default: sliding)"
    )
    parser.add_argument(
        "--split", default="test", choices=["test", "val", "train", "all"],
        help="Which split to evaluate (default: test)"
    )
    parser.add_argument(
        "--model", default=None,
        help="Path to model checkpoint (default: models/production/best_model.pt)"
    )
    parser.add_argument(
        "--wav-dir", default=None,
        help="Path to WAV files directory (required for sliding mode)"
    )
    parser.add_argument(
        "--spectrograms-dir", default=None,
        help="Path to spectrograms_training/ dir (for png mode, default: <repo>/spectrograms_training)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for output files (default: models/production/evaluation/)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for inference (default: 32, png mode only)"
    )
    args = parser.parse_args()

    # Resolve paths
    model_path = Path(args.model) if args.model else REPO_ROOT / "models" / "production" / "best_model.pt"
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "models" / "production" / "evaluation"
    splits_dir = REPO_ROOT / "splits"

    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    # Determine splits to process
    if args.split == "all":
        splits = ["test", "val", "train"]
    else:
        splits = [args.split]

    mode_label = args.mode  # "sliding" or "png"

    if args.mode == "sliding":
        # --- Sliding inference mode: evaluate on whole WAV files ---
        wav_dir = Path(args.wav_dir) if args.wav_dir else REPO_ROOT / "5970 USV"
        if not wav_dir.is_absolute():
            wav_dir = REPO_ROOT / wav_dir
        if not wav_dir.exists():
            print(f"Error: WAV directory not found: {wav_dir}")
            print("Sliding mode requires --wav-dir pointing to the original WAV files")
            sys.exit(1)

        candidates_csv = REPO_ROOT / "data" / "candidates.csv"
        if not candidates_csv.exists():
            print(f"Error: Candidates CSV not found: {candidates_csv}")
            sys.exit(1)

        print(f"Mode: Sliding inference (matches app pipeline)")
        print(f"WAV dir: {wav_dir}")
        print(f"Model: {model_path}")
        print(f"Splits: {splits}")
        print(f"{'='*60}\n")

        y_true, y_proba = evaluate_sliding_inference(
            splits=splits,
            splits_dir=splits_dir,
            candidates_csv=candidates_csv,
            wav_dir=wav_dir,
            model_path=model_path,
        )

    else:
        # --- PNG mode: evaluate on pre-extracted spectrogram images ---
        spectrograms_dir = Path(args.spectrograms_dir) if args.spectrograms_dir else REPO_ROOT / "spectrograms_training"

        print(f"Mode: PNG (isolated CNN evaluation)")
        print(f"Spectrograms: {spectrograms_dir}")
        print(f"Model: {model_path}")
        print(f"Splits: {splits}")
        print(f"{'='*60}\n")

        all_probabilities = []
        all_labels = []

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Load model
        print(f"Loading model from: {model_path}")
        model, checkpoint = load_model_checkpoint(model_path, USVClassifierCNN, str(device))
        print(f"Model loaded (epoch {checkpoint.get('epoch', '?')})")

        for split in splits:
            print(f"\n--- Processing split: {split} ---")

            # Build joined CSV
            eval_csv = output_dir / f"{split}_eval.csv"
            build_evaluation_csv(split, splits_dir, spectrograms_dir, eval_csv)

            # Create dataset and loader
            dataset = USVDataset(eval_csv)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                collate_fn=pad_collate_fn,
            )

            # Run evaluation
            metrics, preds, probas, labels = evaluate_model(model, loader, str(device), verbose=True)

            all_probabilities.extend(probas)
            all_labels.extend(labels)

            print(f"  Samples: {len(labels)}, USV: {int(sum(labels))}, Not USV: {int(len(labels) - sum(labels))}")

        y_true = np.array(all_labels)
        y_proba = np.array(all_probabilities)

    split_label = args.split

    # Label for output files includes mode to distinguish sliding vs png results
    file_label = f"{mode_label}_{split_label}"

    n_usv = int(y_true.sum())
    n_not_usv = len(y_true) - n_usv

    print(f"\n{'='*60}")
    print(f"Generating curves: mode={mode_label}, split={split_label}")
    print(f"Samples: {len(y_true)} (USV: {n_usv}, Not USV: {n_not_usv})")
    print(f"Probability range: [{y_proba.min():.4f}, {y_proba.max():.4f}]")
    print(f"{'='*60}")

    # 1. Annotated ROC curve
    roc_path = output_dir / f"roc_curve_{file_label}.png"
    youden_threshold, roc_auc = plot_roc_curve_annotated(y_true, y_proba, roc_path)
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Youden's J optimal threshold: {youden_threshold:.4f}")

    # 2. Annotated PR curve
    pr_path = output_dir / f"pr_curve_{file_label}.png"
    f1_threshold, pr_auc = plot_pr_curve_annotated(y_true, y_proba, pr_path)
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"Best F1 threshold: {f1_threshold:.4f}")

    # 3. Threshold table
    table_path = output_dir / f"threshold_table_{file_label}.csv"
    generate_threshold_table(y_true, y_proba, table_path)

    print(f"\nOutputs saved to: {output_dir}")
    print(f"  ROC curve: {roc_path.name}")
    print(f"  PR curve:  {pr_path.name}")
    print(f"  Threshold table: {table_path.name}")
    print(f"\nRecommended: inspect the threshold table CSV to pick your decision value (DV).")


if __name__ == "__main__":
    main()
