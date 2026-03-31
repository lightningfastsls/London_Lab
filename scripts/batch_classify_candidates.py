#!/usr/bin/env python3
"""Batch CNN classification of existing detection outputs.

Re-classifies detections from USV_Detections/ (produced by the PyQt6 app)
at a new decision value (DV) threshold chosen from the ROC curve analysis.

Usage:
    # Classify all detections in USV_Detections/ at DV=0.4
    python scripts/batch_classify_candidates.py --detections-dir USV_Detections --threshold 0.4

    # Classify specific subdirectories
    python scripts/batch_classify_candidates.py --detections-dir USV_Detections --threshold 0.4 \
        --files 2024-10-05_01-51-49_0002695 2024-10-05_18-03-44_0004142

    # Re-run CNN inference on PNGs (instead of using existing max_prob from CSV)
    python scripts/batch_classify_candidates.py --detections-dir USV_Detections --threshold 0.4 --reinfer

Input: USV_Detections/<wav_name>/ directories, each containing:
  - detections_summary.csv (with max_prob, mean_prob columns)
  - detection_NNN_*.png spectrogram images

Output CSV: batch_results/<group>/classified_detections.csv
  Columns: wav_file, detection_index, start_time_s, end_time_s, duration_ms,
           max_prob, mean_prob, above_dv, user_action
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

# Add src to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from usv_spectrogram.models.cnn_classifier import USVClassifierCNN
from usv_spectrogram.models.evaluate import load_model_checkpoint


def load_and_preprocess_png(png_path: Path, device: torch.device) -> torch.Tensor:
    """Load a spectrogram PNG and prepare it for CNN inference.

    Matches USVDataset preprocessing: convert to grayscale, per-image normalize to [0,1].
    """
    img = Image.open(png_path).convert('L')
    arr = np.array(img, dtype=np.float32)

    # Per-image normalization (same as USVDataset)
    spec_min = arr.min()
    spec_max = arr.max()
    if spec_max > spec_min:
        arr = (arr - spec_min) / (spec_max - spec_min)
    else:
        arr = np.zeros_like(arr)

    # Shape: (1, 1, H, W) — batch=1, channels=1
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
    return tensor


def main():
    parser = argparse.ArgumentParser(
        description="Batch classify existing USV detections at a new decision value"
    )
    parser.add_argument(
        "--detections-dir", required=True,
        help="Path to USV_Detections/ directory (contains per-WAV subdirectories)"
    )
    parser.add_argument(
        "--threshold", type=float, required=True,
        help="Decision value (DV) — detections with max_prob >= DV are flagged for review"
    )
    parser.add_argument(
        "--files", nargs="*", default=None,
        help="Specific subdirectory names to process (default: all)"
    )
    parser.add_argument(
        "--model", default=None,
        help="Path to CNN checkpoint (only needed with --reinfer)"
    )
    parser.add_argument(
        "--reinfer", action="store_true",
        help="Re-run CNN inference on PNGs instead of using max_prob from CSV"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: batch_results/classified/)"
    )
    parser.add_argument(
        "--group-name", default="classified",
        help="Name for this batch run (default: 'classified')"
    )
    args = parser.parse_args()

    # Resolve paths
    detections_dir = Path(args.detections_dir)
    if not detections_dir.is_absolute():
        detections_dir = REPO_ROOT / detections_dir
    if not detections_dir.exists():
        print(f"Error: Detections directory not found: {detections_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "batch_results" / args.group_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find detection subdirectories
    if args.files:
        subdirs = [detections_dir / f for f in args.files]
        missing = [d for d in subdirs if not d.exists()]
        if missing:
            print(f"Error: Subdirectories not found: {[m.name for m in missing]}")
            sys.exit(1)
    else:
        subdirs = sorted([
            d for d in detections_dir.iterdir()
            if d.is_dir() and (d / "detections_summary.csv").exists()
        ])

    if not subdirs:
        print(f"Error: No detection subdirectories with detections_summary.csv found in {detections_dir}")
        sys.exit(1)

    # Optionally load model for re-inference
    model = None
    device = None
    if args.reinfer:
        model_path = Path(args.model) if args.model else REPO_ROOT / "models" / "production" / "best_model.pt"
        if not model_path.exists():
            print(f"Error: Model not found at {model_path}")
            sys.exit(1)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model on {device}...")
        model, checkpoint = load_model_checkpoint(model_path, USVClassifierCNN, str(device))
        model.eval()

    print(f"Batch Classification")
    print(f"{'='*60}")
    print(f"Detections dir: {detections_dir}")
    print(f"Subdirectories: {len(subdirs)}")
    print(f"Threshold (DV): {args.threshold}")
    print(f"Mode:           {'Re-inference from PNGs' if args.reinfer else 'Using existing max_prob'}")
    print(f"Output:         {output_dir}")
    print(f"{'='*60}\n")

    all_results = []
    file_summaries = []
    total_start = time.time()

    for idx, subdir in enumerate(subdirs, 1):
        wav_name = subdir.name
        csv_path = subdir / "detections_summary.csv"

        print(f"[{idx}/{len(subdirs)}] {wav_name}")

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  Error reading CSV: {e}")
            file_summaries.append({
                'wav_name': wav_name, 'detections': 0,
                'above_dv': 0, 'below_dv': 0, 'error': str(e)
            })
            continue

        if len(df) == 0:
            file_summaries.append({
                'wav_name': wav_name, 'detections': 0,
                'above_dv': 0, 'below_dv': 0, 'error': None
            })
            continue

        above_count = 0
        below_count = 0

        for _, row in df.iterrows():
            # Get probability — either from CSV or re-inference
            if args.reinfer and model is not None:
                # Find the PNG for this detection
                det_idx = int(row.get('detection_index', 0))
                pngs = list(subdir.glob(f"detection_{det_idx:03d}_*.png"))
                if not pngs:
                    continue
                with torch.no_grad():
                    tensor = load_and_preprocess_png(pngs[0], device)
                    prob = model.predict_proba(tensor).item()
            else:
                prob = float(row.get('max_prob', 0))

            above_dv = prob >= args.threshold

            if above_dv:
                above_count += 1
            else:
                below_count += 1

            all_results.append({
                'wav_file': row.get('wav_file', wav_name),
                'detection_index': int(row.get('detection_index', 0)),
                'start_time_s': round(float(row.get('start_time_s', 0)), 6),
                'end_time_s': round(float(row.get('end_time_s', 0)), 6),
                'duration_ms': round(float(row.get('duration_ms', 0)), 2),
                'max_prob': round(prob, 6),
                'mean_prob': round(float(row.get('mean_prob', 0)), 6),
                'above_dv': above_dv,
                'user_action': row.get('user_action', ''),
            })

        print(f"  {len(df)} detections: {above_count} above DV, {below_count} below DV")

        file_summaries.append({
            'wav_name': wav_name,
            'detections': len(df),
            'above_dv': above_count,
            'below_dv': below_count,
            'error': None,
        })

    # Save results
    results_csv = output_dir / f"classified_{args.group_name}.csv"
    if all_results:
        with open(results_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

    # Save above-DV only (the review list)
    review_csv = output_dir / f"review_list_{args.group_name}.csv"
    above_results = [r for r in all_results if r['above_dv']]
    if above_results:
        with open(review_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=above_results[0].keys())
            writer.writeheader()
            writer.writerows(above_results)

    # Save summary
    summary_csv = output_dir / f"file_summary_{args.group_name}.csv"
    if file_summaries:
        with open(summary_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=file_summaries[0].keys())
            writer.writeheader()
            writer.writerows(file_summaries)

    total_elapsed = time.time() - total_start
    total_detections = sum(s['detections'] for s in file_summaries)
    total_above = sum(s['above_dv'] for s in file_summaries)
    total_below = sum(s['below_dv'] for s in file_summaries)

    print(f"\n{'='*60}")
    print(f"BATCH CLASSIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"Files processed:       {len(subdirs)}")
    print(f"Total detections:      {total_detections}")
    print(f"Above DV (>={args.threshold}):  {total_above} ({total_above/max(total_detections,1)*100:.1f}%)")
    print(f"Below DV (<{args.threshold}):   {total_below} ({total_below/max(total_detections,1)*100:.1f}%)")
    print(f"Total time:            {total_elapsed:.1f}s")
    print(f"\nAll detections:        {results_csv}")
    print(f"Review list (above DV): {review_csv}")
    print(f"File summary:          {summary_csv}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
