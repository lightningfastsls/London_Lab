#!/usr/bin/env python3
"""Extract training-mode spectrogram PNGs from existing app detections.

The PyQt6 app saves review-mode PNGs (with axes/colorbars) that the CNN
can't use. This script goes back to the original WAV files and extracts
training-mode spectrograms matching the CNN's training format.

Usage:
    # Extract from all 5970 detections
    python scripts/extract_detection_spectrograms.py \
        --detections-dir USV_Detections/5970 \
        --wav-dir "5970 USV" \
        --output-dir batch_results/5970_cnn_eval

    # Extract + run CNN inference in one shot
    python scripts/extract_detection_spectrograms.py \
        --detections-dir USV_Detections/5970 \
        --wav-dir "5970 USV" \
        --output-dir batch_results/5970_cnn_eval \
        --classify --threshold 0.4

Output:
  <output-dir>/spectrograms/    — training-mode PNGs
  <output-dir>/detections_with_cnn.csv — original detection data + cnn_probability column
"""

import argparse
import csv
import json
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

from usv_spectrogram.detection.candidate import Candidate
from usv_spectrogram.detection.spectrogram_extractor import SpectrogramExtractor
from usv_spectrogram.models.cnn_classifier import USVClassifierCNN
from usv_spectrogram.models.evaluate import load_model_checkpoint


# Context padding to match training data extraction
CONTEXT_BEFORE_MS = 50.0
CONTEXT_AFTER_MS = 50.0


def load_and_preprocess_png(png_path: Path, device: torch.device) -> torch.Tensor:
    """Load a spectrogram PNG and prepare it for CNN inference.

    Matches USVDataset preprocessing: grayscale, per-image normalize to [0,1].
    """
    img = Image.open(png_path).convert('L')
    arr = np.array(img, dtype=np.float32)

    spec_min = arr.min()
    spec_max = arr.max()
    if spec_max > spec_min:
        arr = (arr - spec_min) / (spec_max - spec_min)
    else:
        arr = np.zeros_like(arr)

    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
    return tensor


def detection_to_candidate(row: dict, wav_name: str) -> Candidate:
    """Convert a detection CSV row into a Candidate for spectrogram extraction.

    Applies the same context padding (50ms each side) used when generating
    the training data, so the CNN sees the same framing it was trained on.
    """
    start_ms = float(row['start_time_s']) * 1000.0
    end_ms = float(row['end_time_s']) * 1000.0
    duration_ms = end_ms - start_ms

    context_start_ms = max(0.0, start_ms - CONTEXT_BEFORE_MS)
    context_end_ms = end_ms + CONTEXT_AFTER_MS

    det_idx = int(row.get('detection_index', 0))
    candidate_id = f"{wav_name}_{det_idx:05d}"

    return Candidate(
        source_file=Path(f"{wav_name}.wav"),
        candidate_id=candidate_id,
        start_ms=start_ms,
        end_ms=end_ms,
        duration_ms=duration_ms,
        context_start_ms=context_start_ms,
        context_end_ms=context_end_ms,
        peak_freq_hz=0.0,  # Not available from app detections
        peak_energy_db=0.0,
    )


def find_wav_file(wav_name: str, wav_dir: Path) -> Path:
    """Find WAV file, trying common naming patterns."""
    candidates = [
        wav_dir / f"{wav_name}.wav",
        wav_dir / f"{wav_name}.WAV",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract training-mode spectrograms from app detections + optional CNN inference"
    )
    parser.add_argument(
        "--detections-dir", required=True,
        help="Path to detections directory (e.g., USV_Detections/5970)"
    )
    parser.add_argument(
        "--wav-dir", required=True,
        help="Path to directory containing the original WAV files"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for extracted spectrograms and results CSV"
    )
    parser.add_argument(
        "--classify", action="store_true",
        help="Also run CNN inference on extracted spectrograms"
    )
    parser.add_argument(
        "--model", default=None,
        help="Path to CNN checkpoint (default: models/production/best_model.pt)"
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Decision value for above_dv flag (requires --classify)"
    )
    args = parser.parse_args()

    # Resolve paths
    detections_dir = Path(args.detections_dir)
    if not detections_dir.is_absolute():
        detections_dir = REPO_ROOT / detections_dir
    wav_dir = Path(args.wav_dir)
    if not wav_dir.is_absolute():
        wav_dir = REPO_ROOT / wav_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    spectrograms_dir = output_dir / "spectrograms"
    output_dir.mkdir(parents=True, exist_ok=True)
    spectrograms_dir.mkdir(parents=True, exist_ok=True)

    if not detections_dir.exists():
        print(f"Error: Detections directory not found: {detections_dir}")
        sys.exit(1)
    if not wav_dir.exists():
        print(f"Error: WAV directory not found: {wav_dir}")
        sys.exit(1)

    # Find detection subdirectories
    subdirs = sorted([
        d for d in detections_dir.iterdir()
        if d.is_dir() and (d / "detections_summary.csv").exists()
    ])

    if not subdirs:
        print(f"Error: No detection subdirectories found in {detections_dir}")
        sys.exit(1)

    # Optionally load CNN
    model = None
    device = None
    if args.classify:
        model_path = Path(args.model) if args.model else REPO_ROOT / "models" / "production" / "best_model.pt"
        if not model_path.exists():
            print(f"Error: Model not found at {model_path}")
            sys.exit(1)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading CNN model on {device}...")
        model, _ = load_model_checkpoint(model_path, USVClassifierCNN, str(device))
        model.eval()

    extractor = SpectrogramExtractor()

    print(f"Detection Spectrogram Extraction")
    print(f"{'='*60}")
    print(f"Detections:     {detections_dir}")
    print(f"WAV files:      {wav_dir}")
    print(f"Output:         {output_dir}")
    print(f"Subdirectories: {len(subdirs)}")
    print(f"CNN classify:   {args.classify}")
    if args.threshold is not None:
        print(f"Threshold (DV): {args.threshold}")
    print(f"{'='*60}\n")

    all_results = []
    total_extracted = 0
    total_failed = 0
    total_start = time.time()

    for idx, subdir in enumerate(subdirs, 1):
        wav_name = subdir.name
        csv_path = subdir / "detections_summary.csv"

        # Find the WAV file
        wav_path = find_wav_file(wav_name, wav_dir)
        if wav_path is None:
            print(f"[{idx}/{len(subdirs)}] {wav_name} — WAV not found, skipping")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[{idx}/{len(subdirs)}] {wav_name} — CSV error: {e}")
            continue

        if len(df) == 0:
            continue

        extracted = 0
        failed = 0

        for _, row in df.iterrows():
            # Build Candidate from detection row
            candidate = detection_to_candidate(row.to_dict(), wav_name)

            # Extract training-mode spectrogram
            spec_path = extractor.extract_single(
                candidate, wav_dir, spectrograms_dir, render_mode="training"
            )

            if spec_path is None:
                failed += 1
                continue

            extracted += 1

            # Build result row
            result = {
                'wav_file': wav_name,
                'detection_index': int(row.get('detection_index', 0)),
                'start_time_s': float(row.get('start_time_s', 0)),
                'end_time_s': float(row.get('end_time_s', 0)),
                'duration_ms': float(row.get('duration_ms', 0)),
                'app_max_prob': float(row.get('max_prob', 0)),
                'app_mean_prob': float(row.get('mean_prob', 0)),
                'user_action': row.get('user_action', ''),
                'spectrogram_path': str(spec_path),
            }

            # CNN inference if requested
            if model is not None:
                with torch.no_grad():
                    tensor = load_and_preprocess_png(spec_path, device)
                    cnn_prob = model.predict_proba(tensor).item()
                result['cnn_probability'] = round(cnn_prob, 6)
                if args.threshold is not None:
                    result['above_dv'] = cnn_prob >= args.threshold

            all_results.append(result)

        total_extracted += extracted
        total_failed += failed
        print(f"[{idx}/{len(subdirs)}] {wav_name}: {extracted} extracted, {failed} failed ({len(df)} detections)")

    # Save results
    results_csv = output_dir / "detections_with_spectrograms.csv"
    if all_results:
        with open(results_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

    total_elapsed = time.time() - total_start

    print(f"\n{'='*60}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Subdirectories processed: {len(subdirs)}")
    print(f"Spectrograms extracted:   {total_extracted}")
    print(f"Extraction failures:      {total_failed}")
    print(f"Total time:               {total_elapsed:.1f}s")
    print(f"\nResults CSV: {results_csv}")
    print(f"Spectrograms: {spectrograms_dir}")

    if args.classify and args.threshold is not None and all_results:
        above = sum(1 for r in all_results if r.get('above_dv', False))
        below = len(all_results) - above
        print(f"\nCNN Classification (DV={args.threshold}):")
        print(f"  Above DV: {above} ({above/len(all_results)*100:.1f}%)")
        print(f"  Below DV: {below} ({below/len(all_results)*100:.1f}%)")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
