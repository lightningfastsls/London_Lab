"""Mine hard negatives from CNN false positives.

Identifies regions where the CNN predicts high USV probability (> threshold)
but humans did not save them as detections. These are the model's mistakes
and make excellent hard negatives for retraining.

This implements active learning - using model errors to guide data collection.

Workflow:
1. Load trained CNN model
2. Scan unlabeled/partially-labeled recordings with sliding window
3. Load human-saved detections from JSON files
4. Find high-probability regions NOT saved by humans
5. Export candidates for manual review
6. Confirmed false positives - add to training data as hard negatives

Usage:
    python scripts/mine_hard_negatives.py \
        --model models/full_retrained_cnn/best_model.pt \
        --wav-dir "5970 USV" \
        --labeled-detections-dir data/labeled_detections \
        --output-dir data/hard_negative_candidates \
        --probability-threshold 0.3 \
        --buffer-ms 100
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from usv_spectrogram.detection.candidate import Candidate
from usv_spectrogram.detection.extraction_config import ExtractionConfig
from usv_spectrogram.detection.spectrogram_extractor import SpectrogramExtractor
from usv_spectrogram.io_wav import load_wav_mono
from usv_spectrogram.models.cnn_classifier import USVClassifierCNN
from usv_spectrogram._stft_core import compute_stft_frames_db


class SavedDetection(NamedTuple):
    """Represents a human-saved detection region."""
    start_time_s: float
    end_time_s: float

    def overlaps_with(self, time_s: float, window_ms: float, buffer_ms: float) -> bool:
        """Check if this detection overlaps with a candidate window + buffer.

        Args:
            time_s: Candidate center time in seconds
            window_ms: Extraction window size
            buffer_ms: Buffer zone to avoid near-misses

        Returns:
            True if regions overlap
        """
        window_s = window_ms / 1000.0
        buffer_s = buffer_ms / 1000.0

        # Candidate window with buffer
        cand_start = time_s - (window_s / 2) - buffer_s
        cand_end = time_s + (window_s / 2) + buffer_s

        # Check for overlap
        return not (cand_end < self.start_time_s or cand_start > self.end_time_s)


def load_saved_detections(detections_dir: Path, wav_filename: str) -> list[SavedDetection]:
    """Load saved detections for a specific WAV file.

    Args:
        detections_dir: Base directory containing detection subdirectories
        wav_filename: WAV filename (with .wav extension)

    Returns:
        List of SavedDetection objects
    """
    wav_stem = Path(wav_filename).stem
    detection_subdir = detections_dir / wav_stem

    detections = []

    # Check if detection directory exists
    if not detection_subdir.exists():
        return detections

    # Load from JSON files (format: detection_NNN_XXXs-YYYs.json)
    for json_path in detection_subdir.glob("detection_*.json"):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Extract core time (the actual USV boundaries)
            if "core_time" in data:
                start_s = data["core_time"]["start_s"]
                end_s = data["core_time"]["end_s"]
            else:
                # Fallback: parse from filename (detection_001_1.234s-1.456s.json)
                filename = json_path.stem
                times_part = filename.split('_', 2)[-1]  # Get "1.234s-1.456s" part
                start_str, end_str = times_part.replace('s', '').split('-')
                start_s = float(start_str)
                end_s = float(end_str)

            detections.append(SavedDetection(start_s, end_s))

        except (json.JSONDecodeError, KeyError, ValueError, IndexError) as e:
            print(f"Warning: Failed to parse {json_path.name}: {e}")
            continue

    return detections


def run_sliding_window_inference(
    audio: np.ndarray,
    sr: int,
    model: USVClassifierCNN,
    config: ExtractionConfig,
    device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run CNN inference on sliding windows across audio.

    Args:
        audio: Audio waveform
        sr: Sample rate
        model: Trained CNN model
        config: Extraction configuration
        device: Torch device

    Returns:
        Tuple of (probabilities, times, column_indices)
    """
    # Compute full spectrogram
    spec_db, times, freqs = compute_stft_frames_db(
        audio, sr,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        window='hann'
    )

    # Filter to target frequency range
    freq_mask = (freqs >= config.freq_min_hz) & (freqs <= config.freq_max_hz)
    spec_db_filtered = spec_db[freq_mask, :]

    n_cols = spec_db_filtered.shape[1]
    window_cols = config.window_width_cols

    probabilities = []
    window_times = []
    column_indices = []

    # Slide window across spectrogram
    for col_idx in range(n_cols - window_cols + 1):
        # Extract window
        window = spec_db_filtered[:, col_idx:col_idx + window_cols]

        # Convert to tensor (1, 1, H, W) for CNN
        window_tensor = torch.from_numpy(window).float().unsqueeze(0).unsqueeze(0)
        window_tensor = window_tensor.to(device)

        # Run inference
        with torch.no_grad():
            output = model(window_tensor)
            prob = torch.sigmoid(output).item()

        probabilities.append(prob)

        # Calculate center time of window
        center_col = col_idx + window_cols // 2
        center_time = times[center_col]
        window_times.append(center_time)
        column_indices.append(col_idx)

    return (
        np.array(probabilities),
        np.array(window_times),
        np.array(column_indices)
    )


def find_false_positive_candidates(
    probabilities: np.ndarray,
    times: np.ndarray,
    saved_detections: list[SavedDetection],
    threshold: float,
    window_ms: float,
    buffer_ms: float
) -> list[tuple[float, float]]:
    """Find high-probability regions not saved by humans.

    Args:
        probabilities: CNN predictions for each window
        times: Time in seconds for each window
        saved_detections: Human-saved detection regions
        threshold: Minimum probability to consider
        window_ms: Extraction window size
        buffer_ms: Buffer to avoid near-misses

    Returns:
        List of (time, probability) tuples for false positive candidates
    """
    candidates = []

    for time_s, prob in zip(times, probabilities):
        # Skip if below threshold
        if prob < threshold:
            continue

        # Check if overlaps with any saved detection
        is_saved = any(
            det.overlaps_with(time_s, window_ms, buffer_ms)
            for det in saved_detections
        )

        if not is_saved:
            # High probability but not saved = potential false positive
            candidates.append((time_s, prob))

    return candidates


def export_candidate(
    candidate_time_s: float,
    candidate_prob: float,
    candidate_idx: int,
    audio: np.ndarray,
    sr: int,
    wav_filename: str,
    output_dir: Path,
    config: ExtractionConfig
) -> tuple[Path, Path]:
    """Export a false positive candidate as PNG + JSON.

    Args:
        candidate_time_s: Center time of candidate
        candidate_prob: CNN probability
        candidate_idx: Index for naming
        audio: Full audio waveform
        sr: Sample rate
        wav_filename: Source WAV filename
        output_dir: Output directory
        config: Extraction configuration

    Returns:
        Tuple of (png_path, json_path)
    """
    # Create candidate object
    candidate = Candidate.create(
        start_time_s=candidate_time_s,
        duration_s=0.0,  # Will be overridden by extraction
        source_file=wav_filename,
        expand_ms=0.0
    )

    # Extract spectrogram
    extractor = SpectrogramExtractor(config=config)
    spec_result = extractor.extract_single(
        candidate=candidate,
        audio_data=audio,
        sr=sr,
        render_mode="training"
    )

    if spec_result is None:
        raise ValueError(f"Failed to extract spectrogram for candidate at {candidate_time_s:.3f}s")

    # Create output subdirectory for this WAV
    wav_stem = Path(wav_filename).stem
    output_subdir = output_dir / wav_stem
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    base_name = f"fp_candidate_{candidate_idx:03d}_{candidate_time_s:.3f}s_p{candidate_prob:.3f}"

    # Save PNG
    png_path = output_subdir / f"{base_name}.png"
    img = Image.fromarray(spec_result.spectrogram_uint8)
    img.save(png_path)

    # Save JSON metadata
    json_path = output_subdir / f"{base_name}.json"
    metadata = {
        "candidate_index": candidate_idx,
        "center_time_s": candidate_time_s,
        "cnn_probability": candidate_prob,
        "source_wav": wav_filename,
        "extraction_window": {
            "start_s": spec_result.extraction_start_s,
            "end_s": spec_result.extraction_end_s,
            "duration_ms": (spec_result.extraction_end_s - spec_result.extraction_start_s) * 1000
        },
        "spectrogram_shape": {
            "height": spec_result.spectrogram_uint8.shape[0],
            "width": spec_result.spectrogram_uint8.shape[1]
        },
        "label": "hard_negative_candidate",
        "note": "High CNN probability but not saved by human - potential false positive"
    }

    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return png_path, json_path


def main():
    parser = argparse.ArgumentParser(
        description="Mine hard negatives from CNN false positives",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained CNN model checkpoint (.pt file)"
    )

    parser.add_argument(
        "--wav-dir",
        type=Path,
        required=True,
        help="Directory containing WAV files to scan"
    )

    parser.add_argument(
        "--labeled-detections-dir",
        type=Path,
        required=True,
        help="Directory containing saved detection JSONs (subdirs per WAV)"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for hard negative candidates"
    )

    parser.add_argument(
        "--probability-threshold",
        type=float,
        default=0.3,
        help="Minimum CNN probability to consider (default: 0.3)"
    )

    parser.add_argument(
        "--buffer-ms",
        type=float,
        default=100.0,
        help="Buffer around saved detections to avoid near-misses (default: 100ms)"
    )

    parser.add_argument(
        "--max-candidates-per-file",
        type=int,
        default=50,
        help="Maximum candidates to export per WAV file (default: 50)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device for inference (default: auto-detect)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.model.exists():
        print(f"Error: Model file not found: {args.model}")
        return 1

    if not args.wav_dir.exists():
        print(f"Error: WAV directory not found: {args.wav_dir}")
        return 1

    if not args.labeled_detections_dir.exists():
        print(f"Warning: Labeled detections directory not found: {args.labeled_detections_dir}")
        print("Will treat all high-probability regions as candidates.")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.model}...")
    device = torch.device(args.device)

    checkpoint = torch.load(args.model, map_location=device)
    model = USVClassifierCNN()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded on {device}")

    # Extraction config (must match training data generation)
    config = ExtractionConfig()

    # Find WAV files
    wav_files = sorted(args.wav_dir.glob("*.wav"))
    if not wav_files:
        print(f"Error: No WAV files found in {args.wav_dir}")
        return 1

    print(f"Found {len(wav_files)} WAV files to scan")
    print(f"Probability threshold: {args.probability_threshold}")
    print(f"Buffer around saved detections: {args.buffer_ms}ms")

    # Process each WAV file
    total_candidates = 0
    summary_rows = []

    for wav_path in tqdm(wav_files, desc="Scanning WAV files"):
        wav_filename = wav_path.name

        # Load saved detections for this file
        saved_detections = load_saved_detections(
            args.labeled_detections_dir,
            wav_filename
        )

        # Load audio
        try:
            audio, sr = load_wav_mono(str(wav_path))
        except Exception as e:
            print(f"\nError loading {wav_filename}: {e}")
            continue

        # Run sliding window inference
        probabilities, times, col_indices = run_sliding_window_inference(
            audio, sr, model, config, device
        )

        # Find false positive candidates
        fp_candidates = find_false_positive_candidates(
            probabilities,
            times,
            saved_detections,
            threshold=args.probability_threshold,
            window_ms=config.window_ms,
            buffer_ms=args.buffer_ms
        )

        # Sort by probability (highest first) and limit
        fp_candidates.sort(key=lambda x: x[1], reverse=True)
        fp_candidates = fp_candidates[:args.max_candidates_per_file]

        # Export candidates
        for idx, (cand_time, cand_prob) in enumerate(fp_candidates):
            try:
                png_path, json_path = export_candidate(
                    candidate_time_s=cand_time,
                    candidate_prob=cand_prob,
                    candidate_idx=idx,
                    audio=audio,
                    sr=sr,
                    wav_filename=wav_filename,
                    output_dir=args.output_dir,
                    config=config
                )

                summary_rows.append({
                    'wav_file': wav_filename,
                    'candidate_index': idx,
                    'center_time_s': cand_time,
                    'cnn_probability': cand_prob,
                    'png_path': str(png_path.relative_to(args.output_dir)),
                    'json_path': str(json_path.relative_to(args.output_dir))
                })

                total_candidates += 1

            except Exception as e:
                print(f"\nError exporting candidate at {cand_time:.3f}s: {e}")
                continue

    # Write summary CSV
    csv_path = args.output_dir / "hard_negative_candidates_summary.csv"
    with open(csv_path, 'w', newline='') as f:
        if summary_rows:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"\n✓ Mining complete!")
    print(f"  Total candidates found: {total_candidates}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Summary CSV: {csv_path}")
    print(f"\nNext steps:")
    print(f"  1. Review candidates manually")
    print(f"  2. Confirm false positives")
    print(f"  3. Add confirmed FPs to training data with label='Not USV'")

    return 0


if __name__ == "__main__":
    sys.exit(main())
