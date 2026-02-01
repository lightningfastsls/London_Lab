"""Batch USV detection for clustering analysis.

Simple approach:
1. Load WAV → compute spectrogram with global MAD normalization (AudioLoader)
2. Slide through in ~40ms windows, get per-window probability (SlidingInference)
3. If prob > threshold → positive chunk (NO hysteresis)
4. Merge adjacent positive chunks into detections
5. Filter by duration (10-500ms)
6. Extract spectrograms for final detections
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from usv_spectrogram.app.core.audio_loader import AudioLoader
from usv_spectrogram.app.core.sliding_inference import SlidingInference
from usv_spectrogram.detection.spectrogram_extractor import SpectrogramExtractor
from usv_spectrogram.detection.extraction_config import ExtractionConfig
from usv_spectrogram.detection.candidate import Candidate


def merge_positive_chunks(
    probabilities: np.ndarray,
    times: np.ndarray,
    threshold: float,
    min_duration_ms: float = 10.0,
    max_duration_ms: float = 500.0,
) -> list[dict]:
    """Merge adjacent positive chunks into detections.

    Simple logic: consecutive windows with prob > threshold form a detection.
    No hysteresis - just binary threshold.

    Args:
        probabilities: Per-window probabilities from SlidingInference
        times: Per-window center times in seconds
        threshold: Probability threshold for positive classification
        min_duration_ms: Minimum detection duration (reject shorter)
        max_duration_ms: Maximum detection duration (reject longer)

    Returns:
        List of detection dicts with start_time, end_time, max_prob, mean_prob
    """
    if len(probabilities) == 0:
        return []

    # Find positive windows
    positive_mask = probabilities >= threshold

    # Find runs of consecutive positive windows
    detections = []
    in_detection = False
    detection_start_idx = 0

    for i in range(len(positive_mask)):
        if positive_mask[i] and not in_detection:
            # Start new detection
            in_detection = True
            detection_start_idx = i
        elif not positive_mask[i] and in_detection:
            # End detection
            in_detection = False
            detection_end_idx = i - 1

            # Create detection
            det = _create_detection(
                probabilities, times, detection_start_idx, detection_end_idx,
                min_duration_ms, max_duration_ms
            )
            if det is not None:
                detections.append(det)

    # Handle detection at end of file
    if in_detection:
        detection_end_idx = len(positive_mask) - 1
        det = _create_detection(
            probabilities, times, detection_start_idx, detection_end_idx,
            min_duration_ms, max_duration_ms
        )
        if det is not None:
            detections.append(det)

    return detections


def _create_detection(
    probabilities: np.ndarray,
    times: np.ndarray,
    start_idx: int,
    end_idx: int,
    min_duration_ms: float,
    max_duration_ms: float,
) -> dict | None:
    """Create a detection dict from index range, applying duration filter.

    Returns None if detection doesn't pass duration filter.
    """
    # Get time bounds (times are window centers, so extend by half window on each side)
    # Window is ~43ms, so half is ~21.5ms
    window_half_ms = 21.5

    start_time = times[start_idx] - window_half_ms / 1000.0
    end_time = times[end_idx] + window_half_ms / 1000.0

    # Ensure non-negative start
    start_time = max(0.0, start_time)

    duration_ms = (end_time - start_time) * 1000.0

    # Apply duration filter
    if duration_ms < min_duration_ms:
        return None
    if duration_ms > max_duration_ms:
        return None

    # Compute probability stats for this detection
    det_probs = probabilities[start_idx:end_idx + 1]
    max_prob = float(np.max(det_probs))
    mean_prob = float(np.mean(det_probs))

    return {
        'start_time': start_time,
        'end_time': end_time,
        'duration_ms': duration_ms,
        'max_prob': max_prob,
        'mean_prob': mean_prob,
        'n_windows': end_idx - start_idx + 1,
    }


def batch_detect_usvs(
    wav_dir: Path,
    model_path: Path,
    output_dir: Path,
    threshold: float = 0.90,
    min_duration_ms: float = 10.0,
    max_duration_ms: float = 500.0,
    device: str = 'cpu',
    batch_size: int = 32,
    n_files: int | None = None,
):
    """Batch detect USVs from directory of WAV files.

    Args:
        wav_dir: Directory containing WAV files
        model_path: Path to trained CNN model
        output_dir: Output directory for spectrograms and metadata
        threshold: Probability threshold for positive classification (default 0.90)
        min_duration_ms: Minimum detection duration in ms (default 10)
        max_duration_ms: Maximum detection duration in ms (default 500)
        device: Device for inference ('cpu' or 'cuda')
        batch_size: Batch size for CNN inference
        n_files: Limit processing to first N files (for testing)
    """
    # Create output directories
    spec_dir = output_dir / "spectrograms"
    spec_dir.mkdir(parents=True, exist_ok=True)

    print(f"Batch USV Detection for Clustering")
    print(f"=" * 60)
    print(f"WAV directory: {wav_dir}")
    print(f"Model: {model_path}")
    print(f"Output directory: {output_dir}")
    print(f"Threshold: {threshold}")
    print(f"Duration filter: {min_duration_ms}-{max_duration_ms} ms")
    print(f"Device: {device}")
    print(f"=" * 60)

    # Initialize components
    print("\nInitializing components...")

    # Audio loader (computes globally normalized spectrogram)
    audio_loader = AudioLoader()

    # CNN inference (slides through spectrogram)
    sliding_inference = SlidingInference(
        model_path=str(model_path),
        device=device,
        window_width_px=100,  # ~43ms windows
        hop_px=10,            # ~4.3ms hop
        batch_size=batch_size,
        energy_threshold=0.35,
        enable_per_window_norm=False  # Don't double-normalize
    )

    # Spectrogram extractor (training mode - matches CNN training pipeline)
    extraction_config = ExtractionConfig(
        sample_rate=300_000,
        n_fft=512,
        hop_length=128,
        window="hann",
        freq_min_hz=25_000,   # Match training
        freq_max_hz=110_000,  # Match training
        image_height_px=256,
        pixels_per_ms=2.0,
        min_width_px=128,
        max_width_px=512,
        db_floor=-80.0,
        db_ceiling=0.0,
        colormap="inferno"    # Match training
    )
    extractor = SpectrogramExtractor(config=extraction_config)

    # Find all WAV files
    wav_files = sorted(wav_dir.glob("*.wav"))
    total_files = len(wav_files)

    # Limit number of files for testing
    if n_files is not None:
        wav_files = wav_files[:n_files]

    print(f"\nFound {total_files} WAV files, processing {len(wav_files)}")

    if len(wav_files) == 0:
        print("No WAV files found! Exiting.")
        return

    # Process each WAV file
    all_detections = []
    total_detections = 0
    total_rejected_short = 0
    total_rejected_long = 0

    for wav_file in tqdm(wav_files, desc="Processing WAV files"):
        try:
            # Load audio and compute spectrogram
            audio_data = audio_loader.load(str(wav_file))

            # Run CNN inference (get per-window probabilities)
            inference_result = sliding_inference.infer(
                audio_data.spectrogram_db,
                audio_data.times
            )

            # Simple threshold + merge (NO hysteresis)
            detections = merge_positive_chunks(
                inference_result.probabilities,
                inference_result.times,
                threshold=threshold,
                min_duration_ms=min_duration_ms,
                max_duration_ms=max_duration_ms,
            )

            # Count stats
            n_positive_windows = np.sum(inference_result.probabilities >= threshold)
            print(f"\n[{wav_file.name}] Positive windows: {n_positive_windows}/{len(inference_result.probabilities)}")
            print(f"[{wav_file.name}] Detections after merge+filter: {len(detections)}")

            # Skip if no detections
            if len(detections) == 0:
                continue

            # Extract spectrogram for each detection
            for i, det in enumerate(detections):
                # Convert to milliseconds
                start_ms = det['start_time'] * 1000
                end_ms = det['end_time'] * 1000

                # Create Candidate using factory method
                candidate = Candidate.create(
                    source_file=wav_file,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    peak_freq_hz=0.0,    # Not available
                    peak_energy_db=0.0,  # Not available
                    context_before_ms=50.0,
                    context_after_ms=50.0,
                )

                # Extract spectrogram in training mode
                output_path = extractor.extract_single(
                    candidate=candidate,
                    wav_dir=wav_file.parent,
                    output_dir=spec_dir,
                    render_mode="training"
                )

                # Record metadata (only if extraction succeeded)
                if output_path is not None:
                    all_detections.append({
                        'candidate_id': candidate.candidate_id,
                        'source_file': wav_file.name,
                        'start_ms': candidate.start_ms,
                        'end_ms': candidate.end_ms,
                        'duration_ms': candidate.duration_ms,
                        'max_prob': det['max_prob'],
                        'mean_prob': det['mean_prob'],
                        'n_windows': det['n_windows'],
                        'spectrogram_path': str(output_path.absolute())
                    })

            total_detections += len(detections)

        except Exception as e:
            print(f"\nError processing {wav_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save metadata CSV
    if len(all_detections) > 0:
        df = pd.DataFrame(all_detections)
        csv_path = output_dir / "detections.csv"
        df.to_csv(csv_path, index=False)

        print(f"\n" + "=" * 60)
        print(f"Detection complete!")
        print(f"Total detections: {total_detections}")
        print(f"Spectrograms saved to: {spec_dir}")
        print(f"Metadata saved to: {csv_path}")
        print(f"=" * 60)
    else:
        print("\nNo detections found across all files!")


def main():
    parser = argparse.ArgumentParser(
        description="Batch USV detection for clustering analysis"
    )
    parser.add_argument(
        "--wav-dir",
        type=Path,
        required=True,
        help="Directory containing WAV files to process"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("checkpoints/best_model.pt"),
        help="Path to trained CNN model (default: checkpoints/best_model.pt)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/clustering/auto_detected"),
        help="Output directory (default: analysis/clustering/auto_detected)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Probability threshold for positive classification (default: 0.90)"
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=10.0,
        help="Minimum detection duration in ms (default: 10)"
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=500.0,
        help="Maximum detection duration in ms (default: 500)"
    )
    parser.add_argument(
        "--n-files",
        type=int,
        default=None,
        help="Limit to N files for testing (default: all files)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help="Device for inference (default: cpu)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for CNN inference (default: 32)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.wav_dir.exists():
        print(f"Error: WAV directory not found: {args.wav_dir}")
        sys.exit(1)

    if not args.model.exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    # Run batch detection
    batch_detect_usvs(
        wav_dir=args.wav_dir,
        model_path=args.model,
        output_dir=args.output_dir,
        threshold=args.threshold,
        min_duration_ms=args.min_duration,
        max_duration_ms=args.max_duration,
        device=args.device,
        batch_size=args.batch_size,
        n_files=args.n_files,
    )


if __name__ == "__main__":
    main()
