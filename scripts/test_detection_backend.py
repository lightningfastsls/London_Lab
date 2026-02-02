"""Test script for USV detection backend (Phase 1).

Runs the complete detection pipeline:
1. Load WAV file and compute spectrogram
2. Run sliding window CNN inference
3. Apply hysteresis detection
4. Save results to JSON and export annotated image
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from usv_spectrogram.app.core.audio_loader import AudioLoader
from usv_spectrogram.app.core.sliding_inference import SlidingInference
from usv_spectrogram.app.core.detection_logic import HysteresisDetector
from usv_spectrogram.app.core.label_storage import LabelStorage


def main():
    parser = argparse.ArgumentParser(
        description="Test USV detection backend pipeline"
    )
    parser.add_argument(
        "--wav",
        required=True,
        help="Path to WAV file"
    )
    parser.add_argument(
        "--model",
        default="models/production/best_model.pt",
        help="Path to trained CNN model (default: models/production/best_model.pt)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.40,
        help="High threshold for detection (default: 0.40)"
    )
    parser.add_argument(
        "--output",
        help="Output JSON path (default: <wav_stem>_detections.json)"
    )
    parser.add_argument(
        "--export-image",
        action="store_true",
        help="Export annotated spectrogram image"
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=150,
        help="Sliding window width in pixels (default: 150)"
    )
    parser.add_argument(
        "--hop",
        type=int,
        default=10,
        help="Sliding window hop in pixels (default: 10)"
    )

    args = parser.parse_args()

    wav_path = Path(args.wav)
    if not wav_path.exists():
        print(f"Error: WAV file not found: {wav_path}")
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model checkpoint not found: {model_path}")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = wav_path.parent / f"{wav_path.stem}_detections.json"

    print(f"Loading WAV file: {wav_path}")
    print(f"Model: {model_path}")
    print(f"High threshold: {args.threshold:.2f}")
    print()

    # Step 1: Load audio and compute spectrogram
    print("[1/4] Loading audio and computing spectrogram...")
    audio_loader = AudioLoader()
    audio_data = audio_loader.load(wav_path)
    print(f"  Duration: {audio_data.duration_s:.2f} s")
    print(f"  Sample rate: {audio_data.sample_rate} Hz")
    print(f"  Spectrogram shape: {audio_data.spectrogram_db.shape}")
    print()

    # Step 2: Run sliding window inference
    print(f"[2/4] Running CNN inference (window={args.window_width}px, hop={args.hop}px)...")
    inference = SlidingInference(
        model_path=model_path,
        window_width_px=args.window_width,
        hop_px=args.hop,
        batch_size=32
    )
    inference_result = inference.infer(
        audio_data.spectrogram_db,
        audio_data.times
    )
    print(f"  Processed {len(inference_result.probabilities)} windows")
    print(f"  Probability range: [{inference_result.probabilities.min():.3f}, "
          f"{inference_result.probabilities.max():.3f}]")
    print()

    # Step 3: Apply hysteresis detection
    print("[3/4] Applying hysteresis detection...")
    detector = HysteresisDetector(
        high_threshold=args.threshold,
        low_threshold=None,  # Auto: 0.7 × high
        merge_gap_columns=3,
        min_sustained_prob=0.0  # Disabled for retrained model (outputs 0.05-0.16 range)
    )
    detection_result = detector.detect(
        inference_result.probabilities,
        inference_result.column_indices,
        inference_result.times
    )
    print(f"  Detected {len(detection_result.usvs)} USV events")
    print()

    # Step 4: Save results
    print(f"[4/4] Saving results to: {output_path}")
    LabelStorage.save(
        output_path=output_path,
        audio_data=audio_data,
        detection_result=detection_result,
        wav_path=wav_path,
        model_path=model_path,
        high_threshold=args.threshold,
        low_threshold=detector.low_threshold
    )
    print(f"  [OK] Saved JSON labels")

    # Export annotated image if requested
    if args.export_image:
        image_path = output_path.with_suffix('.png')
        print(f"  Exporting annotated image: {image_path}")
        LabelStorage.export_annotated_image(
            output_path=image_path,
            audio_data=audio_data,
            detection_result=detection_result,
            high_threshold=args.threshold,
            low_threshold=detector.low_threshold
        )
        print(f"  [OK] Saved annotated PNG")

    print()
    print("Done!")
    print()

    # Print summary of detections
    if detection_result.usvs:
        print("Detected USVs:")
        for i, usv in enumerate(detection_result.usvs, 1):
            duration_ms = (usv.end_time_s - usv.start_time_s) * 1000
            print(f"  {i}. {usv.start_time_s:.3f}s - {usv.end_time_s:.3f}s "
                  f"({duration_ms:.1f}ms), P_max={usv.max_probability:.3f}")


if __name__ == "__main__":
    main()
