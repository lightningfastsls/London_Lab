"""Evaluate CNN retraining experiment with three-scenario testing.

This script evaluates the experiment model on three distinct scenarios:
1. Labeled USV samples (should still recognize USVs - maintain >0.8 accuracy)
2. Labeled "Not USV" samples (should better recognize non-USVs - target <0.5)
3. Fresh random chunks (KEY TEST - should recognize as background - target <0.5)

The fresh random chunks test is critical because it tests generalization to
truly novel data that wasn't in the training set.

Success Criteria:
- USV mean probability: >0.8 (maintains USV recognition)
- Not USV mean probability: <0.5 (better than baseline 0.684)
- Fresh random chunks mean probability: <0.5 (vs baseline 0.997) ← KEY METRIC

Outputs:
- Three-panel histogram plot comparing probability distributions
- JSON with detailed statistics
- Verdict text file (SUCCESS/PARTIAL/FAILED)

Usage:
    python scripts/evaluate_experiment.py \\
        --model models/experiment_random_negatives/best_model.pt \\
        --test-csv spectrograms_training/test.csv \\
        --wav-dir "5970 USV" \\
        --labels-csv labels.csv \\
        --output-dir data/experiment_dataset
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from usv_spectrogram.detection.candidate import Candidate
from usv_spectrogram.detection.extraction_config import ExtractionConfig
from usv_spectrogram.detection.spectrogram_extractor import SpectrogramExtractor
from usv_spectrogram.io_wav import load_wav_mono
from usv_spectrogram.models.cnn_classifier import USVClassifierCNN

# Import the USVRegion and helper functions from generate_random_negatives
# Add scripts to path for cross-script imports
sys.path.insert(0, str(Path(__file__).parent))
from generate_random_negatives import load_usv_regions, overlaps_usv, USVRegion


def load_model(model_path: Path, device: torch.device) -> USVClassifierCNN:
    """Load trained CNN model from checkpoint.

    Parameters
    ----------
    model_path:
        Path to model checkpoint (.pt file)
    device:
        Device to load model on

    Returns
    -------
    Loaded model in eval mode
    """
    print(f"Loading model from {model_path}...")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Extract model state dict (handle both direct state_dict and wrapped checkpoint)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Initialize model (use default architecture)
    model = USVClassifierCNN()
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully")
    return model


def load_and_predict(
    model: USVClassifierCNN,
    device: torch.device,
    spec_path: Path,
) -> float:
    """Load a single spectrogram and predict probability.

    Matches USVDataset preprocessing:
    - Convert to grayscale
    - Per-image normalization to [0, 1]
    - Add batch and channel dimensions

    Parameters
    ----------
    model:
        Trained CNN model
    device:
        Device for inference
    spec_path:
        Path to spectrogram PNG

    Returns
    -------
    Predicted probability (0-1)
    """
    # Load image and convert to grayscale (matches USVDataset)
    img = Image.open(spec_path).convert("L")
    spectrogram = np.array(img, dtype=np.float32)

    # Per-image normalization (matches USVDataset)
    spec_min = spectrogram.min()
    spec_max = spectrogram.max()
    if spec_max > spec_min:
        spectrogram = (spectrogram - spec_min) / (spec_max - spec_min)
    else:
        spectrogram = np.zeros_like(spectrogram)

    # Convert to tensor: (1, 1, H, W) - batch, channel, height, width
    spectrogram = torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0)
    spectrogram = spectrogram.to(device)

    # Predict
    with torch.no_grad():
        prob = model.predict_proba(spectrogram)

    return prob.item()


def test_labeled_samples(
    model: USVClassifierCNN,
    device: torch.device,
    test_csv: Path,
    label_filter: str,
    repo_root: Path,
) -> np.ndarray:
    """Test model on labeled samples from test.csv.

    Parameters
    ----------
    model:
        Trained CNN model
    device:
        Device for inference
    test_csv:
        Path to test.csv
    label_filter:
        Label to filter for ("USV" or "Not USV")
    repo_root:
        Repository root for resolving relative paths

    Returns
    -------
    Array of predicted probabilities
    """
    import pandas as pd

    print(f"\nTesting on labeled samples (label={label_filter})...")

    # Load test CSV and filter by label
    df = pd.read_csv(test_csv)
    df_filtered = df[df["label"] == label_filter].copy()

    print(f"  Found {len(df_filtered)} samples")

    if len(df_filtered) == 0:
        return np.array([])

    # Predict on each sample
    probabilities = []

    for idx, row in df_filtered.iterrows():
        spec_path = Path(row["spectrogram_path"])

        # Resolve relative paths
        if not spec_path.is_absolute():
            spec_path = repo_root / spec_path

        if not spec_path.exists():
            print(f"  Warning: Spectrogram not found: {spec_path}")
            continue

        try:
            prob = load_and_predict(model, device, spec_path)
            probabilities.append(prob)
        except Exception as e:
            print(f"  Warning: Error predicting {spec_path}: {e}")
            continue

    print(f"  Predicted on {len(probabilities)} samples")

    return np.array(probabilities)


def test_random_chunks(
    model: USVClassifierCNN,
    device: torch.device,
    wav_dir: Path,
    labels_csv: Path,
    n_samples: int,
    duration_ms: float = 40.0,
    seed: int = 42,
) -> np.ndarray:
    """Test model on fresh random chunks (not in training data).

    This is the KEY TEST - these random chunks are generated at evaluation time
    and are completely novel to the model.

    Parameters
    ----------
    model:
        Trained CNN model
    device:
        Device for inference
    wav_dir:
        Directory containing WAV files
    labels_csv:
        Path to labels.csv (to avoid USV regions)
    n_samples:
        Number of random chunks to generate
    duration_ms:
        Duration of each chunk in milliseconds
    seed:
        Random seed for reproducibility

    Returns
    -------
    Array of predicted probabilities
    """
    print(f"\nGenerating and testing fresh random chunks (n={n_samples})...")

    random.seed(seed)
    np.random.seed(seed)

    # Load USV regions to avoid
    usv_regions_map = load_usv_regions(labels_csv, buffer_ms=50.0)

    # Find all WAV files
    wav_files = list(wav_dir.glob("*.wav"))
    if not wav_files:
        raise ValueError(f"No WAV files found in {wav_dir}")

    # Create extraction config matching training
    extraction_config = ExtractionConfig(
        sample_rate=300_000,
        n_fft=512,
        hop_length=128,
        freq_min_hz=20_000,
        freq_max_hz=120_000,
        colormap="magma",
        dynamic_range_method="mad",
        default_render_mode="training",
    )

    extractor = SpectrogramExtractor(config=extraction_config)

    # Create temporary directory for spectrograms
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        probabilities = []
        generated_count = 0
        max_attempts = n_samples * 100  # Prevent infinite loop

        for attempt in range(max_attempts):
            if generated_count >= n_samples:
                break

            # Pick random WAV file
            wav_path = random.choice(wav_files)

            # Load WAV to get duration
            try:
                samples, sample_rate = load_wav_mono(wav_path)
            except Exception as e:
                continue

            if sample_rate != extraction_config.sample_rate:
                continue

            total_duration_ms = len(samples) / sample_rate * 1000.0

            # Check if file is long enough
            max_start_ms = total_duration_ms - duration_ms
            if max_start_ms <= 0:
                continue

            # Random start time
            start_ms = random.uniform(0, max_start_ms)
            end_ms = start_ms + duration_ms

            # Check for USV overlap
            usv_regions = usv_regions_map.get(wav_path.name, [])
            if overlaps_usv(start_ms, end_ms, usv_regions):
                continue

            # Create candidate
            candidate = Candidate.create(
                source_file=wav_path,
                start_ms=start_ms,
                end_ms=end_ms,
                peak_freq_hz=0.0,
                peak_energy_db=0.0,
                context_before_ms=0.0,
                context_after_ms=0.0,
                interference_flag=False,
            )

            # Extract spectrogram
            try:
                spec_path = extractor.extract_single(
                    candidate=candidate,
                    wav_dir=wav_dir,
                    output_dir=temp_dir,
                    render_mode="training",
                )
            except Exception as e:
                continue

            if spec_path is None:
                continue

            # Predict
            try:
                prob = load_and_predict(model, device, spec_path)
                probabilities.append(prob)
                generated_count += 1

                if generated_count % 10 == 0:
                    print(f"  Generated {generated_count}/{n_samples}...")

            except Exception as e:
                continue

        print(f"  Generated and predicted on {len(probabilities)} samples")

    return np.array(probabilities)


def compute_statistics(probabilities: np.ndarray) -> dict:
    """Compute statistics for probability distribution.

    Parameters
    ----------
    probabilities:
        Array of predicted probabilities

    Returns
    -------
    Dictionary of statistics
    """
    if len(probabilities) == 0:
        return {
            "n_samples": 0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "pct_above_0.5": 0.0,
            "pct_above_0.9": 0.0,
        }

    return {
        "n_samples": len(probabilities),
        "mean": float(probabilities.mean()),
        "median": float(np.median(probabilities)),
        "std": float(probabilities.std()),
        "min": float(probabilities.min()),
        "max": float(probabilities.max()),
        "pct_above_0.5": float((probabilities > 0.5).mean() * 100),
        "pct_above_0.9": float((probabilities > 0.9).mean() * 100),
    }


def generate_histogram_plot(
    usv_probs: np.ndarray,
    not_usv_probs: np.ndarray,
    random_probs: np.ndarray,
    output_path: Path,
) -> None:
    """Generate three-panel histogram plot.

    Parameters
    ----------
    usv_probs:
        Probabilities for USV samples
    not_usv_probs:
        Probabilities for Not USV samples
    random_probs:
        Probabilities for fresh random chunks
    output_path:
        Path to save plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot parameters
    bins = np.linspace(0, 1, 21)
    alpha = 0.7

    # Panel 1: USV samples
    axes[0].hist(usv_probs, bins=bins, alpha=alpha, color="green", edgecolor="black")
    axes[0].axvline(usv_probs.mean(), color="darkgreen", linestyle="--", linewidth=2, label=f"Mean: {usv_probs.mean():.3f}")
    axes[0].axvline(0.5, color="red", linestyle=":", linewidth=1, label="Threshold: 0.5")
    axes[0].set_xlabel("Predicted Probability")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Labeled USV Samples (n={len(usv_probs)})")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Panel 2: Not USV samples
    axes[1].hist(not_usv_probs, bins=bins, alpha=alpha, color="orange", edgecolor="black")
    axes[1].axvline(not_usv_probs.mean(), color="darkorange", linestyle="--", linewidth=2, label=f"Mean: {not_usv_probs.mean():.3f}")
    axes[1].axvline(0.5, color="red", linestyle=":", linewidth=1, label="Threshold: 0.5")
    axes[1].set_xlabel("Predicted Probability")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Labeled Not USV Samples (n={len(not_usv_probs)})")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Panel 3: Fresh random chunks
    axes[2].hist(random_probs, bins=bins, alpha=alpha, color="blue", edgecolor="black")
    axes[2].axvline(random_probs.mean(), color="darkblue", linestyle="--", linewidth=2, label=f"Mean: {random_probs.mean():.3f}")
    axes[2].axvline(0.5, color="red", linestyle=":", linewidth=1, label="Threshold: 0.5")
    axes[2].set_xlabel("Predicted Probability")
    axes[2].set_ylabel("Count")
    axes[2].set_title(f"Fresh Random Chunks (n={len(random_probs)})")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nHistogram plot saved to {output_path}")


def run_evaluation(
    model_path: Path,
    test_csv: Path,
    wav_dir: Path,
    labels_csv: Path,
    output_dir: Path,
    n_random_samples: int = 100,
    random_seed: int = 42,
) -> None:
    """Run full evaluation on experiment model.

    Parameters
    ----------
    model_path:
        Path to trained model checkpoint
    test_csv:
        Path to test.csv with labeled samples
    wav_dir:
        Directory containing WAV files
    labels_csv:
        Path to labels.csv (to avoid USV regions when generating random chunks)
    output_dir:
        Directory to save results
    n_random_samples:
        Number of random chunks to generate for testing
    random_seed:
        Random seed for reproducibility
    """
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(model_path, device)

    # Test on three scenarios
    print("\n" + "=" * 60)
    print("SCENARIO 1: Labeled USV Samples")
    print("=" * 60)
    usv_probs = test_labeled_samples(model, device, test_csv, "USV", repo_root)
    usv_stats = compute_statistics(usv_probs)

    print("\n" + "=" * 60)
    print("SCENARIO 2: Labeled Not USV Samples")
    print("=" * 60)
    not_usv_probs = test_labeled_samples(model, device, test_csv, "Not USV", repo_root)
    not_usv_stats = compute_statistics(not_usv_probs)

    print("\n" + "=" * 60)
    print("SCENARIO 3: Fresh Random Chunks (KEY TEST)")
    print("=" * 60)
    random_probs = test_random_chunks(
        model, device, wav_dir, labels_csv, n_random_samples, seed=random_seed
    )
    random_stats = compute_statistics(random_probs)

    # Generate histogram plot
    if len(usv_probs) > 0 and len(not_usv_probs) > 0 and len(random_probs) > 0:
        histogram_path = output_dir / "evaluation_results.png"
        generate_histogram_plot(usv_probs, not_usv_probs, random_probs, histogram_path)

    # Save metrics JSON
    metrics = {
        "model_path": str(model_path),
        "test_csv": str(test_csv),
        "usv_samples": usv_stats,
        "not_usv_samples": not_usv_stats,
        "random_chunks": random_stats,
    }

    metrics_path = output_dir / "experiment_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # Determine verdict
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\n1. USV Samples:")
    print(f"   Mean probability: {usv_stats['mean']:.3f}")
    print(f"   Target: >0.80 (maintain USV recognition)")
    print(f"   Status: {'✓ PASS' if usv_stats['mean'] > 0.80 else '✗ FAIL'}")

    print(f"\n2. Not USV Samples:")
    print(f"   Mean probability: {not_usv_stats['mean']:.3f}")
    print(f"   Target: <0.50 (better than baseline 0.684)")
    print(f"   Status: {'✓ PASS' if not_usv_stats['mean'] < 0.50 else '✗ FAIL'}")

    print(f"\n3. Fresh Random Chunks (KEY METRIC):")
    print(f"   Mean probability: {random_stats['mean']:.3f}")
    print(f"   Target: <0.50 (vs baseline 0.997)")
    print(f"   Status: {'✓ PASS' if random_stats['mean'] < 0.50 else '✗ FAIL'}")

    # Overall verdict
    usv_pass = usv_stats["mean"] > 0.80
    random_pass = random_stats["mean"] < 0.50

    if usv_pass and random_pass:
        verdict = "SUCCESS ✓"
        verdict_text = (
            "Experiment SUCCESSFUL!\n"
            f"Random negatives taught the CNN what 'no USV' looks like.\n"
            f"Random chunk probability dropped from 0.997 to {random_stats['mean']:.3f}\n"
            f"while maintaining USV recognition at {usv_stats['mean']:.3f}.\n\n"
            f"Next step: Full retraining with 1000+ comprehensive negatives."
        )
    elif random_stats["mean"] < 0.70:
        verdict = "PARTIAL SUCCESS"
        verdict_text = (
            "Experiment shows PARTIAL SUCCESS.\n"
            f"Random chunk probability improved to {random_stats['mean']:.3f} (vs 0.997 baseline)\n"
            f"but didn't reach target <0.50.\n\n"
            f"Next step: Generate 500-1000 random negatives and retrain."
        )
    else:
        verdict = "FAILED"
        verdict_text = (
            "Experiment FAILED.\n"
            f"Random chunk probability is still high: {random_stats['mean']:.3f}\n\n"
            f"Next step: Investigate spectrogram generation consistency,\n"
            f"normalization pipeline, or consider different approach."
        )

    print(f"\n{'=' * 60}")
    print(f"VERDICT: {verdict}")
    print(f"{'=' * 60}")
    print(verdict_text)

    # Save verdict
    verdict_path = output_dir / "verdict.txt"
    with open(verdict_path, "w") as f:
        f.write(f"VERDICT: {verdict}\n\n")
        f.write(verdict_text)
    print(f"\nVerdict saved to {verdict_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CNN retraining experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        required=True,
        help="Path to test.csv with labeled samples",
    )
    parser.add_argument(
        "--wav-dir",
        type=Path,
        required=True,
        help='Directory containing WAV files (e.g., "5970 USV")',
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        required=True,
        help="Path to labels.csv (to avoid USV regions when generating random chunks)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--n-random-samples",
        type=int,
        default=100,
        help="Number of fresh random chunks to generate for testing (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    run_evaluation(
        model_path=args.model,
        test_csv=args.test_csv,
        wav_dir=args.wav_dir,
        labels_csv=args.labels_csv,
        output_dir=args.output_dir,
        n_random_samples=args.n_random_samples,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
