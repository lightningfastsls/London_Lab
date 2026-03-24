"""Find potential label errors by detecting model-label disagreement.

Identifies training samples where the CNN strongly disagrees with the human label.
These are potential labeling errors that should be manually reviewed.

Disagreement Types:
1. False Negative Candidates: Label="USV" but model says <0.3 (model thinks NOT USV)
2. False Positive Candidates: Label="Not USV" but model says >0.7 (model thinks IS USV)

High disagreement suggests either:
- Labeling error (human made a mistake)
- Edge case (genuinely ambiguous sample)
- Model weakness (needs more training data for this pattern)

Workflow:
1. Load trained CNN model
2. Load training data CSV (with labels and spectrogram paths)
3. Run inference on all samples
4. Find high-disagreement cases
5. Export outliers for manual review
6. Review using existing labeling app to correct errors

Usage:
    python scripts/find_label_outliers.py \
        --model models/full_retrained_cnn/best_model.pt \
        --data-csv data/full_training_dataset/train.csv \
        --spec-dir data/full_training_dataset/spectrograms \
        --output-dir analysis/label_outliers \
        --threshold 0.7
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from usv_spectrogram.models.cnn_classifier import USVClassifierCNN


def load_training_data(csv_path: Path, spec_dir: Path) -> list[dict]:
    """Load training data from CSV.

    Args:
        csv_path: Path to training CSV file
        spec_dir: Directory containing spectrogram PNG files

    Returns:
        List of sample dictionaries with keys:
            - candidate_id: Sample identifier
            - label: "USV" or "Not USV"
            - spectrogram_path: Path to PNG file
    """
    samples = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            candidate_id = row['candidate_id']
            label = row['label']

            # Handle different possible column names for spectrogram path
            if 'spectrogram_path' in row:
                spec_rel_path = row['spectrogram_path']
            elif 'image_path' in row:
                spec_rel_path = row['image_path']
            else:
                # Assume standard naming: {candidate_id}.png
                spec_rel_path = f"{candidate_id}.png"

            spec_path = spec_dir / spec_rel_path

            if not spec_path.exists():
                print(f"Warning: Spectrogram not found: {spec_path}")
                continue

            samples.append({
                'candidate_id': candidate_id,
                'label': label,
                'spectrogram_path': spec_path
            })

    return samples


def run_inference_on_samples(
    samples: list[dict],
    model: USVClassifierCNN,
    device: torch.device
) -> list[dict]:
    """Run CNN inference on all samples.

    Args:
        samples: List of sample dictionaries
        model: Trained CNN model
        device: Torch device

    Returns:
        List of samples with added 'probability' key
    """
    samples_with_probs = []

    for sample in tqdm(samples, desc="Running inference"):
        # Load spectrogram image
        img = Image.open(sample['spectrogram_path']).convert('L')  # Grayscale
        img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]

        # Convert to tensor (1, 1, H, W)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
        img_tensor = img_tensor.to(device)

        # Run inference
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.sigmoid(output).item()

        # Add probability to sample
        sample_with_prob = sample.copy()
        sample_with_prob['probability'] = prob
        samples_with_probs.append(sample_with_prob)

    return samples_with_probs


def find_outliers(
    samples: list[dict],
    threshold: float
) -> tuple[list[dict], list[dict]]:
    """Find samples with high model-label disagreement.

    Args:
        samples: List of samples with labels and probabilities
        threshold: Disagreement threshold (e.g., 0.7)

    Returns:
        Tuple of (false_negative_candidates, false_positive_candidates)
        - FN candidates: Label=USV but model prob < (1 - threshold)
        - FP candidates: Label=Not USV but model prob > threshold
    """
    fn_candidates = []  # Human says USV, model says NOT
    fp_candidates = []  # Human says NOT USV, model says USV

    for sample in samples:
        label = sample['label']
        prob = sample['probability']

        if label == "USV":
            # Should have high probability, but doesn't
            if prob < (1.0 - threshold):
                disagreement = 1.0 - prob  # How much model disagrees
                sample['disagreement'] = disagreement
                sample['disagreement_type'] = 'false_negative_candidate'
                fn_candidates.append(sample)

        elif label == "Not USV":
            # Should have low probability, but doesn't
            if prob > threshold:
                disagreement = prob  # How much model disagrees
                sample['disagreement'] = disagreement
                sample['disagreement_type'] = 'false_positive_candidate'
                fp_candidates.append(sample)

    # Sort by disagreement (highest first)
    fn_candidates.sort(key=lambda x: x['disagreement'], reverse=True)
    fp_candidates.sort(key=lambda x: x['disagreement'], reverse=True)

    return fn_candidates, fp_candidates


def export_outliers(
    outliers: list[dict],
    output_dir: Path,
    outlier_type: str
) -> Path:
    """Export outlier samples for review.

    Args:
        outliers: List of outlier samples
        output_dir: Output directory
        outlier_type: "false_negative_candidates" or "false_positive_candidates"

    Returns:
        Path to CSV summary file
    """
    # Create subdirectories
    type_dir = output_dir / outlier_type
    spec_dir = type_dir / "spectrograms"
    spec_dir.mkdir(parents=True, exist_ok=True)

    # Copy spectrograms
    for sample in tqdm(outliers, desc=f"Exporting {outlier_type}"):
        src_path = sample['spectrogram_path']
        dst_path = spec_dir / src_path.name
        shutil.copy2(src_path, dst_path)

    # Write CSV summary
    csv_path = type_dir / "outliers_summary.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'candidate_id',
            'label',
            'model_probability',
            'disagreement',
            'spectrogram_path'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for sample in outliers:
            writer.writerow({
                'candidate_id': sample['candidate_id'],
                'label': sample['label'],
                'model_probability': f"{sample['probability']:.4f}",
                'disagreement': f"{sample['disagreement']:.4f}",
                'spectrogram_path': str(spec_dir / sample['spectrogram_path'].name)
            })

    return csv_path


def generate_summary_report(
    total_samples: int,
    fn_candidates: list[dict],
    fp_candidates: list[dict],
    output_dir: Path
):
    """Generate summary statistics report.

    Args:
        total_samples: Total number of samples processed
        fn_candidates: False negative candidates
        fp_candidates: False positive candidates
        output_dir: Output directory
    """
    report_path = output_dir / "summary_report.txt"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("LABEL OUTLIER DETECTION SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total samples processed: {total_samples}\n\n")

        # False Negative Candidates
        f.write("-" * 60 + "\n")
        f.write("FALSE NEGATIVE CANDIDATES\n")
        f.write("(Label=USV but model thinks NOT USV)\n")
        f.write("-" * 60 + "\n")
        f.write(f"Count: {len(fn_candidates)}\n")
        f.write(f"Percentage: {len(fn_candidates) / total_samples * 100:.2f}%\n\n")

        if fn_candidates:
            f.write("Top 10 by disagreement:\n")
            for i, sample in enumerate(fn_candidates[:10], 1):
                f.write(f"  {i}. {sample['candidate_id']}: "
                       f"prob={sample['probability']:.4f}, "
                       f"disagreement={sample['disagreement']:.4f}\n")
        f.write("\n")

        # False Positive Candidates
        f.write("-" * 60 + "\n")
        f.write("FALSE POSITIVE CANDIDATES\n")
        f.write("(Label=Not USV but model thinks USV)\n")
        f.write("-" * 60 + "\n")
        f.write(f"Count: {len(fp_candidates)}\n")
        f.write(f"Percentage: {len(fp_candidates) / total_samples * 100:.2f}%\n\n")

        if fp_candidates:
            f.write("Top 10 by disagreement:\n")
            for i, sample in enumerate(fp_candidates[:10], 1):
                f.write(f"  {i}. {sample['candidate_id']}: "
                       f"prob={sample['probability']:.4f}, "
                       f"disagreement={sample['disagreement']:.4f}\n")
        f.write("\n")

        # Recommendations
        f.write("-" * 60 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 60 + "\n")
        total_outliers = len(fn_candidates) + len(fp_candidates)
        outlier_rate = total_outliers / total_samples * 100

        if outlier_rate > 10:
            f.write("⚠ HIGH OUTLIER RATE (>10%)\n")
            f.write("- Review top outliers for systematic labeling errors\n")
            f.write("- Consider if model needs more training data\n")
            f.write("- Check labeling guidelines for consistency\n")
        elif outlier_rate > 5:
            f.write("⚠ MODERATE OUTLIER RATE (5-10%)\n")
            f.write("- Review high-disagreement samples\n")
            f.write("- May indicate edge cases or ambiguous samples\n")
        else:
            f.write("✓ LOW OUTLIER RATE (<5%)\n")
            f.write("- Labels appear consistent with model\n")
            f.write("- Review top outliers to catch remaining errors\n")

        f.write("\n")
        f.write("Next steps:\n")
        f.write("1. Review outliers using labeling app\n")
        f.write("2. Correct confirmed labeling errors\n")
        f.write("3. Retrain model with corrected labels\n")
        f.write("4. Re-run outlier detection to verify improvements\n")

    print(f"\nSummary report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Find potential label errors via model-label disagreement",
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
        "--data-csv",
        type=Path,
        required=True,
        help="Path to training data CSV (with labels and spectrogram paths)"
    )

    parser.add_argument(
        "--spec-dir",
        type=Path,
        required=True,
        help="Directory containing spectrogram PNG files"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for outlier analysis"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Disagreement threshold (default: 0.7). Higher = stricter filtering."
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

    if not args.data_csv.exists():
        print(f"Error: Data CSV not found: {args.data_csv}")
        return 1

    if not args.spec_dir.exists():
        print(f"Error: Spectrogram directory not found: {args.spec_dir}")
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.model}...")
    device = torch.device(args.device)

    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model = USVClassifierCNN()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded on {device}")

    # Load training data
    print(f"Loading training data from {args.data_csv}...")
    samples = load_training_data(args.data_csv, args.spec_dir)
    print(f"Loaded {len(samples)} samples")

    # Run inference
    print("Running inference on all samples...")
    samples_with_probs = run_inference_on_samples(samples, model, device)

    # Find outliers
    print(f"Finding outliers (threshold={args.threshold})...")
    fn_candidates, fp_candidates = find_outliers(samples_with_probs, args.threshold)

    print(f"\nOutliers found:")
    print(f"  False Negative Candidates (Label=USV, Model disagrees): {len(fn_candidates)}")
    print(f"  False Positive Candidates (Label=Not USV, Model disagrees): {len(fp_candidates)}")

    # Export outliers
    if fn_candidates:
        print("\nExporting false negative candidates...")
        fn_csv = export_outliers(fn_candidates, args.output_dir, "false_negative_candidates")
        print(f"  Exported to: {fn_csv}")

    if fp_candidates:
        print("\nExporting false positive candidates...")
        fp_csv = export_outliers(fp_candidates, args.output_dir, "false_positive_candidates")
        print(f"  Exported to: {fp_csv}")

    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(
        len(samples_with_probs),
        fn_candidates,
        fp_candidates,
        args.output_dir
    )

    print(f"\n✓ Outlier detection complete!")
    print(f"  Output directory: {args.output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Review summary report: {args.output_dir}/summary_report.txt")
    print(f"  2. Check outlier spectrograms in subdirectories")
    print(f"  3. Use labeling app to review and correct errors")
    print(f"  4. Retrain model with corrected labels")

    return 0


if __name__ == "__main__":
    sys.exit(main())
