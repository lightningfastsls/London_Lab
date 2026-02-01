"""Extract CNN embeddings from USV spectrograms for clustering analysis.

This script extracts 128-dimensional embeddings from both:
1. Labeled USVs from splits/ (596 samples)
2. Auto-detected USVs from batch detection (~ 1900 samples)

Combines into a single embeddings_all.csv for downstream clustering.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from usv_spectrogram.clustering.feature_extractor import FeatureExtractor


def prepare_labeled_csv(splits_dir: Path, output_path: Path) -> Path:
    """Combine train/val/test splits into single CSV for labeled USVs.

    Args:
        splits_dir: Directory containing train.csv, val.csv, test.csv
        output_path: Path to save combined CSV

    Returns:
        Path to combined CSV
    """
    print("Preparing labeled USV dataset...")

    # Load all splits
    dfs = []
    for split_name in ['train', 'val', 'test']:
        csv_path = splits_dir / f"{split_name}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Filter to USV only (exclude 'Not USV')
            df = df[df['label'] == 'USV'].copy()
            # Add source_file column (extract from spectrogram_path)
            df['source_file'] = df['spectrogram_path'].apply(
                lambda x: Path(x).stem.rsplit('_', 1)[0] + '.wav'
            )
            dfs.append(df)
            print(f"  {split_name}: {len(df)} USVs")

    # Combine
    combined = pd.concat(dfs, ignore_index=True)

    # Keep only necessary columns
    combined = combined[['candidate_id', 'spectrogram_path', 'source_file', 'label']]

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Combined labeled USVs: {len(combined)} samples")
    print(f"Saved to: {output_path}\n")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract CNN embeddings for clustering analysis"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("checkpoints/best_model.pt"),
        help="Path to trained CNN model (default: checkpoints/best_model.pt)"
    )
    parser.add_argument(
        "--labeled-csv-dir",
        type=Path,
        default=Path("splits"),
        help="Directory with train/val/test splits for labeled USVs (default: splits/)"
    )
    parser.add_argument(
        "--auto-detected-csv",
        type=Path,
        default=Path("analysis/clustering/auto_detected/detections.csv"),
        help="CSV from batch detection (default: analysis/clustering/auto_detected/detections.csv)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/clustering"),
        help="Output directory (default: analysis/clustering/)"
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
        help="Batch size for embedding extraction (default: 32)"
    )
    parser.add_argument(
        "--labeled-only",
        action='store_true',
        help="Extract only labeled USVs (skip auto-detected)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.model.exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    if not args.labeled_csv_dir.exists():
        print(f"Error: Labeled CSV directory not found: {args.labeled_csv_dir}")
        sys.exit(1)

    print("=" * 70)
    print("CNN Embedding Extraction for Clustering")
    print("=" * 70)

    # Initialize feature extractor
    extractor = FeatureExtractor(
        model_path=args.model,
        device=args.device,
        batch_size=args.batch_size
    )

    # Prepare CSV sources
    csv_sources = []

    # 1. Labeled USVs
    labeled_csv = args.output_dir / "labeled_usvs.csv"
    prepare_labeled_csv(args.labeled_csv_dir, labeled_csv)
    csv_sources.append((labeled_csv, "labeled"))

    # 2. Auto-detected USVs (if available and not skipped)
    if not args.labeled_only:
        if args.auto_detected_csv.exists():
            print(f"Auto-detected USVs: {args.auto_detected_csv}")
            df_auto = pd.read_csv(args.auto_detected_csv)
            print(f"  Found {len(df_auto)} auto-detected USVs\n")
            csv_sources.append((args.auto_detected_csv, "auto_detected"))
        else:
            print(f"Warning: Auto-detected CSV not found: {args.auto_detected_csv}")
            print("Run batch_detect_for_clustering.py first to generate auto-detected USVs.\n")
            print("Proceeding with labeled USVs only...\n")

    # Extract embeddings from all sources
    output_path = args.output_dir / "embeddings_all.csv"
    embeddings_df = extractor.extract_from_multiple_csvs(
        csv_paths=csv_sources,
        output_path=output_path
    )

    # Validation
    print("\n" + "=" * 70)
    print("Validation:")
    print("=" * 70)

    # Check for NaNs
    nan_count = embeddings_df.iloc[:, 3:].isna().sum().sum()
    if nan_count == 0:
        print("[PASS] No NaN values in embeddings")
    else:
        print(f"[FAIL] Found {nan_count} NaN values in embeddings!")

    # Check dimensions
    expected_embedding_cols = 128
    actual_embedding_cols = len([c for c in embeddings_df.columns if c.startswith('embedding_')])
    if actual_embedding_cols == expected_embedding_cols:
        print(f"[PASS] Embedding dimension: {actual_embedding_cols}")
    else:
        print(f"[FAIL] Expected {expected_embedding_cols} embedding columns, found {actual_embedding_cols}")

    # Summary
    print(f"\nTotal samples: {len(embeddings_df)}")
    print("Data sources:")
    for source, count in embeddings_df['data_source'].value_counts().items():
        print(f"  {source}: {count}")

    print("\n" + "=" * 70)
    print("Feature extraction complete!")
    print(f"Embeddings saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
