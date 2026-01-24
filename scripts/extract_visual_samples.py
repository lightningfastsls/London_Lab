"""Extract spectrogram samples for visual inspection.

Extracts samples from best and worst performing recordings for manual comparison.
"""

import argparse
import numpy as np
import pandas as pd
import torch
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from usv_spectrogram.models.cnn_classifier import USVClassifierCNN
from usv_spectrogram.models.evaluate import load_model_checkpoint, evaluate_model
from usv_spectrogram.models.data_loader import USVDataset, pad_collate_fn
from torch.utils.data import DataLoader


def extract_samples(
    model_path,
    data_csv,
    best_recording,
    worst_recording,
    threshold=0.5,
    n_samples=5,
    output_dir=None
):
    """Extract spectrogram samples for visual inspection."""
    if output_dir is None:
        output_dir = Path('analysis/visual_comparison')
    else:
        output_dir = Path(output_dir)

    best_dir = output_dir / 'best_correct'
    worst_dir = output_dir / 'worst_incorrect'
    best_dir.mkdir(exist_ok=True, parents=True)
    worst_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 80)
    print("EXTRACTING VISUAL SAMPLES")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Data: {data_csv}")
    print(f"Best recording: {best_recording}")
    print(f"Worst recording: {worst_recording}")
    print(f"Threshold: {threshold}")
    print()

    # Load dataset
    df = pd.read_csv(data_csv)
    df = df[df['label'].isin(['USV', 'Not USV'])].copy()
    df['spec_path'] = df['spectrogram_path'].apply(Path)
    missing = ~df['spec_path'].apply(lambda p: p.exists())
    df = df[~missing].copy()
    df = df.reset_index(drop=True)

    # Load model and run evaluation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, checkpoint = load_model_checkpoint(
        Path(model_path),
        USVClassifierCNN,
        device=device
    )

    dataset = USVDataset(data_csv)
    data_loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=pad_collate_fn
    )

    print("Running model evaluation...")
    metrics, y_pred_binary, y_proba, y_true = evaluate_model(
        model,
        data_loader,
        device=device,
        verbose=False
    )

    # Apply custom threshold
    y_pred = (y_proba >= threshold).astype(int)

    # Add predictions to dataframe
    df['y_true'] = y_true
    df['y_pred'] = y_pred
    df['y_proba'] = y_proba
    df['correct'] = (y_pred == y_true).astype(int)

    manifest_rows = []

    # Extract from best recording: correct USV predictions with high confidence
    print(f"\nExtracting from BEST recording ({best_recording})...")
    best_df = df[df['source_file'] == best_recording].copy()
    best_usv_correct = best_df[
        (best_df['label'] == 'USV') &
        (best_df['correct'] == 1)
    ].sort_values('y_proba', ascending=False)

    print(f"  Found {len(best_usv_correct)} correct USV predictions")

    for i, (idx, row) in enumerate(best_usv_correct.head(n_samples).iterrows()):
        src_path = row['spec_path']
        dst_filename = f"{row['candidate_id']}_conf{row['y_proba']:.2f}.png"
        dst_path = best_dir / dst_filename

        shutil.copy2(src_path, dst_path)
        print(f"    {i+1}. {dst_filename} (confidence: {row['y_proba']:.3f})")

        manifest_rows.append({
            'category': 'best_correct',
            'recording': best_recording,
            'candidate_id': row['candidate_id'],
            'label': row['label'],
            'predicted': 'USV',
            'confidence': row['y_proba'],
            'correct': True,
            'file': dst_filename,
        })

    # Extract from worst recording: incorrect USV predictions (false negatives)
    print(f"\nExtracting from WORST recording ({worst_recording})...")
    worst_df = df[df['source_file'] == worst_recording].copy()
    worst_usv_incorrect = worst_df[
        (worst_df['label'] == 'USV') &
        (worst_df['correct'] == 0)
    ].sort_values('y_proba', ascending=True)  # Lowest confidence FNs

    print(f"  Found {len(worst_usv_incorrect)} incorrect USV predictions (false negatives)")

    for i, (idx, row) in enumerate(worst_usv_incorrect.head(n_samples).iterrows()):
        src_path = row['spec_path']
        dst_filename = f"{row['candidate_id']}_conf{row['y_proba']:.2f}.png"
        dst_path = worst_dir / dst_filename

        shutil.copy2(src_path, dst_path)
        print(f"    {i+1}. {dst_filename} (confidence: {row['y_proba']:.3f})")

        manifest_rows.append({
            'category': 'worst_incorrect',
            'recording': worst_recording,
            'candidate_id': row['candidate_id'],
            'label': row['label'],
            'predicted': 'Not USV',
            'confidence': row['y_proba'],
            'correct': False,
            'file': dst_filename,
        })

    # Save manifest
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / 'manifest.csv'
    manifest_df.to_csv(manifest_path, index=False)

    print(f"\nSaved manifest to: {manifest_path}")
    print(f"Extracted {len(manifest_rows)} samples to: {output_dir}")
    print("=" * 80)

    return manifest_df


def main():
    parser = argparse.ArgumentParser(description='Extract visual samples for inspection')
    parser.add_argument('--model', type=str, default='models/clean_test/best_model.pt')
    parser.add_argument('--data', type=str, default='splits/test.csv')
    parser.add_argument('--best-recording', type=str,
                       default='2024-09-30_11-20-07_0000020.wav',
                       help='Best performing recording')
    parser.add_argument('--worst-recording', type=str,
                       default='2024-09-30_11-19-09_0000008.wav',
                       help='Worst performing recording')
    parser.add_argument('--threshold', type=float, default=0.25)
    parser.add_argument('--n-samples', type=int, default=5,
                       help='Number of samples to extract per category')
    parser.add_argument('--output-dir', type=str, default='analysis/visual_comparison')

    args = parser.parse_args()

    extract_samples(
        model_path=args.model,
        data_csv=args.data,
        best_recording=args.best_recording,
        worst_recording=args.worst_recording,
        threshold=args.threshold,
        n_samples=args.n_samples,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
