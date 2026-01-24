"""Analyze per-recording performance of CNN classifier.

Identifies which recordings perform well/poorly and compares to training set
characteristics.
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from usv_spectrogram.models.cnn_classifier import USVClassifierCNN
from usv_spectrogram.models.evaluate import load_model_checkpoint, evaluate_model
from usv_spectrogram.models.data_loader import USVDataset, pad_collate_fn
from torch.utils.data import DataLoader


def compute_spectrogram_stats(spec_path):
    """Compute pixel statistics from spectrogram image."""
    try:
        img = Image.open(spec_path).convert('L')
        pixels = np.array(img, dtype=np.float32)
        return {
            'pixel_mean': pixels.mean(),
            'pixel_std': pixels.std(),
            'pixel_min': pixels.min(),
            'pixel_max': pixels.max(),
        }
    except Exception as e:
        return {
            'pixel_mean': np.nan,
            'pixel_std': np.nan,
            'pixel_min': np.nan,
            'pixel_max': np.nan,
        }


def analyze_recording_performance(
    model_path,
    data_csv,
    threshold=0.5,
    batch_size=16,
    device=None,
    output_dir=None
):
    """Analyze per-recording performance."""
    if output_dir is None:
        output_dir = Path('analysis')
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 80)
    print("RECORDING-LEVEL PERFORMANCE ANALYSIS")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Data: {data_csv}")
    print(f"Threshold: {threshold}")
    print()

    # Load dataset CSV to get source_file info
    df = pd.read_csv(data_csv)
    print(f"Loaded {len(df)} samples from CSV")

    # Filter valid labels
    valid_labels = ['USV', 'Not USV']
    df = df[df['label'].isin(valid_labels)].copy()

    # Check spectrograms exist
    df['spec_path'] = df['spectrogram_path'].apply(Path)
    missing = ~df['spec_path'].apply(lambda p: p.exists())
    if missing.any():
        print(f"Warning: {missing.sum()} spectrograms not found, removing")
        df = df[~missing].copy()

    df = df.reset_index(drop=True)
    print(f"Using {len(df)} samples with valid labels and spectrograms")
    print()

    # Load model and run evaluation
    print("Running model evaluation...")
    model, checkpoint = load_model_checkpoint(
        Path(model_path),
        USVClassifierCNN,
        device=device
    )

    dataset = USVDataset(data_csv)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=pad_collate_fn
    )

    metrics, y_pred_binary, y_proba, y_true = evaluate_model(
        model,
        data_loader,
        device=device,
        verbose=False
    )

    # Apply custom threshold
    y_pred = (y_proba >= threshold).astype(int)

    print(f"Evaluation complete: {len(y_pred)} predictions")
    print()

    # Add predictions to dataframe
    df['y_true'] = y_true
    df['y_pred'] = y_pred
    df['y_proba'] = y_proba
    df['correct'] = (y_pred == y_true).astype(int)

    # Compute per-recording statistics
    print("Computing per-recording statistics...")
    recording_stats = []

    for source_file, group in df.groupby('source_file'):
        n_samples = len(group)
        n_usv = (group['label'] == 'USV').sum()
        n_not_usv = (group['label'] == 'Not USV').sum()
        usv_ratio = n_usv / n_samples if n_samples > 0 else 0

        # Compute metrics
        y_t = group['y_true'].values
        y_p = group['y_pred'].values

        tp = ((y_p == 1) & (y_t == 1)).sum()
        fp = ((y_p == 1) & (y_t == 0)).sum()
        fn = ((y_p == 0) & (y_t == 1)).sum()
        tn = ((y_p == 0) & (y_t == 0)).sum()

        accuracy = (y_p == y_t).mean()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Mean confidence
        mean_conf = group['y_proba'].mean()
        mean_conf_correct = group[group['correct'] == 1]['y_proba'].mean()
        mean_conf_incorrect = group[group['correct'] == 0]['y_proba'].mean()

        # Pixel statistics (sample first spectrogram)
        first_spec = group.iloc[0]['spec_path']
        pixel_stats = compute_spectrogram_stats(first_spec)

        recording_stats.append({
            'recording': source_file,
            'n_samples': n_samples,
            'n_usv': n_usv,
            'n_not_usv': n_not_usv,
            'usv_ratio': usv_ratio,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mean_conf': mean_conf,
            'mean_conf_correct': mean_conf_correct if not np.isnan(mean_conf_correct) else 0.0,
            'mean_conf_incorrect': mean_conf_incorrect if not np.isnan(mean_conf_incorrect) else 0.0,
            **pixel_stats,
        })

    rec_df = pd.DataFrame(recording_stats)
    rec_df = rec_df.sort_values('accuracy', ascending=True)

    # Save results
    csv_path = output_dir / 'recording_comparison.csv'
    rec_df.to_csv(csv_path, index=False)
    print(f"Saved recording statistics to: {csv_path}")
    print()

    # Print results
    print("=" * 80)
    print("PER-RECORDING PERFORMANCE")
    print("=" * 80)
    print(rec_df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print("=" * 80)
    print()

    # Identify worst and best recordings
    n_worst = min(3, len(rec_df))
    n_best = min(3, len(rec_df))

    worst_recordings = rec_df.head(n_worst)
    best_recordings = rec_df.tail(n_best)

    print(f"WORST {n_worst} RECORDINGS (by accuracy):")
    for _, row in worst_recordings.iterrows():
        print(f"  {row['recording']}: {row['accuracy']:.1%} accuracy "
              f"({row['n_samples']} samples, {row['usv_ratio']:.1%} USV)")
    print()

    print(f"BEST {n_best} RECORDINGS (by accuracy):")
    for _, row in best_recordings.iterrows():
        print(f"  {row['recording']}: {row['accuracy']:.1%} accuracy "
              f"({row['n_samples']} samples, {row['usv_ratio']:.1%} USV)")
    print()

    # Overall statistics
    print("OVERALL STATISTICS:")
    print(f"  Mean accuracy: {rec_df['accuracy'].mean():.3f} +/- {rec_df['accuracy'].std():.3f}")
    print(f"  Mean F1: {rec_df['f1'].mean():.3f} +/- {rec_df['f1'].std():.3f}")
    print(f"  Mean confidence: {rec_df['mean_conf'].mean():.3f} +/- {rec_df['mean_conf'].std():.3f}")
    print(f"  Mean pixel mean: {rec_df['pixel_mean'].mean():.1f} +/- {rec_df['pixel_mean'].std():.1f}")
    print(f"  Mean pixel std: {rec_df['pixel_std'].mean():.1f} +/- {rec_df['pixel_std'].std():.1f}")
    print()

    return rec_df, worst_recordings, best_recordings


def compare_train_vs_test(train_csv, test_csv, output_dir=None):
    """Compare recording characteristics between train and test sets."""
    if output_dir is None:
        output_dir = Path('analysis')
    else:
        output_dir = Path(output_dir)

    print("=" * 80)
    print("TRAIN VS TEST RECORDING COMPARISON")
    print("=" * 80)

    # Load train set
    train_df = pd.read_csv(train_csv)
    train_df = train_df[train_df['label'].isin(['USV', 'Not USV'])].copy()
    train_df['spec_path'] = train_df['spectrogram_path'].apply(Path)
    missing = ~train_df['spec_path'].apply(lambda p: p.exists())
    train_df = train_df[~missing].copy()

    # Load test set
    test_df = pd.read_csv(test_csv)
    test_df = test_df[test_df['label'].isin(['USV', 'Not USV'])].copy()
    test_df['spec_path'] = test_df['spectrogram_path'].apply(Path)
    missing = ~test_df['spec_path'].apply(lambda p: p.exists())
    test_df = test_df[~missing].copy()

    print(f"Train recordings: {train_df['source_file'].nunique()}")
    print(f"Test recordings: {test_df['source_file'].nunique()}")
    print()

    # Compute per-recording statistics for train set
    train_stats = []
    for source_file, group in train_df.groupby('source_file'):
        n_samples = len(group)
        usv_ratio = (group['label'] == 'USV').sum() / n_samples
        first_spec = group.iloc[0]['spec_path']
        pixel_stats = compute_spectrogram_stats(first_spec)

        train_stats.append({
            'recording': source_file,
            'n_samples': n_samples,
            'usv_ratio': usv_ratio,
            **pixel_stats,
        })

    train_stats_df = pd.DataFrame(train_stats)

    # Compute per-recording statistics for test set
    test_stats = []
    for source_file, group in test_df.groupby('source_file'):
        n_samples = len(group)
        usv_ratio = (group['label'] == 'USV').sum() / n_samples
        first_spec = group.iloc[0]['spec_path']
        pixel_stats = compute_spectrogram_stats(first_spec)

        test_stats.append({
            'recording': source_file,
            'n_samples': n_samples,
            'usv_ratio': usv_ratio,
            **pixel_stats,
        })

    test_stats_df = pd.DataFrame(test_stats)

    # Compare distributions
    comparison = pd.DataFrame({
        'metric': ['pixel_mean', 'pixel_std', 'usv_ratio', 'samples_per_recording'],
        'train_mean': [
            train_stats_df['pixel_mean'].mean(),
            train_stats_df['pixel_std'].mean(),
            train_stats_df['usv_ratio'].mean(),
            train_stats_df['n_samples'].mean(),
        ],
        'train_std': [
            train_stats_df['pixel_mean'].std(),
            train_stats_df['pixel_std'].std(),
            train_stats_df['usv_ratio'].std(),
            train_stats_df['n_samples'].std(),
        ],
        'test_mean': [
            test_stats_df['pixel_mean'].mean(),
            test_stats_df['pixel_std'].mean(),
            test_stats_df['usv_ratio'].mean(),
            test_stats_df['n_samples'].mean(),
        ],
        'test_std': [
            test_stats_df['pixel_mean'].std(),
            test_stats_df['pixel_std'].std(),
            test_stats_df['usv_ratio'].std(),
            test_stats_df['n_samples'].std(),
        ],
    })

    print(comparison.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print()

    # Check if test recordings are outliers
    print("OUTLIER ANALYSIS (test recordings > 2σ from train mean):")
    for _, row in test_stats_df.iterrows():
        outliers = []
        if abs(row['pixel_mean'] - train_stats_df['pixel_mean'].mean()) > 2 * train_stats_df['pixel_mean'].std():
            outliers.append('pixel_mean')
        if abs(row['pixel_std'] - train_stats_df['pixel_std'].mean()) > 2 * train_stats_df['pixel_std'].std():
            outliers.append('pixel_std')

        if outliers:
            print(f"  {row['recording']}: {', '.join(outliers)}")

    # Save comparison
    csv_path = output_dir / 'train_vs_test_recordings.csv'
    comparison.to_csv(csv_path, index=False)
    print(f"\nSaved train vs test comparison to: {csv_path}")
    print("=" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(description='Analyze per-recording performance')
    parser.add_argument('--model', type=str, default='models/clean_test/best_model.pt')
    parser.add_argument('--test-csv', type=str, default='splits/test.csv')
    parser.add_argument('--train-csv', type=str, default='splits/train.csv')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold')
    parser.add_argument('--output-dir', type=str, default='analysis')

    args = parser.parse_args()

    # Analyze test set performance
    rec_df, worst, best = analyze_recording_performance(
        model_path=args.model,
        data_csv=args.test_csv,
        threshold=args.threshold,
        output_dir=args.output_dir
    )

    # Compare train vs test
    compare_train_vs_test(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
