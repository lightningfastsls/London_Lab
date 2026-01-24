"""Compare probability distributions between validation and test sets."""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from usv_spectrogram.models.cnn_classifier import USVClassifierCNN
from usv_spectrogram.models.evaluate import load_model_checkpoint, evaluate_model
from usv_spectrogram.models.data_loader import USVDataset, pad_collate_fn
from torch.utils.data import DataLoader


def get_probabilities(model_path, data_csv, batch_size=16, device=None):
    """Get model predictions and probabilities for a dataset."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        pin_memory=True,
        collate_fn=pad_collate_fn
    )

    metrics, y_pred, y_proba, y_true = evaluate_model(
        model,
        data_loader,
        device=device,
        verbose=False
    )

    return y_proba, y_true, y_pred


def main():
    parser = argparse.ArgumentParser(description='Compare probability distributions')
    parser.add_argument('--model', type=str, default='models/clean_test/best_model.pt')
    parser.add_argument('--val-csv', type=str, default='splits/val.csv')
    parser.add_argument('--test-csv', type=str, default='splits/test.csv')
    parser.add_argument('--output', type=str, default='analysis/probability_distributions.png')

    args = parser.parse_args()

    print("Loading validation set probabilities...")
    val_proba, val_true, val_pred = get_probabilities(args.model, args.val_csv)

    print("Loading test set probabilities...")
    test_proba, test_true, test_pred = get_probabilities(args.model, args.test_csv)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: Overall probability distributions
    ax = axes[0, 0]
    ax.hist(val_proba, bins=30, alpha=0.6, label=f'Validation (n={len(val_proba)})', color='blue', edgecolor='black')
    ax.hist(test_proba, bins=30, alpha=0.6, label=f'Test (n={len(test_proba)})', color='orange', edgecolor='black')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Frequency')
    ax.set_title('Probability Distribution: Val vs Test')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Default threshold')

    # Plot 2: Probability distributions by true label (validation)
    ax = axes[0, 1]
    val_usv_proba = val_proba[val_true == 1]
    val_not_usv_proba = val_proba[val_true == 0]
    ax.hist(val_usv_proba, bins=30, alpha=0.6, label=f'USV (n={len(val_usv_proba)})', color='green', edgecolor='black')
    ax.hist(val_not_usv_proba, bins=30, alpha=0.6, label=f'Not USV (n={len(val_not_usv_proba)})', color='red', edgecolor='black')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Frequency')
    ax.set_title('Validation Set: By True Label')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(0.5, color='black', linestyle='--', alpha=0.3)

    # Plot 3: Probability distributions by true label (test)
    ax = axes[0, 2]
    test_usv_proba = test_proba[test_true == 1]
    test_not_usv_proba = test_proba[test_true == 0]
    ax.hist(test_usv_proba, bins=30, alpha=0.6, label=f'USV (n={len(test_usv_proba)})', color='green', edgecolor='black')
    ax.hist(test_not_usv_proba, bins=30, alpha=0.6, label=f'Not USV (n={len(test_not_usv_proba)})', color='red', edgecolor='black')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Frequency')
    ax.set_title('Test Set: By True Label')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(0.5, color='black', linestyle='--', alpha=0.3)

    # Plot 4: Box plots by correctness (validation)
    ax = axes[1, 0]
    val_correct = (val_pred == val_true)
    val_correct_proba = val_proba[val_correct]
    val_incorrect_proba = val_proba[~val_correct]
    ax.boxplot([val_correct_proba, val_incorrect_proba],
               labels=['Correct', 'Incorrect'],
               patch_artist=True)
    ax.set_ylabel('Predicted Probability')
    ax.set_title('Validation: Probability by Correctness')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.3)

    # Plot 5: Box plots by correctness (test)
    ax = axes[1, 1]
    test_correct = (test_pred == test_true)
    test_correct_proba = test_proba[test_correct]
    test_incorrect_proba = test_proba[~test_correct]
    ax.boxplot([test_correct_proba, test_incorrect_proba],
               labels=['Correct', 'Incorrect'],
               patch_artist=True)
    ax.set_ylabel('Predicted Probability')
    ax.set_title('Test: Probability by Correctness')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.3)

    # Plot 6: Summary statistics table
    ax = axes[1, 2]
    ax.axis('off')

    stats_data = [
        ['Metric', 'Validation', 'Test'],
        ['', '', ''],
        ['Min Probability', f'{val_proba.min():.4f}', f'{test_proba.min():.4f}'],
        ['Max Probability', f'{val_proba.max():.4f}', f'{test_proba.max():.4f}'],
        ['Mean Probability', f'{val_proba.mean():.4f}', f'{test_proba.mean():.4f}'],
        ['Std Probability', f'{val_proba.std():.4f}', f'{test_proba.std():.4f}'],
        ['', '', ''],
        ['USV Mean Prob', f'{val_usv_proba.mean():.4f}', f'{test_usv_proba.mean():.4f}'],
        ['Not USV Mean Prob', f'{val_not_usv_proba.mean():.4f}', f'{test_not_usv_proba.mean():.4f}'],
        ['', '', ''],
        ['% Above 0.5', f'{(val_proba > 0.5).mean()*100:.1f}%', f'{(test_proba > 0.5).mean()*100:.1f}%'],
        ['% Above 0.7', f'{(val_proba > 0.7).mean()*100:.1f}%', f'{(test_proba > 0.7).mean()*100:.1f}%'],
        ['% Above 0.9', f'{(val_proba > 0.9).mean()*100:.1f}%', f'{(test_proba > 0.9).mean()*100:.1f}%'],
    ]

    table = ax.table(cellText=stats_data, cellLoc='left', loc='center',
                    colWidths=[0.35, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax.set_title('Probability Statistics Comparison', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nSaved probability distribution comparison to: {args.output}")

    # Print summary
    print("\n" + "=" * 80)
    print("PROBABILITY DISTRIBUTION SUMMARY")
    print("=" * 80)
    print(f"\nValidation Set:")
    print(f"  Range: [{val_proba.min():.4f}, {val_proba.max():.4f}]")
    print(f"  Mean: {val_proba.mean():.4f} ± {val_proba.std():.4f}")
    print(f"  USV mean: {val_usv_proba.mean():.4f}, Not USV mean: {val_not_usv_proba.mean():.4f}")
    print(f"  % Above 0.5: {(val_proba > 0.5).mean()*100:.1f}%")

    print(f"\nTest Set:")
    print(f"  Range: [{test_proba.min():.4f}, {test_proba.max():.4f}]")
    print(f"  Mean: {test_proba.mean():.4f} ± {test_proba.std():.4f}")
    print(f"  USV mean: {test_usv_proba.mean():.4f}, Not USV mean: {test_not_usv_proba.mean():.4f}")
    print(f"  % Above 0.5: {(test_proba > 0.5).mean()*100:.1f}%")

    print("\n" + "=" * 80)
    print("DIAGNOSIS:")
    if val_proba.max() < 0.7 and test_proba.max() < 0.7:
        print("❌ MODEL CALIBRATION ISSUE - Both val and test have compressed probabilities")
        print("   Root cause is in training, not data distribution shift")
    elif test_proba.max() < val_proba.max() - 0.2:
        print("⚠️  DISTRIBUTION SHIFT - Test probabilities more compressed than validation")
        print("   Test recordings may differ from training data")
    else:
        print("✅ PROBABILITIES LOOK REASONABLE")

    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
