"""Plot training curves from saved training history.

Usage:
    python scripts/plot_training_curves.py
    python scripts/plot_training_curves.py --history checkpoints/training_history.json
    python scripts/plot_training_curves.py --history exp1/training_history.json --output exp1/curves.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from usv_spectrogram.models.evaluate import plot_training_history


def main():
    parser = argparse.ArgumentParser(
        description='Plot training curves from training history JSON'
    )

    parser.add_argument(
        '--history',
        type=Path,
        default=Path('checkpoints/training_history.json'),
        help='Path to training_history.json (default: checkpoints/training_history.json)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output path for plot (default: same directory as history file, named training_curves.png)'
    )

    parser.add_argument(
        '--show',
        action='store_true',
        help='Display plot interactively instead of saving'
    )

    args = parser.parse_args()

    # Verify history file exists
    if not args.history.exists():
        print(f"Error: Training history not found: {args.history}")
        print("\nExpected format: training_history.json with keys:")
        print("  - train_loss, train_acc, val_loss, val_acc")
        print("  - val_precision, val_recall, val_f1, learning_rates")
        sys.exit(1)

    # Determine output path
    if args.output is None and not args.show:
        args.output = args.history.parent / 'training_curves.png'

    # Generate plot
    print(f"Loading training history from: {args.history}")
    plot_training_history(
        history_path=args.history,
        output_path=args.output,
        show=args.show
    )

    if args.output:
        print(f"Plot saved to: {args.output}")


if __name__ == '__main__':
    main()
