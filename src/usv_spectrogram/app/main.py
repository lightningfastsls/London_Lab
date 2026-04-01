"""Main entry point for USV Detection App."""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from .main_window import MainWindow


def main():
    """Launch the USV Detection App."""
    app = QApplication(sys.argv)
    app.setApplicationName("USV Detection")
    app.setOrganizationName("USV Lab")

    # Set default model path
    # Use hard-negative retrained CNN (mid-C [32,96,192], trained 2026-03-31)
    # Retrained with 620 hard negatives + 144 hard positives from manual review.
    # Test: precision 90.55%, recall 88.54%, F1 89.53% (vs old: 87.2%/93.2%/90.1%)
    # Batch validation: 16/18 known noise files eliminated, 98.7% USV rate in manual review
    repo_root = Path(__file__).resolve().parents[3]
    default_model = repo_root / "models" / "hard_neg_retrain" / "best_model.pt"

    window = MainWindow(default_model_path=default_model)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
