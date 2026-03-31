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
    # Use matched-windows CNN (mid-C [32,96,192], trained 2026-03-27)
    repo_root = Path(__file__).resolve().parents[3]
    default_model = repo_root / "models" / "matched_windows" / "best_model.pt"

    window = MainWindow(default_model_path=default_model)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
