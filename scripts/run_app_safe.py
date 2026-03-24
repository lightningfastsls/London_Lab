"""Launch the USV Detection App with explicit import order."""

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

# Import torch FIRST (before PyQt6)
import torch
print(f"Loaded PyTorch {torch.__version__}")

# Now import and run the app
from usv_spectrogram.app.main import main

if __name__ == "__main__":
    main()
