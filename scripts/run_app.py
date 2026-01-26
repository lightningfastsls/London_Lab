"""Launch the USV Detection App."""

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from usv_spectrogram.app.main import main

if __name__ == "__main__":
    main()
