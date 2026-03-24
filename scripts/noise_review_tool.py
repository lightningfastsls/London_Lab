"""Launch the Noise Sample Review Tool.

Usage:
    python scripts/noise_review_tool.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    app_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "usv_spectrogram"
        / "labeling"
        / "noise_review_app.py"
    )

    if not app_path.exists():
        print(f"Error: App not found at {app_path}", file=sys.stderr)
        return 1

    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
