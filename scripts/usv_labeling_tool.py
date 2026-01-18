"""Streamlit entrypoint for the USV Labeling Tool.

Run with: streamlit run scripts/usv_labeling_tool.py
"""

from __future__ import annotations

from pathlib import Path
import sys

# Add src to path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.labeling.labeling_app import run

if __name__ == "__main__":
    run()
