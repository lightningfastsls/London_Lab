"""Streamlit entrypoint for the USV Parameter Lab.

DEPRECATED: This Streamlit app is no longer in active use.
The primary UI is now the PyQt6 desktop app (usv_spectrogram.app.main).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch USV Parameter Lab - interactive spectrogram exploration"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8501,
        help="Port to run on (default: 8501)",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host to bind to (default: localhost)",
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        default=True,
        help="Open browser automatically (default: True)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Import streamlit here to avoid slow import if just checking --help
    from streamlit.web import cli as stcli

    # Build streamlit arguments
    script_path = str(Path(__file__).parent.parent / "src" / "usv_spectrogram" / "param_lab" / "app.py")

    st_args = [
        "streamlit", "run", script_path,
        "--server.port", str(args.port),
        "--server.address", args.host,
    ]

    if args.no_browser:
        st_args.extend(["--server.headless", "true"])

    # Run streamlit
    sys.argv = st_args
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
