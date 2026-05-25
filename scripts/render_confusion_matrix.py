"""Render confusion_matrix.png from a results/.../metrics.json file.

Standalone helper to back-fill the PNG for a training run whose process
finished before `scripts/train_lab_classifier.py` was patched to render
inline, or for re-rendering with a different title.

Usage
-----
    python scripts/render_confusion_matrix.py \\
        --metrics  results/lab_classifier_v1/metrics.json \\
        --output   results/lab_classifier_v1/confusion_matrix.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from usv_spectrogram.classifier.dataset import GRIMSLEY_12_CLASSES

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
from train_lab_classifier import _render_confusion_matrix_png  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metrics", required=True, type=Path,
                        help="Path to metrics.json from a finished training run")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output PNG path")
    parser.add_argument(
        "--title",
        default="Lab classifier — test confusion matrix",
        help="Figure title",
    )
    args = parser.parse_args(argv)

    if not args.metrics.exists():
        print(f"Error: --metrics path does not exist: {args.metrics}",
              file=sys.stderr)
        return 1

    with open(args.metrics) as f:
        data = json.load(f)
    if "confusion_matrix" not in data:
        print("Error: metrics.json missing 'confusion_matrix' key",
              file=sys.stderr)
        return 2

    _render_confusion_matrix_png(
        data["confusion_matrix"],
        GRIMSLEY_12_CLASSES,
        args.output,
        title=args.title,
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
