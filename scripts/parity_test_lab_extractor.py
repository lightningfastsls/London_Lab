#!/usr/bin/env python3
"""Parity test: lab extractor's DatasetAssembler path vs wild PNGs.

Picks a known wild hard-negative PNG produced by extract_hard_negatives.py
and reproduces it via the same DatasetAssembler-based path the lab
extractor uses. If the bytes / pixels match, the lab extractor sits on
the CNN training grid.

Test target:
    PNG: data/training/hard_noises/spectrograms/2024-09-30_12-19-37_0000491_000600.png
    Source row in data/manual_review_labels.csv:
        stem=2024-09-30_12-19-37_0000491, detection_idx=0, label=noise,
        start_time_s=0.2568533, end_time_s=0.2696533, jitter_idx=0 (centered).
    Source WAV: 5970_manual_review_reviewed/2024-09-30_12-19-37_0000491.wav

Pass criteria:
    - byte-identical PNG, OR
    - per-pixel max abs diff <= 1 in uint8 (PNG encoder non-determinism is
      possible with some libpng versions; per-pixel exactness on the array
      is what matters).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.dataset.assembler import AssemblyConfig, DatasetAssembler

WAV = REPO_ROOT / "5970_manual_review_reviewed" / "2024-09-30_12-19-37_0000491.wav"
EXISTING_PNG = (
    REPO_ROOT
    / "data"
    / "training"
    / "hard_noises"
    / "spectrograms"
    / "2024-09-30_12-19-37_0000491_000600.png"
)
START_TIME_S = 0.2568533333333333
END_TIME_S = 0.2696533333333333
WINDOW_COLUMNS = 100
HALF_WINDOW = WINDOW_COLUMNS // 2

OUT_DIR = REPO_ROOT / "tmp" / "parity_test"
OUT_PNG = OUT_DIR / "reproduced.png"


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main() -> int:
    if not WAV.exists():
        print(f"FAIL: source WAV missing: {WAV}", file=sys.stderr)
        return 2
    if not EXISTING_PNG.exists():
        print(f"FAIL: reference PNG missing: {EXISTING_PNG}", file=sys.stderr)
        return 2

    cfg = AssemblyConfig(
        unified_labels_path=Path("unused.csv"),
        use_global_mad=True,
        window_columns=WINDOW_COLUMNS,
    )
    asm = DatasetAssembler(cfg)

    spec_norm, _ = asm._load_global_spectrogram(WAV)

    # Replicate extract_hard_negatives.py's centered-window logic.
    start_col = asm._time_to_col(START_TIME_S)
    end_col = asm._time_to_col(END_TIME_S)
    center_col = (start_col + end_col) // 2
    win_start = max(0, center_col - HALF_WINDOW)
    n_cols = spec_norm.shape[1]
    win_start = min(win_start, max(0, n_cols - WINDOW_COLUMNS))
    win_end = win_start + WINDOW_COLUMNS

    print(f"start_col={start_col}, end_col={end_col}, center={center_col}")
    print(f"window=[{win_start},{win_end}], n_cols={n_cols}")

    window = spec_norm[:, win_start:win_end]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    asm._render_window_png(window, OUT_PNG)

    # Compare
    ref_md5 = md5(EXISTING_PNG)
    new_md5 = md5(OUT_PNG)
    print(f"reference md5: {ref_md5}")
    print(f"reproduced md5: {new_md5}")

    ref_arr = np.asarray(Image.open(EXISTING_PNG))
    new_arr = np.asarray(Image.open(OUT_PNG))
    print(f"reference shape: {ref_arr.shape} dtype={ref_arr.dtype}")
    print(f"reproduced shape: {new_arr.shape} dtype={new_arr.dtype}")

    if ref_arr.shape != new_arr.shape:
        print("FAIL: shape mismatch", file=sys.stderr)
        return 1

    diff = np.abs(ref_arr.astype(np.int16) - new_arr.astype(np.int16))
    max_diff = int(diff.max())
    n_changed = int((diff > 0).sum())
    total = int(diff.size)
    print(f"max abs pixel diff: {max_diff}")
    print(f"changed pixels:     {n_changed}/{total} "
          f"({100 * n_changed / total:.4f}%)")

    if ref_md5 == new_md5:
        print("PASS: byte-identical PNGs")
        return 0
    if max_diff <= 1:
        print("PASS: per-pixel diff <= 1 (PNG encoder noise tolerated)")
        return 0
    print("FAIL: pixels differ beyond tolerance — extractor drift detected",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
