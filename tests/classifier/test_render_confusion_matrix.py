"""Tests for the confusion-matrix renderer.

Covers the inline `_render_confusion_matrix_png` helper in
`scripts/train_lab_classifier.py` and the standalone CLI in
`scripts/render_confusion_matrix.py`.

Per Module 18.3 Stream V handoff (Step 8) the renderer is intentionally
inline in the training CLI; this test file is a NEW addition (handoff
§"Files NOT to touch" permits new test files) and does not modify any
existing test expectations.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from train_lab_classifier import _render_confusion_matrix_png  # noqa: E402
from usv_spectrogram.classifier.dataset import GRIMSLEY_12_CLASSES  # noqa: E402


def _make_12x12_cm() -> list[list[int]]:
    """Strong-diagonal 12x12 matrix with a few off-diagonal counts."""
    n = 12
    cm = [[0] * n for _ in range(n)]
    for i in range(n):
        cm[i][i] = 100 - i * 5
    cm[0][3] = 7
    cm[4][9] = 5
    return cm


def test_renderer_writes_png_file(tmp_path: Path) -> None:
    """The renderer creates a non-empty PNG at the requested path."""
    out = tmp_path / "cm.png"
    _render_confusion_matrix_png(
        _make_12x12_cm(), GRIMSLEY_12_CLASSES, out,
    )
    assert out.exists(), f"PNG was not created at {out}"
    assert out.stat().st_size > 1024, "PNG seems suspiciously small"
    with open(out, "rb") as f:
        magic = f.read(8)
    assert magic.startswith(b"\x89PNG"), "Output does not have PNG signature"


def test_renderer_creates_parent_dir(tmp_path: Path) -> None:
    """If output dir doesn't exist, the renderer creates it (no FileNotFound)."""
    out = tmp_path / "nested" / "sub" / "cm.png"
    _render_confusion_matrix_png(
        _make_12x12_cm(), GRIMSLEY_12_CLASSES, out,
    )
    assert out.exists()


def test_renderer_rejects_shape_mismatch(tmp_path: Path) -> None:
    """A 10x10 matrix with 12 class names raises ValueError."""
    out = tmp_path / "cm.png"
    bad = [[0] * 10 for _ in range(10)]
    with pytest.raises(ValueError, match="does not match"):
        _render_confusion_matrix_png(bad, GRIMSLEY_12_CLASSES, out)


def test_renderer_handles_all_zero_matrix(tmp_path: Path) -> None:
    """All-zero matrix is renderable (threshold==0 branch, no NaN colours)."""
    out = tmp_path / "cm.png"
    zero = [[0] * 12 for _ in range(12)]
    _render_confusion_matrix_png(zero, GRIMSLEY_12_CLASSES, out)
    assert out.exists()


def test_renderer_accepts_numpy_array(tmp_path: Path) -> None:
    """np.ndarray input works (round-trip via np.asarray internally)."""
    import numpy as np

    out = tmp_path / "cm.png"
    arr = np.asarray(_make_12x12_cm(), dtype=np.int64)
    _render_confusion_matrix_png(arr, GRIMSLEY_12_CLASSES, out)
    assert out.exists()


def test_helper_cli_missing_metrics_exits_nonzero(tmp_path: Path) -> None:
    """`render_confusion_matrix.py` fails clearly when --metrics is absent."""
    out = tmp_path / "cm.png"
    proc = subprocess.run(
        [sys.executable, str(_SCRIPTS / "render_confusion_matrix.py"),
         "--metrics", str(tmp_path / "missing.json"),
         "--output", str(out)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 1
    assert "does not exist" in proc.stderr


def test_helper_cli_missing_cm_key_exits_nonzero(tmp_path: Path) -> None:
    """`render_confusion_matrix.py` fails when metrics.json lacks confusion_matrix."""
    bad_metrics = tmp_path / "metrics.json"
    bad_metrics.write_text(json.dumps({"macro_f1_val": 0.5}))
    out = tmp_path / "cm.png"
    proc = subprocess.run(
        [sys.executable, str(_SCRIPTS / "render_confusion_matrix.py"),
         "--metrics", str(bad_metrics), "--output", str(out)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 2
    assert "missing" in proc.stderr.lower()


def test_helper_cli_happy_path(tmp_path: Path) -> None:
    """End-to-end: valid metrics.json → CLI emits PNG, exits 0."""
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"confusion_matrix": _make_12x12_cm()}))
    out = tmp_path / "cm.png"
    proc = subprocess.run(
        [sys.executable, str(_SCRIPTS / "render_confusion_matrix.py"),
         "--metrics", str(metrics), "--output", str(out)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    assert out.exists()
    assert f"Wrote {out}" in proc.stdout
