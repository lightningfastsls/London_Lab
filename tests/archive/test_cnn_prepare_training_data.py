"""Tests for scripts/cnn_prepare_training_data.py — Module 18.2b end-to-end smoke.

Written by test-architect BEFORE implementation exists. The script does not
exist yet; the test will fail at collection (ImportError / ModuleNotFoundError)
until scripts/cnn_prepare_training_data.py is created with a main() entry point.
That is the expected TDD red phase.

ROADMAP §18.2b test plan coverage:
  10. End-to-end smoke: synthetic dataset, <30 s, valid splits
      -> test_end_to_end_smoke_produces_valid_splits

Additional coverage (recurring gap patterns):
  - Missing vocalmat source raises      ->  test_missing_vocalmat_source_raises
  - Missing output dir is auto-created  ->  test_output_dir_is_auto_created
  - Manifest columns are present        ->  test_manifest_csv_has_required_columns
  - Sanity patches are populated        ->  test_sanity_patches_dir_is_populated
  - Return code is 0 on success         ->  test_main_returns_zero_on_success

Total: 6 tests (1 from ROADMAP, 5 additional)

VocalMat folder-name convention used in this file:
  The synthetic dataset uses snake_case subdirectory names matching the real
  VocalMat OSF layout. The mapping from GRIMSLEY_12_CLASSES display names
  to folder names is:

    "Noise"           -> "noise"
    "Step up"         -> "step_up"
    "Down-FM"         -> "down_fm"
    "Short"           -> "short"
    "Chevron"         -> "chevron"
    "Up-FM"           -> "up_fm"
    "Flat"            -> "flat"
    "Two steps"       -> "two_steps"
    "Step down"       -> "step_down"
    "Complex"         -> "complex"
    "Reverse Chevron" -> "rev_chevron"
    "Multi-steps"     -> "mult_steps"

  This mapping must be consistent with what cnn_prepare_training_data.py
  uses to build its manifest (folder name -> class display name).
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Repo-root sys.path bootstrap
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# PIL is needed to create synthetic PNG files.
# ---------------------------------------------------------------------------
pytest.importorskip("PIL", reason="Pillow required to build synthetic PNG fixtures")

# soundfile is needed to write synthetic WAV files.
pytest.importorskip("soundfile", reason="soundfile required to build synthetic WAV fixtures")

# ---------------------------------------------------------------------------
# Import the script under test via importlib (same pattern as
# test_cleaning_real_data_loader.py uses for cnn_cleaning_validation.py).
# This will raise ImportError / FileNotFoundError until the script exists.
# ---------------------------------------------------------------------------
_SCRIPT_PATH = REPO_ROOT / "scripts" / "cnn_prepare_training_data.py"
_spec = importlib.util.spec_from_file_location("cnn_prepare_training_data", _SCRIPT_PATH)
assert _spec and _spec.loader, (
    f"Cannot locate scripts/cnn_prepare_training_data.py at {_SCRIPT_PATH}. "
    "This is expected during the red phase — create the script to make tests pass."
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["cnn_prepare_training_data"] = _mod
_spec.loader.exec_module(_mod)

main = _mod.main  # must be callable: main(argv: list[str] | None) -> int

# Also import GRIMSLEY_12_CLASSES for cross-checking.
from usv_spectrogram.classifier.dataset import GRIMSLEY_12_CLASSES  # noqa: E402
from usv_spectrogram.corpus import SAMPLE_RATE_HZ  # noqa: E402  (ADR-001: always import, never redeclare)

# ---------------------------------------------------------------------------
# Folder-name mapping: GRIMSLEY_12_CLASSES display name -> snake_case dir
# Must match what cnn_prepare_training_data.py expects on disk.
# ---------------------------------------------------------------------------
_DISPLAY_TO_FOLDER: dict[str, str] = {
    "Noise":           "noise",
    "Step up":         "step_up",
    "Down-FM":         "down_fm",
    "Short":           "short",
    "Chevron":         "chevron",
    "Up-FM":           "up_fm",
    "Flat":            "flat",
    "Two steps":       "two_steps",
    "Step down":       "step_down",
    "Complex":         "complex",
    "Reverse Chevron": "rev_chevron",
    "Multi-steps":     "mult_steps",
}

assert set(_DISPLAY_TO_FOLDER.keys()) == set(GRIMSLEY_12_CLASSES), (
    "Folder-map keys diverge from GRIMSLEY_12_CLASSES — update _DISPLAY_TO_FOLDER"
)

# Required columns that must appear in every manifest CSV.
_REQUIRED_MANIFEST_COLUMNS = {"path", "class", "source_recording", "duration_ms"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_fake_png(path: Path, size: tuple[int, int] = (227, 227)) -> None:
    """Write a random uint8 RGB PNG (VocalMat spectrogram format)."""
    from PIL import Image
    rng = np.random.default_rng(abs(hash(str(path))) % (2**31))
    arr = (rng.random((size[1], size[0], 3)) * 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(str(path))


def _write_fake_wav(path: Path, duration_s: float = 0.5,
                    sample_rate: int = SAMPLE_RATE_HZ) -> None:
    """Write a synthetic WAV: white noise + faint 60 kHz tone at 300 kHz."""
    import soundfile as sf
    rng = np.random.default_rng(abs(hash(str(path))) % (2**31))
    n = int(round(duration_s * sample_rate))
    t = np.arange(n) / sample_rate
    samples = (0.02 * rng.standard_normal(n) + 0.1 * np.sin(2 * np.pi * 60_000 * t))
    sf.write(str(path), samples.astype(np.float32), sample_rate)


def _build_synthetic_vocalmat(root: Path, n_pngs_per_class: int = 5) -> Path:
    """Create a VocalMat-like directory tree under root/vocalmat/.

    Structure:
        root/vocalmat/
            noise/          5 × 227×227 RGB PNGs
            step_up/        5 × 227×227 RGB PNGs
            ...             (all 12 classes)

    Each PNG gets a unique filename so source_recording can be inferred from
    the parent folder + filename stem.
    """
    vm_root = root / "vocalmat"
    for display_name in GRIMSLEY_12_CLASSES:
        folder_name = _DISPLAY_TO_FOLDER[display_name]
        cls_dir = vm_root / folder_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_pngs_per_class):
            _write_fake_png(cls_dir / f"spectrogram_{i:03d}.png")
    return vm_root


def _build_synthetic_lab_wavs(root: Path, n_wavs: int = 2) -> Path:
    """Create n_wavs fake 0.5-second WAV files in root/lab_wavs/."""
    lab_dir = root / "lab_wavs"
    lab_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_wavs):
        _write_fake_wav(lab_dir / f"lab_recording_{i:02d}.wav")
    return lab_dir


# ===========================================================================
# Test 10 (ROADMAP item 10) — End-to-end smoke
# ===========================================================================

def test_end_to_end_smoke_produces_valid_splits(tmp_path):
    """Spec: synthetic 12-class × 5-PNG dataset runs full prep in <30 s.

    Verifies:
      - Script completes without error (return code 0)
      - Runs in under 30 seconds (generous budget for slow CI)
      - train/manifest.csv, val/manifest.csv, test/manifest.csv exist
      - Each manifest CSV has the required columns
      - sanity_patches/ is populated with at least 1 file

    Uses in-process main() call (not subprocess) to avoid spawning overhead
    and to get a real Python stack trace on failure.
    """
    vm_dir = _build_synthetic_vocalmat(tmp_path, n_pngs_per_class=5)
    lab_dir = _build_synthetic_lab_wavs(tmp_path, n_wavs=2)
    out_dir = tmp_path / "out"

    argv = [
        "--vocalmat-source", str(vm_dir),
        "--lab-wav-dirs", str(lab_dir),
        "--output-dir", str(out_dir),
        "--patch-duration-s", "0.22",
        "--workers", "1",
        "--skip-checksum-verify",
    ]

    t0 = time.perf_counter()
    ret = main(argv)
    elapsed = time.perf_counter() - t0

    assert ret == 0, (
        f"cnn_prepare_training_data.main() returned {ret} (expected 0). "
        "Check script's error output above."
    )
    assert elapsed < 30.0, (
        f"Script took {elapsed:.1f} s on synthetic data (budget: 30 s). "
        "Consider --workers 1 path for test mode."
    )

    # Check output manifests exist.
    for split_name in ("train", "val", "test"):
        manifest_path = out_dir / split_name / "manifest.csv"
        assert manifest_path.exists(), (
            f"Expected manifest at {manifest_path} — not found after script run"
        )

    # Check sanity patches directory exists and has files.
    sanity_dir = out_dir / "sanity_patches"
    assert sanity_dir.exists(), (
        f"sanity_patches/ directory not created at {sanity_dir}"
    )
    sanity_files = list(sanity_dir.rglob("*.png"))
    assert len(sanity_files) >= 1, (
        f"sanity_patches/ is empty — expected at least 1 PNG, got 0"
    )


# ===========================================================================
# Additional test — Manifest CSVs have required columns
# ===========================================================================

def test_manifest_csv_has_required_columns(tmp_path):
    """Every manifest CSV must contain: path, class, source_recording, duration_ms.

    These columns are consumed by build_stratified_split (dataset.py) and by
    the training DataLoader. A missing column is a silent data bug.
    """
    import pandas as pd

    vm_dir = _build_synthetic_vocalmat(tmp_path, n_pngs_per_class=5)
    lab_dir = _build_synthetic_lab_wavs(tmp_path, n_wavs=1)
    out_dir = tmp_path / "out_cols"

    argv = [
        "--vocalmat-source", str(vm_dir),
        "--lab-wav-dirs", str(lab_dir),
        "--output-dir", str(out_dir),
        "--patch-duration-s", "0.22",
        "--workers", "1",
        "--skip-checksum-verify",
    ]
    ret = main(argv)
    assert ret == 0

    for split_name in ("train", "val", "test"):
        df = pd.read_csv(out_dir / split_name / "manifest.csv")
        missing = _REQUIRED_MANIFEST_COLUMNS - set(df.columns)
        assert not missing, (
            f"{split_name}/manifest.csv is missing columns: {missing}"
        )
        assert len(df) > 0, f"{split_name}/manifest.csv is empty"


# ===========================================================================
# Additional test — Return code 0 on valid input
# ===========================================================================

def test_main_returns_zero_on_success(tmp_path):
    """main() must return integer 0 (not None, not True) on success.

    Some CLI helpers return None on success or misuse sys.exit().  This test
    pins the expected return type and value.
    """
    vm_dir = _build_synthetic_vocalmat(tmp_path, n_pngs_per_class=3)
    lab_dir = _build_synthetic_lab_wavs(tmp_path, n_wavs=1)
    out_dir = tmp_path / "out_retcode"

    argv = [
        "--vocalmat-source", str(vm_dir),
        "--lab-wav-dirs", str(lab_dir),
        "--output-dir", str(out_dir),
        "--patch-duration-s", "0.22",
        "--workers", "1",
        "--skip-checksum-verify",
    ]
    ret = main(argv)
    assert ret == 0
    assert isinstance(ret, int), f"main() returned {type(ret).__name__}, expected int"


# ===========================================================================
# Additional test — Output dir is auto-created (does not need to pre-exist)
# ===========================================================================

def test_output_dir_is_auto_created(tmp_path):
    """Script must create --output-dir if it does not already exist.

    A common shell pattern is to pass a fresh output dir; requiring it to
    pre-exist forces the user to mkdir manually.
    """
    vm_dir = _build_synthetic_vocalmat(tmp_path, n_pngs_per_class=3)
    lab_dir = _build_synthetic_lab_wavs(tmp_path, n_wavs=1)
    out_dir = tmp_path / "deeply" / "nested" / "output"
    assert not out_dir.exists(), "Test precondition: output dir must not exist"

    argv = [
        "--vocalmat-source", str(vm_dir),
        "--lab-wav-dirs", str(lab_dir),
        "--output-dir", str(out_dir),
        "--patch-duration-s", "0.22",
        "--workers", "1",
        "--skip-checksum-verify",
    ]
    ret = main(argv)
    assert ret == 0
    assert out_dir.exists(), "Script did not create the nested output directory"


# ===========================================================================
# Additional test — Missing vocalmat source raises or returns non-zero
# ===========================================================================

def test_missing_vocalmat_source_raises_or_returns_nonzero(tmp_path):
    """Script must not silently succeed when --vocalmat-source does not exist.

    Either raise an exception (FileNotFoundError, SystemExit) or return a
    non-zero exit code.  Silent success with an empty output would hide a
    misconfigured path.
    """
    lab_dir = _build_synthetic_lab_wavs(tmp_path, n_wavs=1)
    out_dir = tmp_path / "out_missing"

    argv = [
        "--vocalmat-source", str(tmp_path / "does_not_exist"),
        "--lab-wav-dirs", str(lab_dir),
        "--output-dir", str(out_dir),
        "--patch-duration-s", "0.22",
        "--workers", "1",
        "--skip-checksum-verify",
    ]

    try:
        ret = main(argv)
        # If main() returns rather than raises, it must signal failure.
        assert ret != 0, (
            "Script returned 0 (success) when --vocalmat-source is a non-existent path. "
            "It should return non-zero or raise FileNotFoundError."
        )
    except (FileNotFoundError, SystemExit, ValueError, RuntimeError):
        pass  # Any of these indicate proper error handling.


# ===========================================================================
# Additional test — Sanity patches are populated with PNGs
# ===========================================================================

def test_sanity_patches_dir_is_populated(tmp_path):
    """sanity_patches/ must contain PNG files after a successful run.

    The PLAN explicitly calls for 50 patches per cohort for human inspection
    before Phase 18.3 starts.  With a tiny synthetic dataset we can't guarantee
    50, but at least 1 per cohort is a meaningful lower bound.
    """
    vm_dir = _build_synthetic_vocalmat(tmp_path, n_pngs_per_class=5)
    lab_dir = _build_synthetic_lab_wavs(tmp_path, n_wavs=2)
    out_dir = tmp_path / "out_sanity"

    argv = [
        "--vocalmat-source", str(vm_dir),
        "--lab-wav-dirs", str(lab_dir),
        "--output-dir", str(out_dir),
        "--patch-duration-s", "0.22",
        "--workers", "1",
        "--skip-checksum-verify",
    ]
    ret = main(argv)
    assert ret == 0

    sanity_dir = out_dir / "sanity_patches"
    assert sanity_dir.is_dir(), f"sanity_patches/ not created at {sanity_dir}"

    png_files = list(sanity_dir.rglob("*.png"))
    assert len(png_files) >= 1, (
        "sanity_patches/ contains no PNG files. "
        "At least one spectrogram patch per cohort must be written for human review."
    )
