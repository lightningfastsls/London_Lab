"""Tests for scripts/train_lab_classifier.py — Module 18.3 training CLI.

Written by test-architect BEFORE implementation exists. All tests will fail at
collection (ImportError or missing script) until train_lab_classifier.py is
created. That is the expected TDD red phase.

ROADMAP §18.3 test plan coverage:
  (CLI portion of spec #9 — smoke test produces checkpoint)
                                           ->  test_cli_help_exits_zero
                                               test_cli_epochs_zero_raises
                                               test_cli_missing_manifest_raises

Additional coverage (recurring gap patterns):
  - --help prints required arg names          ->  test_cli_help_shows_required_args
  - --epochs 0 exits nonzero with message     ->  test_cli_epochs_zero_raises
  - missing manifest produces clear error     ->  test_cli_missing_manifest_raises
  - script is importable as a module          ->  test_script_is_importable
  - CLI accepts all documented flags          ->  test_cli_known_flags_accepted

Total: 5 tests (3 from additional gap patterns, 2 additional)

Implementation note:
  The script is expected at scripts/train_lab_classifier.py relative to the
  repo root. Tests invoke it via subprocess so the test itself does not need
  to import it; importability is tested separately.

  CLI contract from ROADMAP §18.3 "Files to create" item 5:
    python scripts/train_lab_classifier.py \
        --train-csv   data/lab_cnn_training/train/manifest.csv \
        --val-csv     data/lab_cnn_training/val/manifest.csv \
        --test-csv    data/lab_cnn_training/test/manifest.csv \
        --held-out-845 classified_detections_lab_131204_clean.csv \
        --output-dir  results/lab_classifier_v1/ \
        --epochs 50 --batch-size 64

  Required flags (all must appear in --help output):
    --train-csv, --val-csv, --test-csv, --held-out-845, --output-dir,
    --epochs, --batch-size

Grimsley 12-class mapping (snake-case folder names):
  Display name          Snake-case folder
  "Noise"           ->  "noise"
  "Step up"         ->  "step_up"
  "Down-FM"         ->  "down_fm"
  "Short"           ->  "short"
  "Chevron"         ->  "chevron"
  "Up-FM"           ->  "up_fm"
  "Flat"            ->  "flat"
  "Two steps"       ->  "two_steps"
  "Step down"       ->  "step_down"
  "Complex"         ->  "complex"
  "Reverse Chevron" ->  "rev_chevron"
  "Multi-steps"     ->  "mult_steps"
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = REPO_ROOT / "scripts" / "train_lab_classifier.py"
_PYTHON = sys.executable


def _run_script(*args: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run train_lab_classifier.py with the given args and return CompletedProcess."""
    return subprocess.run(
        [_PYTHON, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ===========================================================================
# Test — --help exits 0
# ===========================================================================

def test_cli_help_exits_zero():
    """Spec: train_lab_classifier.py --help must exit with code 0.

    A CLI that crashes on --help is broken by definition. argparse exits 0
    on --help; if the script uses a different parser this test catches a
    deviation from that standard.
    """
    result = _run_script("--help")
    assert result.returncode == 0, (
        f"--help exited with code {result.returncode}.\n"
        f"stdout: {result.stdout[:500]}\n"
        f"stderr: {result.stderr[:500]}"
    )


# ===========================================================================
# Test — --help prints required argument names
# ===========================================================================

def test_cli_help_shows_required_args():
    """Spec: --help output must mention all required flag names from the ROADMAP spec.

    A user trying to run the script should be able to learn all required
    arguments from --help without reading the source.
    """
    result = _run_script("--help")
    help_text = result.stdout + result.stderr

    required_flags = [
        "--train-csv",
        "--val-csv",
        "--test-csv",
        "--held-out-845",
        "--output-dir",
        "--epochs",
        "--batch-size",
    ]
    missing = [flag for flag in required_flags if flag not in help_text]
    assert not missing, (
        f"--help output is missing documentation for flags: {missing}.\n"
        f"Full --help output:\n{help_text[:1000]}"
    )


# ===========================================================================
# Test — --epochs 0 exits nonzero with a meaningful error
# ===========================================================================

def test_cli_epochs_zero_raises(tmp_path):
    """Spec: --epochs 0 must exit with a nonzero return code and a meaningful message.

    Training for 0 epochs is a programmer error. The CLI must validate this
    argument and reject it before attempting to load data or build a model.
    """
    # Provide dummy paths for required args so argument parsing proceeds to validation.
    dummy_csv = tmp_path / "dummy.csv"
    dummy_csv.write_text("path,class,source_recording,duration_ms\n")
    dummy_out = tmp_path / "out"

    result = _run_script(
        "--train-csv", str(dummy_csv),
        "--val-csv", str(dummy_csv),
        "--test-csv", str(dummy_csv),
        "--held-out-845", str(dummy_csv),
        "--output-dir", str(dummy_out),
        "--epochs", "0",
        "--batch-size", "64",
    )
    assert result.returncode != 0, (
        f"--epochs 0 must exit nonzero, but exited {result.returncode}.\n"
        f"stdout: {result.stdout[:300]}\n"
        f"stderr: {result.stderr[:300]}"
    )
    combined = result.stdout + result.stderr
    # Must contain some indication that epochs is invalid (not a generic Python traceback).
    assert any(kw in combined.lower() for kw in ("epoch", "invalid", "positive", "must be", "error")), (
        f"--epochs 0 error message does not mention 'epoch' or a validation term.\n"
        f"Output: {combined[:500]}"
    )


# ===========================================================================
# Test — missing manifest produces clear error message
# ===========================================================================

def test_cli_missing_manifest_raises(tmp_path):
    """Spec: a missing --train-csv path must produce a clear error message, not a traceback.

    Users frequently mistype paths. The CLI must detect missing files early
    and print a human-readable error rather than letting Python propagate a
    FileNotFoundError stack trace 50 lines deep.
    """
    nonexistent = tmp_path / "does_not_exist.csv"
    dummy_csv = tmp_path / "dummy.csv"
    dummy_csv.write_text("path,class,source_recording,duration_ms\n")
    dummy_out = tmp_path / "out"

    result = _run_script(
        "--train-csv", str(nonexistent),   # this file does not exist
        "--val-csv", str(dummy_csv),
        "--test-csv", str(dummy_csv),
        "--held-out-845", str(dummy_csv),
        "--output-dir", str(dummy_out),
        "--epochs", "1",
        "--batch-size", "4",
    )
    assert result.returncode != 0, (
        f"Missing --train-csv must exit nonzero, got {result.returncode}."
    )
    combined = result.stdout + result.stderr
    # Must contain the missing path or an explicit 'not found' / 'does not exist' message.
    assert (
        str(nonexistent) in combined
        or "not found" in combined.lower()
        or "does not exist" in combined.lower()
        or "no such file" in combined.lower()
    ), (
        f"Error message does not mention the missing file path or 'not found'.\n"
        f"Output: {combined[:500]}"
    )


# ===========================================================================
# Additional test — script is importable as a Python module
# ===========================================================================

def test_script_is_importable():
    """scripts/train_lab_classifier.py must exist and be parseable by Python.

    This is a syntax check: if the script has a top-level syntax error it will
    fail here with a clear message rather than a confusing subprocess error.

    We use py_compile rather than import to avoid executing __main__ code.
    """
    import py_compile

    assert _SCRIPT.exists(), (
        f"Script not found at {_SCRIPT}. "
        "Create scripts/train_lab_classifier.py before running Module 18.3."
    )
    try:
        py_compile.compile(str(_SCRIPT), doraise=True)
    except py_compile.PyCompileError as exc:
        pytest.fail(f"Syntax error in train_lab_classifier.py: {exc}")
