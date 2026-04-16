"""Tests for imsa_classifier — written by test-architect BEFORE implementation.

ROADMAP test plan coverage (module 17.4, ROADMAP_SIS_BENCHMARK.md lines 396-409):
  1. Pure flat tone (constant FM)             -> test_classify_imsa_constant_fm_returns_flat
  2. Monotonic rising FM                      -> test_classify_imsa_monotonic_rising_returns_up
  3. Monotonic falling FM                     -> test_classify_imsa_monotonic_falling_returns_down
  4. V-shaped FM (down then up)               -> test_classify_imsa_v_shape_returns_u_shape
  5. Inverted-V FM (up then down)             -> test_classify_imsa_inverted_v_returns_inverted_u
  6. FM with one >10 kHz jump                 -> test_classify_imsa_pitch_jump_returns_complex
  7. All-NaN FM                               -> test_classify_imsa_all_nan_returns_flat
  8. 2-column FM (too short)                  -> test_classify_imsa_below_min_valid_cols_returns_flat
  9. Noisy FM with small oscillations         -> test_classify_imsa_small_oscillations_returns_flat
  10. IMSAConfig validation: threshold <= 0   -> test_imsa_config_rejects_nonpositive_threshold
  11. End-to-end on synthetic WAV             -> test_end_to_end_script_label_distribution

Additional coverage (recurring gap patterns):
  - IMSALabel enum completeness               -> test_imsa_label_enum_has_all_six_members
  - IMSAConfig defaults match spec            -> test_imsa_config_defaults_match_spec
  - IMSAConfig is frozen (immutable)          -> test_imsa_config_is_frozen
  - classify_imsa with interior NaN stripped  -> test_classify_imsa_ignores_interior_nan
  - classify_imsa with exactly min_valid_cols -> test_classify_imsa_at_exactly_min_valid_cols
  - Complex not triggered by sub-threshold d  -> test_classify_imsa_sub_threshold_delta_not_complex
  - Output is always an IMSALabel instance    -> test_classify_imsa_always_returns_imsalabel
  - U-shape vs Inverted-U direction           -> test_classify_imsa_u_shape_not_confused_with_inverted_u

Total: 19 tests (11 from ROADMAP, 8 additional)

All tests MUST fail until implementation exists (ImportError is the correct initial
failure mode).  Expected initial state: collection fails with
    ModuleNotFoundError: No module named 'usv_spectrogram.features'
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Pattern 8: import bootstrap — tests/ is one level below REPO_ROOT
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# ---------------------------------------------------------------------------
# Import module under test — will fail with ImportError until implemented
# ---------------------------------------------------------------------------
from usv_spectrogram.features.imsa_classifier import (  # noqa: E402
    IMSAConfig,
    IMSALabel,
    classify_imsa,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# ADR-001: 300 kHz sample rate; ADR-002: hop_length=128
SR = 300_000
HOP_LENGTH = 128
HOP_S: float = HOP_LENGTH / SR  # ≈ 4.2667e-4 s

# Instantiated at module level; also exercises IMSAConfig() construction
DEFAULT_CFG = IMSAConfig()

# Spec-defined thresholds — hand-computed reference values used in assertions
FLAT_THRESHOLD_HZ_PER_S = 500_000.0  # 5 kHz per 10 ms (ROADMAP line 315)
JUMP_THRESHOLD_HZ = 10_000.0         # Hertz 2020 / ROADMAP line 311


# ─────────────────────────────────────────────────────────────────────────────
# Enum & Config correctness
# ─────────────────────────────────────────────────────────────────────────────


def test_imsa_label_enum_has_all_six_members() -> None:
    """Verifies the spec-mandated six label types exist as IMSALabel enum members.

    ROADMAP line 319: "Aim for 6: Flat, Up, Down, U, InvertedU, Complex".
    Checks enum *values* (not names) because the spec defines string values.
    """
    expected_values = {"Flat", "Up", "Down", "U", "InvertedU", "Complex"}
    actual_values = {label.value for label in IMSALabel}
    assert actual_values == expected_values, (
        f"Expected label values {expected_values}, got {actual_values}"
    )


def test_imsa_config_defaults_match_spec() -> None:
    """Verifies IMSAConfig default field values match ROADMAP lines 342-345.

    Hand-checked expected values:
      pitch_jump_threshold_hz       = 10_000.0   (Hertz 2020 used ~10 kHz)
      flat_slope_threshold_hz_per_s = 500_000.0  (5 kHz per 10 ms)
      min_valid_cols                = 3
      smooth_before_slope           = True
    """
    cfg = IMSAConfig()
    assert cfg.pitch_jump_threshold_hz == 10_000.0
    assert cfg.flat_slope_threshold_hz_per_s == 500_000.0
    assert cfg.min_valid_cols == 3
    assert cfg.smooth_before_slope is True


def test_imsa_config_is_frozen() -> None:
    """Verifies IMSAConfig is immutable (Pattern 1: frozen=True dataclass).

    Any assignment to a field after construction must raise an exception
    (dataclasses.FrozenInstanceError, which is a subclass of AttributeError in
    Python 3.11+).
    """
    cfg = IMSAConfig()
    with pytest.raises(Exception):  # FrozenInstanceError / AttributeError
        cfg.pitch_jump_threshold_hz = 5_000.0  # type: ignore[misc]


def test_imsa_config_rejects_nonpositive_threshold() -> None:
    """ROADMAP test plan #10: pitch_jump_threshold_hz <= 0 must raise ValueError.

    ROADMAP line 349: "if self.pitch_jump_threshold_hz <= 0: raise ValueError(...)".
    Both exact-zero and negative values must be rejected.
    """
    with pytest.raises(ValueError):
        IMSAConfig(pitch_jump_threshold_hz=0.0)

    with pytest.raises(ValueError):
        IMSAConfig(pitch_jump_threshold_hz=-500.0)


# ─────────────────────────────────────────────────────────────────────────────
# classify_imsa — shape-label tests  (ROADMAP test plan #1–9)
# ─────────────────────────────────────────────────────────────────────────────


def test_classify_imsa_constant_fm_returns_flat() -> None:
    """ROADMAP test plan #1: constant FM trajectory must produce Flat.

    Hand-computed:
      fm = [70_000] * 20  →  all deltas = 0  →  overall slope = 0 Hz/s
      0  <  FLAT_THRESHOLD = 500_000 Hz/s  →  Flat
      No delta > 10_000 Hz  →  not Complex
    """
    n_cols = 20
    fm_hz = np.full(n_cols, 70_000.0)
    am = np.ones(n_cols)
    result = classify_imsa(fm_hz, am, HOP_S, DEFAULT_CFG)
    assert result == IMSALabel.FLAT, f"Expected FLAT for constant FM, got {result}"


def test_classify_imsa_monotonic_rising_returns_up() -> None:
    """ROADMAP test plan #2: monotonic rising FM must produce Up.

    Hand-computed:
      fm: 50 kHz → 80 kHz over 20 cols
      Duration = 20 * HOP_S ≈ 8.53 ms
      Slope = 30_000 / 8.53e-3 ≈ 3.52e6 Hz/s  >>  500_000  →  not Flat
      All 19 deltas positive, no sign change  →  Up
      Max delta = 30_000 / 19 ≈ 1_578 Hz  <  10_000 Hz  →  not Complex
    """
    n_cols = 20
    fm_hz = np.linspace(50_000.0, 80_000.0, n_cols)
    am = np.ones(n_cols)
    result = classify_imsa(fm_hz, am, HOP_S, DEFAULT_CFG)
    assert result == IMSALabel.UP, f"Expected UP for monotonic rising FM, got {result}"


def test_classify_imsa_monotonic_falling_returns_down() -> None:
    """ROADMAP test plan #3: monotonic falling FM must produce Down.

    Hand-computed:
      fm: 80 kHz → 50 kHz over 20 cols
      Slope ≈ -3.52e6 Hz/s  <<  -500_000  →  not Flat
      All 19 deltas negative  →  Down
      Max |delta| ≈ 1_578 Hz  <  10_000  →  not Complex
    """
    n_cols = 20
    fm_hz = np.linspace(80_000.0, 50_000.0, n_cols)
    am = np.ones(n_cols)
    result = classify_imsa(fm_hz, am, HOP_S, DEFAULT_CFG)
    assert result == IMSALabel.DOWN, f"Expected DOWN for monotonic falling FM, got {result}"


def test_classify_imsa_v_shape_returns_u_shape() -> None:
    """ROADMAP test plan #4: V-shaped (down then up) FM must produce U_SHAPE.

    Hand-computed:
      fm descends 70→50 kHz (15 cols) then ascends back to 70 kHz (15 cols).
      First half: all deltas negative.  Second half: all deltas positive.
      Exactly one sign change in delta direction  →  U_SHAPE (valley shape).
      Step size per col ≈ 20_000/14 ≈ 1_429 Hz  <  10_000 Hz  →  not Complex.
    """
    n_cols = 30
    half = n_cols // 2
    descend = np.linspace(70_000.0, 50_000.0, half)
    ascend = np.linspace(50_000.0, 70_000.0, n_cols - half + 1)[1:]  # skip duplicate midpoint
    fm_hz = np.concatenate([descend, ascend])
    am = np.ones(len(fm_hz))
    result = classify_imsa(fm_hz, am, HOP_S, DEFAULT_CFG)
    assert result == IMSALabel.U_SHAPE, (
        f"Expected U_SHAPE for V-shaped (down-then-up) FM, got {result}"
    )


def test_classify_imsa_inverted_v_returns_inverted_u() -> None:
    """ROADMAP test plan #5: inverted-V (up then down) FM must produce INVERTED_U.

    Hand-computed:
      fm ascends 50→70 kHz (15 cols) then descends back to 50 kHz (15 cols).
      First half: all deltas positive.  Second half: all deltas negative.
      Exactly one sign change  →  INVERTED_U (arch shape).
      Step size per col ≈ 1_429 Hz  <  10_000 Hz  →  not Complex.
    """
    n_cols = 30
    half = n_cols // 2
    ascend = np.linspace(50_000.0, 70_000.0, half)
    descend = np.linspace(70_000.0, 50_000.0, n_cols - half + 1)[1:]
    fm_hz = np.concatenate([ascend, descend])
    am = np.ones(len(fm_hz))
    result = classify_imsa(fm_hz, am, HOP_S, DEFAULT_CFG)
    assert result == IMSALabel.INVERTED_U, (
        f"Expected INVERTED_U for inverted-V FM, got {result}"
    )


def test_classify_imsa_pitch_jump_returns_complex() -> None:
    """ROADMAP test plan #6: FM with one delta > 10 kHz must produce Complex.

    Hand-computed:
      fm is mostly flat around 60 kHz; index 5 jumps +15_000 Hz to 77 kHz.
      |delta[4]| = 15_000 Hz  >  10_000 Hz threshold  →  Complex immediately.

    The test also self-validates the synthetic input so the assertion cannot
    silently regress if the array is accidentally changed.
    """
    fm_hz = np.array([
        60_000.0, 60_500.0, 61_000.0, 61_500.0,
        62_000.0,
        77_000.0,  # jump of 15_000 Hz from previous value
        77_500.0, 78_000.0, 78_500.0, 79_000.0,
    ], dtype=float)
    am = np.ones(len(fm_hz))

    diffs = np.abs(np.diff(fm_hz))
    assert np.any(diffs > JUMP_THRESHOLD_HZ), (
        "Test setup error: no delta in fm_hz exceeds the 10 kHz jump threshold"
    )

    result = classify_imsa(fm_hz, am, HOP_S, DEFAULT_CFG)
    assert result == IMSALabel.COMPLEX, (
        f"Expected COMPLEX for FM containing a {diffs.max():.0f} Hz pitch jump, got {result}"
    )


def test_classify_imsa_all_nan_returns_flat() -> None:
    """ROADMAP test plan #7: all-NaN FM must return Flat (degenerate case).

    ROADMAP line 364: "if len(fm_valid) < cfg.min_valid_cols: return Flat".
    All-NaN  →  0 valid points  <  3  →  degenerate Flat.
    """
    fm_hz = np.full(10, np.nan)
    am = np.ones(10)
    result = classify_imsa(fm_hz, am, HOP_S, DEFAULT_CFG)
    assert result == IMSALabel.FLAT, (
        f"Expected FLAT for all-NaN FM (degenerate), got {result}"
    )


def test_classify_imsa_below_min_valid_cols_returns_flat() -> None:
    """ROADMAP test plan #8: FM with only 2 valid columns must return Flat.

    ROADMAP line 364: need >= cfg.min_valid_cols non-NaN cols to classify.
    Default min_valid_cols = 3; with only 2 valid values the call is degenerate.
    """
    fm_hz = np.array([np.nan, 60_000.0, 80_000.0, np.nan, np.nan])
    am = np.ones(len(fm_hz))
    result = classify_imsa(fm_hz, am, HOP_S, DEFAULT_CFG)
    assert result == IMSALabel.FLAT, (
        f"Expected FLAT for 2-valid-col FM (below min_valid_cols=3), got {result}"
    )


def test_classify_imsa_small_oscillations_returns_flat() -> None:
    """ROADMAP test plan #9: noisy FM with small net drift must stay Flat.

    Design:
      Base trajectory drifts only 1 kHz over 20 cols (≈8.53 ms).
      Overall slope = 1_000 / 8.53e-3 ≈ 117_000 Hz/s  <  500_000 Hz/s  →  Flat.
      Added Gaussian noise std=50 Hz (max ≈ 200 Hz, well below 10 kHz jump threshold).

    Seed is fixed to 42 for reproducibility.
    """
    rng = np.random.default_rng(42)
    n_cols = 20
    fm_base = np.linspace(70_000.0, 71_000.0, n_cols)  # 1 kHz total drift
    noise = rng.normal(0.0, 50.0, n_cols)               # 50 Hz std
    fm_hz = fm_base + noise
    am = np.ones(n_cols)

    result = classify_imsa(fm_hz, am, HOP_S, DEFAULT_CFG)
    assert result == IMSALabel.FLAT, (
        f"Expected FLAT for low-drift FM (slope << flat_threshold), got {result}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Additional gap-pattern tests
# ─────────────────────────────────────────────────────────────────────────────


def test_classify_imsa_ignores_interior_nan() -> None:
    """Verifies interior NaN values are stripped before classification.

    A monotonically rising sequence with two NaN gaps must still be
    classified as Up; the 8 valid points all increase and the overall slope
    is ≫ flat threshold.
    """
    fm_hz = np.array([
        50_000.0, 55_000.0, np.nan, 65_000.0, 70_000.0, np.nan,
        80_000.0, 85_000.0, 90_000.0, 95_000.0,
    ])
    am = np.ones(len(fm_hz))
    result = classify_imsa(fm_hz, am, HOP_S, DEFAULT_CFG)
    assert result == IMSALabel.UP, (
        f"Expected UP after stripping interior NaN from monotonic-rising FM, got {result}"
    )


def test_classify_imsa_at_exactly_min_valid_cols() -> None:
    """Boundary: exactly min_valid_cols (=3) valid columns must reach shape logic.

    The degenerate guard fires at < min_valid_cols, so exactly 3 valid cols must
    proceed to shape classification and return a meaningful label.

    Hand-computed:
      Valid points after NaN-strip: [50_000, 65_000, 80_000]
      Slope = (80_000 - 50_000) / (2 * HOP_S)
            = 30_000 / 8.53e-4 ≈ 35.1e6 Hz/s  >>  500_000 Hz/s  →  not Flat
      Two deltas both positive, no sign change  →  Up
    """
    fm_hz = np.array([np.nan, 50_000.0, 65_000.0, 80_000.0, np.nan])
    am = np.ones(len(fm_hz))
    result = classify_imsa(fm_hz, am, HOP_S, DEFAULT_CFG)
    assert result == IMSALabel.UP, (
        f"Expected UP for steep 3-col rising FM (slope >> flat threshold), got {result}"
    )


def test_classify_imsa_sub_threshold_delta_not_complex() -> None:
    """Verifies deltas strictly below the jump threshold do not trigger Complex.

    Catches off-by-one bugs in jump detection (e.g., >= vs >).
    Largest step = 4_999 Hz  <  10_000 Hz  →  must classify by shape, not Complex.
    """
    fm_hz = np.array([
        50_000.0, 54_999.0, 59_998.0, 64_997.0, 69_996.0,
        74_995.0, 79_994.0, 84_993.0,
    ], dtype=float)
    am = np.ones(len(fm_hz))

    diffs = np.diff(fm_hz)
    assert np.all(np.abs(diffs) < JUMP_THRESHOLD_HZ), (
        "Test setup error: a delta in fm_hz reaches or exceeds the jump threshold"
    )

    result = classify_imsa(fm_hz, am, HOP_S, DEFAULT_CFG)
    assert result != IMSALabel.COMPLEX, (
        f"Expected non-Complex for max |delta|={np.abs(diffs).max():.0f} Hz "
        f"(< {JUMP_THRESHOLD_HZ:.0f} Hz threshold), got {result}"
    )


def test_classify_imsa_u_shape_not_confused_with_inverted_u() -> None:
    """Verifies U_SHAPE and INVERTED_U are never confused with each other.

    Both shapes have exactly one slope sign change; only the direction differs.
    Running both through classify_imsa and asserting different outputs catches
    implementations that swap the direction logic.
    """
    n_cols = 20
    half = n_cols // 2

    # U-shape: falls first then rises
    descend = np.linspace(70_000.0, 55_000.0, half)
    ascend = np.linspace(55_000.0, 70_000.0, n_cols - half + 1)[1:]
    fm_u = np.concatenate([descend, ascend])

    # Inverted-U: rises first then falls
    asc2 = np.linspace(55_000.0, 70_000.0, half)
    desc2 = np.linspace(70_000.0, 55_000.0, n_cols - half + 1)[1:]
    fm_inv = np.concatenate([asc2, desc2])

    am = np.ones(n_cols)
    label_u = classify_imsa(fm_u, am, HOP_S, DEFAULT_CFG)
    label_inv = classify_imsa(fm_inv, am, HOP_S, DEFAULT_CFG)

    assert label_u == IMSALabel.U_SHAPE, f"Expected U_SHAPE, got {label_u}"
    assert label_inv == IMSALabel.INVERTED_U, f"Expected INVERTED_U, got {label_inv}"
    assert label_u != label_inv, (
        "U_SHAPE and INVERTED_U must not map to the same label value"
    )


def test_classify_imsa_always_returns_imsalabel() -> None:
    """Verifies classify_imsa always returns an IMSALabel instance for any input.

    Tests five qualitatively different inputs (constant, rising, falling,
    all-NaN, too-short).  Each must return a proper IMSALabel member — never
    a string, None, or raw integer.
    """
    test_cases = [
        np.full(10, 70_000.0),               # constant
        np.linspace(50_000.0, 80_000.0, 15), # rising
        np.linspace(80_000.0, 50_000.0, 15), # falling
        np.full(10, np.nan),                 # all NaN
        np.array([60_000.0, 75_000.0]),      # too short (2 valid < 3)
    ]
    for fm_hz in test_cases:
        am = np.ones(len(fm_hz))
        result = classify_imsa(fm_hz, am, HOP_S, DEFAULT_CFG)
        assert isinstance(result, IMSALabel), (
            f"classify_imsa returned {type(result).__name__!r} instead of IMSALabel "
            f"for input length={len(fm_hz)}, first_val={fm_hz[0]}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end script test  (ROADMAP test plan #11)
# ─────────────────────────────────────────────────────────────────────────────


def _build_synthetic_classified_csv(
    tmp_path: Path,
    create_tone_wav,
    tones: list[dict],
) -> tuple[Path, Path]:
    """Helper: create a classified_detections_full.csv pointing at real WAV files.

    Each entry in ``tones`` is a dict with keys freq_hz and duration_ms.
    Returns (csv_path, wav_dir).  The create_tone_wav fixture handles cleanup.
    """
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir(exist_ok=True)

    rows = []
    for i, tone_spec in enumerate(tones):
        wav_path = create_tone_wav(
            freq_hz=tone_spec["freq_hz"],
            duration_ms=tone_spec["duration_ms"],
            amplitude=0.5,
            sample_rate=SR,
            noise_level=0.001,
            start_offset_ms=10.0,
        )
        dest = wav_dir / f"call_{i:03d}.wav"
        shutil.copy(wav_path, dest)

        rows.append({
            "call_id": f"call_{i:03d}",
            "file": str(dest),
            "begin_time_s": 0.01,
            "end_time_s": 0.01 + tone_spec["duration_ms"] / 1000.0,
            "cnn_label": "USV",
        })

    csv_path = tmp_path / "classified_detections_full.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return csv_path, wav_dir


@pytest.mark.slow
def test_end_to_end_script_label_distribution(tmp_path: Path, create_tone_wav) -> None:
    """ROADMAP test plan #11: end-to-end script produces a valid label distribution.

    Creates 8 synthetic WAV calls (all pure constant-frequency tones at varying
    pitches) and runs scripts/run_imsa_labeling.py via subprocess.

    Verified outputs:
      1. Script exits with return code 0
      2. output_dir/imsa_labels.csv is created
      3. File contains exactly 8 rows
      4. Every row's imsa_label is a valid member of the IMSALabel enum
      5. The 3 constant 70 kHz tones yield >= 3 Flat labels

    Note: the "Up" and "Down" calls here are pure constant-frequency tones —
    they will produce Flat labels from the ridge-tracker.  The primary value of
    this test is verifying that the pipeline runs end-to-end without crashing
    and produces a well-formed output file.
    """
    tones = [
        # 3 constant 70 kHz tones — must yield Flat
        {"freq_hz": 70_000.0, "duration_ms": 30.0},
        {"freq_hz": 70_000.0, "duration_ms": 30.0},
        {"freq_hz": 70_000.0, "duration_ms": 30.0},
        # 5 tones at different pitches to exercise the full pipeline
        {"freq_hz": 60_000.0, "duration_ms": 20.0},
        {"freq_hz": 65_000.0, "duration_ms": 20.0},
        {"freq_hz": 60_000.0, "duration_ms": 20.0},
        {"freq_hz": 85_000.0, "duration_ms": 20.0},
        {"freq_hz": 80_000.0, "duration_ms": 20.0},
    ]

    csv_path, wav_dir = _build_synthetic_classified_csv(tmp_path, create_tone_wav, tones)

    output_dir = tmp_path / "imsa_output"
    script_path = REPO_ROOT / "scripts" / "run_imsa_labeling.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--classified-csv", str(csv_path),
            "--wav-search-dirs", str(wav_dir),
            "--output-dir", str(output_dir),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert proc.returncode == 0, (
        f"run_imsa_labeling.py exited with code {proc.returncode}.\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )

    imsa_csv = output_dir / "imsa_labels.csv"
    assert imsa_csv.exists(), f"Expected imsa_labels.csv at {imsa_csv}"

    with open(imsa_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 8, f"Expected 8 rows in imsa_labels.csv, got {len(rows)}"

    # Every label must be a valid IMSALabel string value
    valid_labels = {label.value for label in IMSALabel}
    for row in rows:
        assert "imsa_label" in row, f"Row missing 'imsa_label' column: {row}"
        assert row["imsa_label"] in valid_labels, (
            f"Unexpected imsa_label value {row['imsa_label']!r} — "
            f"valid values are {valid_labels}"
        )

    labels = [row["imsa_label"] for row in rows]
    flat_count = labels.count("Flat")
    assert flat_count >= 3, (
        f"Expected >= 3 Flat labels (3 constant 70 kHz tones), got {flat_count}. "
        f"Full distribution: {sorted(labels)}"
    )
