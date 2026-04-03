"""Adversarial / gap-filling tests for classify_traditional_taxonomy.

Written by test-adversary AFTER the initial 55 tests existed and passing.

Gap categories addressed here:
  A. Negative feature values (negative duration, sinuosity, bandwidth, slope)
  B. Infinite feature values (Inf / -Inf) bypass the NaN guard
  C. Zero values for all four features
  D. Flat confidence path — negative slope branch and sinuosity=1.3 boundary
  E. Down call confidence at the call level (abs(slope) detail)
  F. Frequency_Jump confidence at the call level
  G. Chevron confidence: sinuosity drives minimum vs. bandwidth drives minimum
  H. _confidence called with value on wrong side of threshold (negative distance)
  I. _confidence called with threshold=0 (zero-threshold edge case)
  J. classify_dataframe with non-default / duplicate index
  K. classify_dataframe with extra irrelevant columns (realistic input shape)
  L. Large input — no crash / correct row count
  M. Flat high-confidence boundary: sinuosity exactly at 1.3
  N. Priority gap: Short + Chevron features combined
  O. Priority gap: Short + Frequency_Jump features combined
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from classify_traditional_taxonomy import (
    CONFIDENCE_MARGIN,
    THRESH_CHEVRON_BANDWIDTH,
    THRESH_CHEVRON_SINUOSITY,
    THRESH_COMPLEX_SINUOSITY,
    THRESH_FREQJUMP_BANDWIDTH,
    THRESH_FREQJUMP_SINUOSITY,
    THRESH_SHORT_DURATION_S,
    THRESH_SLOPE_DIRECTIONAL,
    _confidence,
    _min_confidence,
    classify_call,
    classify_dataframe,
)

VALID_TYPES = {"Short", "Complex", "Chevron", "Frequency_Jump", "Up", "Down", "Flat", "unclassified"}
VALID_CONFIDENCES = {"high", "medium", "low", "none"}


def _make_row(
    call_length_s: float = 0.08,
    slope: float = 0.0,
    sinuosity: float = 1.5,
    bandwidth_hz: float = 20.0,
) -> pd.Series:
    return pd.Series(
        {
            "call_length_s": call_length_s,
            "slope": slope,
            "sinuosity": sinuosity,
            "bandwidth_hz": bandwidth_hz,
        }
    )


# ---------------------------------------------------------------------------
# A. Negative feature values
# ---------------------------------------------------------------------------


def test_negative_duration_classifies_as_short():
    """A negative duration satisfies duration < 0.015, so classify_call returns Short.

    Negative durations are physically impossible but can appear in malformed data.
    The classifier must not crash and must return a valid (type, confidence) pair.
    The expected behaviour is Short (the strict-less-than guard fires) — this test
    documents the actual behaviour rather than asserting it is desirable.
    """
    row = _make_row(call_length_s=-0.001)
    syllable_type, confidence = classify_call(row)
    assert syllable_type in VALID_TYPES
    assert confidence in VALID_CONFIDENCES
    # Document: negative duration currently triggers Short (distance from threshold is large)
    assert syllable_type == "Short", (
        "Negative duration satisfies duration < THRESH_SHORT_DURATION_S — "
        "currently classified as Short. If this should return 'unclassified', add a guard."
    )


def test_negative_bandwidth_does_not_trigger_chevron_or_freqjump():
    """Negative bandwidth_hz must not trigger Chevron (>25) or Frequency_Jump (>55).

    Both thresholds are positive, so negative bandwidth fails both checks and
    the call should fall through to slope-based or Flat classification.
    """
    # Slope=0, sinuosity=1.5 — should land on Flat
    row = _make_row(call_length_s=0.08, slope=0.0, sinuosity=1.5, bandwidth_hz=-10.0)
    syllable_type, confidence = classify_call(row)
    assert syllable_type not in {"Chevron", "Frequency_Jump"}, (
        f"Negative bandwidth should not trigger Chevron or Frequency_Jump, got {syllable_type}"
    )
    assert syllable_type in VALID_TYPES
    assert confidence in VALID_CONFIDENCES


def test_negative_sinuosity_does_not_trigger_complex_or_chevron():
    """Negative sinuosity must not trigger Complex (>3.5) or Chevron (>1.8).

    Sinuosity is always >= 1 in practice (it is a ratio of path length to chord),
    but malformed data could supply a negative value. The classifier must not crash
    and must not incorrectly fire the sinuosity-based rules.
    """
    # Large bandwidth, negative sinuosity — sinuosity < 1.8 so Freq_Jump CAN fire
    row = _make_row(call_length_s=0.08, slope=0.0, sinuosity=-2.0, bandwidth_hz=70.0)
    syllable_type, confidence = classify_call(row)
    assert syllable_type not in {"Complex", "Chevron"}, (
        f"Negative sinuosity should not trigger Complex or Chevron, got {syllable_type}"
    )
    assert syllable_type in VALID_TYPES
    assert confidence in VALID_CONFIDENCES


def test_negative_sinuosity_with_low_bandwidth_lands_on_slope_or_flat():
    """Negative sinuosity + low bandwidth must not crash and must produce a valid label."""
    row = _make_row(call_length_s=0.08, slope=50.0, sinuosity=-1.0, bandwidth_hz=10.0)
    syllable_type, confidence = classify_call(row)
    assert syllable_type in VALID_TYPES
    assert confidence in VALID_CONFIDENCES


# ---------------------------------------------------------------------------
# B. Infinite feature values bypass the NaN guard
# ---------------------------------------------------------------------------


def test_inf_duration_does_not_crash():
    """duration=Inf is not NaN, so it passes the guard and must produce a valid label.

    Inf < 0.015 is False, so the Short check is skipped.  The call falls through
    to whatever rule next fires.
    """
    row = _make_row(call_length_s=float("inf"))
    syllable_type, confidence = classify_call(row)
    assert syllable_type in VALID_TYPES
    assert confidence in VALID_CONFIDENCES
    # Short should NOT fire: Inf is not < 0.015
    assert syllable_type != "Short", (
        "duration=Inf should not classify as Short (Inf < 0.015 is False)"
    )


def test_inf_sinuosity_classifies_as_complex():
    """sinuosity=Inf is not NaN; Inf > 3.5 is True, so Complex should fire."""
    row = _make_row(call_length_s=0.08, sinuosity=float("inf"), slope=50.0, bandwidth_hz=20.0)
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Complex", (
        f"sinuosity=Inf should trigger Complex, got {syllable_type}"
    )
    assert confidence in VALID_CONFIDENCES


def test_inf_slope_classifies_as_up():
    """slope=Inf is not NaN; Inf > 200 is True, so Up should fire (assuming no prior rule fires)."""
    row = _make_row(call_length_s=0.08, sinuosity=1.3, slope=float("inf"), bandwidth_hz=20.0)
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Up", (
        f"slope=Inf should trigger Up, got {syllable_type}"
    )
    assert confidence in VALID_CONFIDENCES


def test_neg_inf_slope_classifies_as_down():
    """slope=-Inf is not NaN; -Inf < -200 is True, so Down should fire."""
    row = _make_row(call_length_s=0.08, sinuosity=1.3, slope=float("-inf"), bandwidth_hz=20.0)
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Down", (
        f"slope=-Inf should trigger Down, got {syllable_type}"
    )
    assert confidence in VALID_CONFIDENCES


def test_inf_bandwidth_with_low_sinuosity_classifies_as_freqjump():
    """bandwidth_hz=Inf with sinuosity<1.8 should trigger Frequency_Jump (Inf > 55)."""
    row = _make_row(call_length_s=0.08, sinuosity=1.3, slope=50.0, bandwidth_hz=float("inf"))
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Frequency_Jump", (
        f"bandwidth=Inf + sinuosity=1.3 should be Frequency_Jump, got {syllable_type}"
    )
    assert confidence in VALID_CONFIDENCES


# ---------------------------------------------------------------------------
# C. Zero values for all four features
# ---------------------------------------------------------------------------


def test_all_zeros_classifies_as_short():
    """All four features zero: duration=0 < 0.015, so Short fires first.

    This documents the cascade behaviour with a degenerate all-zero row.
    """
    row = _make_row(call_length_s=0.0, slope=0.0, sinuosity=0.0, bandwidth_hz=0.0)
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Short", (
        f"duration=0 satisfies duration < 0.015 → should be Short, got {syllable_type}"
    )
    assert confidence in VALID_CONFIDENCES


def test_zero_slope_zero_sinuosity_normal_duration_classifies_as_flat():
    """slope=0, sinuosity=0 (duration above Short threshold) should reach Flat.

    abs_slope=0 < 160 AND sinuosity=0 < 1.3 → high confidence.
    """
    row = _make_row(call_length_s=0.08, slope=0.0, sinuosity=0.0, bandwidth_hz=20.0)
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Flat", (
        f"slope=0, sinuosity=0 with normal duration should be Flat, got {syllable_type}"
    )
    assert confidence == "high", (
        f"slope=0 sinuosity=0 should yield high-confidence Flat, got '{confidence}'"
    )


# ---------------------------------------------------------------------------
# D. Flat confidence — negative slope branch and sinuosity boundary at 1.3
# ---------------------------------------------------------------------------


def test_flat_negative_slope_high_confidence():
    """A flat call with a small NEGATIVE slope (e.g. -30) should also reach high confidence.

    The Flat path uses abs(slope), so a symmetric negative value must produce
    the same confidence as its positive counterpart.
    """
    row = _make_row(call_length_s=0.08, slope=-30.0, sinuosity=1.2, bandwidth_hz=20.0)
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Flat"
    assert confidence == "high", (
        f"slope=-30 (abs=30 < 160, sinuosity=1.2 < 1.3) should be high-confidence Flat, got '{confidence}'"
    )


def test_flat_negative_slope_medium_confidence():
    """slope=-165 should reach medium-confidence Flat (mirrors positive slope=-165 case)."""
    row = _make_row(call_length_s=0.08, slope=-165.0, sinuosity=1.5, bandwidth_hz=20.0)
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Flat"
    assert confidence == "medium", (
        f"slope=-165 should be medium-confidence Flat, got '{confidence}'"
    )


def test_flat_sinuosity_at_boundary_1_3_is_not_high_confidence():
    """sinuosity exactly 1.3 fails the strict-less-than condition for high confidence.

    Flat high confidence requires: abs_slope < 160 AND sinuosity < 1.3 (strict).
    At sinuosity=1.3 the high-confidence branch is NOT taken; the call must be
    medium or low confidence instead.
    """
    # abs_slope=30 < 160 — would be high if sinuosity < 1.3, but sinuosity=1.3 exactly
    row = _make_row(call_length_s=0.08, slope=30.0, sinuosity=1.3, bandwidth_hz=20.0)
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Flat"
    assert confidence != "high", (
        f"sinuosity=1.3 (not strictly < 1.3) should NOT yield high confidence, got '{confidence}'"
    )


def test_flat_sinuosity_just_below_1_3_is_high_confidence():
    """sinuosity=1.29 is strictly less than 1.3 — high confidence must fire."""
    row = _make_row(call_length_s=0.08, slope=30.0, sinuosity=1.29, bandwidth_hz=20.0)
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Flat"
    assert confidence == "high", (
        f"sinuosity=1.29 < 1.3 with abs_slope=30 < 160 should be high confidence, got '{confidence}'"
    )


# ---------------------------------------------------------------------------
# E. Down call confidence at the call level (uses abs(slope))
# ---------------------------------------------------------------------------


def test_down_call_high_confidence():
    """A strongly negative slope should produce Down with high confidence.

    The Down path calls _confidence(abs(slope), THRESH_SLOPE_DIRECTIONAL, below=False).
    T=200, margin=40, high cutoff = 80.
    slope=-400 → abs=400, distance=200 > 80 → high.
    """
    row = _make_row(call_length_s=0.08, sinuosity=1.3, slope=-400.0, bandwidth_hz=20.0)
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Down"
    assert confidence == "high", f"Expected high confidence for slope=-400, got '{confidence}'"


def test_down_call_medium_confidence():
    """A moderately negative slope should produce Down with medium confidence.

    T=200, margin=40, medium cutoff=20, high cutoff=80.
    slope=-250 → abs=250, distance=50 → 20 < 50 ≤ 80 → medium.
    """
    row = _make_row(call_length_s=0.08, sinuosity=1.3, slope=-250.0, bandwidth_hz=20.0)
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Down"
    assert confidence == "medium", f"Expected medium confidence for slope=-250, got '{confidence}'"


def test_down_call_low_confidence():
    """A barely negative slope (just past -200) should produce Down with low confidence.

    T=200, margin=40, medium cutoff=20.
    slope=-205 → abs=205, distance=5 ≤ 20 → low.
    """
    row = _make_row(call_length_s=0.08, sinuosity=1.3, slope=-205.0, bandwidth_hz=20.0)
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Down"
    assert confidence == "low", f"Expected low confidence for slope=-205, got '{confidence}'"


def test_down_uses_abs_slope_for_confidence():
    """slope=-400 and slope=-401 should both give the same confidence level (abs symmetry).

    This verifies that the abs() call in the Down path is actually exercised
    and not accidentally passing the raw negative value to _confidence.
    """
    row_a = _make_row(call_length_s=0.08, sinuosity=1.3, slope=-400.0, bandwidth_hz=20.0)
    row_b = _make_row(call_length_s=0.08, sinuosity=1.3, slope=-401.0, bandwidth_hz=20.0)
    _, conf_a = classify_call(row_a)
    _, conf_b = classify_call(row_b)
    # Both are comfortably in the high-confidence zone; identical confidence expected
    assert conf_a == conf_b == "high"


# ---------------------------------------------------------------------------
# F. Frequency_Jump confidence at the call level
# ---------------------------------------------------------------------------


def test_freqjump_high_confidence():
    """bandwidth well above 55 kHz should yield Frequency_Jump with high confidence.

    T=55, margin=11, high cutoff=22.
    bandwidth=90 → distance=35 > 22 → high.
    """
    row = _make_row(call_length_s=0.08, sinuosity=1.3, slope=50.0, bandwidth_hz=90.0)
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Frequency_Jump"
    assert confidence == "high", f"Expected high confidence for bandwidth=90, got '{confidence}'"


def test_freqjump_medium_confidence():
    """bandwidth moderately above 55 kHz should yield Frequency_Jump with medium confidence.

    T=55, margin=11, medium cutoff=5.5, high cutoff=22.
    bandwidth=70 → distance=15 → 5.5 < 15 ≤ 22 → medium.
    """
    row = _make_row(call_length_s=0.08, sinuosity=1.3, slope=50.0, bandwidth_hz=70.0)
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Frequency_Jump"
    assert confidence == "medium", f"Expected medium confidence for bandwidth=70, got '{confidence}'"


def test_freqjump_low_confidence():
    """bandwidth just above 55 kHz should yield Frequency_Jump with low confidence.

    T=55, margin=11, medium cutoff=5.5.
    bandwidth=58 → distance=3 ≤ 5.5 → low.
    """
    row = _make_row(call_length_s=0.08, sinuosity=1.3, slope=50.0, bandwidth_hz=58.0)
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Frequency_Jump"
    assert confidence == "low", f"Expected low confidence for bandwidth=58, got '{confidence}'"


# ---------------------------------------------------------------------------
# G. Chevron confidence: which criterion drives the minimum
# ---------------------------------------------------------------------------


def test_chevron_confidence_driven_by_bandwidth():
    """When sinuosity is high-confidence but bandwidth is only low-confidence,
    the Chevron result should be low confidence (minimum wins).

    sinuosity=3.0: T=1.8, margin=0.36, distance=1.2 > 0.72 (high cutoff) → sinuosity=high
    bandwidth=26: T=25, margin=5, distance=1 ≤ 2.5 (medium cutoff) → bandwidth=low
    min(high, low) = low
    """
    row = _make_row(call_length_s=0.08, slope=50.0, sinuosity=3.0, bandwidth_hz=26.0)
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Chevron"
    assert confidence == "low", (
        f"bandwidth barely above 25 should drag Chevron confidence to low, got '{confidence}'"
    )


def test_chevron_confidence_driven_by_sinuosity():
    """When bandwidth is high-confidence but sinuosity is only low-confidence,
    the Chevron result should be low confidence (minimum wins).

    sinuosity=1.9: T=1.8, margin=0.36, distance=0.1 ≤ 0.18 (medium cutoff) → sinuosity=low
    bandwidth=60: T=25, margin=5, distance=35 > 10 (high cutoff) → bandwidth=high
    min(low, high) = low
    """
    row = _make_row(call_length_s=0.08, slope=50.0, sinuosity=1.9, bandwidth_hz=60.0)
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Chevron"
    assert confidence == "low", (
        f"sinuosity barely above 1.8 should drag Chevron confidence to low, got '{confidence}'"
    )


def test_chevron_both_high_confidence():
    """When both sinuosity and bandwidth are clearly above their thresholds,
    Chevron should be high confidence.

    sinuosity=3.0: distance=1.2 > 0.72 → high
    bandwidth=50: T=25, margin=5, distance=25 > 10 → high
    min(high, high) = high
    """
    row = _make_row(call_length_s=0.08, slope=50.0, sinuosity=3.0, bandwidth_hz=50.0)
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Chevron"
    assert confidence == "high", (
        f"Both sinuosity and bandwidth well above thresholds should yield high confidence, got '{confidence}'"
    )


# ---------------------------------------------------------------------------
# H. _confidence called with value on wrong side of threshold (negative distance)
# ---------------------------------------------------------------------------


def test_confidence_negative_distance_below_returns_low():
    """When below=True but value > threshold, distance is negative.

    margin = T * 0.20 > 0.  Negative distance < margin*2 AND < margin*0.5, so result is "low".
    This tests the contract-violation branch: caller passes a value that already exceeds
    the threshold while asking for a below=True confidence.
    """
    # value=0.020 > threshold=0.015 with below=True → distance = 0.015 - 0.020 = -0.005
    result = _confidence(0.020, 0.015, below=True)
    # Distance is negative: neither high nor medium cutoff is exceeded → "low"
    assert result == "low", (
        f"Negative distance (value on wrong side) should return 'low', got '{result}'"
    )


def test_confidence_negative_distance_above_returns_low():
    """When below=False but value < threshold, distance is negative → "low"."""
    # value=3.0 < threshold=3.5 with below=False → distance = 3.0 - 3.5 = -0.5
    result = _confidence(3.0, 3.5, below=False)
    assert result == "low", (
        f"Negative distance (value below threshold, below=False) should return 'low', got '{result}'"
    )


# ---------------------------------------------------------------------------
# I. _confidence with threshold=0 (zero-threshold edge case)
# ---------------------------------------------------------------------------


def test_confidence_zero_threshold_any_positive_value_is_high():
    """When threshold=0, margin=0 and high cutoff = 0*2=0.
    distance > 0 is True for any positive distance, so the result should be 'high'.
    """
    result = _confidence(1.0, 0.0, below=False)
    assert result == "high", (
        f"With threshold=0, distance=1.0 > 0 = high cutoff → expected 'high', got '{result}'"
    )


def test_confidence_zero_threshold_zero_value_below_true():
    """threshold=0, value=0, below=True → distance = 0 - 0 = 0.
    0 > margin*2 = 0 is False (not strictly greater).
    0 > margin*0.5 = 0 is False.
    Result should be 'low'.
    """
    result = _confidence(0.0, 0.0, below=True)
    assert result == "low", (
        f"threshold=0 value=0: distance=0, not > 0 → expected 'low', got '{result}'"
    )


# ---------------------------------------------------------------------------
# J. classify_dataframe with non-default / duplicate index
# ---------------------------------------------------------------------------


def test_classify_dataframe_non_default_index_preserved():
    """A DataFrame whose index starts at 100 must have its index preserved after classification."""
    df = pd.DataFrame(
        [
            {"call_length_s": 0.005, "slope": 50.0, "sinuosity": 1.5, "bandwidth_hz": 20.0},
            {"call_length_s": 0.08, "slope": 400.0, "sinuosity": 1.3, "bandwidth_hz": 20.0},
        ],
        index=[100, 200],
    )
    result = classify_dataframe(df)
    assert list(result.index) == [100, 200], (
        f"Original index must be preserved, got {list(result.index)}"
    )
    assert result.loc[100, "syllable_type"] == "Short"
    assert result.loc[200, "syllable_type"] == "Up"


def test_classify_dataframe_duplicate_index_no_nan_types():
    """A DataFrame with duplicate index values must still classify correctly without producing NaN types."""
    df = pd.DataFrame(
        [
            {"call_length_s": 0.005, "slope": 50.0, "sinuosity": 1.5, "bandwidth_hz": 20.0},
            {"call_length_s": 0.08, "slope": 400.0, "sinuosity": 1.3, "bandwidth_hz": 20.0},
            {"call_length_s": 0.08, "slope": -400.0, "sinuosity": 1.3, "bandwidth_hz": 20.0},
        ],
        index=[0, 0, 0],  # all duplicates
    )
    result = classify_dataframe(df)
    assert len(result) == 3
    assert result["syllable_type"].notna().all(), (
        "Duplicate index must not cause NaN syllable_type via misaligned concat"
    )
    assert list(result["syllable_type"]) == ["Short", "Up", "Down"]


def test_classify_dataframe_string_index():
    """A DataFrame with a string index must classify correctly."""
    df = pd.DataFrame(
        [
            {"call_length_s": 0.005, "slope": 50.0, "sinuosity": 1.5, "bandwidth_hz": 20.0},
            {"call_length_s": 0.08, "slope": 0.0, "sinuosity": 1.2, "bandwidth_hz": 20.0},
        ],
        index=["call_a", "call_b"],
    )
    result = classify_dataframe(df)
    assert result.loc["call_a", "syllable_type"] == "Short"
    assert result.loc["call_b", "syllable_type"] == "Flat"


# ---------------------------------------------------------------------------
# K. classify_dataframe with extra irrelevant columns (realistic input shape)
# ---------------------------------------------------------------------------


def test_classify_dataframe_extra_columns_preserved():
    """A DataFrame with many extra columns (as in real DeepSqueak output) must
    have ALL original columns preserved in the output."""
    df = pd.DataFrame(
        [
            {
                "call_length_s": 0.005,
                "slope": 50.0,
                "sinuosity": 1.5,
                "bandwidth_hz": 20.0,
                "wav_stem": "rec_001",
                "begin_time_s": 1.234,
                "end_time_s": 1.239,
                "principal_freq_hz": 70.0,
                "low_freq_hz": 60.0,
                "high_freq_hz": 80.0,
                "tonality": 0.95,
                "label": "cluster_3",
                "score": 0.87,
            }
        ]
    )
    original_cols = set(df.columns)
    result = classify_dataframe(df)
    missing = original_cols - set(result.columns)
    assert not missing, f"Columns dropped from realistic input: {missing}"
    # Two new columns must be added
    assert "syllable_type" in result.columns
    assert "classification_confidence" in result.columns
    assert len(result.columns) == len(df.columns) + 2


def test_classify_dataframe_extra_columns_correct_classification():
    """Classification result must be correct regardless of extra columns present."""
    df = pd.DataFrame(
        [
            {
                "call_length_s": 0.08,
                "slope": 400.0,
                "sinuosity": 1.3,
                "bandwidth_hz": 20.0,
                "extra_col_a": "foo",
                "extra_col_b": 999,
            }
        ]
    )
    result = classify_dataframe(df)
    assert result.iloc[0]["syllable_type"] == "Up"


# ---------------------------------------------------------------------------
# L. Large input — no crash and correct row count
# ---------------------------------------------------------------------------


def test_classify_dataframe_large_input_no_crash():
    """Classifying 5000 rows must complete without error and return the correct row count."""
    rng = np.random.default_rng(seed=42)
    n = 5000
    df = pd.DataFrame(
        {
            "call_length_s": rng.uniform(0.005, 0.200, n),
            "slope": rng.uniform(-500.0, 500.0, n),
            "sinuosity": rng.uniform(1.0, 6.0, n),
            "bandwidth_hz": rng.uniform(5.0, 100.0, n),
        }
    )
    result = classify_dataframe(df)
    assert len(result) == n, f"Expected {n} rows, got {len(result)}"
    assert result["syllable_type"].notna().all()
    assert result["classification_confidence"].notna().all()
    # All produced types must be valid
    assert set(result["syllable_type"].unique()).issubset(VALID_TYPES)
    assert set(result["classification_confidence"].unique()).issubset(VALID_CONFIDENCES)


def test_classify_dataframe_large_input_all_types_represented():
    """Over 5000 randomly-sampled rows spanning the full feature space, all 7 classified
    types (Short, Complex, Chevron, Frequency_Jump, Up, Down, Flat) should appear at
    least once.  If any type is absent the thresholds or the random seed are suspect.
    """
    rng = np.random.default_rng(seed=42)
    n = 5000
    df = pd.DataFrame(
        {
            "call_length_s": rng.uniform(0.005, 0.200, n),
            "slope": rng.uniform(-600.0, 600.0, n),
            "sinuosity": rng.uniform(1.0, 7.0, n),
            "bandwidth_hz": rng.uniform(5.0, 120.0, n),
        }
    )
    result = classify_dataframe(df)
    found_types = set(result["syllable_type"].unique()) - {"unclassified"}
    expected_types = {"Short", "Complex", "Chevron", "Frequency_Jump", "Up", "Down", "Flat"}
    missing = expected_types - found_types
    assert not missing, (
        f"Expected all 7 types to appear in 5000 rows, missing: {missing}"
    )


# ---------------------------------------------------------------------------
# M. Priority gaps: Short wins over Chevron and Frequency_Jump
# ---------------------------------------------------------------------------


def test_priority_short_over_chevron():
    """A very short call with Chevron-qualifying features must be Short, not Chevron.

    Short (priority 1) must beat Chevron (priority 3).
    """
    row = _make_row(
        call_length_s=0.005,   # Short: < 0.015
        sinuosity=2.5,          # Chevron: > 1.8
        bandwidth_hz=40.0,      # Chevron: > 25
        slope=50.0,
    )
    syllable_type, _ = classify_call(row)
    assert syllable_type == "Short", (
        f"Short must beat Chevron in priority cascade, got {syllable_type}"
    )


def test_priority_short_over_freqjump():
    """A very short call with Frequency_Jump-qualifying features must be Short, not Frequency_Jump.

    Short (priority 1) must beat Frequency_Jump (priority 4).
    """
    row = _make_row(
        call_length_s=0.005,   # Short: < 0.015
        sinuosity=1.3,          # Freq_Jump: < 1.8
        bandwidth_hz=70.0,      # Freq_Jump: > 55
        slope=50.0,
    )
    syllable_type, _ = classify_call(row)
    assert syllable_type == "Short", (
        f"Short must beat Frequency_Jump in priority cascade, got {syllable_type}"
    )


def test_priority_short_over_down():
    """A very short call with a strongly negative slope must be Short, not Down."""
    row = _make_row(
        call_length_s=0.005,
        sinuosity=1.3,
        slope=-400.0,
        bandwidth_hz=20.0,
    )
    syllable_type, _ = classify_call(row)
    assert syllable_type == "Short", (
        f"Short must beat Down in priority cascade, got {syllable_type}"
    )


def test_priority_complex_over_freqjump():
    """A call with both Complex sinuosity AND Frequency_Jump bandwidth must be Complex.

    sinuosity=5.0 > 3.5 (Complex) AND sinuosity=5.0 > 1.8 (fails Freq_Jump sinuosity < 1.8).
    This test confirms that even when bandwidth would qualify for Freq_Jump,
    Complex takes priority if sinuosity is high enough.
    Note: sinuosity=5.0 > 1.8 means Freq_Jump's sinuosity<1.8 condition is not met anyway,
    but we're confirming the cascade stops at Complex.
    """
    row = _make_row(
        call_length_s=0.08,
        sinuosity=5.0,          # Complex: > 3.5
        bandwidth_hz=70.0,      # would qualify bandwidth for Freq_Jump
        slope=50.0,
    )
    syllable_type, _ = classify_call(row)
    assert syllable_type == "Complex", (
        f"Complex must beat any lower-priority rule, got {syllable_type}"
    )


def test_priority_complex_over_up():
    """A call with Complex sinuosity AND Up-qualifying slope must be Complex."""
    row = _make_row(
        call_length_s=0.08,
        sinuosity=5.0,
        slope=400.0,    # would qualify as Up
        bandwidth_hz=20.0,
    )
    syllable_type, _ = classify_call(row)
    assert syllable_type == "Complex"


def test_priority_complex_over_down():
    """A call with Complex sinuosity AND Down-qualifying slope must be Complex."""
    row = _make_row(
        call_length_s=0.08,
        sinuosity=5.0,
        slope=-400.0,   # would qualify as Down
        bandwidth_hz=20.0,
    )
    syllable_type, _ = classify_call(row)
    assert syllable_type == "Complex"


# ---------------------------------------------------------------------------
# N. Up confidence levels (parallel to Down, verifies Up branch directly)
# ---------------------------------------------------------------------------


def test_up_call_high_confidence():
    """slope=400 (well above 200) should yield Up with high confidence.

    T=200, margin=40, high cutoff=80.
    distance=200 > 80 → high.
    """
    row = _make_row(call_length_s=0.08, sinuosity=1.3, slope=400.0, bandwidth_hz=20.0)
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Up"
    assert confidence == "high", f"Expected high confidence for slope=400, got '{confidence}'"


def test_up_call_medium_confidence():
    """slope=250 should yield Up with medium confidence.

    T=200, margin=40, medium cutoff=20, high cutoff=80.
    distance=50 → 20 < 50 ≤ 80 → medium.
    """
    row = _make_row(call_length_s=0.08, sinuosity=1.3, slope=250.0, bandwidth_hz=20.0)
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Up"
    assert confidence == "medium", f"Expected medium confidence for slope=250, got '{confidence}'"


def test_up_call_low_confidence():
    """slope=205 (just above 200) should yield Up with low confidence.

    T=200, margin=40, medium cutoff=20.
    distance=5 ≤ 20 → low.
    """
    row = _make_row(call_length_s=0.08, sinuosity=1.3, slope=205.0, bandwidth_hz=20.0)
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Up"
    assert confidence == "low", f"Expected low confidence for slope=205, got '{confidence}'"
