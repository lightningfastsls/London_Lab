"""Tests for classify_traditional_taxonomy — written by test-architect BEFORE implementation.

ROADMAP test plan coverage:
  1. Each type classification (7 types + unclassified)
       -> test_classify_short_call
       -> test_classify_complex_call
       -> test_classify_chevron_call
       -> test_classify_frequency_jump_call
       -> test_classify_up_call
       -> test_classify_down_call
       -> test_classify_flat_call
       -> test_classify_nan_features_returns_unclassified
  2. Priority ordering (Short wins over Complex)
       -> test_priority_short_over_complex
       -> test_priority_complex_over_chevron
       -> test_priority_chevron_over_freqjump
       -> test_priority_chevron_over_up
       -> test_priority_freqjump_over_up
  3. Boundary cases — values exactly at thresholds
       -> test_duration_exactly_at_short_threshold_is_not_short
       -> test_sinuosity_exactly_at_complex_threshold_is_not_complex
       -> test_slope_exactly_at_directional_threshold_is_not_up
       -> test_slope_exactly_at_negative_directional_threshold_is_not_down
  4. NaN handling
       -> test_partial_nan_returns_unclassified
       -> test_all_nan_returns_unclassified
       -> test_none_value_returns_unclassified
  5. Confidence levels with hand-computed expected values
       -> test_confidence_high_below
       -> test_confidence_medium_below
       -> test_confidence_low_below
       -> test_confidence_high_above
       -> test_confidence_medium_above
       -> test_confidence_low_above
       -> test_short_call_high_confidence
       -> test_short_call_medium_confidence
       -> test_short_call_low_confidence
       -> test_complex_call_high_confidence
       -> test_complex_call_medium_confidence
       -> test_complex_call_low_confidence
  6. _min_confidence ordering
       -> test_min_confidence_high_and_medium_returns_medium
       -> test_min_confidence_high_and_low_returns_low
       -> test_min_confidence_both_high_returns_high
       -> test_min_confidence_both_low_returns_low
       -> test_min_confidence_medium_and_low_returns_low
  7. classify_dataframe — column addition, shape, no NaN in output
       -> test_classify_dataframe_adds_syllable_type_column
       -> test_classify_dataframe_adds_confidence_column
       -> test_classify_dataframe_preserves_original_columns
       -> test_classify_dataframe_no_nan_syllable_type
       -> test_classify_dataframe_row_count_unchanged
  8. Distribution not degenerate over varied data
       -> test_varied_dataframe_produces_multiple_types
       -> test_varied_dataframe_at_least_four_types

Additional coverage (recurring gap patterns):
  - Empty DataFrame input -> test_classify_dataframe_empty_input
  - Single-row DataFrame  -> test_classify_dataframe_single_row
  - Return type is always (str, str) tuple -> test_classify_call_return_type
  - Confidence value always in known set -> test_classify_call_confidence_is_valid_value
  - Syllable type always in known set -> test_classify_call_type_is_valid_value
  - Flat confidence grading (no active threshold, special-cased in impl)
       -> test_flat_call_high_confidence
       -> test_flat_call_medium_confidence
       -> test_flat_call_low_confidence
  - Chevron requires BOTH sinuosity AND bandwidth to be high
       -> test_chevron_requires_both_sinuosity_and_bandwidth
  - Frequency Jump requires BOTH high bandwidth AND low sinuosity
       -> test_freqjump_requires_low_sinuosity
  - classify_dataframe on DataFrame with NaN rows
       -> test_classify_dataframe_with_some_nan_rows

Total: 43 tests (8 ROADMAP item groups, 35 additional / gap-pattern)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Bootstrap: scripts/ must be on sys.path so we can import the classifier
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# The import will fail until the script exists; that is the expected red state.
# Once implemented the tests should pass or reveal real bugs.
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_TYPES = {"Short", "Complex", "Chevron", "Frequency_Jump", "Up", "Down", "Flat", "unclassified"}
VALID_CONFIDENCES = {"high", "medium", "low", "none"}


def _make_row(
    call_length_s: float = 0.08,
    slope: float = 0.0,
    sinuosity: float = 1.5,
    bandwidth_hz: float = 20.0,
) -> pd.Series:
    """Convenience factory for a minimal feature row."""
    return pd.Series(
        {
            "call_length_s": call_length_s,
            "slope": slope,
            "sinuosity": sinuosity,
            "bandwidth_hz": bandwidth_hz,
        }
    )


# ---------------------------------------------------------------------------
# 1. Each type classification
# ---------------------------------------------------------------------------


def test_classify_short_call():
    """A call well below 15 ms must be classified as Short."""
    row = _make_row(call_length_s=0.005)  # 5 ms — clearly short
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Short", f"Expected Short, got {syllable_type}"
    assert confidence != "none"


def test_classify_complex_call():
    """A call with sinuosity > 3.5 (and duration >= 15 ms) must be Complex."""
    row = _make_row(call_length_s=0.08, sinuosity=5.0, slope=50.0, bandwidth_hz=20.0)
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Complex", f"Expected Complex, got {syllable_type}"


def test_classify_chevron_call():
    """A call with sinuosity > 1.8 AND bandwidth > 25 (and not short/complex) is Chevron."""
    # sinuosity=2.5 (above 1.8, below 3.5) and bandwidth=40 (above 25)
    row = _make_row(call_length_s=0.08, sinuosity=2.5, slope=50.0, bandwidth_hz=40.0)
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Chevron", f"Expected Chevron, got {syllable_type}"


def test_classify_frequency_jump_call():
    """A call with bandwidth > 55 AND sinuosity < 1.8 (not short/complex/chevron) is Frequency_Jump."""
    # bandwidth=70 (>55), sinuosity=1.4 (<1.8), slope=50 (well within flat range)
    row = _make_row(call_length_s=0.08, sinuosity=1.4, slope=50.0, bandwidth_hz=70.0)
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Frequency_Jump", f"Expected Frequency_Jump, got {syllable_type}"


def test_classify_up_call():
    """A call with slope > 200 (not short/complex/chevron/freqjump) is Up."""
    row = _make_row(call_length_s=0.08, sinuosity=1.4, slope=400.0, bandwidth_hz=20.0)
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Up", f"Expected Up, got {syllable_type}"


def test_classify_down_call():
    """A call with slope < -200 (not short/complex/chevron/freqjump) is Down."""
    row = _make_row(call_length_s=0.08, sinuosity=1.4, slope=-400.0, bandwidth_hz=20.0)
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Down", f"Expected Down, got {syllable_type}"


def test_classify_flat_call():
    """A call with |slope| < 200, low sinuosity, no other special features is Flat."""
    row = _make_row(call_length_s=0.08, sinuosity=1.3, slope=50.0, bandwidth_hz=20.0)
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Flat", f"Expected Flat, got {syllable_type}"


def test_classify_nan_features_returns_unclassified():
    """Any row with a NaN feature must return ('unclassified', 'none')."""
    row = _make_row(call_length_s=float("nan"))
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "unclassified"
    assert confidence == "none"


# ---------------------------------------------------------------------------
# 2. Priority ordering
# ---------------------------------------------------------------------------


def test_priority_short_over_complex():
    """A very short call with extreme sinuosity must be classified as Short, not Complex.

    This verifies the cascade order: priority 1 (Short) must beat priority 2 (Complex).
    """
    # duration=0.005 qualifies as Short AND sinuosity=8 qualifies as Complex
    row = _make_row(call_length_s=0.005, sinuosity=8.0)
    syllable_type, _ = classify_call(row)
    assert syllable_type == "Short", (
        "Short must take priority over Complex — duration check comes first in cascade"
    )


def test_priority_complex_over_chevron():
    """A long call with sinuosity > 3.5 AND bandwidth > 25 must be Complex, not Chevron.

    Complex (priority 2) must beat Chevron (priority 3).
    """
    row = _make_row(call_length_s=0.08, sinuosity=4.0, bandwidth_hz=40.0, slope=50.0)
    syllable_type, _ = classify_call(row)
    assert syllable_type == "Complex"


def test_priority_chevron_over_freqjump():
    """A call qualifying for both Chevron and Frequency_Jump must be classified Chevron.

    Chevron (priority 3) must beat Frequency_Jump (priority 4).
    sinuosity=2.0 (>1.8 for Chevron), bandwidth=60 (>55 for FreqJump and >25 for Chevron).
    For Freq_Jump we also need sinuosity < 1.8 — this row has sinuosity=2.0 so Freq_Jump
    doesn't actually apply here. Instead verify Chevron is preferred over FreqJump
    when only Chevron qualifies.
    """
    # sinuosity=2.0 (>THRESH_CHEVRON_SINUOSITY=1.8), bandwidth=60 (>THRESH_CHEVRON_BANDWIDTH=25)
    # sinuosity=2.0 is NOT < THRESH_FREQJUMP_SINUOSITY=1.8, so Freq_Jump doesn't apply
    row = _make_row(call_length_s=0.08, sinuosity=2.0, bandwidth_hz=60.0, slope=50.0)
    syllable_type, _ = classify_call(row)
    assert syllable_type == "Chevron", (
        "Chevron rule (sinuosity>1.8 AND bandwidth>25) must match before FreqJump is checked"
    )


def test_priority_chevron_over_up():
    """A call with chevron features plus positive slope must be Chevron, not Up.

    Chevron (priority 3) must beat Up (priority 5).
    """
    row = _make_row(call_length_s=0.08, sinuosity=2.5, bandwidth_hz=40.0, slope=400.0)
    syllable_type, _ = classify_call(row)
    assert syllable_type == "Chevron"


def test_priority_freqjump_over_up():
    """A call with large bandwidth, low sinuosity AND positive slope must be Frequency_Jump, not Up.

    Frequency_Jump (priority 4) must beat Up (priority 5).
    """
    # bandwidth=70 (>55), sinuosity=1.3 (<1.8), slope=400 (>200) — qualifies for FreqJump AND Up
    row = _make_row(call_length_s=0.08, sinuosity=1.3, bandwidth_hz=70.0, slope=400.0)
    syllable_type, _ = classify_call(row)
    assert syllable_type == "Frequency_Jump"


# ---------------------------------------------------------------------------
# 3. Boundary conditions
# ---------------------------------------------------------------------------


def test_duration_exactly_at_short_threshold_is_not_short():
    """A call at exactly 0.015 s must NOT be Short (boundary is strictly less-than).

    Spec: duration < THRESH_SHORT_DURATION_S → Short
    At exactly the threshold value, strict inequality means it falls through to later rules.
    """
    row = _make_row(call_length_s=THRESH_SHORT_DURATION_S, sinuosity=1.3, slope=50.0, bandwidth_hz=20.0)
    syllable_type, _ = classify_call(row)
    assert syllable_type != "Short", (
        f"At exactly {THRESH_SHORT_DURATION_S} s the call should NOT be Short (strict <)"
    )


def test_sinuosity_exactly_at_complex_threshold_is_not_complex():
    """A call at exactly sinuosity=3.5 must NOT be Complex (boundary is strictly greater-than).

    Spec: sinuosity > THRESH_COMPLEX_SINUOSITY → Complex
    """
    row = _make_row(
        call_length_s=0.08,
        sinuosity=THRESH_COMPLEX_SINUOSITY,
        slope=50.0,
        bandwidth_hz=20.0,
    )
    syllable_type, _ = classify_call(row)
    assert syllable_type != "Complex", (
        f"At sinuosity exactly {THRESH_COMPLEX_SINUOSITY} the call should NOT be Complex (strict >)"
    )


def test_slope_exactly_at_directional_threshold_is_not_up():
    """A call at exactly slope=200 must NOT be Up (boundary is strictly greater-than)."""
    row = _make_row(
        call_length_s=0.08,
        sinuosity=1.3,
        slope=THRESH_SLOPE_DIRECTIONAL,
        bandwidth_hz=20.0,
    )
    syllable_type, _ = classify_call(row)
    assert syllable_type != "Up", (
        f"At slope exactly {THRESH_SLOPE_DIRECTIONAL} the call should NOT be Up (strict >)"
    )


def test_slope_exactly_at_negative_directional_threshold_is_not_down():
    """A call at exactly slope=-200 must NOT be Down (boundary is strictly less-than)."""
    row = _make_row(
        call_length_s=0.08,
        sinuosity=1.3,
        slope=-THRESH_SLOPE_DIRECTIONAL,
        bandwidth_hz=20.0,
    )
    syllable_type, _ = classify_call(row)
    assert syllable_type != "Down"


# ---------------------------------------------------------------------------
# 4. NaN handling
# ---------------------------------------------------------------------------


def test_partial_nan_returns_unclassified():
    """A row with only one NaN feature must still return unclassified."""
    row = _make_row(sinuosity=float("nan"))
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "unclassified"
    assert confidence == "none"


def test_all_nan_returns_unclassified():
    """A row where all features are NaN must return ('unclassified', 'none')."""
    row = pd.Series(
        {
            "call_length_s": float("nan"),
            "slope": float("nan"),
            "sinuosity": float("nan"),
            "bandwidth_hz": float("nan"),
        }
    )
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "unclassified"
    assert confidence == "none"


def test_none_value_returns_unclassified():
    """A row with Python None (missing key) must return unclassified.

    pd.Series.get() returns None for missing keys, which pd.isna() treats as NaN.
    """
    row = pd.Series({"call_length_s": 0.08, "slope": 50.0})  # missing sinuosity, bandwidth_hz
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "unclassified"
    assert confidence == "none"


# ---------------------------------------------------------------------------
# 5. _confidence — hand-computed exact values
#
# For threshold T and CONFIDENCE_MARGIN M=0.20:
#   margin = T * 0.20
#   high   : distance > margin * 2  = T * 0.40
#   medium : distance > margin * 0.5 = T * 0.10
#   low    : otherwise
# ---------------------------------------------------------------------------


def test_confidence_high_below():
    """Value well below threshold should produce 'high' confidence.

    Hand-check: T=0.015, M=0.20 → margin=0.003, high cutoff = margin*2 = 0.006
    value=0.005 → distance = 0.015-0.005 = 0.010 > 0.006 → high
    """
    result = _confidence(0.005, 0.015, below=True)
    assert result == "high", f"Expected 'high', got '{result}'"


def test_confidence_medium_below():
    """Value moderately below threshold should produce 'medium' confidence.

    Hand-check: T=0.015, M=0.20 → margin=0.003
    medium cutoff = margin*0.5 = 0.0015; high cutoff = margin*2 = 0.006
    value=0.013 → distance = 0.015-0.013 = 0.002 → 0.0015 < 0.002 ≤ 0.006 → medium
    """
    result = _confidence(0.013, 0.015, below=True)
    assert result == "medium", f"Expected 'medium', got '{result}'"


def test_confidence_low_below():
    """Value just barely below threshold should produce 'low' confidence.

    Hand-check: T=0.015, M=0.20 → margin=0.003, medium cutoff=0.0015
    value=0.0145 → distance = 0.015-0.0145 = 0.0005 ≤ 0.0015 → low
    """
    result = _confidence(0.0145, 0.015, below=True)
    assert result == "low", f"Expected 'low', got '{result}'"


def test_confidence_high_above():
    """Value well above threshold should produce 'high' confidence.

    Hand-check: T=3.5, M=0.20 → margin=0.7, high cutoff = margin*2 = 1.4
    value=5.1 → distance = 5.1-3.5 = 1.6 > 1.4 → high
    """
    result = _confidence(5.1, 3.5, below=False)
    assert result == "high", f"Expected 'high', got '{result}'"


def test_confidence_medium_above():
    """Value moderately above threshold should produce 'medium' confidence.

    Hand-check: T=3.5, M=0.20 → margin=0.7
    medium cutoff = margin*0.5 = 0.35; high cutoff = margin*2 = 1.4
    value=4.0 → distance = 4.0-3.5 = 0.5 → 0.35 < 0.5 ≤ 1.4 → medium
    """
    result = _confidence(4.0, 3.5, below=False)
    assert result == "medium", f"Expected 'medium', got '{result}'"


def test_confidence_low_above():
    """Value just barely above threshold should produce 'low' confidence.

    Hand-check: T=3.5, M=0.20 → margin=0.7, medium cutoff = 0.35
    value=3.6 → distance = 3.6-3.5 = 0.1 ≤ 0.35 → low
    """
    result = _confidence(3.6, 3.5, below=False)
    assert result == "low", f"Expected 'low', got '{result}'"


def test_short_call_high_confidence():
    """A 5 ms call should be classified Short with high confidence.

    THRESH_SHORT_DURATION_S = 0.015
    margin = 0.015 * 0.20 = 0.003
    high cutoff = 0.003 * 2 = 0.006
    distance = 0.015 - 0.005 = 0.010 > 0.006 → high
    """
    row = _make_row(call_length_s=0.005)
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Short"
    assert confidence == "high", f"Expected 'high', got '{confidence}'"


def test_short_call_medium_confidence():
    """A 13 ms call should be classified Short with medium confidence.

    distance = 0.015 - 0.013 = 0.002 → medium cutoff=0.0015, high cutoff=0.006
    0.0015 < 0.002 ≤ 0.006 → medium
    """
    row = _make_row(call_length_s=0.013)
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Short"
    assert confidence == "medium", f"Expected 'medium', got '{confidence}'"


def test_short_call_low_confidence():
    """A 14.5 ms call should be classified Short with low confidence.

    distance = 0.015 - 0.0145 = 0.0005 ≤ 0.0015 (medium cutoff) → low
    """
    row = _make_row(call_length_s=0.0145)
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Short"
    assert confidence == "low", f"Expected 'low', got '{confidence}'"


def test_complex_call_high_confidence():
    """sinuosity=5.1 should yield Complex with high confidence.

    T=3.5, margin=0.7, high cutoff = 1.4
    distance = 5.1-3.5 = 1.6 > 1.4 → high
    """
    row = _make_row(call_length_s=0.08, sinuosity=5.1)
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Complex"
    assert confidence == "high"


def test_complex_call_medium_confidence():
    """sinuosity=4.0 should yield Complex with medium confidence.

    distance = 4.0-3.5 = 0.5; medium cutoff=0.35, high cutoff=1.4
    0.35 < 0.5 ≤ 1.4 → medium
    """
    row = _make_row(call_length_s=0.08, sinuosity=4.0)
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Complex"
    assert confidence == "medium"


def test_complex_call_low_confidence():
    """sinuosity=3.6 should yield Complex with low confidence.

    distance = 3.6-3.5 = 0.1 ≤ 0.35 (medium cutoff) → low
    """
    row = _make_row(call_length_s=0.08, sinuosity=3.6)
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Complex"
    assert confidence == "low"


# ---------------------------------------------------------------------------
# 6. _min_confidence ordering
# ---------------------------------------------------------------------------


def test_min_confidence_high_and_medium_returns_medium():
    """The lower of 'high' and 'medium' is 'medium'."""
    assert _min_confidence("high", "medium") == "medium"
    assert _min_confidence("medium", "high") == "medium"


def test_min_confidence_high_and_low_returns_low():
    """The lower of 'high' and 'low' is 'low'."""
    assert _min_confidence("high", "low") == "low"
    assert _min_confidence("low", "high") == "low"


def test_min_confidence_both_high_returns_high():
    """Both 'high' should return 'high'."""
    assert _min_confidence("high", "high") == "high"


def test_min_confidence_both_low_returns_low():
    """Both 'low' should return 'low'."""
    assert _min_confidence("low", "low") == "low"


def test_min_confidence_medium_and_low_returns_low():
    """The lower of 'medium' and 'low' is 'low'."""
    assert _min_confidence("medium", "low") == "low"
    assert _min_confidence("low", "medium") == "low"


# ---------------------------------------------------------------------------
# 7. classify_dataframe — column structure and completeness
# ---------------------------------------------------------------------------


def _make_varied_df() -> pd.DataFrame:
    """Build a small DataFrame with one row per syllable type."""
    return pd.DataFrame(
        [
            # Short: duration well below 15 ms
            {"call_length_s": 0.005, "slope": 50.0, "sinuosity": 1.5, "bandwidth_hz": 20.0},
            # Complex: sinuosity > 3.5
            {"call_length_s": 0.08, "slope": 50.0, "sinuosity": 5.0, "bandwidth_hz": 20.0},
            # Chevron: sinuosity > 1.8 AND bandwidth > 25
            {"call_length_s": 0.08, "slope": 50.0, "sinuosity": 2.5, "bandwidth_hz": 40.0},
            # Frequency_Jump: bandwidth > 55 AND sinuosity < 1.8
            {"call_length_s": 0.08, "slope": 50.0, "sinuosity": 1.3, "bandwidth_hz": 70.0},
            # Up: slope > 200, low sinuosity, low bandwidth
            {"call_length_s": 0.08, "slope": 400.0, "sinuosity": 1.3, "bandwidth_hz": 20.0},
            # Down: slope < -200, low sinuosity, low bandwidth
            {"call_length_s": 0.08, "slope": -400.0, "sinuosity": 1.3, "bandwidth_hz": 20.0},
            # Flat: low slope, low sinuosity
            {"call_length_s": 0.08, "slope": 50.0, "sinuosity": 1.3, "bandwidth_hz": 20.0},
        ]
    )


def test_classify_dataframe_adds_syllable_type_column():
    """classify_dataframe must add a 'syllable_type' column."""
    df = _make_varied_df()
    result = classify_dataframe(df)
    assert "syllable_type" in result.columns


def test_classify_dataframe_adds_confidence_column():
    """classify_dataframe must add a 'classification_confidence' column."""
    df = _make_varied_df()
    result = classify_dataframe(df)
    assert "classification_confidence" in result.columns


def test_classify_dataframe_preserves_original_columns():
    """classify_dataframe must not drop any original column."""
    df = _make_varied_df()
    original_cols = set(df.columns)
    result = classify_dataframe(df)
    assert original_cols.issubset(set(result.columns)), (
        f"Columns dropped: {original_cols - set(result.columns)}"
    )


def test_classify_dataframe_no_nan_syllable_type():
    """Every row in the output must have a non-NaN syllable_type.

    This is an acceptance criterion from the handoff spec.
    """
    df = _make_varied_df()
    result = classify_dataframe(df)
    nan_mask = result["syllable_type"].isna()
    assert not nan_mask.any(), f"{nan_mask.sum()} rows have NaN syllable_type"


def test_classify_dataframe_row_count_unchanged():
    """classify_dataframe must not add or drop rows."""
    df = _make_varied_df()
    result = classify_dataframe(df)
    assert len(result) == len(df)


def test_classify_dataframe_empty_input():
    """classify_dataframe on an empty DataFrame must return an empty DataFrame with the two new columns."""
    df = pd.DataFrame(columns=["call_length_s", "slope", "sinuosity", "bandwidth_hz"])
    result = classify_dataframe(df)
    assert len(result) == 0
    assert "syllable_type" in result.columns
    assert "classification_confidence" in result.columns


def test_classify_dataframe_single_row():
    """classify_dataframe must work correctly on a single-row DataFrame."""
    df = pd.DataFrame(
        [{"call_length_s": 0.005, "slope": 50.0, "sinuosity": 1.5, "bandwidth_hz": 20.0}]
    )
    result = classify_dataframe(df)
    assert len(result) == 1
    assert result.iloc[0]["syllable_type"] == "Short"


def test_classify_dataframe_with_some_nan_rows():
    """Rows with NaN features must be 'unclassified'; non-NaN rows must be labelled normally."""
    df = pd.DataFrame(
        [
            {"call_length_s": 0.005, "slope": 50.0, "sinuosity": 1.5, "bandwidth_hz": 20.0},
            {"call_length_s": float("nan"), "slope": 50.0, "sinuosity": 1.5, "bandwidth_hz": 20.0},
            {"call_length_s": 0.08, "slope": 400.0, "sinuosity": 1.3, "bandwidth_hz": 20.0},
        ]
    )
    result = classify_dataframe(df)
    assert result.iloc[0]["syllable_type"] == "Short"
    assert result.iloc[1]["syllable_type"] == "unclassified"
    assert result.iloc[1]["classification_confidence"] == "none"
    assert result.iloc[2]["syllable_type"] == "Up"


# ---------------------------------------------------------------------------
# 8. Distribution non-degeneracy on varied data
# ---------------------------------------------------------------------------


def test_varied_dataframe_produces_multiple_types():
    """A DataFrame with deliberately varied features must yield at least 4 distinct types."""
    df = _make_varied_df()
    result = classify_dataframe(df)
    n_types = result["syllable_type"].nunique()
    assert n_types >= 4, f"Expected >= 4 types, got {n_types}: {result['syllable_type'].unique()}"


def test_varied_dataframe_at_least_four_types():
    """Larger synthetic dataset (20 rows) spanning all types must yield ≥ 4 type categories.

    This mirrors the handoff acceptance criterion:
    'distribution is not degenerate — at least 4 types have > 5% of calls'.
    """
    rows = []
    # 3 Short
    for _ in range(3):
        rows.append({"call_length_s": 0.005, "slope": 10.0, "sinuosity": 1.4, "bandwidth_hz": 18.0})
    # 4 Flat
    for _ in range(4):
        rows.append({"call_length_s": 0.08, "slope": 30.0, "sinuosity": 1.2, "bandwidth_hz": 18.0})
    # 4 Up
    for _ in range(4):
        rows.append({"call_length_s": 0.08, "slope": 450.0, "sinuosity": 1.3, "bandwidth_hz": 18.0})
    # 4 Down
    for _ in range(4):
        rows.append({"call_length_s": 0.08, "slope": -450.0, "sinuosity": 1.3, "bandwidth_hz": 18.0})
    # 3 Complex
    for _ in range(3):
        rows.append({"call_length_s": 0.08, "slope": 50.0, "sinuosity": 5.0, "bandwidth_hz": 20.0})
    # 2 Chevron
    for _ in range(2):
        rows.append({"call_length_s": 0.08, "slope": 50.0, "sinuosity": 2.5, "bandwidth_hz": 40.0})

    df = pd.DataFrame(rows)
    result = classify_dataframe(df)
    total = len(result)
    type_counts = result["syllable_type"].value_counts()
    types_above_threshold = (type_counts / total > 0.05).sum()
    assert types_above_threshold >= 4, (
        f"Only {types_above_threshold} types had > 5% share: {type_counts.to_dict()}"
    )


# ---------------------------------------------------------------------------
# Additional gap-pattern tests
# ---------------------------------------------------------------------------


def test_classify_call_return_type():
    """classify_call must always return a tuple of exactly 2 strings."""
    row = _make_row()
    result = classify_call(row)
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 2
    assert isinstance(result[0], str)
    assert isinstance(result[1], str)


def test_classify_call_type_is_valid_value():
    """The syllable_type returned by classify_call must always be in the known type set."""
    test_rows = [
        _make_row(call_length_s=0.005),   # Short
        _make_row(sinuosity=5.0),          # Complex
        _make_row(sinuosity=2.5, bandwidth_hz=40.0),   # Chevron
        _make_row(sinuosity=1.3, bandwidth_hz=70.0),   # Frequency_Jump
        _make_row(slope=400.0),            # Up
        _make_row(slope=-400.0),           # Down
        _make_row(),                       # Flat
        _make_row(call_length_s=float("nan")),  # unclassified
    ]
    for i, row in enumerate(test_rows):
        syllable_type, _ = classify_call(row)
        assert syllable_type in VALID_TYPES, (
            f"Row {i}: syllable_type '{syllable_type}' not in valid set {VALID_TYPES}"
        )


def test_classify_call_confidence_is_valid_value():
    """The confidence returned by classify_call must always be in the known confidence set."""
    test_rows = [
        _make_row(call_length_s=0.005),
        _make_row(sinuosity=5.0),
        _make_row(call_length_s=float("nan")),
    ]
    for i, row in enumerate(test_rows):
        _, confidence = classify_call(row)
        assert confidence in VALID_CONFIDENCES, (
            f"Row {i}: confidence '{confidence}' not in valid set {VALID_CONFIDENCES}"
        )


def test_chevron_requires_both_sinuosity_and_bandwidth():
    """A call with high sinuosity but low bandwidth must NOT be Chevron.

    Spec: sinuosity > 1.8 AND bandwidth > 25 kHz → Chevron (both required).
    """
    # sinuosity=2.5 (> 1.8) but bandwidth=10 (< 25) — only one criterion met
    row = _make_row(call_length_s=0.08, sinuosity=2.5, bandwidth_hz=10.0, slope=50.0)
    syllable_type, _ = classify_call(row)
    assert syllable_type != "Chevron", (
        "Chevron requires BOTH sinuosity > 1.8 AND bandwidth > 25; bandwidth=10 should fail"
    )


def test_freqjump_requires_low_sinuosity():
    """A call with large bandwidth but high sinuosity must NOT be Frequency_Jump.

    Spec: bandwidth > 55 AND sinuosity < 1.8 → Frequency_Jump (both required).
    """
    # bandwidth=70 (> 55) but sinuosity=2.5 (> 1.8) — sinuosity condition fails
    row = _make_row(call_length_s=0.08, sinuosity=2.5, bandwidth_hz=70.0, slope=50.0)
    syllable_type, _ = classify_call(row)
    # Should be Chevron (sinuosity > 1.8 AND bandwidth > 25) not Frequency_Jump
    assert syllable_type != "Frequency_Jump", (
        "Frequency_Jump requires sinuosity < 1.8; sinuosity=2.5 should disqualify"
    )


def test_flat_call_high_confidence():
    """A call well within Flat region (very low slope, very low sinuosity) should have high confidence.

    Spec: abs_slope < THRESH * (1 - MARGIN) AND sinuosity < 1.3 → high
    0.08 * 200 → abs_slope must be < 200*(1-0.20) = 160 AND sinuosity < 1.3
    """
    row = _make_row(call_length_s=0.08, slope=30.0, sinuosity=1.2, bandwidth_hz=20.0)
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Flat"
    assert confidence == "high", f"Expected high confidence for clearly flat call, got '{confidence}'"


def test_flat_call_medium_confidence():
    """A call with moderate slope (but still below threshold) and moderate sinuosity should be medium.

    abs_slope < 200*(1-0.20*0.5) = 180 but above 160 → medium confidence
    """
    row = _make_row(call_length_s=0.08, slope=165.0, sinuosity=1.5, bandwidth_hz=20.0)
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Flat"
    assert confidence == "medium", f"Expected medium confidence, got '{confidence}'"


def test_flat_call_low_confidence():
    """A call just barely classified as Flat (slope close to 200) should have low confidence."""
    row = _make_row(call_length_s=0.08, slope=195.0, sinuosity=1.5, bandwidth_hz=20.0)
    syllable_type, confidence = classify_call(row)
    assert syllable_type == "Flat"
    assert confidence == "low", f"Expected low confidence for borderline flat call, got '{confidence}'"
