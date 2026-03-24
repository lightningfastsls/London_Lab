"""Test script for boundary adjustment feature.

Tests the boundary adjustment functionality without requiring GUI interaction.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from usv_spectrogram.app.core.detection_logic import DetectedUSV, DetectionResult


def test_detected_usv_adjustment_fields():
    """Test that DetectedUSV has the new adjustment fields."""
    print("\n" + "="*60)
    print("TEST 1: DetectedUSV Adjustment Fields")
    print("="*60)

    # Create a DetectedUSV with adjustment fields
    usv = DetectedUSV(
        start_time_s=1.0,
        end_time_s=1.2,
        start_col=100,
        end_col=120,
        max_probability=0.95,
        mean_probability=0.90,
        user_adjusted=True,
        original_start_time_s=0.95,
        original_end_time_s=1.25
    )

    # Verify fields exist and have correct values
    assert hasattr(usv, 'user_adjusted'), "Missing user_adjusted field"
    assert hasattr(usv, 'original_start_time_s'), "Missing original_start_time_s field"
    assert hasattr(usv, 'original_end_time_s'), "Missing original_end_time_s field"

    assert usv.user_adjusted == True, "user_adjusted should be True"
    assert usv.original_start_time_s == 0.95, "original_start_time_s incorrect"
    assert usv.original_end_time_s == 1.25, "original_end_time_s incorrect"

    print("PASS: All adjustment fields present and correct")
    print(f"  - user_adjusted: {usv.user_adjusted}")
    print(f"  - original_start_time_s: {usv.original_start_time_s}")
    print(f"  - original_end_time_s: {usv.original_end_time_s}")

    return True


def test_default_values():
    """Test that adjustment fields have correct defaults."""
    print("\n" + "="*60)
    print("TEST 2: Default Values for Adjustment Fields")
    print("="*60)

    # Create DetectedUSV without adjustment fields (backward compatibility)
    usv = DetectedUSV(
        start_time_s=1.0,
        end_time_s=1.2,
        start_col=100,
        end_col=120,
        max_probability=0.95,
        mean_probability=0.90
    )

    # Verify defaults
    assert usv.user_adjusted == False, "user_adjusted should default to False"
    assert usv.original_start_time_s is None, "original_start_time_s should default to None"
    assert usv.original_end_time_s is None, "original_end_time_s should default to None"

    print("PASS: Default values correct")
    print(f"  - user_adjusted: {usv.user_adjusted} (default)")
    print(f"  - original_start_time_s: {usv.original_start_time_s} (default)")
    print(f"  - original_end_time_s: {usv.original_end_time_s} (default)")

    return True


def test_boundary_adjustment_logic():
    """Test the boundary adjustment logic."""
    print("\n" + "="*60)
    print("TEST 3: Boundary Adjustment Logic")
    print("="*60)

    # Create original detection
    original = DetectedUSV(
        start_time_s=1.0,
        end_time_s=1.2,
        start_col=100,
        end_col=120,
        max_probability=0.95,
        mean_probability=0.90
    )

    print(f"Original detection: {original.start_time_s}s - {original.end_time_s}s")

    # Simulate boundary adjustment (what MainWindow._on_boundary_adjusted does)
    new_start = 0.95
    new_end = 1.25

    # Create adjusted detection
    adjusted = DetectedUSV(
        start_time_s=new_start,
        end_time_s=new_end,
        start_col=95,  # Would be calculated from time
        end_col=125,
        max_probability=original.max_probability,
        mean_probability=original.mean_probability,
        user_adjusted=True,
        original_start_time_s=original.start_time_s,
        original_end_time_s=original.end_time_s
    )

    print(f"Adjusted detection: {adjusted.start_time_s}s - {adjusted.end_time_s}s")

    # Verify adjustment
    assert adjusted.user_adjusted == True, "Should be marked as adjusted"
    assert adjusted.original_start_time_s == 1.0, "Should preserve original start"
    assert adjusted.original_end_time_s == 1.2, "Should preserve original end"
    assert adjusted.start_time_s == 0.95, "New start time incorrect"
    assert adjusted.end_time_s == 1.25, "New end time incorrect"
    assert adjusted.max_probability == original.max_probability, "Should preserve probability"

    print("PASS: Boundary adjustment logic correct")
    print(f"  - Original times preserved: {adjusted.original_start_time_s}s - {adjusted.original_end_time_s}s")
    print(f"  - New times applied: {adjusted.start_time_s}s - {adjusted.end_time_s}s")
    print(f"  - Metadata preserved: max_prob={adjusted.max_probability}")

    return True


def test_sequential_adjustments():
    """Test multiple sequential adjustments preserve first original."""
    print("\n" + "="*60)
    print("TEST 4: Sequential Adjustments")
    print("="*60)

    # Original detection
    detection = DetectedUSV(
        start_time_s=1.0,
        end_time_s=1.2,
        start_col=100,
        end_col=120,
        max_probability=0.95,
        mean_probability=0.90
    )

    print(f"Original: {detection.start_time_s}s - {detection.end_time_s}s")

    # First adjustment
    first_adjusted = DetectedUSV(
        start_time_s=0.95,
        end_time_s=1.25,
        start_col=95,
        end_col=125,
        max_probability=detection.max_probability,
        mean_probability=detection.mean_probability,
        user_adjusted=True,
        original_start_time_s=detection.start_time_s,  # Store original
        original_end_time_s=detection.end_time_s
    )

    print(f"After 1st adjustment: {first_adjusted.start_time_s}s - {first_adjusted.end_time_s}s")

    # Second adjustment (simulating what MainWindow does)
    second_adjusted = DetectedUSV(
        start_time_s=0.90,
        end_time_s=1.30,
        start_col=90,
        end_col=130,
        max_probability=first_adjusted.max_probability,
        mean_probability=first_adjusted.mean_probability,
        user_adjusted=True,
        # Keep original from FIRST adjustment, not intermediate
        original_start_time_s=(
            first_adjusted.start_time_s
            if not first_adjusted.user_adjusted
            else first_adjusted.original_start_time_s
        ),
        original_end_time_s=(
            first_adjusted.end_time_s
            if not first_adjusted.user_adjusted
            else first_adjusted.original_end_time_s
        )
    )

    print(f"After 2nd adjustment: {second_adjusted.start_time_s}s - {second_adjusted.end_time_s}s")

    # Verify that original times are from the FIRST detection, not intermediate
    assert second_adjusted.original_start_time_s == 1.0, "Should preserve FIRST original start"
    assert second_adjusted.original_end_time_s == 1.2, "Should preserve FIRST original end"
    assert second_adjusted.start_time_s == 0.90, "Should have latest start time"
    assert second_adjusted.end_time_s == 1.30, "Should have latest end time"

    print("PASS: Sequential adjustments preserve first original")
    print(f"  - Original (preserved): {second_adjusted.original_start_time_s}s - {second_adjusted.original_end_time_s}s")
    print(f"  - Current: {second_adjusted.start_time_s}s - {second_adjusted.end_time_s}s")

    return True


def test_validation_constraints():
    """Test that validation constraints work correctly."""
    print("\n" + "="*60)
    print("TEST 5: Validation Constraints")
    print("="*60)

    # Test minimum duration constraint
    detection = DetectedUSV(
        start_time_s=1.0,
        end_time_s=1.2,
        start_col=100,
        end_col=120,
        max_probability=0.95,
        mean_probability=0.90
    )

    # Simulate dragging start past end (what the UI prevents)
    new_start = 1.25  # Beyond end time
    new_end = detection.end_time_s

    # UI should clamp this to (end - 0.001)
    clamped_start = min(new_start, new_end - 0.001)

    adjusted = DetectedUSV(
        start_time_s=clamped_start,
        end_time_s=new_end,
        start_col=119,
        end_col=120,
        max_probability=detection.max_probability,
        mean_probability=detection.mean_probability,
        user_adjusted=True,
        original_start_time_s=detection.start_time_s,
        original_end_time_s=detection.end_time_s
    )

    duration = adjusted.end_time_s - adjusted.start_time_s

    # Use small tolerance for floating point comparison
    assert duration >= 0.001 - 1e-10, f"Duration {duration*1000}ms should be >= 1ms (within floating point tolerance)"
    assert adjusted.start_time_s < adjusted.end_time_s, "Start should be before end"

    print("PASS: Validation constraints working")
    print(f"  - Attempted start: {new_start}s (past end)")
    print(f"  - Clamped start: {clamped_start}s")
    print(f"  - Final duration: {duration*1000:.3f}ms (>= 1ms, within FP tolerance)")

    return True


def test_json_serialization():
    """Test that adjustment fields can be serialized to dict."""
    print("\n" + "="*60)
    print("TEST 6: JSON Serialization")
    print("="*60)

    from dataclasses import asdict

    usv = DetectedUSV(
        start_time_s=1.0,
        end_time_s=1.2,
        start_col=100,
        end_col=120,
        max_probability=0.95,
        mean_probability=0.90,
        user_adjusted=True,
        original_start_time_s=0.95,
        original_end_time_s=1.25
    )

    # Convert to dict (what LabelStorage.save() does)
    usv_dict = asdict(usv)

    assert 'user_adjusted' in usv_dict, "user_adjusted missing from dict"
    assert 'original_start_time_s' in usv_dict, "original_start_time_s missing from dict"
    assert 'original_end_time_s' in usv_dict, "original_end_time_s missing from dict"

    print("PASS: Serialization includes adjustment fields")
    print(f"  Dict keys: {list(usv_dict.keys())}")

    # Simulate JSON structure
    detection_dict = {
        "start_time_s": usv.start_time_s,
        "end_time_s": usv.end_time_s,
        "start_col": usv.start_col,
        "end_col": usv.end_col,
        "max_probability": usv.max_probability,
        "mean_probability": usv.mean_probability
    }

    # Add adjustment metadata if present
    if hasattr(usv, 'user_adjusted') and usv.user_adjusted:
        detection_dict["user_adjusted"] = True
        detection_dict["original_start_time_s"] = usv.original_start_time_s
        detection_dict["original_end_time_s"] = usv.original_end_time_s

    assert detection_dict["user_adjusted"] == True
    print(f"  JSON-ready dict: {detection_dict}")

    return True


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*70)
    print("BOUNDARY ADJUSTMENT FEATURE - AUTOMATED TESTS")
    print("="*70)

    tests = [
        ("DetectedUSV Adjustment Fields", test_detected_usv_adjustment_fields),
        ("Default Values", test_default_values),
        ("Boundary Adjustment Logic", test_boundary_adjustment_logic),
        ("Sequential Adjustments", test_sequential_adjustments),
        ("Validation Constraints", test_validation_constraints),
        ("JSON Serialization", test_json_serialization),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "PASS" if success else "FAIL", None))
        except Exception as e:
            results.append((name, "FAIL", str(e)))
            print(f"\nFAIL: {e}")

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, status, _ in results if status == "PASS")
    failed = sum(1 for _, status, _ in results if status == "FAIL")

    for name, status, error in results:
        print(f"{status:>6}: {name}")
        if error:
            print(f"         Error: {error}")

    print("-" * 70)
    print(f"Total: {len(results)} tests | Passed: {passed} | Failed: {failed}")

    if failed == 0:
        print("\n SUCCESS: All tests passed!")
        return 0
    else:
        print(f"\n FAILURE: {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
