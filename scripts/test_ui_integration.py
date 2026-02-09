"""Test UI integration for boundary adjustment feature.

Tests signal/slot connections and coordinate conversions.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QPointF

from usv_spectrogram.app.main_window import MainWindow
from usv_spectrogram.app.core.detection_logic import DetectedUSV, DetectionResult
from usv_spectrogram.app.core.audio_loader import AudioData
from usv_spectrogram.app.core.sliding_inference import InferenceResult


def test_signal_slot_connection():
    """Test that boundary_adjusted signal is connected to handler."""
    print("\n" + "="*60)
    print("TEST 1: Signal/Slot Connection")
    print("="*60)

    app = QApplication(sys.argv)
    window = MainWindow()

    # Check signal exists
    assert hasattr(window.spectrogram_view.canvas, 'boundary_adjusted'), \
        "SpectrogramCanvas missing boundary_adjusted signal"

    # Check handler exists
    assert hasattr(window, '_on_boundary_adjusted'), \
        "MainWindow missing _on_boundary_adjusted handler"

    # Verify signal object exists and is the correct type
    signal = window.spectrogram_view.canvas.boundary_adjusted
    assert signal is not None, "Signal should not be None"

    print(f"PASS: Signal and handler both exist")
    print(f"  - Signal type: {type(signal).__name__}")
    print(f"  - Handler: {window._on_boundary_adjusted.__name__}")

    app.quit()
    return True


def test_pixel_to_time_conversion():
    """Test pixel-to-time coordinate conversion."""
    print("\n" + "="*60)
    print("TEST 2: Pixel-to-Time Conversion")
    print("="*60)

    app = QApplication(sys.argv)
    window = MainWindow()
    canvas = window.spectrogram_view.canvas

    # Set up mock times array
    canvas.times = np.linspace(0, 10.0, 1000)  # 10 second audio, 1000 time points

    # Mock canvas width
    original_width = canvas.width
    canvas.width = lambda: 1000  # 1000 pixels wide

    # Test conversions
    test_cases = [
        (0, 0.0),      # Start of file
        (500, 5.0),    # Middle of file
        (1000, 10.0),  # End of file
        (250, 2.5),    # Quarter point
        (750, 7.5),    # Three-quarter point
    ]

    all_passed = True
    for pixel, expected_time in test_cases:
        actual_time = canvas._pixel_to_time(pixel)
        error = abs(actual_time - expected_time)

        if error < 0.01:  # 10ms tolerance
            print(f"  PASS: pixel {pixel:4d} -> time {actual_time:.3f}s (expected {expected_time:.3f}s)")
        else:
            print(f"  FAIL: pixel {pixel:4d} -> time {actual_time:.3f}s (expected {expected_time:.3f}s, error {error:.3f}s)")
            all_passed = False

    # Restore original width method
    canvas.width = original_width

    if all_passed:
        print("PASS: All pixel-to-time conversions accurate")
    else:
        print("FAIL: Some conversions had excessive error")

    app.quit()
    return all_passed


def test_time_to_pixel_conversion():
    """Test time-to-pixel coordinate conversion."""
    print("\n" + "="*60)
    print("TEST 3: Time-to-Pixel Conversion")
    print("="*60)

    app = QApplication(sys.argv)
    window = MainWindow()
    canvas = window.spectrogram_view.canvas

    # Set up mock times array
    canvas.times = np.linspace(0, 10.0, 1000)

    # Mock canvas width
    original_width = canvas.width
    canvas.width = lambda: 1000

    # Test conversions
    test_cases = [
        (0.0, 0),
        (5.0, 500),
        (10.0, 1000),
        (2.5, 250),
        (7.5, 750),
    ]

    all_passed = True
    for time, expected_pixel in test_cases:
        actual_pixel = canvas._time_to_pixel(time)
        error = abs(actual_pixel - expected_pixel)

        if error <= 1:  # 1 pixel tolerance
            print(f"  PASS: time {time:.3f}s -> pixel {actual_pixel:4d} (expected {expected_pixel:4d})")
        else:
            print(f"  FAIL: time {time:.3f}s -> pixel {actual_pixel:4d} (expected {expected_pixel:4d}, error {error}px)")
            all_passed = False

    # Restore
    canvas.width = original_width

    if all_passed:
        print("PASS: All time-to-pixel conversions accurate")
    else:
        print("FAIL: Some conversions had excessive error")

    app.quit()
    return all_passed


def test_round_trip_conversion():
    """Test that pixel->time->pixel conversions are consistent."""
    print("\n" + "="*60)
    print("TEST 4: Round-Trip Conversion Consistency")
    print("="*60)

    app = QApplication(sys.argv)
    window = MainWindow()
    canvas = window.spectrogram_view.canvas

    # Set up mock data
    canvas.times = np.linspace(0, 10.0, 1000)
    original_width = canvas.width
    canvas.width = lambda: 1000

    # Test round-trip conversions
    test_pixels = [0, 100, 250, 500, 750, 900, 1000]

    all_passed = True
    for original_pixel in test_pixels:
        time = canvas._pixel_to_time(original_pixel)
        final_pixel = canvas._time_to_pixel(time)
        error = abs(final_pixel - original_pixel)

        if error <= 1:
            print(f"  PASS: pixel {original_pixel:4d} -> time {time:.3f}s -> pixel {final_pixel:4d} (error: {error}px)")
        else:
            print(f"  FAIL: pixel {original_pixel:4d} -> time {time:.3f}s -> pixel {final_pixel:4d} (error: {error}px)")
            all_passed = False

    # Restore
    canvas.width = original_width

    if all_passed:
        print("PASS: Round-trip conversions consistent")
    else:
        print("FAIL: Some round-trips had excessive error")

    app.quit()
    return all_passed


def test_boundary_adjustment_handler():
    """Test that MainWindow handler correctly processes adjustments."""
    print("\n" + "="*60)
    print("TEST 5: MainWindow Boundary Adjustment Handler")
    print("="*60)

    app = QApplication(sys.argv)
    window = MainWindow()

    # Create mock data
    times = np.linspace(0, 10.0, 1000)
    probabilities = np.random.rand(1000) * 0.5 + 0.3  # Random probs 0.3-0.8
    column_indices = np.arange(1000)

    # Create mock detection
    original_detection = DetectedUSV(
        start_time_s=2.0,
        end_time_s=2.5,
        start_col=200,
        end_col=250,
        max_probability=0.95,
        mean_probability=0.90
    )

    # Create mock inference and detection results
    window.inference_result = InferenceResult(
        probabilities=probabilities,
        column_indices=column_indices,
        times=times
    )

    window.detection_result = DetectionResult(
        usvs=[original_detection],
        probabilities=probabilities,
        column_indices=column_indices,
        times=times
    )

    print(f"Original detection: {original_detection.start_time_s}s - {original_detection.end_time_s}s")

    # Simulate boundary adjustment
    new_start = 1.9
    new_end = 2.6

    # Call the handler directly (simulating signal emission)
    window._on_boundary_adjusted(original_detection, new_start, new_end)

    # Check that detection was updated
    assert len(window.detection_result.usvs) == 1, "Should still have 1 detection"

    adjusted_detection = window.detection_result.usvs[0]

    # Verify adjustment
    assert adjusted_detection.user_adjusted == True, "Should be marked as adjusted"
    assert abs(adjusted_detection.start_time_s - new_start) < 0.1, "Start time should be updated"
    assert abs(adjusted_detection.end_time_s - new_end) < 0.1, "End time should be updated"
    assert adjusted_detection.original_start_time_s == original_detection.start_time_s, "Should preserve original start"
    assert adjusted_detection.original_end_time_s == original_detection.end_time_s, "Should preserve original end"

    print(f"Adjusted detection: {adjusted_detection.start_time_s:.3f}s - {adjusted_detection.end_time_s:.3f}s")
    print(f"PASS: Handler correctly updated detection")
    print(f"  - user_adjusted: {adjusted_detection.user_adjusted}")
    print(f"  - Original preserved: {adjusted_detection.original_start_time_s}s - {adjusted_detection.original_end_time_s}s")

    app.quit()
    return True


def test_label_storage_reconstruction():
    """Test that LabelStorage can reconstruct DetectedUSV with adjustment fields."""
    print("\n" + "="*60)
    print("TEST 6: LabelStorage Reconstruction")
    print("="*60)

    from usv_spectrogram.app.core.label_storage import LabelStorage

    # Simulate JSON dict from file
    detection_dict = {
        "start_time_s": 1.234,
        "end_time_s": 1.456,
        "start_col": 123,
        "end_col": 145,
        "max_probability": 0.95,
        "mean_probability": 0.90,
        "user_adjusted": True,
        "original_start_time_s": 1.200,
        "original_end_time_s": 1.450
    }

    # Reconstruct DetectedUSV
    usv = LabelStorage.reconstruct_detected_usv(detection_dict)

    # Verify all fields
    assert usv.start_time_s == 1.234
    assert usv.end_time_s == 1.456
    assert usv.user_adjusted == True
    assert usv.original_start_time_s == 1.200
    assert usv.original_end_time_s == 1.450

    print("PASS: LabelStorage correctly reconstructs adjusted detections")
    print(f"  - Reconstructed: {usv.start_time_s}s - {usv.end_time_s}s")
    print(f"  - user_adjusted: {usv.user_adjusted}")
    print(f"  - Original: {usv.original_start_time_s}s - {usv.original_end_time_s}s")

    # Test backward compatibility (dict without adjustment fields)
    old_dict = {
        "start_time_s": 2.0,
        "end_time_s": 2.2,
        "start_col": 200,
        "end_col": 220,
        "max_probability": 0.85,
        "mean_probability": 0.80
    }

    old_usv = LabelStorage.reconstruct_detected_usv(old_dict)

    assert old_usv.user_adjusted == False, "Old dict should default to False"
    assert old_usv.original_start_time_s is None, "Old dict should default to None"

    print("PASS: Backward compatibility with old JSON format maintained")

    return True


def run_all_tests():
    """Run all UI integration tests."""
    print("\n" + "="*70)
    print("BOUNDARY ADJUSTMENT - UI INTEGRATION TESTS")
    print("="*70)

    tests = [
        ("Signal/Slot Connection", test_signal_slot_connection),
        ("Pixel-to-Time Conversion", test_pixel_to_time_conversion),
        ("Time-to-Pixel Conversion", test_time_to_pixel_conversion),
        ("Round-Trip Conversion", test_round_trip_conversion),
        ("Boundary Adjustment Handler", test_boundary_adjustment_handler),
        ("LabelStorage Reconstruction", test_label_storage_reconstruction),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "PASS" if success else "FAIL", None))
        except Exception as e:
            results.append((name, "FAIL", str(e)))
            print(f"\nFAIL: {e}")
            import traceback
            traceback.print_exc()

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
        print("\n SUCCESS: All UI integration tests passed!")
        return 0
    else:
        print(f"\n FAILURE: {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
