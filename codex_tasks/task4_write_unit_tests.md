# Codex Task 4: Write Unit Tests for Diagnostic Functions

## Goal
Create pytest tests for the metric computation functions in diagnostic scripts.

## Test File to Create
- `tests/test_diagnostic_metrics.py`

## Functions to Test

### From `scripts/threshold_sweep.py`
**Function:** `compute_metrics_at_threshold(y_true, y_proba, threshold)`

**Test cases:**
1. Perfect predictions (all correct)
2. All wrong predictions
3. Edge case: all predicted as positive
4. Edge case: all predicted as negative
5. Typical case with mixed predictions
6. Different thresholds on same data

### From `scripts/analyze_recording_performance.py`
**Function:** `compute_spectrogram_stats(spec_path)`

**Test cases:**
1. Valid spectrogram image
2. Non-existent file (should handle gracefully)
3. Corrupt image file
4. All-white image (max=255, uniform)
5. All-black image (min=0, uniform)

## Test Structure

```python
import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import functions to test
# (You may need to refactor scripts to make functions importable)


class TestThresholdMetrics:
    """Tests for compute_metrics_at_threshold function."""

    def test_perfect_predictions(self):
        """Test with all correct predictions."""
        y_true = np.array([1, 1, 0, 0])
        y_proba = np.array([0.9, 0.8, 0.1, 0.2])
        threshold = 0.5

        metrics = compute_metrics_at_threshold(y_true, y_proba, threshold)

        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0

    # ... more test methods


class TestSpectrogramStats:
    """Tests for compute_spectrogram_stats function."""

    @pytest.fixture
    def temp_spectrogram(self, tmp_path):
        """Create a temporary spectrogram for testing."""
        from PIL import Image
        img = Image.new('L', (256, 256), color=128)
        path = tmp_path / "test_spec.png"
        img.save(path)
        return path

    def test_valid_spectrogram(self, temp_spectrogram):
        """Test stats computation on valid spectrogram."""
        stats = compute_spectrogram_stats(temp_spectrogram)

        assert 'pixel_mean' in stats
        assert 'pixel_std' in stats
        assert stats['pixel_mean'] == pytest.approx(128.0, abs=1.0)

    # ... more test methods
```

## Notes
- Use `pytest.approx()` for floating-point comparisons
- Use `tmp_path` fixture for temporary test files
- Test both happy paths and error cases
- Each test should be independent (no shared state)
- Use descriptive test names (test_<what>_<condition>_<expected_result>)

## Refactoring Note
You may need to extract functions from scripts into importable modules first, since the scripts are currently monolithic with argparse.

**Option 1:** Keep functions in scripts, import them for testing
**Option 2:** Create `src/usv_spectrogram/diagnostic_utils.py` and move functions there

Use whichever approach is simpler.

## Running Tests
```bash
".venv/Scripts/python.exe" -m pytest tests/test_diagnostic_metrics.py -v
```

## Estimated Time
45-60 minutes
