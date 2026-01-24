# Codex Task 3: Add Type Hints to Diagnostic Scripts

## Goal
Add complete type annotations to all diagnostic scripts created in Session 9.

## Files to Annotate
1. `scripts/threshold_sweep.py`
2. `scripts/compare_probability_distributions.py`
3. `scripts/analyze_recording_performance.py`
4. `scripts/extract_visual_samples.py`

## Requirements

### Coverage
- All function signatures (parameters and return types)
- All class methods if any
- Use `from __future__ import annotations` for forward references
- Import types from `typing` (Dict, List, Tuple, Optional, Union)
- Import `Path` from `pathlib`

### Style Guide
- Follow existing type hint conventions in the codebase
- Use `Optional[X]` for potentially None values
- Use `Union[str, Path]` for flexible path inputs
- Return types: be explicit even for `None` (use `-> None`)
- Use `pd.DataFrame`, `np.ndarray`, `torch.Tensor` for library types

### Example
**Before:**
```python
def compute_metrics_at_threshold(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    # ...
    return metrics
```

**After:**
```python
def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float
) -> Dict[str, Union[float, int]]:
    y_pred = (y_proba >= threshold).astype(int)
    # ...
    return metrics
```

## Testing
After adding type hints, verify with:
```bash
python -m py_compile scripts/threshold_sweep.py
python -m py_compile scripts/compare_probability_distributions.py
python -m py_compile scripts/analyze_recording_performance.py
python -m py_compile scripts/extract_visual_samples.py
```

Optional (if mypy is installed):
```bash
mypy scripts/threshold_sweep.py --ignore-missing-imports
```

## Notes
- Don't change any logic, only add annotations
- If uncertain about a type, use `Any` from `typing` as fallback
- Pandas and numpy types may require `# type: ignore` comments if mypy complains

## Estimated Time
20-30 minutes
