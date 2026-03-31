# Implementation Handoff: Temperature Scaling Calibration

**Module:** Temperature Scaling (ROADMAP_POST_PROCESSING.md §15.3)
**Review Tier:** 2
**Date:** 2026-03-27
**Branch:** main

## What Changed

- Added `TemperatureScaler` dataclass with L-BFGS-B fitting, JSON save/load, and stable binary NLL
- Added `compute_ece()` function for Expected Calibration Error measurement
- Extended `InferenceResult` with optional `logits` field (backward compatible via default `None`)
- Added `return_logits` parameter to `SlidingInference.infer()` — when True, calls `model.forward()` + manual sigmoid instead of `model.predict_proba()`
- Created CLI script following `assemble_training_data.py` pattern

## Files Changed

- `src/usv_spectrogram/postprocessing/calibration.py` (NEW) — Core module: TemperatureScaler, compute_ece, _binary_nll
- `src/usv_spectrogram/postprocessing/__init__.py` (MODIFIED) — Added calibration exports
- `src/usv_spectrogram/app/core/sliding_inference.py` (MODIFIED) — Added `logits` field to InferenceResult, `return_logits` param to `infer()`
- `scripts/calibrate_temperature.py` (NEW) — CLI entry point for fitting T
- `tests/test_calibration.py` (NEW) — 9 tests
- `docs/modules/calibration.md` (NEW) — Module documentation

## Key Decisions Made

1. **Not frozen dataclass** — `TemperatureScaler` mutates during `fit()` (sets temperature, fitted, nll_before, nll_after), same pattern as `USVEvent`.
2. **Caller-side integration** — Calibration is applied by the caller between inference and hysteresis, not baked into SlidingInference. Keeps modules composable and testable independently.
3. **Energy-skipped windows get logits=0.0** — These map to p=0.5 after calibration, but it's safe because hysteresis thresholds filter them out. Simpler than using -inf.
4. **Stable NLL via log-sum-exp** — `max(z, 0) + log1p(exp(-|z|)) - y*z` avoids log(0) for extreme logits.
5. **L-BFGS-B bounds [0.01, 50.0]** — Wide enough for any reasonable model, prevents division by zero.

## What I'm Unsure About

- **ECE bin edge handling** — The first bin includes `prob == 0.0` via a special case. This is standard but worth verifying the bin boundary logic doesn't double-count.
- **`squeeze(dim=1)` on logits** — When `return_logits=True`, we call `model.forward().squeeze(dim=1)`. This matches `evaluate.py:56` but should be verified against the actual model output shape.

## Test Results

```
.venv/bin/python -m pytest tests/test_calibration.py tests/test_hysteresis.py tests/test_dataset_assembler.py -v
37 passed, 0 failed
```

Full suite: 8 pre-existing collection errors (unrelated `anthropic` import in notion_notes tests).

## ROADMAP Exit Criteria Status

- [ ] Fitted T is in reasonable range (0.5-3.0) — **Cannot verify yet** (requires running on real val data)
- [ ] Calibrated probabilities have lower ECE than raw — **Cannot verify yet** (same)
- [x] SlidingInference backward compatible (existing code that doesn't use logits works unchanged)
- [ ] Temperature parameter saved to `models/matched_windows/temperature.json` — **Ready to run** via CLI

## Test Coverage

| # | Scenario | Status |
|---|----------|--------|
| 1 | T=1 calibrate == raw sigmoid | PASS |
| 2 | T=3 softens (probs toward 0.5) | PASS |
| 3 | T=0.5 sharpens (probs away from 0.5) | PASS |
| 4 | fit() reduces NLL on synthetic data | PASS |
| 5 | JSON save/load roundtrip | PASS |
| 6 | Negative T raises ValueError | PASS |
| 7 | InferenceResult without logits → None | PASS |
| 8 | Perfect calibration → ECE ≈ 0 | PASS |
| 9 | Overconfident preds → high ECE | PASS |

## Docs Written/Updated

- `docs/modules/calibration.md` — Created
- `docs/reviews/calibration-handoff.md` — This file

## Next Steps

Run `scripts/calibrate_temperature.py` on the real validation set to verify fitted T is in expected range and ECE improves. Then append entry to `IMPLEMENTATION_PROGRESS.md`.
