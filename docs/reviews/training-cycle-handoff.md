# Phase 10.1: Training Cycle Runner — Handoff

**Date:** 2026-02-21
**Review Tier:** 2 (Standard Implementation Review)

## Summary

Implemented the Active Learning Cycle Runner — an orchestration module that chains 7 steps (assemble, train, evaluate, optimize threshold, mine hard negatives, compare, report) into a single reproducible CLI command. This is the core tool for the scaling roadmap (2K -> 5K -> 10K -> 20K -> 30K label milestones).

## Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `src/usv_spectrogram/training/__init__.py` | 11 | Module init, exports CycleMetrics and generate_cycle_report |
| `src/usv_spectrogram/training/cycle_report.py` | 157 | CycleMetrics dataclass + markdown report generator |
| `scripts/run_training_cycle.py` | 427 | 7-step orchestration CLI script |
| `tests/test_training_cycle.py` | 260 | 27 tests covering metrics, reports, model configs, CLI parsing |

**No existing files were modified.**

## Architecture Decisions

### Integration strategy
- **Direct imports** for library modules: DatasetAssembler, USVClassifierCNN, Trainer, evaluate, create_data_loaders
- **sys.path import** for optimize_threshold.py functions (get_predictions, compute_metrics_at_threshold) — pure functions, clean to import
- **Subprocess** for mine_hard_negatives.py — uses argparse/sys.exit/tqdm, not cleanly importable

### Model loading
- Custom `load_model_with_architecture()` instead of `evaluate.load_model_checkpoint()` — the latter calls `model_class()` with no args, hardcoding default [32,64,128] filters. For medium/large models we need explicit architecture params.

### Threshold optimization
- Runs on **validation set** (not test set) to avoid information leakage — the test set metrics in Step 3 remain unbiased.

### Error handling
- Each step is wrapped in try/except. On failure: log error, set `failed_step`, skip to Step 7 (report) with partial metrics, return exit code 1.

## Test Coverage

| Test Class | Tests | What |
|------------|-------|------|
| TestCycleMetrics | 6 | Defaults, incremental population, JSON roundtrip, field isolation |
| TestCycleReport | 6 | Section presence, data/training/threshold values, mining count |
| TestCycleReportComparison | 2 | Comparison section with positive and negative deltas |
| TestCycleReportMiningSkipped | 2 | Skip message, artifact list without hard_negatives/ |
| TestCycleReportFailure | 1 | Failure banner with step name and completed steps |
| TestModelSizeConfigs | 5 | All sizes instantiate, param count ranges, ordering |
| TestParseArgs | 5 | Required args, defaults, all optional, invalid model size, missing required |

**Total: 27 tests, all passing. Full suite: 461 passed, 0 regressions.**

## Verification

- [x] `py_compile` on all 3 source files
- [x] `pytest tests/test_training_cycle.py -v` — 27 passed
- [x] `pytest tests/ -q` — 461 passed, 0 failures

## Usage

```bash
# Full cycle
python scripts/run_training_cycle.py \
    --labels-dir data/labeled_detections \
    --wav-dir "5970 USV" \
    --cycle-name milestone_1 \
    --output-dir runs/milestone_1

# Quick smoke test
python scripts/run_training_cycle.py \
    --labels-dir data/labeled_detections \
    --wav-dir "5970 USV" \
    --cycle-name smoke_test \
    --output-dir runs/smoke_test \
    --model-size small --epochs 2 --skip-mining

# With model comparison
python scripts/run_training_cycle.py \
    --labels-dir data/labeled_detections \
    --wav-dir "5970 USV" \
    --cycle-name milestone_2 \
    --output-dir runs/milestone_2 \
    --previous-model runs/milestone_1/model/best_model.pt
```

## Known Limitations

1. Label count is approximated as `total_positives // (1 + jitter_n_samples)` — some USVs may not produce all jitter copies
2. Hard negative mining runs as subprocess — errors are logged but don't propagate CycleMetrics details
3. No resume from partial failure — must re-run entire cycle
