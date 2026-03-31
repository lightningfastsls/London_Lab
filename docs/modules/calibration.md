# Temperature Scaling Calibration (Post-Processing)

**Module:** `src/usv_spectrogram/postprocessing/calibration.py`
**Package:** `src/usv_spectrogram/postprocessing/`
**Tests:** `tests/test_calibration.py` (9 tests)
**ROADMAP:** `ROADMAP_POST_PROCESSING.md` §15.3

## Purpose

Post-hoc calibration of CNN probability outputs using temperature scaling (Guo et al., 2017 ICML). Learns a single parameter T on the validation set that divides logits before sigmoid, improving calibration without changing ROC AUC (the transform is monotonic — it preserves rank ordering).

## Why Calibration Matters

A raw sigmoid output of 0.8 doesn't necessarily mean "80% chance this is a USV." CNNs are systematically overconfident. Temperature scaling makes thresholds more interpretable and stable across recordings, which is critical for the batch detection pipeline's triage system (§15.7).

## API

### `TemperatureScaler` (dataclass)

| Field | Default | Description |
|-------|---------|-------------|
| `temperature` | 1.0 | Scaling parameter (T > 1 softens, T < 1 sharpens) |
| `fitted` | False | Whether `fit()` has been called |
| `nll_before` | None | NLL at T=1 (before fitting) |
| `nll_after` | None | NLL at optimal T |

Validates: `temperature > 0` in `__post_init__`.

### `TemperatureScaler.fit(logits, labels) → float`

Fits T by minimizing binary NLL on validation data using L-BFGS-B with bounds [0.01, 50.0]. Returns optimal T.

### `TemperatureScaler.calibrate(logits) → np.ndarray`

Applies `sigmoid(logits / T)` to produce calibrated probabilities.

### `TemperatureScaler.save(path)` / `TemperatureScaler.load(path)`

JSON serialization. Output file: `models/matched_windows/temperature.json`.

### `compute_ece(probabilities, labels, n_bins=15) → float`

Expected Calibration Error — partitions predictions into equal-width bins, returns weighted average of |accuracy - confidence|. Lower is better.

## Integration Point

Calibration sits between `SlidingInference` and `hysteresis_detect` in the pipeline:

```
AudioLoader → SlidingInference(return_logits=True)
    → TemperatureScaler.calibrate(logits)
    → hysteresis_detect(calibrated_probs, ...)
```

The `InferenceResult` dataclass now has an optional `logits` field (default None for backward compatibility).

## CLI

```bash
python scripts/calibrate_temperature.py \
    --model models/matched_windows/best_model.pt \
    --val-csv data/training/matched_windows/val.csv \
    --output models/matched_windows/temperature.json
```

## NLL Formulation

Uses the numerically stable form to avoid `log(0)`:

```
NLL = mean(max(z, 0) + log(1 + exp(-|z|)) - y * z)
```

where `z = logits / T`. This is equivalent to `mean(-y*log(σ(z)) - (1-y)*log(1-σ(z)))` but safe for extreme logit values.
