# Module: False Positive Filter (15.5)

## Purpose

Second-stage logistic regression classifier that filters false positive USV detections. Takes `EventFeatures` from hysteresis-detected events and predicts whether each event is a real USV or a spectral artifact.

## Public Interface

```python
from usv_spectrogram.postprocessing.fp_filter import FalsePositiveFilter

filt = FalsePositiveFilter()
filt.fit(features: List[EventFeatures], labels: List[bool])
predictions: List[bool] = filt.predict(features)
probabilities: np.ndarray = filt.predict_proba(features)  # shape (n, 2)
importances: dict[str, float] = filt.feature_importances()
filt.save(Path("model.pkl"))
loaded = FalsePositiveFilter.load(Path("model.pkl"))
```

## Pipeline Position

```
CNN probabilities → HysteresisDetector → EventFeatures → [FP Filter] → filtered events
```

## Key Decisions

- **LogisticRegression + StandardScaler**: Interpretable coefficients, calibrated probabilities, minimal overfitting on ~hundreds of events. Upgrade to LightGBM only if >1000 labeled events and logistic regression underfits.
- **class_weight='balanced'**: Handles typical imbalance where FPs outnumber TPs.
- **Constant-label fallback**: Degenerate single-class training data produces a constant predictor instead of crashing.

## Training

```bash
python scripts/train_fp_filter.py \
    --model models/matched_windows/best_model.pt \
    --labels data/unified_labels.json \
    --hysteresis-config results/hysteresis_optimization.json \
    --output models/matched_windows/fp_filter.pkl
```

## Integration Points

- **Input**: `List[EventFeatures]` from `event_features.py` (module 15.4)
- **Evaluation**: `match_events_collar()` from `event_scoring.py` (module 15.2) for labeling training events
- **Upstream**: `hysteresis_detect()` from `hysteresis.py` (module 15.1) produces the events
