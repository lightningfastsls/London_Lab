# Event Scoring (Post-Processing)

**Module:** `src/usv_spectrogram/postprocessing/event_scoring.py`
**Package:** `src/usv_spectrogram/postprocessing/`
**Tests:** `tests/test_event_scoring.py` (16 tests)

## Purpose

Pure-function event-level evaluation for USV detection. Matches detected USVEvents against ground-truth intervals using collar-based tolerance, then computes F-beta scores. No I/O, no model loading.

## API

### `EventScoringConfig` (frozen dataclass)

| Field | Default | Description |
|-------|---------|-------------|
| `onset_collar_s` | 0.200 | Tolerance window (seconds) for onset/offset matching |
| `min_iou` | 0.0 | Reserved for future IoU-based matching; not used by collar matching |

### `match_events_collar(detected, ground_truth, collar_s) -> (TP, FP, FN)`

Matches detected `USVEvent` objects to ground-truth `(start_s, end_s)` tuples.

**Match criteria** (any one sufficient):
- `|det.onset - gt.onset| <= collar_s`
- `|det.offset - gt.offset| <= collar_s`
- Any temporal overlap > 0

**Algorithm:** Greedy best-overlap-first. Each detection and ground-truth event matched at most once.

### `compute_f_beta(tp, fp, fn, beta=2.0) -> float`

Standard F-beta score: `(1 + beta^2) * TP / ((1 + beta^2) * TP + beta^2 * FN + FP)`

Returns 0.0 when TP=0.

## Design Decisions

- **Collar over IoU:** USV boundaries are inherently uncertain (annotators disagree by ~100-200ms). Collar tolerance is standard in bioacoustic evaluation (sed_eval, DCASE).
- **Greedy over Hungarian:** Simpler, O(n·m·log(n·m)) vs O(n³). Produces identical results for non-overlapping USVs.
- **F2 default:** Recall weighted ~4x over precision. Missing a real USV is worse than a false alarm for a screening tool.

## Used By

- `scripts/optimize_hysteresis.py` — Cross-validated grid search over hysteresis parameters
- Future: batch pipeline evaluation, model comparison scripts
