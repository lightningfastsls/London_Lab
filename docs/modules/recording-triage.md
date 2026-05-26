# Recording-Level Triage

**Module:** `src/usv_spectrogram/postprocessing/triage.py`
**Package:** `usv_spectrogram.postprocessing`
**Tests:** `tests/test_triage.py`

## Purpose

Assigns each recording to a triage tier after detection, enabling batch processing of 25K+ recordings without manual review of every file:

- **auto_accept** (~60-70%): All detected events have high confidence
- **auto_reject** (~10-20%): No signal detected above noise threshold
- **manual_review** (~15-25%): Ambiguous recordings needing human judgment

## Public Interface

### TriageConfig (frozen dataclass)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `auto_accept_min_peak` | float | 0.90 | Min peak_probability for all events to auto-accept |
| `auto_reject_max_window` | float | 0.10 | Max window probability to auto-reject |
| `noise_floor_p90_threshold` | float | 0.4 | p90 threshold for noise QC flag |
| `outlier_count_zscore` | float | 2.0 | Z-score threshold for outlier event count |
| `max_event_duration_ms` | float | 600.0 | Flag any event longer than this |
| `total_duration_review_ms` | float | 600.0 | Flag recordings/chunks whose summed detected duration exceeds this |
| `high_event_count_threshold` | int | 10 | Flag recordings/chunks with more events than this |
| `max_event_fraction_of_recording` | float | 0.8 | Flag any event spanning at least this fraction of the probability timeline |

Validation: `auto_accept_min_peak > 0`, `auto_reject_max_window >= 0`, `reject < accept`.

### RecordingResult (dataclass)

| Field | Type | Description |
|-------|------|-------------|
| `filepath` | str | Source WAV path |
| `events` | List[USVEvent] | Detected events |
| `tier` | str | 'auto_accept', 'auto_reject', or 'manual_review' |
| `confidence_score` | float | Summary confidence (currently max_confidence) |
| `qc_flags` | List[str] | Machine-readable QC flags |
| `n_events` | int | Event count |
| `max_confidence` | float | Max peak_probability across events |
| `mean_event_confidence` | float | Mean of per-event peak_probability |
| `total_usv_duration_ms` | float | Sum of event durations |
| `noise_floor_p90` | float | 90th percentile of window probabilities |

### triage_recording()

```python
def triage_recording(
    filepath: str,
    events: List[USVEvent],
    probabilities: np.ndarray,
    config: TriageConfig = None,
    batch_stats: dict = None,
) -> RecordingResult
```

## Algorithm

1. Compute QC metrics (n_events, max/mean confidence, total duration)
2. Compute noise_floor_p90 = np.percentile(probabilities, 90)
3. Check QC flags:
   - Outlier event count (if batch_stats provided): z = (n - mean) / std > threshold
   - High noise floor: p90 > noise_floor_p90_threshold
   - High event count: n_events > high_event_count_threshold
   - Long event duration: any event > max_event_duration_ms
   - High total USV duration: sum(event.duration_ms) > total_duration_review_ms
   - Event spans most recording/chunk: event window count / probability count >= max_event_fraction_of_recording
4. Tier assignment (order matters):
   - **auto_reject**: max(probabilities) <= auto_reject_max_window
   - **manual_review**: events exist and a structural artifact flag was raised (`long_event_duration` or `event_spans_most_of_recording`)
   - **auto_accept**: events exist, no structural artifact flags, and all peak_probability >= auto_accept_min_peak
   - **manual_review**: fallback

## Integration Points

- **Input:** `USVEvent` from `hysteresis_detect()`, probabilities from `SlidingInference`
- **Output:** `RecordingResult` consumed by `write_batch_results()`
- **Batch script:** `scripts/run_batch_detection.py` orchestrates the full pipeline

## Usage

```python
from usv_spectrogram.postprocessing import triage_recording, TriageConfig

result = triage_recording(
    filepath="recording.wav",
    events=events,
    probabilities=probs,
    config=TriageConfig(auto_accept_min_peak=0.85),
    batch_stats={"event_count_mean": 5.0, "event_count_std": 2.0},
)
print(f"{result.tier}: {result.n_events} events, flags={result.qc_flags}")
```

## Batch Output

**Module:** `src/usv_spectrogram/postprocessing/batch_output.py`

`write_batch_results()` produces:
- `summary.parquet` — DataFrame with filepath, tier, event/confidence metrics, duration/noise metrics, and qc_flags
- `detections/<stem>.json` — list of ADR-010 dicts per recording

## Key Decisions

- Triage thresholds are initial defaults; recalibration expected after first real batch run
- confidence_score = mean_event_confidence (per ROADMAP Resolved Ambiguity #5)
- mean_event_confidence uses peak_probability (not mean_probability) per event
