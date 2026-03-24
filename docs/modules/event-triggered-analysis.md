# Event-Triggered USV Rate Analysis (PETH)

## Purpose

Computes peri-event time histograms (PETHs) — the first cross-modal analysis
bridging LMT behavioral events with USV detections. For each behavioral event
type, bins USV count in a configurable window around event onset and normalizes
to rate (Hz). Statistical significance is assessed via circular-shift
permutation test.

## Key Concepts

**PETH**: Standard neuroscience analysis. Align many event instances at t=0,
bin vocalization count in a window around onset, divide by (n_events × bin_size)
to get rate. If USVs correlate with behavior, the PETH shows rate modulation.

**Circular-shift permutation**: Shifts all USV times by a random offset
(wrapping at recording boundaries), preserving temporal autocorrelation while
destroying cross-correlation with events. More principled than independent
shuffling.

**Bootstrap CI**: Resamples event indices with replacement to capture
between-event variability. 200 resamples, 2.5th/97.5th percentiles.

## Architecture

```
src/usv_spectrogram/lmt/event_triggered.py
├── PETHConfig (frozen dataclass)     # Window, bins, permutations
├── PETHResult (frozen dataclass)     # Rate, CI, p-value, peak
├── _count_peri_event()               # Inner loop: binary search + histogram
├── compute_peth()                    # Core: one event type
├── compute_all_peths()               # Multiple event types
├── compare_populations()             # Descriptive group comparison
├── plot_peth()                       # Single PETH plot
└── plot_all_peths()                  # Grid layout
```

## Dependencies

- **Imports from**: `usv_spectrogram.lmt.db_loader` (BehavioralEvent)
- **Computation**: numpy (searchsorted, histogram, percentile)
- **Plotting**: matplotlib (lazy import, not required for computation)
- **Detection CSV**: Reads `start_ms` column from batch detection output

## Configuration

| Field | Default | Description |
|-------|---------|-------------|
| `window_before_s` | 2.0 | Seconds before event onset |
| `window_after_s` | 2.0 | Seconds after event onset |
| `bin_size_s` | 0.1 | Bin width (100 ms) |
| `n_permutations` | 1000 | Circular-shift permutations |
| `min_events` | 5 | Minimum events to compute |
| `baseline_method` | "whole_recording" | Baseline rate method |

## CLI Usage

```bash
python scripts/run_event_triggered_analysis.py \
    --lmt-db experiment.sqlite \
    --detections usvs.csv \
    --output results/ \
    --event-types "Oral-oral Contact" "Rearing"
```

Outputs: `results.json`, `event_triggered_report.md`, `peth_all.png`,
`peth_{EventType}.png`.

## Design Decisions

1. **Tuples in PETHResult**: Frozen dataclass needs immutable fields; tuples are
   JSON-serializable unlike ndarray.
2. **Binary search per event**: O(E log U) via `np.searchsorted` — efficient
   for typical event/USV counts.
3. **Conservative p-value**: `(n_exceed + 1) / (n_perm + 1)` avoids zero.
4. **No boundary correction**: Edge events with partial windows average out.

## Test Coverage

23 tests in `tests/test_event_triggered.py` (9 config, 10 compute_peth,
2 compute_all_peths, 2 compare_populations):
- Config validation (defaults, all 5 invalid-value cases, n_bins property, frozen)
- Flat PETH from uniform ~1 Hz data (jittered to avoid aliasing)
- Peak detection from clustered post-event data
- Permutation test: correlated significant, uncorrelated not significant
- CI bounds contain point estimate, pre_event baseline method
- Edge cases: no USVs (zero rate), short recording (no crash), frozen result
- Multi-type: filters types below min_events, handles empty events_by_type
- Population comparison: expected keys, no-common-types returns empty dict
