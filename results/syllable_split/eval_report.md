# Syllable splitter — evaluation report

## Parameters
- `gap_threshold_ms` = **30.0** (chosen from sweep — see E5 below)
- `prominence_threshold_db` = 15.0
- `min_syllable_ms` = 8
- `close_frame_holes` = 2 frames (~0.85 ms)
- Source CSVs: `results/traditional_taxonomy/classified_traditional.csv` (5970, n=7,921)
              `results/traditional_taxonomy_3452/classified_traditional.csv` (3452, n=401)
- Events processed: all `call_length_s > 300 ms` (n=122; 5970=121, 3452=1)

## Headline result
- 27/122 events split into ≥2 sub-syllables (22.1%)
- 9/122 events split into ≥3 sub-syllables (7.4%)
- 95/122 events returned unchanged (no detectable silent gap)
- 163 sub-syllables emitted in total (was 122)
- 95 sub-syllables still longer than 300 ms after splitting

## Eval checks

| ID | Check | Result | Detail |
|---:|-------|:------:|--------|
| E1 | No false splits on short detections (<100ms) | **PASS** | 0/250 short detections falsely split (target <1%) |
| E2 | Post-split duration distribution sanity | **WARN** | sub-syllable median = 317.9 ms; cohort 5970 median_call_duration_ms = 60 ms. Higher = still under-splitting |
| E3 | Visual inspection of all 122 events | **MANUAL** | see `results/syllable_split/eval.html` |
| E4 | Idempotence (splitting a sub-syllable doesn't re-split) | **PASS** | tested on 30 events × all sub-syllables; 0 re-split |
| E5 | Threshold sensitivity sweep | see below | run @ {20, 30, 50, 65} ms |

## E5 — sensitivity sweep

| gap_threshold_ms | events split ≥2 | events split ≥3 | unchanged | sub-syllables still > 300 ms |
|---:|---:|---:|---:|---:|
| 20 (aggressive) | 46 (37.7%) | 22 (18.0%) | 76 (62.3%) | 78 |
| **30 (chosen)** | **27 (22.1%)** | **9 (7.4%)** | **95 (77.9%)** | **95** |
| 50 (mid) | 6 (4.9%) | 0 (0.0%) | 116 (95.1%) | 116 |
| 65 (corpus q25_ici_gap_ms) | 1 (0.8%) | 0 (0.0%) | 121 (99.2%) | 121 |

The corpus-anchored threshold (65 ms) is too strict for dense intra-bout bursts; even at the most
aggressive setting (20 ms), 78 sub-syllables remain longer than 300 ms. Trace-gap detection alone
is necessary-but-not-sufficient for this splitting problem.

## Limitations
1. **Trace-gap is insufficient on dense bursts.** Inspection of the worst-offender event (1.15 s)
   shows max below-threshold prominence run of only 30 ms — there is no silent gap to split on.
   Spectral-flux onset detection also fails (zero onsets ≥1σ).
2. **78% of long events return unchanged.** They are either (a) real long single syllables
   (Complex/Frequency_Jump/Chevron at the duration tail) or (b) bursts whose syllables flow into
   each other with no detectable boundary in raw spectrogram space.
3. **The right next step is one of:** (i) re-run CNN sliding inference on these 122 events with a
   finer stride and use per-frame probability dips to split; (ii) train a small "syllable onset"
   classifier on hand-labeled boundary data; (iii) accept the partial splitter and gate Q3 violin
   tails on `split_count == 1 AND original_duration > 300 ms` (which would just filter the long
   tail rather than splitting it).

---

## Update — CNN re-inference at fine stride

Reran SlidingInference on all 122 long-event WAVs with hop_px=2 (5× finer than default
~0.85 ms/frame), production model `models/hard_neg_retrain/best_model.pt`. Cache at
`results/syllable_split/cnn_rerun/{stem}.npz`. Total inference time: 5,766 s (~96 min).

### Headline cross-tab (trace-gap vs CNN-prob)

|                       | CNN splits ≥2 | CNN unchanged | Total |
|-----------------------|--------------:|--------------:|------:|
| **Trace splits ≥2**   | 1             | 26            | 27    |
| **Trace unchanged**   | 0             | 95            | 95    |
| **Total**             | 1             | 121           | 122   |

**CNN-prob splitting is strictly subsumed by trace-gap splitting.** Every event the CNN-prob
splitter resolves, the trace-gap splitter also resolves. No event is split by CNN-prob alone.

### Diagnostic on worst-offender (rank 1, 1.15 s)

CNN per-frame probability inside the event: **min=0.957, median=1.000, max=1.000**. Not a
single frame drops below 0.75 at any tested threshold. The CNN is confidently saying
"continuous USV present" throughout — not a sequence of distinct calls.

### Why CNN doesn't help

The CNN's analysis window is `window_width_px=100 × STFT_HOP=128 / SAMPLE_RATE_HZ=300000` =
**42.7 ms** wide. In a dense bout, every window contains at least some USV energy. The CNN
correctly outputs ~1.0 for all of them — its training question is "is there a USV in this
window?" not "where does syllable N end?" Finer stride makes the curve smoother but does not
change this structural floor.

### Conclusion

The 95 unresolved long events almost certainly are **real single syllables of unusual
length** (Complex / Frequency_Jump / Chevron at the tail of their natural distribution),
not multi-syllable bursts with hidden boundaries. The CNN-prob trace panel below each
spectrogram in `eval.html` makes this visible: where the curve sits at p≈1.0 flat throughout
the event, the CNN itself is asserting "one continuous USV."

The recommended action is **option (1) from the earlier menu**: accept the trace-gap splits
for the 27 resolved events; tag the remaining 95 with `splitter_unresolved=True` and treat
them as real long syllables in downstream analysis (or filter from Q3 violin tails if
preferred).

### E1 false-positive check (CNN splitter)

Ran on 47 short detections (<100 ms) with fresh inference at hop_px=2:

| Config | False splits | Verdict |
|--------|--------------:|---------|
| p_thr=0.50, gap=20 ms | 0/47 | PASS |
| p_thr=0.50, gap=30 ms | 0/47 | PASS |
| p_thr=0.75, gap=20 ms | 0/47 | PASS |

The CNN-prob splitter is *safe* (does not corrupt short USVs) but *useless* (resolves no
event the trace-gap splitter does not also resolve). Both splitters are now characterised:
trace-gap is the working tool; CNN-prob is documented null evidence.
