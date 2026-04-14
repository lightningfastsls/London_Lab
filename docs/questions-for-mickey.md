# Questions for Mickey

Open questions that need domain expertise or experimental context before we can resolve them in code.

---

## Q1: Bout threshold — should file boundaries be explicit bout breaks?

**Date:** 2026-04-12
**Status:** Open
**Blocks:** Final bout segmentation logic, re-run of A2 sequential analysis

### Context

The WAV files are trigger-based recordings (start on noise, stop ~2s after silence). This means cross-file gaps are not real inter-call intervals — the recorder wasn't listening during the gap. Currently, `detect_bouts()` treats the entire timeline as continuous and applies a single ICI threshold (0.6s) to segment bouts, which mixes two different things:

- **Within-file gaps** (82.5% of transitions): real vocalization timing. Median gap = 78ms.
- **Cross-file gaps** (17.5%): recording system artifacts. Median gap = 15s, but 90 are under 1s.

### What the data shows

We fit a Gaussian mixture model on within-file gap-based ICIs only:

- The distribution is nearly unimodal — dominant peak at 74ms (78% weight)
- A small tail component centered at 184ms (22% weight)
- Crossover between components: **0.143s**
- The sensitivity sweep is flat above ~0.25s (diminishing returns from tighter thresholds)

The original 0.6s threshold came from `3 × median(onset-to-onset ICI)` computed over ALL ICIs (within + cross-file, start-to-start instead of gap). It was wrong in derivation but mostly harmless in practice because file boundaries were silently doing most of the work.

### Proposed two-layer logic

```
Bout boundary if:  (different file)  OR  (same file AND gap > threshold)
```

With threshold around 0.25s (conservative) or 0.14s (data-driven crossover).

### Questions

1. **Should file boundaries always be treated as bout breaks?** The recorder stopping means ≥2s of silence (its timeout). Is that always a bout break in your definition, or could a "bout" span a brief recorder restart (90 cross-file gaps are under 1s)?

2. **What does "bout" mean for your analysis?** Is it:
   - (a) A continuous stream of calls with minimal pauses (→ tighter threshold, ~0.15s)
   - (b) A behaviorally meaningful episode that can include brief pauses (→ looser threshold, ~0.25-0.5s)
   - (c) Something defined by the recording trigger itself (→ file = bout, no within-file splitting)

3. **Is the 2s recorder timeout exact?** If the stop delay varies (e.g., 1-5s), that affects whether short cross-file gaps (<1s) are meaningful.

### Supporting artifacts

- `results/sequential_structure/bout_threshold_analysis.png` — mixture fit on all ICIs (before file-aware correction)
- `results/sequential_structure/bout_threshold_within_file.png` — mixture fit on within-file ICIs only
- `results/temporal_dynamics/ici_distribution.png` — original A1 ICI histogram

---

## Q2: Overlapping detections — real or artifact?

**Date:** 2026-04-12
**Status:** Open
**Blocks:** Data quality confidence for all downstream analyses

### Context

We found 10 cases where a detection starts *before* the previous detection ends (negative gap-based ICI). Two of these are within the same file, 8 are cross-file.

Can USVs genuinely overlap in a single-animal recording? If not, these are likely:
- Duplicate detections (CNN flagging the same call twice with slightly different windows)
- Edge artifacts from the detection pipeline's sliding window

Should we deduplicate or merge these, or are they expected?

---

## Q3: How strong does sequential structure need to be to matter?

**Date:** 2026-04-12
**Status:** Open
**Blocks:** Interpretation of A2 results for the progress report

### Context

The audit corrected two inflated statistics:

| Metric | Old (wrong) | New (corrected) |
|--------|-------------|-----------------|
| Self-transition enrichment | 1.80× over chance | 1.28× over independence |
| Significant idioms | 1,843 | 653 |

The entropy reduction from knowing the current call type is 3.7% — meaning the sequence is ~96% as random as it would be if calls were independent. Self-repetition is the dominant pattern (Flat→Flat, Down→Down, etc.), but the enrichment is modest (25.7% observed vs 20.0% expected under independence).

### Questions

1. **Is 1.28× self-repetition enrichment biologically meaningful for wild mice?** Lab studies with inbred strains might show stronger structure. Is there a threshold or literature reference for what's "interesting"?

2. **The 653 remaining idioms** — even after the shuffle fix, 360 have observed count = 1 (seen once in the entire dataset). Should we apply a minimum occurrence filter (e.g., observed ≥ 3 or ≥ 5) before interpreting them? The top idioms (same-type runs like Complex×5, Down×5) are robust, but the long tail is uncertain.

3. **Does the weak structure match your expectations?** The UMAP/HDBSCAN clustering already showed USVs form a continuum rather than discrete categories. If the categories themselves are fuzzy, weak sequential structure between categories is expected — you can't have strong A→B patterns when A and B aren't sharply defined.

---

## Q4: Is there more than one animal vocalizing in cage 5970?

**Date:** 2026-04-12
**Status:** Open
**Blocks:** Whether sequential analysis should account for caller identity

### Context

All analyses assume a single animal producing all 7,864 calls. If the cage has a pair (or the microphone picks up vocalizations from adjacent cages), then:
- Self-repetition could be partly explained by two animals with different type preferences alternating
- Transition structure would mix two independent sequences
- Bout segmentation might group calls from different individuals

Is cage 5970 a single animal, a pair, or a group? Does the recording setup isolate calls to one source?

---

## Q5: DeepSqueak cross-validation and merged-call detection issue

**Date:** 2026-04-12
**Status:** Informational + open question
**Blocks:** Decision on whether to adopt new FP filter for production

### What we found

We ran DeepSqueak's independent YOLO v2 mouse detector on 197 files from the 5970 dataset to cross-validate our CNN pipeline. Key results:

- **95.6% of CNN detections confirmed** by DeepSqueak (216/226 overlap)
- **87-91% of DeepSqueak detections confirmed** by CNN (depending on filter)
- Two completely independent neural networks largely agree on what's a USV

### The merged-call problem

Our pipeline frequently produces long detections that contain multiple USV calls merged into one event. This happens because:

1. During dense calling bouts, the CNN probability stays high between calls (above the hysteresis sustain threshold of 0.4)
2. Hysteresis merges the adjacent calls into one long event
3. The FP filter was trained on short single-call events (mean 82ms) and learned "long = noise"
4. Result: real USV bouts get rejected by the FP filter

**Example:** File `2024-09-30_12-33-18_0000567` — CNN probability is at ~1.0 across a dense bout, hysteresis correctly finds 10 events, but the FP filter removes 5 of them, including a 1-second merged event containing ~5 separate calls.

### What we did

Retrained the FP filter without `duration_windows` as a feature:

| Metric | Old filter | New filter (no duration) |
|--------|-----------|--------------------------|
| F2 (cross-validation) | 0.823 | **0.833** |
| DS-confirmed detections | 87.0% | **91.0%** |
| DS_ONLY (CNN missed) | 36 | **25** |
| CNN_ONLY unchanged | 10 | 10 |

The new filter (`fp_filter_no_duration.pkl`) recovers real USV bouts while still rejecting the same noise events.

### Questions

1. **Is it expected that USVs in dense bouts are often detected as one long event?** Our typical merged event spans 200-1000ms and contains 2-6 calls with ~50ms inter-call gaps. Should we try to split these into individual calls, or is the "bout-level detection" acceptable for your analysis?

2. **Should we adopt the new FP filter for production?** It performs strictly better in cross-validation (F2 0.833 vs 0.823) and recovers more real calls. The old filter is still at `fp_filter.pkl` as a fallback.

### Supporting artifacts

- `models/hard_neg_retrain/fp_filter_no_duration.pkl` — new filter (10 features, no duration)
- `models/hard_neg_retrain/fp_filter_no_duration.json` — training report
- `models/hard_neg_retrain/fp_filter.pkl` — old filter (unchanged, 11 features)
- `results/deepsqueak_independent/` — DeepSqueak independent detection results + comparison reports
- `results/validation_old_filter/` — batch results with old filter
- `results/validation_new_filter/` — batch results with new filter
- `scripts/deepsqueak_detect_independent.m` — MATLAB detection script
- `scripts/compare_detections.py` — Python comparison tool
