# Multi-Syllable USV Splitting Strategy

**Date:** 2026-01-24
**Objective:** Modify detection algorithm to ensure only single USVs per spectrogram

---

## Current Detection Algorithm Behavior

### Two-Stage Merging Process

The detection algorithm currently merges segments in TWO stages:

**Stage 1: Simple Gap-Based Merge (`merge_gap_ms`)**
- **Location:** `energy_detector.py:242-261` in `_merge_segments()`
- **Default:** 10.0 ms
- **Logic:** If two segments are < 10ms apart, merge them into one
- **Effect:** Combines syllables separated by <10ms gaps

**Stage 2: Continuity-Based Merge (`segment_continuity_enabled`)**
- **Location:** `energy_detector.py:478-569` in `_extend_segments_by_continuity()`
- **Default:** Enabled (True)
- **Max gap:** 20.0 ms (`segment_continuity_max_gap_ms`)
- **Logic:** Bridges gaps up to 20ms if:
  - Peak frequency within ±3000 Hz
  - Peak energy within ±20 dB
  - Gap frames match reference frequency/energy
- **Effect:** Combines syllables separated by <20ms if frequency/energy are similar

### Detection Pipeline

```
1. Threshold to get active frames
2. Group adjacent frames into segments
3. Merge segments (gap < 10ms)              ← CREATES MULTI-SYLLABLE
4. Extend/merge by continuity (gap < 20ms)  ← CREATES MULTI-SYLLABLE
5. Filter by duration (10-500ms)
6. Create candidate objects
```

---

## How Multi-Syllable USVs Are Created

**Example:** Two-syllable call with 15ms gap
- Syllable 1: 30ms duration at 65 kHz
- Gap: 15ms
- Syllable 2: 40ms duration at 68 kHz (similar frequency)

**Current behavior:**
1. Both syllables detected as separate segments
2. Simple merge (Stage 1): Gap 15ms > 10ms → NOT merged
3. Continuity merge (Stage 2): Gap 15ms < 20ms, frequencies similar (3 kHz apart) → MERGED
4. Result: One 85ms candidate containing both syllables

**Confirmed by training data analysis:**
- 29 USVs > 120ms in training set
- Visual inspection shows many contain 2+ distinct syllables
- These were created by the continuity-based merge

---

## Proposed Solution: Aggressive Gap Splitting

### Option A: Disable Continuity Merging (Recommended)

**Configuration changes:**
```python
segment_continuity_enabled = False  # Currently True
merge_gap_ms = 5.0                  # Currently 10.0
```

**Effect:**
- Eliminates 20ms continuity-based merging (main culprit)
- Reduces simple merge gap from 10ms → 5ms
- Only merges frames that are almost adjacent

**Trade-offs:**
- ✅ Prevents multi-syllable calls (>5ms gaps)
- ✅ Simplifies detection logic
- ⚠️ May split single USVs with brief energy dips
- ⚠️ Increases total candidate count (more false positives)

### Option B: Very Small Merge Gap Only

**Configuration changes:**
```python
segment_continuity_enabled = False  # Currently True
merge_gap_ms = 2.0                  # Currently 10.0
```

**Effect:**
- Even more aggressive splitting
- Only merges frames separated by ≤2ms (almost adjacent)

**Trade-offs:**
- ✅ Maximizes splitting of multi-syllable calls
- ⚠️ Higher risk of splitting single USVs
- ⚠️ More candidates to review

### Option C: Zero Merge (Maximum Splitting)

**Configuration changes:**
```python
segment_continuity_enabled = False  # Currently True
merge_gap_ms = 0.0                  # Currently 10.0
```

**Effect:**
- NO merging at all
- Each continuous energy segment = one candidate
- Pure frame-grouping only

**Trade-offs:**
- ✅ Guarantees no multi-syllable merging
- ⚠️ May create many tiny fragments from single USVs
- ⚠️ Significantly more candidates

---

## Validation Strategy

### Step 1: Test on Known Multi-Syllable Examples

Re-run detection on the 29 longest USVs with different merge settings:

```python
# Current config (creates multi-syllable)
config_current = DetectionConfig()

# Proposed config (splits multi-syllable)
config_split = DetectionConfig(
    segment_continuity_enabled=False,
    merge_gap_ms=5.0
)
```

**Expected outcome:** Same audio segment produces 2-3 candidates instead of 1

### Step 2: Compare Candidate Counts

Run detection on all training recordings with both configs:

**Metrics to track:**
- Total candidates detected
- Candidates per recording
- Duration distribution (should shift toward shorter)
- Manual inspection of random sample

**Acceptance criteria:**
- Median duration < 50ms (currently 36ms)
- 95th percentile < 80ms (currently 120ms)
- No obvious single USVs split inappropriately

### Step 3: Visual Verification

Extract spectrograms for:
- 10 longest candidates from new detection
- 10 random candidates from 50-80ms range

**Check:**
- Do any still contain multiple syllables?
- Are single USVs being inappropriately split?

---

## Implementation Plan

### Phase 1: Create Test Script

**File:** `scripts/test_merge_gap_settings.py`

**Function:**
1. Load one of the 29 long USV recordings (e.g., `2024-09-30_11-20-56_0000033.wav`)
2. Run detection with current config
3. Run detection with `merge_gap_ms = [0.0, 2.0, 5.0, 10.0]` and `segment_continuity_enabled = [True, False]`
4. Compare results:
   - Number of candidates
   - Duration distribution
   - Specific candidate at known multi-syllable timestamp
5. Print comparison table

### Phase 2: Update Detection Config

**File:** `src/usv_spectrogram/detection/config.py`

**Changes:**
```python
# Line 57 - reduce from 10.0 to 5.0
merge_gap_ms: float = 5.0

# Line 61 - disable continuity merging
segment_continuity_enabled: bool = False
```

### Phase 3: Re-run Full Detection

**Script:** `scripts/run_detection.py`

**Steps:**
1. Re-run detection on ALL recordings with new config
2. Save new candidates to `candidates_v2.csv` (don't overwrite original)
3. Compare statistics:
   - Total candidates: old vs new
   - Duration distributions
   - Sample count per recording

### Phase 4: Re-extract Spectrograms

**Script:** `scripts/extract_spectrograms.py`

**Steps:**
1. Extract spectrograms for new candidates
2. Save to `spectrograms_training_v2/` (separate from original)
3. Extract noise samples if needed

### Phase 5: Verify Quality

**Manual review:**
1. Open labeling tool with new candidates
2. Review 50-100 random samples
3. Check for:
   - Multi-syllable calls still present? (BAD)
   - Single USVs inappropriately split? (BAD)
   - Clean single-syllable USVs? (GOOD)

---

## Expected Outcomes

### Quantitative Changes

**Before (current config):**
- 587 USV labels
- Median duration: 36ms
- 29 USVs > 120ms (4.9%)

**After (split config):**
- ~650-750 USV labels (estimated +10-25%)
- Median duration: ~30ms (shorter)
- USVs > 120ms: 0-5 (<1%)

**Noise samples:**
- May increase slightly due to more fragments
- Should still be filtered by human labeling

### Qualitative Changes

- Training set will be more uniformly single-syllable
- Model will learn features of individual USV syllables
- Later, can develop "multi-syllable sequence detector" to combine related USVs
- Cleaner dataset = better model generalization

---

## Downstream Implications

### 1. Labeling Workflow

**Impact:** Minimal
- Labelers will see more candidates (slightly longer session)
- But candidates will be cleaner (fewer ambiguous multi-syllable cases)

### 2. Model Training

**Impact:** Positive
- More uniform duration distribution
- Model learns "single USV syllable" features
- Should improve recall on typical short USVs
- Can handle multi-syllable by combining predictions later

### 3. Production Detection

**Impact:** Requires post-processing
- Detection will output individual syllables
- Need downstream logic to combine into multi-syllable sequences (if desired)
- E.g., "combine USVs within 20ms with similar frequency" as post-detection step

---

## Recommendation

**Start with Option A (Conservative Split):**

```python
DetectionConfig(
    segment_continuity_enabled=False,  # Disable 20ms continuity merge
    merge_gap_ms=5.0,                  # Reduce from 10ms to 5ms
    # ... other params unchanged
)
```

**Rationale:**
1. Eliminates main source of multi-syllable merging (continuity logic)
2. Still merges very short gaps (≤5ms) to avoid over-fragmentation
3. Testable and reversible
4. Aligns with "catch individual USVs first, combine later" philosophy

**Next steps:**
1. Create test script to validate on known examples
2. If successful, update default config
3. Re-run detection pipeline
4. Verify quality before full retraining

---

## Success Criteria

✅ **Mission accomplished if:**
1. No USVs > 120ms in new candidate set (or <1%)
2. Median duration remains 30-50ms (single syllables)
3. Visual inspection confirms no obvious multi-syllable patterns
4. No single USVs inappropriately fragmented

⚠️ **Red flags:**
1. Many single USVs split into 2-3 tiny fragments
2. Median duration < 20ms (over-splitting)
3. Candidate count increases by >50% (too many fragments)

---

## Experimental Results

**Test conducted:** 2026-01-24
**Sample:** 5 recordings containing the longest USVs (29 USVs > 120ms)

### Configuration Comparison

| Configuration | Total Candidates | Mean Duration | Median Duration | Max Duration | % > 120ms | % > 80ms |
|---------------|-----------------|---------------|----------------|--------------|-----------|----------|
| **Current** (merge=10ms, continuity=ON) | 40 | 86.1ms | 36.3ms | 446.7ms | **15.0%** | 27.5% |
| **Proposed A** (merge=5ms, continuity=OFF) | 51 | 31.6ms | 24.3ms | 191.6ms | **2.0%** | 3.9% |
| **Proposed B** (merge=2ms, continuity=OFF) | 33 | 28.4ms | 23.9ms | 125.9ms | **3.0%** | 3.0% |
| **Proposed C** (merge=0ms, continuity=OFF) | 22 | 21.1ms | 17.1ms | 65.7ms | **0.0%** | 0.0% |
| Continuity only (merge=0ms, continuity=ON) | 37 | 90.5ms | 36.7ms | 449.7ms | **16.2%** | 32.4% |

### Key Findings

1. **Continuity merging is the main culprit**
   - Disabling continuity (setting `segment_continuity_enabled=False`) is critical
   - "Continuity only" config still produces 16.2% multi-syllable USVs

2. **Proposed A is the sweet spot**
   - Reduces multi-syllable from 15% → 2% (87% reduction)
   - Candidate count increases by 27.5% (reasonable)
   - Median duration 24.3ms (healthy single-syllable range)

3. **Proposed C is too aggressive**
   - Median duration 17.1ms (too short - likely splitting single USVs)
   - Candidate count DECREASED (fragments fall below 10ms minimum)

### Specific Example: Longest USV (446.7ms)

**File:** `2024-09-30_11-20-56_0000033.wav` at 1033ms

**Current config** (merge=10ms, continuity=ON):
- 1 candidate: 1032.5-1479.3ms (446.7ms duration)
- Peak frequency: 66211 Hz

**Proposed A** (merge=5ms, continuity=OFF):
- 4 candidates:
  - 1032.5-1119.6ms (87.0ms) at 66211 Hz
  - 1187.4-1247.6ms (60.2ms) at 60938 Hz
  - 1315.4-1375.6ms (60.2ms) at 60938 Hz
  - 1443.4-1462.6ms (19.2ms) at 86133 Hz

**Result:** Successfully split 446ms multi-syllable call into 4 individual syllables!

---

## Final Recommendation

**Implement Proposed A:**

```python
DetectionConfig(
    segment_continuity_enabled=False,  # Disable 20ms continuity merge
    merge_gap_ms=5.0,                  # Reduce from 10ms to 5ms
)
```

**Justification:**
- ✅ Reduces multi-syllable USVs from 15% to 2% (near elimination)
- ✅ Maintains healthy single-syllable duration distribution (median 24ms)
- ✅ Reasonable increase in candidate count (+27.5%)
- ✅ Successfully splits known 446ms multi-syllable example into 4 syllables
- ✅ Avoids over-fragmentation (median not too short)

**Expected impact on full dataset:**
- Current: 587 USV labels, 29 > 120ms (4.9%)
- After: ~650-700 USV labels, <10 > 120ms (<1.5%)

---

## Continuity Tuning - Final Solution

**Critical User Insight (2026-01-24):**
Disabling continuity entirely would split single USVs at energy dips, creating **partial USVs** in the training set. This would teach the model incomplete patterns - worse than the multi-syllable problem.

**Example:** Single USV with frequency sweep from 40-100+ kHz has brief energy dips. Without continuity, this gets split into fragments instead of staying as one complete USV.

### The Real Problem

Need to distinguish between:
1. **Energy dips within a single USV** (< 5ms) → should merge ✓
2. **Gaps between distinct syllables** (> 5ms) → should NOT merge ✓

### Tuned Parameters (Grid Search Results)

Tested 135 parameter combinations on two test cases:
- Single USV with dips (`2024-09-30_11-18-17_0000001` at 777ms)
- Multi-syllable call (`2024-09-30_11-20-56_0000033` at 1033ms, 446ms total)

**Found 108 configurations with perfect score (2.0/2.0)!**

All perfect configurations shared: `max_gap_ms = 5.0`

**Recommended configuration (most conservative):**
```python
segment_continuity_enabled = True        # Keep enabled!
segment_continuity_max_gap_ms = 5.0      # Reduced from 20.0 - KEY PARAMETER
segment_continuity_freq_tolerance_hz = 1500.0  # Reduced from 3000.0
segment_continuity_energy_tolerance_db = 10.0   # Reduced from 20.0
merge_gap_ms = 3.0                       # Reduced from 10.0
```

### Validation Results

**Test Case 1: Single USV with dips**
- Current (max_gap=20ms): 1 detection (32.0ms) ✓
- Tuned (max_gap=5ms): 1 detection (32.9ms) ✓
- **Result:** Complete USV preserved!

**Test Case 2: Multi-syllable call (446ms)**
- Current (max_gap=20ms): 1 detection (446.7ms) ✗
- Tuned (max_gap=5ms): 4 detections ✓
  - Syllable 1: 87.0ms at 66211 Hz
  - Syllable 2: 91.7ms at 60938 Hz
  - Syllable 3: 91.7ms at 60938 Hz
  - Syllable 4: 51.2ms at 86133 Hz
- **Result:** Multi-syllable successfully split!

### The 5ms Threshold

**Why 5ms works:**
- Energy dips in continuous USVs: Typically < 5ms (brief amplitude modulation)
- Gaps between syllables: Typically > 5-10ms (silence or frequency transitions)
- This threshold naturally separates the two cases

**Expected impact on full dataset:**
- Preserves all single USVs (including complex frequency sweeps)
- Splits multi-syllable calls into individual syllables
- Training data bias: 68% short USVs → remains similar, but no 446ms outliers
- Multi-syllable USVs > 120ms: 4.9% → < 1%

---

**Status:** ✅ Tuning Complete, Configuration Updated
**Next:** Re-run detection pipeline with tuned parameters
