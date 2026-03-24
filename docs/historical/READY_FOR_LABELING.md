# Dataset Ready for Labeling

**Date:** 2026-01-24
**Status:** ✅ Detection and extraction complete

---

## Final Detection Configuration

After extensive tuning and testing, settled on **middle-ground parameters**:

```python
segment_continuity_max_gap_ms = 5.0         # Prevents multi-syllable merging
segment_continuity_freq_tolerance_hz = 1500.0  # Tight frequency matching
segment_continuity_energy_tolerance_db = 15.0  # Captures quiet tails without over-merging
merge_gap_ms = 3.0                          # Conservative simple merging
```

**Rationale:**
- 5ms gap threshold: Splits multi-syllable calls while preserving single USVs with energy dips
- 15dB energy tolerance: Middle ground between capturing complete USVs (including quiet tails) and preventing extreme merging
- User validation: "This is better and I am not sure we can do better than that"

---

## Detection Results

**Total candidates:** 729 (from 50 WAV files)

**Quality metrics:**
- Max duration: 191.6ms (down from 448.9ms) ✅
- Duration range: 10.2 - 191.6ms
- Frequency range: 25.2 - 99.6 kHz
- Interference flagged: 27 candidates

**Compared to original (untuned) detection:**
- Original: 697 candidates, 36 long USVs (>120ms), max 448.9ms
- Final: 729 candidates, ~2-5 long USVs (>120ms), max 191.6ms
- Multi-syllable reduction: 94% ✅

**Estimated quality (based on sample review):**
- ~90-92% complete single USVs
- ~5-8% partial USVs (incomplete boundaries)
- ~2-3% other issues (noise, artifacts, edge cases)

---

## Files Ready

**Candidates:**
- `candidates_final.csv` - Detection results with boundaries
- `candidates_final_extracted.csv` - With extraction metadata

**Spectrograms:**
- `spectrograms_labeling/` - 729 PNG images ready for review
- Mode: "review" (with axes, labels, colorbar for human evaluation)

**Labeling infrastructure:**
- `labels.csv` - Will be created/updated by labeling app
- Labeling app: `scripts/usv_labeling_tool.py`

---

## Labeling Strategy

### For Complete USVs (90-92% of dataset)
**Action:** Label as "USV" ✅
- Clear frequency sweep
- Complete start and end visible
- Single continuous call

### For Partial USVs (5-8% of dataset)
**Action:** Use boundary expansion feature (to be added)
- Extend detection boundaries to capture full USV
- Label as "USV" after adjustment
- Adjusted boundaries saved to labels

**Note:** Boundary expansion feature will be added to labeling app (similar to trim feature in noise review app but opposite direction).

### For Noise/Artifacts
**Action:** Label as "Not USV" ✗
- No USV pattern visible
- Broadband noise
- Artifacts, interference

### For Uncertain Cases
**Action:** Label as "Uncertain" ⚠️
- Use sparingly
- When truly ambiguous
- Can review later

---

## Expected Labeling Time

**Estimated:** 729 candidates × 5-10 seconds/candidate = 1-2 hours total
- Faster for obvious USVs (2-3 sec)
- Slower for partials requiring boundary adjustment (10-15 sec)
- Breaks recommended every 100-150 candidates

**Progress tracking:** Labeling app shows % complete

---

## Starting the Labeling App

```powershell
.\.venv\Scripts\python.exe scripts/usv_labeling_tool.py
```

**On first run:**
- Point to `candidates_final_extracted.csv`
- Point to `spectrograms_labeling/`
- Creates `labels.csv` automatically

**Keyboard shortcuts:**
- `1` = USV
- `2` = Not USV
- `3` = Uncertain
- Arrow keys = Navigate

---

## Next Steps After Labeling

Once labeling is complete:

1. **Dataset split** - Stratified train/val/test split by recording
2. **Training mode extraction** - Re-extract spectrograms without axes for CNN training
3. **Model training** - Train CNN classifier
4. **Evaluation** - Test on held-out test set

---

## Known Issues & Mitigations

### Partial USVs (~5-8%)

**Current:** Small fraction of detections are incomplete (missing quiet tails or cut off mid-call)

**Mitigation:** Boundary expansion feature in labeling app (to be implemented)
- Allows extending start/end times during labeling
- Fixes partial → complete USV
- No need to reject or skip these candidates

### Interference (27 flagged)

**What:** Candidates near 60kHz, 100kHz (electrical interference frequencies)

**Mitigation:** Manual review during labeling
- Most are legitimate USVs near those frequencies
- Some may be true interference → label as "Not USV"

---

## Quality Assurance

**Sample quality check (4 priority recordings, 50 candidates):**
- User review: "This is better"
- ~92% acceptable quality
- Partial USVs identified but fixable with boundary expansion

**Confidence:** High - ready for production labeling

---

**Status:** ✅ Ready for labeling
**Blocker:** Add boundary expansion feature to labeling app (recommended before starting)
**Estimated completion:** Can start labeling immediately if boundary feature is not critical, or wait ~30-60 min for feature implementation
