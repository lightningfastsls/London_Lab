# Phase 0: Verification and Baseline - Test Plan

**Date:** 2026-02-05
**Status:** Ready for execution
**Objective:** Verify app save functions work correctly and establish baseline metrics

---

## Test Scenarios

### 1. Save Current View (Visible Detections Only)

**Purpose:** Verify that "Save Current View" only saves detections visible in viewport

**Steps:**
1. Launch detection app with a WAV file that produces 10+ detections
2. Run detection with moderate thresholds (high=0.06, low=0.04)
3. Scroll to show only 3-6 detections in viewport (not at start or end)
4. Click "Save Current View"
5. Verify confirmation dialog shows correct count
6. Confirm save
7. Check output directory

**Expected Results:**
- Only visible detections are saved (3-6 PNG/JSON/CSV files)
- Detections before/after viewport are NOT saved
- Each detection has: PNG (spectrogram), JSON (metadata), CSV entry
- `_saved_tracking.json` created with saved detection records

**Files to Check:**
```
~/USV_Detections/<wav_name>/
├── detection_000_XXXs-XXXs.png
├── detection_000_XXXs-XXXs.json
├── detection_001_XXXs-XXXs.png
├── detection_001_XXXs-XXXs.json
├── ...
├── detections_summary.csv
└── _saved_tracking.json
```

---

### 2. Save All Detections

**Purpose:** Verify "Save All Detections" saves every detection regardless of viewport

**Steps:**
1. Use same WAV from Test 1 (or new WAV with 10+ detections)
2. Clear any previous saves (delete output directory or use new WAV)
3. Scroll to middle of recording (so some detections out of view)
4. Click "Save All Detections"
5. Verify confirmation shows correct total count
6. Confirm save
7. Check output directory

**Expected Results:**
- ALL detections saved (full count, not just visible)
- PNG + JSON + CSV for each detection
- CSV summary has all detections listed
- Tracking file has all detection time ranges

---

### 3. Duplicate Prevention

**Purpose:** Verify tracker prevents saving same detection twice

**Steps:**
1. Load WAV file
2. Run detection with high=0.08, low=0.06
3. Save current view (save some detections)
4. Change thresholds to high=0.04, low=0.03 (more permissive)
5. Run detection again (should find more detections)
6. Try to save current view again (should include previously saved region)

**Expected Results:**
- Second save attempt shows "Already Saved" or filters out duplicates
- No duplicate files created (same time range not saved twice)
- Tracking file correctly identifies overlapping detections

**Edge Case:** If detection boundaries shift slightly due to threshold change, tracker should still recognize overlap (uses `_time_ranges_overlap` logic)

---

### 4. Metadata Verification

**Purpose:** Verify saved JSON contains all expected fields

**Steps:**
1. Save a few detections using either method
2. Open one of the JSON files
3. Verify schema matches expected structure

**Expected JSON Structure:**
```json
{
  "detection_index": 0,
  "core_time": {
    "start_s": 1.234,
    "end_s": 1.256,
    "duration_ms": 22.0
  },
  "saved_region": {
    "start_s": 1.214,
    "end_s": 1.276,
    "context_ms": 20.0
  },
  "probabilities": {
    "max": 0.123,
    "mean": 0.098
  },
  "spectrogram_columns": {
    "start_col": 123,
    "end_col": 145
  },
  "timestamp": "2026-02-05T14:30:00.123456"
}
```

**Verify:**
- All fields present
- `saved_region` includes context (±20ms by default)
- `core_time` matches actual detection boundaries (without context)
- Probabilities are reasonable (0.0-1.0 range)

---

### 5. CSV Summary Verification

**Purpose:** Verify CSV summary is well-formed and matches saved detections

**Steps:**
1. Save multiple detections
2. Open `detections_summary.csv`
3. Verify headers and data

**Expected CSV Format:**
```csv
wav_file,detection_index,start_time_s,end_time_s,duration_ms,max_prob,mean_prob,timestamp
recording_001,0,1.234000,1.256000,22.00,0.123000,0.098000,2026-02-05T14:30:00.123456
recording_001,1,2.450000,2.478000,28.00,0.156000,0.112000,2026-02-05T14:30:05.654321
```

**Verify:**
- Header row present
- Each saved detection has one row
- Numeric precision preserved (6 decimals for time/prob)
- Timestamps show progression

---

### 6. Tracking File Verification

**Purpose:** Verify `_saved_tracking.json` correctly tracks saved detections

**Steps:**
1. Save a few detections
2. Open `_saved_tracking.json`
3. Close app
4. Reopen app with same WAV
5. Verify tracker loaded previous saves (UI should show saved vs unsaved)

**Expected Tracking File:**
```json
[
  {
    "start_time_s": 1.234,
    "end_time_s": 1.256,
    "save_timestamp": "2026-02-05T14:30:00.123456",
    "output_path": "/path/to/detection_000_1.234s-1.256s.png"
  },
  ...
]
```

**Verify:**
- Each saved detection has one record
- Time ranges match core detection times (not context-extended)
- Tracker persists across app restarts

---

## Baseline Metrics Establishment

### Current Model Evaluation

**Objective:** Run evaluation on current production model to establish baseline for comparison after retraining with jittered data.

**Steps:**
1. Locate current production model:
   - Check `models/production/best_model.pt`
   - Or use latest checkpoint from `experiments/`
2. Run evaluation script:
   ```bash
   .\.venv\Scripts\python.exe scripts/evaluate_model.py \
       --model models/production/best_model.pt \
       --test-csv splits/test.csv \
       --output-dir analysis/baseline_metrics \
       --save-plots \
       --save-predictions analysis/baseline_metrics/predictions.csv
   ```
3. Save metrics to structured JSON for later comparison

**Expected Output Files:**
```
analysis/baseline_metrics/
├── confusion_matrix.png
├── roc_curve.png
├── precision_recall_curve.png
├── predictions.csv
└── metrics.json  # CREATE THIS - structured metrics for comparison
```

**Metrics to Track:**
```json
{
  "model_path": "models/production/best_model.pt",
  "evaluation_date": "2026-02-05",
  "test_set": "splits/test.csv",
  "metrics": {
    "accuracy": 0.XXX,
    "precision": 0.XXX,
    "recall": 0.XXX,
    "f1_score": 0.XXX,
    "auc_roc": 0.XXX,
    "auc_pr": 0.XXX
  },
  "confusion_matrix": {
    "true_positives": XXX,
    "false_positives": XXX,
    "true_negatives": XXX,
    "false_negatives": XXX
  },
  "notes": "Baseline before constrained jittering implementation"
}
```

---

## Bug Tracking

**If bugs found during testing, document in `BUGS.md`:**

```markdown
# Bugs Found During Phase 0 Verification

## Bug 1: [Short Description]

**Severity:** Critical / High / Medium / Low
**Test Scenario:** [Which test scenario exposed this]
**Expected Behavior:** [What should happen]
**Actual Behavior:** [What actually happens]
**Steps to Reproduce:**
1. ...
2. ...

**Files Involved:**
- `src/usv_spectrogram/app/...`

**Proposed Fix:**
[Brief description]

---

## Bug 2: ...
```

---

## Success Criteria

Phase 0 is complete when:
- [ ] All 6 test scenarios executed
- [ ] No critical bugs found (or bugs documented with workarounds)
- [ ] Baseline metrics established and saved to `analysis/baseline_metrics/metrics.json`
- [ ] Save functions verified to work as designed
- [ ] Tracking system prevents duplicates
- [ ] Metadata/CSV formats validated

---

## Notes

- **Manual Testing Required:** These tests require launching the PyQt6 app and interacting with UI
- **Test WAV File:** Use a file with 10+ detections for thorough testing (e.g., `5970 USV/recording_001.wav`)
- **Output Directory:** Default is `~/USV_Detections`, check this location after each test
- **If App Doesn't Exist:** First need to verify app can be launched with:
  ```bash
  .\.venv\Scripts\python.exe -m usv_spectrogram.app
  ```

---

## Next Phase

After Phase 0 complete → **Phase 4A: Training Curves** (training infrastructure before data generation)
