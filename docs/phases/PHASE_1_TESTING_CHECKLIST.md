# Phase 1: Boundary Adjustment - Testing Checklist

**Date:** 2026-02-06
**Feature:** Draggable detection boundaries in USV Detection App

---

## Pre-Testing Setup

- [ ] Python environment activated (`.\.venv\Scripts\Activate.ps1`)
- [ ] Detection app can launch: `python -m usv_spectrogram.app.main_window`
- [ ] Sample WAV file available for testing
- [ ] CNN model checkpoint available (default path configured)

---

## Test Suite

### Test 1: Basic Drag Functionality ⬜

**Steps:**
1. Launch detection app
2. Load WAV file (File → Open WAV or Ctrl+O)
3. Run detection (Run Detection button)
4. Hover over a **green** start boundary line
5. Click and drag left/right
6. Release mouse

**Expected Results:**
- [ ] Cursor changes to resize cursor (↔) when hovering
- [ ] Boundary line follows mouse during drag
- [ ] Yellow highlight appears on selected boundary
- [ ] Both spectrogram and probability views update together
- [ ] Boundary stays at new position after release
- [ ] Status bar shows: `Adjusted detection: X.XXXs - X.XXXs (duration: XXX.Xms)`

**Actual Results:**
```
[Record observations here]
```

---

### Test 2: End Boundary Drag ⬜

**Steps:**
1. With detections visible, hover over a **cyan** end boundary line (dashed)
2. Click and drag left/right
3. Release mouse

**Expected Results:**
- [ ] Cursor changes to resize cursor
- [ ] End boundary moves independently of start
- [ ] Yellow highlight on end boundary
- [ ] Views update together
- [ ] Status bar shows updated duration

**Actual Results:**
```
[Record observations here]
```

---

### Test 3: Validation - Prevent Start >= End ⬜

**Steps:**
1. Click on **start** boundary
2. Try to drag it **past the end boundary** (to the right)
3. Observe behavior

**Expected Results:**
- [ ] Start boundary cannot move past end
- [ ] Boundary clamps at (end_time - 0.001s)
- [ ] Minimum 1ms duration enforced
- [ ] No crash or error

**Actual Results:**
```
[Record observations here]
```

---

### Test 4: Validation - Prevent End <= Start ⬜

**Steps:**
1. Click on **end** boundary
2. Try to drag it **before the start boundary** (to the left)
3. Observe behavior

**Expected Results:**
- [ ] End boundary cannot move before start
- [ ] Boundary clamps at (start_time + 0.001s)
- [ ] Minimum 1ms duration enforced
- [ ] No crash or error

**Actual Results:**
```
[Record observations here]
```

---

### Test 5: Escape Key Cancel ⬜

**Steps:**
1. Click on any boundary and drag to a new position
2. **Before releasing**, press **Escape** key
3. Release mouse

**Expected Results:**
- [ ] Boundary reverts to original position
- [ ] Yellow highlight disappears
- [ ] Cursor returns to normal
- [ ] Status bar clears adjustment message

**Actual Results:**
```
[Record observations here]
```

---

### Test 6: Click Tolerance ⬜

**Steps:**
1. Click 3-4 pixels to the **left** of a boundary line
2. Click 3-4 pixels to the **right** of a boundary line
3. Click 10+ pixels away from any boundary

**Expected Results:**
- [ ] Clicks within ±5px select the boundary
- [ ] Cursor changes within tolerance zone
- [ ] Clicks outside 5px range deselect (no highlight)
- [ ] No crash or error

**Actual Results:**
```
[Record observations here]
```

---

### Test 7: View Synchronization ⬜

**Steps:**
1. Drag a boundary to a new position
2. Observe both views during and after drag:
   - Spectrogram view (top panel)
   - Probability view (bottom panel)

**Expected Results:**
- [ ] Spectrogram boundary line moves in real-time
- [ ] Probability view shaded region updates simultaneously
- [ ] No desynchronization or lag
- [ ] Both views show identical time boundaries

**Actual Results:**
```
[Record observations here]
```

---

### Test 8: Save and Persistence ⬜

**Steps:**
1. Adjust a boundary (note the new time values)
2. Save labels (File → Save Labels or Ctrl+S)
3. Choose output location and confirm
4. Open the saved JSON file in a text editor
5. Find the adjusted detection in the `"detections"` array

**Expected Results:**
- [ ] JSON file saves successfully
- [ ] Detection has `"user_adjusted": true`
- [ ] `"original_start_time_s"` shows old start time
- [ ] `"original_end_time_s"` shows old end time
- [ ] `"start_time_s"` and `"end_time_s"` show new values
- [ ] `"duration_s"` matches (new_end - new_start)

**Example JSON snippet:**
```json
{
  "start_time_s": 1.234,
  "end_time_s": 1.456,
  "user_adjusted": true,
  "original_start_time_s": 1.200,
  "original_end_time_s": 1.450
}
```

**Actual Results:**
```
[Paste JSON snippet or describe]
```

---

### Test 9: Multiple Sequential Adjustments ⬜

**Steps:**
1. Adjust a boundary once
2. Adjust the **same** boundary again (move it further)
3. Adjust the **other** boundary of the same detection
4. Save labels

**Expected Results:**
- [ ] Each adjustment works correctly
- [ ] Original times preserved from **first** adjustment
- [ ] JSON shows original from initial detection, not intermediate values
- [ ] No corruption of detection data

**Actual Results:**
```
[Record observations here]
```

---

### Test 10: Adjust Different Detections ⬜

**Steps:**
1. Load a WAV with multiple detections (at least 3)
2. Adjust boundary on detection #1
3. Adjust boundary on detection #2
4. Adjust boundary on detection #3
5. Verify all adjustments persist

**Expected Results:**
- [ ] Each detection adjusts independently
- [ ] No interference between detections
- [ ] All adjustments visible in both views
- [ ] Save captures all adjustments

**Actual Results:**
```
[Record observations here]
```

---

### Test 11: Edge Cases - File Boundaries ⬜

**Steps:**
1. Find a detection near the **start** of the file (< 0.5s)
2. Try to drag start boundary **before** time = 0
3. Find a detection near the **end** of the file
4. Try to drag end boundary **past** audio duration

**Expected Results:**
- [ ] Start boundary clamps at time = 0 (file start)
- [ ] End boundary clamps at audio duration (file end)
- [ ] No crash or negative times
- [ ] No times beyond file duration

**Actual Results:**
```
[Record observations here]
```

---

### Test 12: Zoom and Scroll ⬜

**Steps:**
1. Load a long WAV file (> 10 seconds)
2. Scroll to middle of file
3. Adjust a boundary
4. Zoom in/out using scroll bars
5. Adjust another boundary at high zoom

**Expected Results:**
- [ ] Adjustments work correctly at any zoom level
- [ ] Scrolling doesn't break selection
- [ ] High zoom provides finer control (as expected)
- [ ] Coordinate conversions accurate at all zoom levels

**Actual Results:**
```
[Record observations here]
```

---

## Bug Tracking

### Bugs Found:

| # | Severity | Description | Steps to Reproduce | Status |
|---|----------|-------------|-------------------|--------|
| 1 |          |             |                   |        |
| 2 |          |             |                   |        |
| 3 |          |             |                   |        |

---

## Performance Observations

- **Responsiveness during drag:**
- **View update lag:**
- **Memory usage:**
- **CPU usage during drag:**

---

## UX Feedback

### What Worked Well:
-

### What Could Be Improved:
-

### Feature Requests:
-

---

## Test Results Summary

- **Tests Passed:** _____ / 12
- **Tests Failed:** _____ / 12
- **Bugs Found:** _____
- **Critical Bugs:** _____

**Overall Assessment:**
```
[Pass / Fail / Needs Revision]
```

**Tester Name:** _________________
**Date Tested:** _________________

---

## Next Steps

- [ ] Fix critical bugs (if any)
- [ ] Document known issues
- [ ] Update user guide based on testing feedback
- [ ] Proceed to Phase 2 (CNN model scaling)
