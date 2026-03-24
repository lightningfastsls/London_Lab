# USV Detection App - Boundary Adjustment User Guide

## Overview

The boundary adjustment feature allows you to manually refine detection boundaries by clicking and dragging the boundary lines in the spectrogram view.

---

## How to Use

### 1. Basic Adjustment

1. **Load a WAV file** and **run detection** first
2. **Hover** over a boundary line (green for start, cyan for end)
   - Cursor will change to a resize cursor (↔)
3. **Click and drag** the boundary line left or right
4. **Release** to finalize the adjustment

### 2. Visual Feedback

- **Green line:** Start boundary (solid)
- **Cyan line:** End boundary (dashed)
- **Yellow highlight:** Currently selected boundary
- **Resize cursor (↔):** Appears when hovering near boundary
- **Status bar:** Shows updated time range and duration

### 3. Validation

The app prevents invalid adjustments:
- **Start boundary** cannot move past end boundary
- **End boundary** cannot move before start boundary
- **Minimum duration:** 1 millisecond enforced
- **Time range:** Boundaries clamped to audio file duration

### 4. Cancel Adjustment

- Press **Escape** while dragging to revert to original position
- The boundary will snap back to where it was before dragging

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **Escape** | Cancel current adjustment and revert to original |

---

## Tips

### Precise Adjustment
- **Zoom in** using the scroll bars for finer control
- The 5-pixel click tolerance makes it easy to select boundaries even at high zoom

### Checking Your Work
- Watch the **probability view** (bottom panel) - the shaded region updates in real-time
- The **status bar** shows the exact time range: `Adjusted detection: 1.234s - 1.456s (duration: 222.0ms)`

### Saving Adjusted Detections
- Adjusted boundaries are **automatically saved** when you save labels
- The JSON file includes:
  - New boundary times
  - `"user_adjusted": true` flag
  - Original boundary times for reference

---

## Example Workflow

### Fixing a Conservative Detection

If the detector missed part of a USV:

1. Find the detection in the spectrogram (green/cyan lines)
2. Look at the probability curve - does it extend beyond the boundaries?
3. Click the **start** or **end** boundary
4. Drag to align with the actual USV extent
5. Release and verify in both views
6. Save labels (Ctrl+S)

### Fixing an Aggressive Detection

If the detector included noise at the edges:

1. Examine the spectrogram and probability curve
2. Click the boundary that extends too far
3. Drag inward to exclude the noisy region
4. Verify the duration makes sense (USVs are typically 10-200ms)
5. Save labels (Ctrl+S)

---

## What Gets Saved?

When you save labels, the JSON file includes:

```json
{
  "detections": [
    {
      "start_time_s": 1.234,
      "end_time_s": 1.456,
      "duration_s": 0.222,
      "user_adjusted": true,
      "original_start_time_s": 1.200,
      "original_end_time_s": 1.450,
      ...
    }
  ]
}
```

This preserves:
- ✅ Adjusted boundary times
- ✅ Flag indicating manual adjustment
- ✅ Original detector output for comparison
- ✅ Full probability metadata

---

## Troubleshooting

### "Cursor doesn't change when hovering"
- Make sure you've run detection first
- Try hovering directly over the boundary line (not between lines)
- The click tolerance is 5 pixels - get close to the line

### "Boundary won't move past a certain point"
- You've hit a validation limit:
  - Start cannot pass end (1ms minimum duration enforced)
  - Boundaries cannot exceed audio file duration

### "Escape key doesn't cancel"
- Press Escape **while the boundary is selected** (before clicking elsewhere)
- The yellow highlight shows which boundary is selected

### "Changes not saved"
- Make sure to **Save Labels** (Ctrl+S) after adjusting
- The app does not auto-save adjustments

---

## Future Enhancements (Planned)

- **Visual handles:** Small rectangles at boundaries for better discoverability
- **Arrow key adjustment:** Fine-tune boundaries with keyboard (1px, 10px increments)
- **Multi-level undo:** Undo/redo for sequential adjustments
- **Batch adjustment:** Select and adjust multiple detections at once

---

## Technical Details

For developers and power users:

- **Click tolerance:** ±5 pixels from boundary line
- **Minimum duration:** 1 millisecond (prevents zero-duration detections)
- **Coordinate precision:** Pixel positions mapped to time via linear interpolation
- **Update frequency:** Real-time during drag (immediate visual feedback)
- **Data model:** Immutable DetectedUSV objects (creates new instance on adjustment)

---

**Questions or feedback?** Report issues at: https://github.com/anthropics/claude-code/issues
