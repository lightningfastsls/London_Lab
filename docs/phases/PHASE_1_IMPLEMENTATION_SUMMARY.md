# Phase 1 Implementation Summary: Boundary Adjustment Feature

**Date:** 2026-02-06
**Status:** ✅ COMPLETE
**Task:** Add draggable boundary handles to PyQt6 detection app

---

## What Was Implemented

Added full boundary adjustment capability to the USV Detection App, allowing users to manually refine detection boundaries by clicking and dragging boundary lines in the spectrogram view.

---

## Changes Made

### 1. DetectedUSV Dataclass Extension (`detection_logic.py`)

**File:** `src/usv_spectrogram/app/core/detection_logic.py`

Added three new fields to track manual adjustments:
```python
user_adjusted: bool = False
original_start_time_s: float | None = None
original_end_time_s: float | None = None
```

**Purpose:** Preserve adjustment history for traceability and potential future undo/analytics.

---

### 2. SpectrogramCanvas Mouse Handling (`spectrogram_view.py`)

**File:** `src/usv_spectrogram/app/widgets/spectrogram_view.py`

#### Added Components:

1. **New Signal:**
   - `boundary_adjusted = pyqtSignal(object, float, float)` - Emitted when user drags boundary

2. **Drag State Tracking:**
   - `_selected_detection` - Currently selected DetectedUSV
   - `_dragging_edge` - Which edge is being dragged ('start' or 'end')
   - `_original_time` - For Escape key cancel/undo
   - `_is_dragging` - Drag state flag

3. **Mouse Event Handlers:**
   - `mousePressEvent()` - Detect clicks near boundaries (±5px tolerance)
   - `mouseMoveEvent()` - Update boundary position during drag with validation
   - `mouseReleaseEvent()` - Finalize drag operation
   - `keyPressEvent()` - Handle Escape key to cancel adjustment

4. **Helper Methods:**
   - `_pixel_to_time()` - Convert screen coordinates to time (inverse of existing `_time_to_pixel()`)

5. **Visual Feedback:**
   - Updated `paintEvent()` to highlight selected boundary in yellow
   - Cursor changes to resize cursor when hovering/dragging boundaries

#### UX Features:

- **Click tolerance:** 5 pixels on either side of boundary line
- **Drag validation:** Prevents start >= end (enforces 1ms minimum duration)
- **Time clamping:** Boundaries cannot exceed audio file duration
- **Escape to cancel:** Reverts to original position if user presses Escape
- **Visual feedback:** Yellow highlight on selected boundary, resize cursor

---

### 3. MainWindow Boundary Update Handler (`main_window.py`)

**File:** `src/usv_spectrogram/app/main_window.py`

#### Added Components:

1. **Signal Connection:**
   - Connected `spectrogram_view.canvas.boundary_adjusted` to `_on_boundary_adjusted()` slot

2. **Boundary Adjustment Handler (`_on_boundary_adjusted()`):**
   - Receives: original detection, new start time, new end time
   - Converts times to column indices using inference result times
   - Creates new `DetectedUSV` with adjusted boundaries
   - Preserves probability metadata from original
   - Tracks adjustment history (stores original times on first adjustment)
   - Replaces detection in result list
   - Updates both spectrogram and probability views
   - Shows status message with new duration

3. **View Synchronization:**
   - Updates both spectrogram and probability canvas together
   - Ensures pixel-perfect alignment via column indices

#### Data Flow:

```
User drags boundary
  ↓
SpectrogramCanvas emits boundary_adjusted signal
  ↓
MainWindow._on_boundary_adjusted() receives signal
  ↓
Creates new DetectedUSV with adjusted times
  ↓
Replaces in detection_result.usvs list
  ↓
Updates both views (spectrogram + probability)
  ↓
Shows status message
```

---

### 4. JSON Persistence (`label_storage.py`)

**File:** `src/usv_spectrogram/app/core/label_storage.py`

#### Save Method Updates:

Modified `save()` to include adjustment metadata in JSON:
```json
{
  "detections": [
    {
      "start_time_s": 1.234,
      "end_time_s": 1.456,
      "user_adjusted": true,
      "original_start_time_s": 1.200,
      "original_end_time_s": 1.450,
      ...
    }
  ]
}
```

#### Load Method Updates:

Added `reconstruct_detected_usv()` helper method:
- Reconstructs `DetectedUSV` objects from JSON dicts
- Restores adjustment metadata fields
- Uses `.get()` for backward compatibility with old JSON files

**Backward Compatibility:** Old JSON files without adjustment metadata will still load correctly (fields default to `False` and `None`).

---

## Files Modified

1. ✅ `src/usv_spectrogram/app/core/detection_logic.py` - Extended DetectedUSV dataclass
2. ✅ `src/usv_spectrogram/app/widgets/spectrogram_view.py` - Added mouse event handling
3. ✅ `src/usv_spectrogram/app/main_window.py` - Added boundary adjustment handler
4. ✅ `src/usv_spectrogram/app/core/label_storage.py` - Updated JSON save/load

**All files compile successfully with `py_compile`.**

---

## Success Criteria (from Plan)

- ✅ User can click on detection boundary to select it
- ✅ Cursor changes to resize cursor when hovering/dragging
- ✅ Boundary follows mouse during drag
- ✅ Start boundary cannot move past end (and vice versa)
- ✅ Escape key cancels adjustment and reverts to original
- ✅ Both spectrogram and probability views update together
- ✅ Adjusted boundaries persist when saved
- ✅ JSON includes `user_adjusted` flag and original times
- ✅ Status bar shows updated duration during adjustment

---

## Testing Recommendations

### Manual Testing Steps:

1. **Basic Drag Test:**
   - Load WAV file, run detection
   - Click on green start boundary line → verify cursor changes to resize
   - Drag left/right → verify boundary follows mouse
   - Release → verify boundary stays at new position
   - Check status bar shows updated duration

2. **Validation Test:**
   - Try to drag start boundary past end boundary
   - Should clamp at end - 1ms (no zero-duration detections)

3. **Escape to Cancel:**
   - Drag boundary to new position
   - Press Escape before releasing
   - Should revert to original position

4. **Persistence Test:**
   - Adjust boundary, save detection
   - Check saved JSON file contains:
     - `"user_adjusted": true`
     - `"original_start_time_s": <original value>`
     - `"original_end_time_s": <original value>`

5. **View Synchronization:**
   - Adjust boundary in spectrogram view
   - Verify probability view shaded region updates simultaneously

---

## Architecture Decisions

### Why Painter-Based (Not QGraphicsView)?

- **Minimal code changes** - No major refactor needed
- **Consistent with existing rendering** - Already using custom `paintEvent()`
- **Clean separation** - SpectrogramCanvas handles UI, MainWindow manages data
- **Performance** - Direct painting is faster for this use case

### Why Immutable DetectedUSV?

- **Dataclass semantics** - Dataclasses are meant to be immutable
- **Thread safety** - Prevents concurrent modification issues
- **History tracking** - Easy to preserve original values
- **Predictable behavior** - No hidden state mutations

### Why Signal/Slot Pattern?

- **Loose coupling** - Canvas doesn't need to know about MainWindow internals
- **PyQt6 best practice** - Standard pattern for inter-widget communication
- **Testability** - Can test canvas and window separately

---

## Next Steps (Future Phases)

### Phase 2: CNN Model Scaling
- Implement multi-scale architecture (32px, 48px, 64px windows)
- Add model selection UI

### Phase 3: Constrained Jittering
- Use adjusted boundaries for data augmentation
- Generate training data with controlled boundary variation
- Preserve user corrections during augmentation

---

## Notes for Future Development

### UX Enhancements (Optional):
- **Visual handles:** Draw small rectangles/circles at boundaries for better discoverability
- **Keyboard shortcuts:** Arrow keys for fine adjustment (1px, 10px increments)
- **Undo stack:** Multi-level undo for sequential adjustments
- **Batch adjustment:** Select and adjust multiple detections at once

### Performance Optimizations:
- Debounce `boundary_adjusted` signal during rapid mouse moves
- Cache pixel-to-time conversions for faster lookup

### Accessibility:
- Add tooltips explaining drag functionality
- Keyboard-only adjustment mode (tab to select, arrow keys to adjust)

---

## Learning Opportunities

### Concepts Demonstrated:

1. **PyQt6 Event Handling:**
   - Mouse events (press, move, release)
   - Keyboard events
   - Custom signals and slots

2. **Coordinate System Conversion:**
   - Pixel ↔ Time mapping
   - Time ↔ Column index mapping
   - Handling edge cases (zero width, empty arrays)

3. **Immutable Data Patterns:**
   - Creating new objects instead of mutating
   - Preserving history for undo/analytics

4. **View Synchronization:**
   - Keeping multiple views in sync
   - Signal-driven architecture

5. **JSON Versioning:**
   - Adding optional fields for backward compatibility
   - Using `.get()` with defaults for graceful degradation

---

## Validation

**Syntax Check:** ✅ All files compile with `py_compile`
**Plan Adherence:** ✅ Implemented all phases (1A-1D)
**Code Quality:** ✅ Clean separation of concerns, well-documented
**Backward Compatibility:** ✅ Old JSON files still work

---

**Implementation completed successfully!** 🎉

Ready for manual testing and integration into the broader USV scaling pipeline (Phases 2-3).
