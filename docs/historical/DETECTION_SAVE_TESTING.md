# Detection Saving Feature - Testing Guide

## Implementation Summary

Added functionality to save individual detections with context, track saved detections, and warn about unsaved work.

### New Components

1. **SavedDetectionTracker** (`src/usv_spectrogram/app/core/saved_detection_tracker.py`)
   - Tracks which detections have been saved by time range
   - Persists to JSON file (`_saved_tracking.json`) in output directory
   - Prevents duplicate saves based on time overlap

2. **DetectionExporter** (`src/usv_spectrogram/app/core/detection_exporter.py`)
   - Exports detections as annotated PNG (spectrogram with detection boundaries)
   - Saves JSON metadata (time ranges, probabilities, context info)
   - Appends to CSV summary file

3. **MainWindow Enhancements** (`src/usv_spectrogram/app/main_window.py`)
   - "Save Current View" button - saves detections visible in viewport
   - "Save All Detections" button - batch saves all unsaved detections
   - Unsaved detection warnings when switching files or closing app
   - Output directory setting (File → Set Output Directory)

## File Output Structure

```
{output_dir}/
  {wav_filename}/
    detection_001_1.234s-1.456s.png      # Annotated spectrogram with ±20ms context
    detection_001_1.234s-1.456s.json     # Metadata (times, probabilities, etc.)
    detection_002_2.345s-2.567s.png
    detection_002_2.345s-2.567s.json
    ...
    detections_summary.csv               # All detections in one CSV
    _saved_tracking.json                 # Internal tracking file
```

## Testing Checklist

### Test 1: Save Current View (Multiple Detections)
**Purpose**: Verify that multiple visible detections are handled correctly

1. ✅ Load a WAV file and run detection
2. ✅ Scroll to a region showing 2-3 detections
3. ✅ Click "Save Current View"
4. ✅ Verify confirmation dialog shows correct count (e.g., "Save 3 detection(s)?")
5. ✅ Click Yes and verify progress (if multiple)
6. ✅ Check output directory:
   - PNG files created with proper naming
   - JSON files have correct metadata
   - CSV summary exists with all detections
7. ✅ Scroll to same region, click "Save Current View" again
8. ✅ Verify message: "All detections in view already saved"

### Test 2: Save All Detections
**Purpose**: Verify batch save with duplicate prevention

1. ✅ Load WAV with 10+ detections
2. ✅ Manually save 2 detections from current view
3. ✅ Click "Save All Detections"
4. ✅ Verify confirmation shows only unsaved count (e.g., "Save 8 unsaved detection(s)?")
5. ✅ Verify progress dialog appears
6. ✅ Check output directory has all detections
7. ✅ Click "Save All Detections" again
8. ✅ Verify message: "All detections already saved"

### Test 3: Unsaved Detection Warning (File Switch)
**Purpose**: Verify warning when switching files with unsaved work

1. ✅ Load WAV, run detection
2. ✅ Save one detection (not all)
3. ✅ Try to open a different WAV file (Ctrl+O or File → Open)
4. ✅ Verify warning popup appears: "You have N unsaved detection(s)"
5. ✅ Click "Review Unsaved"
6. ✅ Verify viewport scrolls to first unsaved detection
7. ✅ Verify file did NOT switch
8. ✅ Try opening file again, click "Discard"
9. ✅ Verify file switches without saving

### Test 4: Unsaved Detection Warning (App Close)
**Purpose**: Verify warning when closing app with unsaved work

1. ✅ Load WAV, run detection, save some (not all)
2. ✅ Try to close app (Ctrl+Q or close window)
3. ✅ Verify warning appears
4. ✅ Click "Review Unsaved"
5. ✅ Verify app stays open and scrolls to unsaved detection
6. ✅ Try closing again, click "Cancel"
7. ✅ Verify app stays open
8. ✅ Try closing again, click "Discard"
9. ✅ Verify app closes

### Test 5: Context and Duplicate Detection
**Purpose**: Verify 20ms context is included and time-based duplicate checking works

1. ✅ Load WAV, run detection
2. ✅ Note a detection at time T (e.g., 1.5-1.6s)
3. ✅ Save it, check JSON metadata:
   - `core_time`: Should match detection bounds (1.5-1.6s)
   - `saved_region`: Should include ±20ms context (~1.48-1.62s)
4. ✅ Adjust threshold to create overlapping detection (e.g., 1.55-1.65s)
5. ✅ Try to save - verify marked as already saved (time overlap)

### Test 6: Output Files Quality
**Purpose**: Verify generated files are correct and useful

1. ✅ Save a detection
2. ✅ Open PNG file:
   - Spectrogram visible with magma colormap
   - Time axis (seconds) at bottom
   - Frequency axis (kHz) on left
   - Cyan dashed line at detection start
   - Lime dashed line at detection end
   - Title shows time range, duration, probabilities
   - Colorbar shows dB scale
3. ✅ Open JSON file:
   - Verify all fields present and correct
   - Timestamp is ISO format
4. ✅ Open CSV file:
   - Header row present
   - All saved detections listed
   - Timestamps match

### Test 7: Output Directory Setting
**Purpose**: Verify custom output directory can be configured

1. ✅ File → Set Output Directory
2. ✅ Select a custom directory
3. ✅ Save a detection
4. ✅ Verify files appear in custom directory
5. ✅ Close and reopen app
6. ✅ Verify output directory setting persisted

### Test 8: Edge Cases

1. **No detections in view**
   - Scroll to region with no detections
   - Click "Save Current View"
   - Verify: "No detections in current view"

2. **Empty detection result**
   - Run detection on file with no USVs
   - Click "Save All"
   - Verify: "No detections to save"

3. **Cancel batch save**
   - Start "Save All" with many detections
   - Click Cancel in progress dialog
   - Verify save stops, partial saves are tracked

## Known Behavior

- **Multiple detections in view**: "Save Current View" saves ALL visible detections, not just one
  - User gets confirmation dialog showing count
  - Progress bar shows for multiple detections

- **Duplicate checking**: Based on core detection time overlap (not including context)
  - Two detections with overlapping time ranges = duplicate
  - Same detection saved with different context lengths = duplicate

- **File structure**: One subdirectory per WAV file
  - All detections from same WAV go in same folder
  - Tracking file is per-WAV (allows same detection across different WAVs)

## Success Criteria

- ✅ Can save detections visible in current viewport
- ✅ Saved files include PNG + JSON + CSV entry
- ✅ Duplicate detections not saved (time-based checking)
- ✅ "Save All" batch saves all unsaved detections with progress dialog
- ✅ Warning popup shows when switching files with unsaved detections
- ✅ Can scroll to first unsaved detection from warning
- ✅ Settings persist output directory
- ✅ File structure organized by WAV filename
