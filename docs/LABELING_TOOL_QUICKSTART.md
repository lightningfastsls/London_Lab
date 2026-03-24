# USV Labeling Tool - Quick Start Guide

## What Was Created

A complete Streamlit-based labeling tool for reviewing and labeling USV candidate spectrograms.

### New Files

1. **`src/usv_spectrogram/labeling/labeling_app.py`** (300 lines)
   - Main Streamlit UI application
   - Handles candidate loading, navigation, labeling, and persistence

2. **`src/usv_spectrogram/labeling/__init__.py`**
   - Package initialization file

3. **`scripts/usv_labeling_tool.py`**
   - Launcher script (similar to usv_parameter_lab.py)

4. **`src/usv_spectrogram/labeling/README.md`**
   - Detailed user guide for the labeling tool

### Modified Files

- **`CLAUDE.md`** - Added labeling tool to project structure
- **`IMPLEMENTATION_PROGRESS.md`** - Updated Phase 3 status

## How to Run

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Run the labeling tool
.\.venv\Scripts\streamlit.exe run scripts/usv_labeling_tool.py

# Or use the Python launcher
.\.venv\Scripts\python.exe scripts/usv_labeling_tool.py
```

The tool will launch on **port 8502** by default (Parameter Lab uses 8501).

## Features

### Core Functionality
- Loads candidates from `candidates_optimized.csv`
- Displays PNG spectrograms from `spectrograms_review/`
- One-at-a-time labeling workflow
- Three label categories: **USV**, **Not USV**, **Uncertain**
- Saves to `labels.csv` (candidate_id, label, labeled_at)

### Navigation
- **Previous** / **Next** buttons
- **Jump to Unlabeled** - finds next unlabeled candidate
- Progress tracking: "X of Y labeled"

### User Experience
- **Keyboard shortcuts**: Press 1 (USV), 2 (Not USV), 3 (Uncertain)
- In-app labeling guide with criteria
- Sidebar statistics:
  - Total labeled count
  - Progress percentage
  - Breakdown by label type
- Wide layout for optimal spectrogram viewing

### Data Persistence
- Labels saved **immediately** after each click
- No data loss on refresh or crash

### Selecting Custom Candidates + Spectrograms
If your spectrograms were generated from a different candidates CSV, use the sidebar
**Controls** section to set:
- **Candidates CSV**: path to the CSV used for extraction
- **Spectrograms folder**: directory containing the PNGs

Click **Use paths** to reload. Paths can be absolute or repo-relative.

### Refresh After Regenerating Spectrograms
If you regenerate `spectrograms_review/` while the app is running, use the sidebar
**Controls ? Reload data from disk** button to clear cached state and reload the
latest PNGs.

### Restart / Archive Labeling Pass
Use this when you want to re-label from scratch or keep an older pass.

1. Open the app and go to the sidebar **Archive / Reset** section.
2. Check the confirmation box and click **Archive labels + labeled spectrograms**.
3. The app creates `labeling_archives/labeling_archive_<timestamp>/` with:
   - `labels.csv` (backup of your labels)
   - `spectrograms/` (all labeled PNGs moved from `spectrograms_review/`)
4. `labels.csv` is cleared and labeling starts fresh.

If you want to re-label everything with improved images, regenerate `spectrograms_review/` after archiving (e.g., re-run `scripts/extract_spectrograms.py`).
- Resume where you left off - existing labels loaded on startup

## Expected Files

The tool expects these files in the repository root:

- `candidates_optimized.csv` - 490 candidates (already exists)
- `spectrograms_review/` - 490 PNG files (already exists)
- `labels.csv` - Will be created automatically

## Labeling Workflow

1. **Launch the tool** - Streamlit loads all candidates
2. **Review spectrogram** - Current candidate's image is displayed
3. **Read metadata** - Duration, frequency, energy, etc.
4. **Apply label** - Click button or press 1/2/3
5. **Navigate** - Use Next/Previous or Jump to Unlabeled
6. **Track progress** - Check sidebar statistics

## Labeling Criteria (Summary)

### USV
- Clear frequency sweep or harmonic pattern
- Frequency: 30-110 kHz
- Duration: 5-100 ms
- Smooth, continuous energy

### Not USV
- Random noise or interference
- Background hum or electrical noise
- Equipment artifacts
- Non-biological signals

### Uncertain
- Ambiguous patterns
- Very faint signals
- Overlapping signals
- Cases needing expert review

See the in-app "Labeling Guide" expander for full details.

## Next Steps

1. **Test the tool** - Label 20-30 candidates to verify workflow
2. **Adjust if needed** - Report any UI/UX issues
3. **Label dataset** - Work through all 490 candidates
4. **Proceed to Phase 4** - Dataset preparation with stratified splits

## Technical Notes

- Built with Streamlit (same framework as Parameter Lab)
- Uses session state for navigation and caching
- Sorts candidates by candidate_id for consistent order
- CSV writer updates entire file on each label (safe, simple)
- Image paths constructed from candidate_id + ".png"

## Troubleshooting

**"Candidates file not found"**
- Check that `candidates_optimized.csv` exists in repo root
- Verify path in labeling_app.py line 224

**"Spectrograms directory not found"**
- Check that `spectrograms_review/` directory exists
- Run `scripts/extract_spectrograms.py` if needed

**"Spectrogram not found"**
- Verify PNG filename matches candidate_id
- Check that extraction completed for all 490 candidates

**Keyboard shortcuts not working**
- Streamlit keyboard shortcuts require clicking buttons
- They're visual hints, not true hotkeys
- Use mouse clicks or tab navigation

## Files Created Summary

```
src/usv_spectrogram/labeling/
  __init__.py                  # Package init
  labeling_app.py              # Main UI (300 lines)
  README.md                    # User guide

scripts/
  usv_labeling_tool.py         # Launcher script

repo_root/
  labels.csv                   # Output file (created on first label)
```

All files follow existing project patterns and coding standards.