# Parameter Lab Refactoring Summary

## Overview
Refactored the 654-line `app_legacy.py` into a modular structure for better maintainability and reusability.

## New Structure

```
src/usv_spectrogram/param_lab/
├── app.py                 # New thin entry point (~35 lines)
├── app_legacy.py          # Original monolithic implementation (preserved)
├── plotting.py            # Plotting functions (~110 lines)
├── state.py               # Caching and config helpers (~105 lines)
└── ui/
    ├── __init__.py
    ├── components.py      # Reusable parameter controls (~80 lines)
    ├── sidebar.py         # Complete sidebar rendering (~230 lines)
    └── main_content.py    # Main content rendering (~280 lines)
```

## Module Responsibilities

### `app.py` (New Entry Point)
- Sets page config and title
- Creates default configuration
- Orchestrates sidebar and main content rendering
- **35 lines** vs original 654 lines

### `plotting.py`
Extracted from lines 63-162 of app_legacy.py:
- `plot_spectrogram()` - Render spectrogram with overlays
- `plot_difference()` - Render difference view

### `state.py`
Extracted from lines 31-213 of app_legacy.py:
- `read_wav_info()` - Cached WAV metadata loading
- `compute_segment_spectrogram()` - Cached spectrogram computation
- `config_from_controls()` - Build config from UI values
- `stft_cache_key()` - Generate cache keys
- `parse_sweep_values()` - Parse sweep parameter values

### `ui/components.py`
New reusable component:
- `render_parameter_controls()` - Eliminates duplication between baseline/variant controls
- Uses consistent key naming: `f"{prefix}.param_name"`

### `ui/sidebar.py`
Extracted from lines 227-454 of app_legacy.py:
- `render_sidebar()` - Complete sidebar rendering
- Returns dict with all user selections
- Uses `render_parameter_controls()` for DRY parameter UI

### `ui/main_content.py`
Extracted from lines 486-653 of app_legacy.py:
- `render_spectrogram_panel()` - Reusable spectrogram display
- `render_diff_view()` - Difference visualization
- `render_parameter_explanations()` - Expandable docs
- `render_sweep_export()` - Sweep interface
- `render_main_content()` - Main orchestrator

## Key Improvements

1. **Modularity**: Clear separation of concerns (plotting, state, UI)
2. **Reusability**: `render_parameter_controls()` eliminates 100+ lines of duplication
3. **Testability**: Pure functions easier to unit test
4. **Maintainability**: ~35 line entry point vs 654 line monolith
5. **Consistency**: Unified key naming scheme (`prefix.param_name`)
6. **UX**: Added `st.spinner("Computing spectrograms...")` for visual feedback

## Preserved Features

All original functionality is preserved:
- WAV file selection and validation
- Baseline and variant parameter controls
- Display gain/range controls
- Heuristic overlay detection
- Difference view
- Parameter explanations
- Sweep export
- Lock baseline checkbox

## Running the App

The launcher script remains unchanged:
```powershell
.\.venv\Scripts\streamlit.exe run scripts/usv_parameter_lab.py
```

## Migration Path

The original `app_legacy.py` is preserved for reference. Once the new modular version is validated, `app_legacy.py` can be removed.

## File Size Comparison

| Module | Lines | Responsibility |
|--------|-------|----------------|
| app.py | 35 | Entry point, orchestration |
| plotting.py | 110 | Visualization |
| state.py | 105 | Caching, config management |
| ui/components.py | 80 | Reusable widgets |
| ui/sidebar.py | 230 | Sidebar rendering |
| ui/main_content.py | 280 | Main content rendering |
| **Total** | **840** | **(vs 654 original, +186 for modularity)** |

The slight increase in total lines provides significant maintainability benefits through clear module boundaries and reusable components.
