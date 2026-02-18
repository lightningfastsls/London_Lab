# USV Detection App - Keyboard Shortcuts

## File Operations

| Shortcut | Action |
|----------|--------|
| `Ctrl+O` | Open WAV file |
| `Ctrl+S` | Save labels to JSON |
| `Ctrl+E` | Export annotated image |
| `Ctrl+Q` | Quit application |

## Detection & Threshold Adjustment

| Shortcut | Action |
|----------|--------|
| `Space` | Run detection (when WAV loaded) |
| `↑` (Up Arrow) | Increase high threshold by 0.01 |
| `↓` (Down Arrow) | Decrease high threshold by 0.01 |
| `→` (Right Arrow) | Increase low threshold by 0.01 |
| `←` (Left Arrow) | Decrease low threshold by 0.01 |

## Features

### Synchronized Scrolling
- Horizontal scrolling is automatically synchronized between spectrogram and probability views
- Scroll in either view to navigate long recordings

### Settings Persistence
- Window size and position are remembered across sessions
- Last used thresholds are restored on startup
- Settings stored platform-specifically (Registry on Windows)

## Tips

- **Threshold Exploration:** Use arrow keys for rapid threshold adjustment while viewing results
- **Quick Workflow:** Load file → Press Space to detect → Use arrows to refine thresholds
- **Long Files:** Scroll horizontally to inspect different time regions
- **Default Thresholds:** High = 0.40, Low = 0.28 (based on CNN optimal threshold)

## Hysteresis Detection Logic

The app uses hysteresis thresholding:
1. USV **starts** when probability crosses **high threshold** (default 0.40)
2. USV **continues** until probability drops below **low threshold** (default 0.28)
3. Nearby detections (< 3 columns apart) are merged

This reduces false positives while maintaining sensitivity to continuous USVs.
