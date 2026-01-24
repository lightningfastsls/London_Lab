# Codex Task 2: Polish Diagnostic Visualizations

## Goal
Improve the aesthetics and readability of diagnostic plots created during Session 9.

## Files to Enhance
1. `scripts/threshold_sweep.py` - threshold sweep plot
2. `scripts/compare_probability_distributions.py` - probability distribution plots

## Improvements Needed

### Threshold Sweep Plot
**Current state:** Functional but basic matplotlib plots
**Enhancements:**
- Improve color palette (use colorblind-friendly colors)
- Add better gridlines (subtle, not distracting)
- Improve legend positioning
- Add annotations for key thresholds (0.25 optimal, 0.50 default)
- Increase font sizes for better readability
- Add title with metadata (model name, dataset)

### Probability Distribution Plots
**Current state:** 6-subplot figure with histograms and box plots
**Enhancements:**
- Use consistent color scheme across subplots
- Improve histogram binning (maybe 50 bins instead of 30)
- Add vertical reference lines at 0.25 and 0.5 thresholds
- Improve table formatting in subplot 6
- Add subtle background colors to distinguish sections
- Ensure all labels are readable at 150 DPI

## Design Principles
- **Accessibility:** Use colorblind-friendly palettes (e.g., viridis, Set2)
- **Clarity:** Emphasize key findings (e.g., compressed probabilities)
- **Professionalism:** Publication-quality plots
- **Consistency:** Same style across all diagnostic plots

## Example Color Palette
```python
COLORS = {
    'primary': '#2E86AB',    # Blue
    'secondary': '#A23B72',  # Purple
    'positive': '#06A77D',   # Green
    'negative': '#D7263D',   # Red
    'neutral': '#F18F01',    # Orange
}
```

## Testing
After making changes, run:
```bash
".venv/Scripts/python.exe" scripts/threshold_sweep.py
".venv/Scripts/python.exe" scripts/compare_probability_distributions.py
```

Check output images in `analysis/` folder.

## Estimated Time
30-45 minutes
