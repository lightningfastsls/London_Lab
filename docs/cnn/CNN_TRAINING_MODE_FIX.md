# CNN Training Mode Fix - Critical Issue Resolved

## Problem Discovered

The CNN was not learning properly (stuck at ~55-60% accuracy) because it was being trained on **review mode spectrograms** instead of **training mode spectrograms**.

### What Review Mode Includes

Review mode spectrograms (used for human labeling) contain:
- ❌ Matplotlib axes, labels, titles, and colorbars
- ❌ **50-67% of pixels are GREEN LINES** marking detection boundaries
- ❌ White backgrounds from matplotlib rendering
- ❌ Text annotations
- ✅ The actual spectrogram data (buried under all the above)

**Example:** A 612×306 review mode image had **95,006 green pixels (50.73%)**!

### Why This is Catastrophic for Learning

1. **Confounding Features**: CNN learns matplotlib artifacts instead of USV acoustic features
2. **Green Line Variation**: Position of green lines varies with USV duration, adding noise
3. **Variable Aspects**: White borders and axes take up different amounts of space
4. **Wrong Size**: Images are RGBA 612×306 (variable width) instead of clean RGB 512×256

### What the CNN Was Learning

Instead of learning:
- ✅ Frequency contours of USVs
- ✅ Harmonic structures
- ✅ Temporal patterns

It was learning:
- ❌ "Green lines at positions X and Y"
- ❌ "White matplotlib borders"
- ❌ "Colorbar on the right side"
- ❌ Axis tick marks and labels

## Solution Implemented

### 1. Re-extracted All Spectrograms in Training Mode

```powershell
# Extract USV candidates
python scripts/extract_spectrograms.py --candidates candidates_with_onsets.csv --wav-dir "5970 USV" --output-dir spectrograms_training --mode training -v

# Extract noise samples
python scripts/extract_spectrograms.py --candidates noise_samples/noise_samples_final.csv --wav-dir "5970 USV" --output-dir noise_samples_training --mode training -v
```

### 2. Training Mode Advantages

Training mode spectrograms are:
- ✅ **Clean RGB images** (no axes/labels/titles)
- ✅ **No green lines** (3.20% green pixels vs 50.73%)
- ✅ **Fixed dimensions**: 512×256 pixels (consistent for CNN)
- ✅ **Pure colormap data** - just the spectrogram
- ✅ **Smaller file size** and faster loading

### 3. Updated Dataset Splits

Updated all split CSVs to point to training mode directories:
- `spectrograms_training/` for USV candidates
- `noise_samples_training/` for noise samples

Some samples were removed (48 total) due to overlap pruning - these were duplicate/overlapping detections.

**New dataset size:**
- **Train**: 706 samples (was 740) - 433 USV / 273 Not USV
- **Val**: 172 samples (was 178) - 108 USV / 64 Not USV
- **Test**: 121 samples (was 129) - 68 USV / 53 Not USV
- **Total**: 999 samples (was 1,047)

## Expected Improvements

With clean training data, expect:

1. **Better Learning**: CNN can focus on acoustic features
2. **Faster Convergence**: No confounding visual artifacts
3. **Higher Accuracy**: Target 80-90% (vs previous 55-60%)
4. **Better Generalization**: Model learns actual USV characteristics

## Verification

Compare training curves before and after:

**Before (review mode):**
- Epoch 1: 55% train, 61% val
- Epoch 10: 61% train, 54% val
- **Stuck, not improving**

**After (training mode):**
- Expect steady improvement over 20-30 epochs
- Target: 80-90% validation accuracy
- Lower train-val gap (less overfitting to matplotlib artifacts)

## Technical Details

### Review Mode Rendering (_render_review)
```python
# Creates matplotlib figure with axes
fig, ax = plt.subplots(figsize=(width, height), dpi=100)
ax.pcolormesh(times, freqs, spec_db, cmap='magma')

# Adds green detection markers
ax.axvline(start_ms, color='lime', linewidth=1, linestyle='--')
ax.axvline(end_ms, color='lime', linewidth=1, linestyle='--')

# Adds labels, title, colorbar
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Frequency (kHz)")
fig.colorbar(mesh, ax=ax, label="dB")
```

### Training Mode Rendering (_render_training)
```python
# Applies colormap directly to numpy array
cmap = plt.get_cmap('magma')
rgb = cmap(normalized)[:, :, :3]  # RGB only, no alpha

# Resize to fixed dimensions
img = Image.fromarray(rgb_uint8)
img_resized = img.resize((512, 256), Image.LANCZOS)
img_resized.save(output_path)  # Clean PNG
```

## Files Created/Modified

**Created:**
- `spectrograms_training/` - 697 clean spectrogram PNGs
- `noise_samples_training/` - 412 clean noise sample PNGs (26 overlap-pruned)
- `update_splits_paths.py` - Script to update CSV paths

**Modified:**
- `splits/train.csv` - Updated paths to training mode
- `splits/val.csv` - Updated paths to training mode
- `splits/test.csv` - Updated paths to training mode

## Lesson Learned

**Never train CNNs on visualization-ready images!**

Always use raw/clean data:
- No axes, labels, or annotations
- No decorative elements (borders, grids, colorbars)
- Fixed dimensions when possible
- Minimal preprocessing (just normalization)

Review mode is for humans, training mode is for machines.

## Next Steps

1. ✅ Re-extracted spectrograms in training mode
2. ✅ Updated dataset splits
3. 🔄 Running test training (15 epochs)
4. ⏳ Full production training (100 epochs with early stopping)
5. ⏳ Evaluate on test set

---

**Date Fixed:** 2026-01-22
**Impact:** Critical - Blocking all CNN training progress
**Status:** Resolved
