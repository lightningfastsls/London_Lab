# CNN Batch Detection Analysis - Root Cause and Options

## Executive Summary

**Problem**: The CNN cannot be used for direct batch detection of USVs from WAV files. It classifies ~100% of random audio chunks as USV (mean probability 0.997).

**Root Cause**: The CNN was trained as a **candidate classifier**, not a **chunk detector**. It learned to recognize "acoustic energy in USV frequency band" rather than distinguishing USVs from other sounds.

**Recommendation**: Either (A) use Energy Detector + CNN in sequence, or (B) retrain the CNN with proper negative samples.

---

## Diagnostic Results

### Test 1: Labeled USV Samples (n=34)
- Mean probability: **0.992**
- 100% above threshold 0.5
- 100% above threshold 0.9
- ✓ CNN correctly identifies labeled USVs

### Test 2: Labeled "Not USV" Samples (n=5)
- Mean probability: **0.684**
- 80% above threshold 0.5
- 40% above threshold 0.9
- ⚠️ CNN gives HIGH probability to "Not USV" samples!

### Test 3: Random 40ms Chunks from WAV Files (n=100)
- Mean probability: **0.997**
- 100% above threshold 0.5
- 100% above threshold 0.9
- ❌ CNN thinks EVERYTHING is USV!

---

## Root Cause Analysis

### How the CNN Was Trained

```
Training Pipeline:
1. Energy Detector → finds regions with high energy in 25-110 kHz band
2. Human Labeling → marks each candidate as "USV" or "Not USV"
3. SpectrogramExtractor → creates centered spectrogram images
4. CNN Training → binary classification on these images
```

### What the CNN Actually Learned

The CNN learned to recognize **"acoustic structure in the USV frequency band"** because:

1. **All training samples came from energy-detector candidates**
   - Even "Not USV" samples had high energy (that's why the energy detector selected them)
   - True quiet/random audio was never shown to the CNN

2. **USVs were always centered and prominent**
   - Training spectrograms were created with USV in the center
   - The CNN expects the signal to be "front and center"

3. **Limited negative diversity**
   - "Not USV" samples were mostly interference or broadband noise
   - Not random positions from quiet audio regions

### Why Batch Detection Fails

```
What batch detection asks:     What CNN learned:
"Is there a USV in this        "Does this look like
 arbitrary 40ms chunk?"    →    energy-detector output?"

Random audio chunk         →    Has acoustic structure
                          →    CNN says "Yes, USV!"
```

---

## Options Going Forward

### Option A: Use Energy Detector + CNN (Recommended for Quick Results)

This is what the pipeline was designed for:

```python
# Step 1: Energy detector finds candidate regions
candidates = energy_detector.detect(wav_file)  # Returns 10-500ms regions

# Step 2: CNN classifies each candidate
for candidate in candidates:
    spectrogram = extract_spectrogram(candidate)  # Centered on candidate
    prob = cnn.predict(spectrogram)
    if prob > 0.9:
        save_as_usv(candidate)
```

**Pros:**
- Works with current CNN (no retraining)
- Energy detector pre-filters to relevant regions
- This is the original design intent

**Cons:**
- Requires tuning energy detector parameters
- Two-stage pipeline more complex
- "Ruins the point of CNN" (per user feedback)

### Option B: Retrain CNN with Proper Negatives

Create a new training dataset with:

1. **Positive class (USV)**: Keep existing labeled USVs
2. **Negative class (Not USV)**: Add samples from:
   - Random positions in recordings (not just energy-detector candidates)
   - Quiet regions with no acoustic activity
   - Regions between known USVs
   - Different noise profiles

```python
# Generate new negative samples
for wav_file in wav_files:
    # Get known USV regions
    usv_regions = get_labeled_usv_regions(wav_file)

    # Sample random positions OUTSIDE USV regions
    for _ in range(samples_per_file):
        start_ms = random_position_outside(usv_regions)
        spectrogram = extract_chunk(wav_file, start_ms, duration_ms=40)
        save_as_negative(spectrogram)
```

**Pros:**
- CNN would learn true USV vs background discrimination
- Direct batch detection would work
- Simpler inference pipeline

**Cons:**
- Requires retraining (time, compute)
- Need to ensure negatives don't accidentally contain USVs
- May need more training data overall

### Option C: Use Labeled Dataset for Clustering (No Expansion)

Skip batch detection entirely and cluster the existing labeled USVs:

- You have 458 labeled USVs
- This is enough for meaningful clustering analysis
- Guaranteed quality (human-verified)

```powershell
# Just run clustering on existing labeled data
python scripts/extract_features.py --csv splits/train.csv --output analysis/clustering
python scripts/visualize_clusters.py --features analysis/clustering/features.npy
python scripts/run_clustering.py --features analysis/clustering/features.npy
```

**Pros:**
- Works immediately with existing data
- No false positives (all human-verified)
- Simpler, faster

**Cons:**
- Limited to 458 samples
- No expansion from unlabeled data
- May miss rare USV types

---

## Technical Details for Retraining (Option B)

If you choose to retrain the CNN, here's what the negative sampling should look like:

### Negative Sample Categories

| Category | Description | How to Generate |
|----------|-------------|-----------------|
| Quiet regions | No acoustic activity | Sample from low-energy time periods |
| Inter-USV gaps | Silence between USVs | Sample from gaps between labeled USVs |
| Random positions | Arbitrary audio chunks | Random start times, any energy level |
| Background noise | Recording artifacts | Sample from file start/end |

### Recommended Dataset Balance

```
Current dataset:
- USV: 458 samples
- Not USV: 374 samples (mostly energy-detector negatives)

Proposed expansion for Not USV:
- Keep existing 374 "Not USV" (energy-detector false positives)
- Add ~500 random quiet chunks
- Add ~300 inter-USV gap chunks
- Add ~200 background noise chunks

New dataset:
- USV: 458 samples
- Not USV: ~1,374 samples (3:1 ratio)
```

### Training Considerations

1. **Use class weighting** to handle imbalance
2. **Augmentation** for USV class (time shift, noise injection)
3. **Validation** should include both candidate-based and random negatives
4. **Test on batch detection** before deployment

---

## Files for Reference

| File | Description |
|------|-------------|
| `scripts/diagnose_cnn_batch_detection.py` | Diagnostic script that produced these results |
| `analysis/cnn_diagnosis/cnn_diagnostic_plot.png` | Visualization of probability distributions |
| `analysis/cnn_diagnosis/*.npy` | Raw probability arrays |
| `src/usv_spectrogram/detection/energy_detector.py` | Energy detector implementation |
| `scripts/train_cnn.py` | Training script (for Option B) |

---

## Conclusion

The CNN works as designed - it's a **candidate classifier** that confirms whether an energy-detector candidate is a real USV. It was never trained to find USVs in arbitrary audio chunks.

For batch detection to work, either:
1. Use Energy Detector first (as originally designed)
2. Retrain with proper negative samples
3. Skip expansion and cluster existing labeled data

The diagnostic plot in `analysis/cnn_diagnosis/cnn_diagnostic_plot.png` clearly shows the issue - random chunks get 0.997 mean probability while even labeled "Not USV" samples get 0.684.
