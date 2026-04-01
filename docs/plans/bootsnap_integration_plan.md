# Plan: Integrate BootSnap Pretrained Classifier with USV Detection Pipeline

## Context
Shachar has a CNN-based USV detection pipeline that identifies mouse ultrasonic vocalizations from audio recordings. He needs syllable-level classification (typing each detected USV into categories like 'simple', 'complex', 'frequency-jump', etc.) for comparing courtship vocalizations between wild field mice and laboratory mice. Deadline: May presentation.

**BootSnap** (Abbasi et al., 2022, PLOS Comp Bio) is a pretrained CNN ensemble classifier for mouse USV syllables that was specifically validated on both wild-derived and lab mice. It classifies into ~12 categories including: 'c' (complex), 'h' (complex+harmonics), 'c2' (2 jumps), 'c3' (3+ jumps), 'up', 'd' (down), 'ui' (up-inverted), 's' (short), 'f' (flat), 'u' (unstructured), 'us' (ultra-short), 'composite', and 'FP' (false positive — useful for denoising).

Repo: https://github.com/ReyhanehAbbasi/BootSnap  
License: Apache-2.0  
Language: Python

## Goal
Use BootSnap's **pretrained** classifier (not retrain) to classify USVs that Shachar's CNN detector has already identified. This is the fastest path to syllable-typed data for the May presentation.

---

## Step 1: Clone and explore the BootSnap repo

```bash
git clone https://github.com/ReyhanehAbbasi/BootSnap.git
```

Explore the directory structure. Key directories:
- `applying_classifier_on_EV_data/` — THIS IS THE ONE WE WANT. Contains code for applying a pretrained model to new data.
- `train_classifier/` — For training from scratch (skip for now, but note for later).

**Action items:**
- Read the README.md thoroughly
- List all files in both directories
- Identify the main entry-point script in `applying_classifier_on_EV_data/`
- Check for any pretrained model weights (`.h5`, `.pt`, `.pkl`, `.pth`, or similar files) — they may be in the repo or may need to be downloaded separately
- Identify required Python dependencies (look for `requirements.txt`, `setup.py`, or imports at the top of scripts)
- Check what Python version / framework is used (likely TensorFlow/Keras or PyTorch based on the paper's CNN description)

## Step 2: Understand BootSnap's expected input format

From the paper, BootSnap expects:
1. **Audio segments** of detected USVs (time-windowed)
2. Converted to **Gammatone spectrograms** (NOT standard STFT spectrograms)
   - Gammatone filter bank with ~128 filters
   - Center frequency midpoint optimized at 68 kHz
   - Frequency range: 20 kHz to 120 kHz
3. Time-frequency windowed to the region of interest around each detected USV

**Action items:**
- Read the preprocessing/spectrogram generation code in the repo
- Document the exact input tensor shape expected by the classifier (height × width × channels)
- Document any normalization or scaling applied to the spectrograms
- Check if A-MUD detection output format is expected (timestamp + frequency bounds) or if any detection format works
- Identify how USV segments are extracted from full audio files

## Step 3: Map Shachar's CNN detector output to BootSnap input

Shachar's existing pipeline outputs detected USVs with:
- Start time and end time (or start time + duration)
- Frequency bounds (min/max frequency of the bounding box)
- Source audio file path

We need a bridge script that:
1. Takes Shachar's detection output (format TBD — check his existing pipeline output format)
2. For each detected USV, extracts the corresponding audio segment from the source .wav file
3. Computes the Gammatone spectrogram using BootSnap's preprocessing code
4. Feeds it to the pretrained classifier
5. Outputs: original detection info + predicted syllable category + confidence score

**Action items:**
- Check Shachar's current detection output format (CSV? JSON? .mat files? Ask him if unclear)
- Write a bridge/adapter module that converts his format → BootSnap's expected input
- Handle edge cases: very short USVs, USVs near file boundaries, overlapping USVs

## Step 4: Run BootSnap on a small test batch

Before running on the full dataset:
1. Select ~100 detected USVs from both wild and lab mice recordings
2. Run them through the BootSnap pipeline
3. Manually verify a subset of classifications against spectrograms (sanity check)
4. Check the distribution of predicted categories — does it look reasonable?
5. Check that the 'FP' class catches actual noise detections (validates denoising capability)

**Action items:**
- Create a test script that processes a small batch
- Generate a summary table: USV_id, predicted_class, confidence, source_file, mouse_type (wild/lab)
- Visually inspect ~20 classified USVs to verify the labels make sense
- Note any systematic issues (e.g., all USVs classified as one type, low confidence scores)

## Step 5: Run on full dataset and generate comparison data

Once validated:
1. Run BootSnap classifier on ALL detected USVs from both wild and lab mouse recordings
2. Output a master CSV with columns: file, usv_id, start_time, end_time, freq_min, freq_max, predicted_class, confidence, mouse_type (wild/lab)
3. Generate summary statistics:
   - Syllable repertoire distribution per group (wild vs. lab)
   - Proportion of each syllable type
   - Any notable differences in complexity (simple vs. complex call ratios)

**Action items:**
- Batch processing script for full dataset
- Summary statistics and basic visualizations (bar charts of syllable distributions)
- Export results in a format suitable for statistical analysis

## Step 6: Assess and document limitations

Important caveats to note for the presentation:
- BootSnap was trained on specific wild-derived and lab mouse strains — performance on Shachar's specific mice may differ
- The pretrained model was not fine-tuned on Shachar's recording setup/conditions
- The 'FP' class predictions could be used to flag questionable detections from the CNN pipeline
- Classification confidence scores should be reported — low-confidence predictions could be excluded or flagged

---

## Dependencies to install (likely needed — verify from repo)
```bash
pip install numpy scipy matplotlib tensorflow keras librosa  # or pytorch
pip install gammatone  # or check if BootSnap includes its own implementation
pip install scikit-learn pandas
```

## Key questions for Shachar (ask before starting if unclear)
1. What is the exact output format of your CNN detector? (CSV columns, file format)
2. Where are the raw .wav audio files stored relative to the detection outputs?
3. What is the sampling rate of your recordings? (likely 250 kHz but confirm)
4. Do you want to filter by confidence threshold before classification?
5. Are your detection files organized by mouse/session already, or is there a metadata file mapping files → wild/lab?

## Risk mitigation
- **If pretrained weights are missing from the repo**: Email Reyhaneh Abbasi (reyhaneh.abbasi@oeaw.ac.at) requesting the trained model files. Mention the PLOS Comp Bio paper and that you're using it for wild vs. lab mouse USV comparison.
- **If input format is incompatible**: Write a custom Gammatone spectrogram generator based on the paper's specifications (filter order, center frequencies, bandwidth equations).
- **If classification quality is poor on your data**: Fall back to using BootSnap's broader category groupings (e.g., 'no-jump' vs 'jumps' vs 'FP' — 3 classes instead of 12) which showed ~95% F1 in the paper.
- **If Python version conflicts arise**: Create a dedicated conda environment for BootSnap.
