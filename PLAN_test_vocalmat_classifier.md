# Plan: Test VocalMat Classifier on 5970 Data

## Goal

Quick parallel experiment: run VocalMat's pre-trained CNN classifier on one session from the 5970 cohort and evaluate whether its classifications are usable for wild mouse USVs. This is a half-day test, not a commitment — if results are bad, drop it and move on.

## Why

VocalMat (Fonseca et al. 2021, eLife) ships a pre-trained CNN that classifies USV spectrogram patches into 12 classes (11 syllable types + noise). If it works on our wild mouse data, we get a supervised classifier for free without needing to solve the label noise problem ourselves. It was trained on lab strain data, so it probably won't transfer well — but it costs very little to check.

## Background

- VocalMat repo: https://github.com/ahof1704/VocalMat
- VocalMat is MATLAB-based. Requires: Signal Processing Toolbox, Deep Learning Toolbox, Image Processing Toolbox, Statistics and Machine Learning Toolbox.
- VocalMat expects 250 kHz sampling rate (Fmax = 125 kHz). Our data is 300 kHz — needs resampling.
- VocalMat taxonomy uses Grimsley et al. 2011 (12 types): short, flat, chevron, reverse chevron, down FM, up FM, complex, multi steps, two steps, step down, step up, noise.
- Our taxonomy uses Holy & Guo 2005 (7 types): Flat, Down, Chevron, Short, Complex, Frequency_Jump, Up.
- Rough mapping: VocalMat flat ≈ our Flat, VocalMat down FM ≈ our Down, VocalMat up FM ≈ our Up, VocalMat chevron ≈ our Chevron, VocalMat short ≈ our Short, VocalMat complex/multi steps ≈ our Complex, VocalMat step down/step up/two steps ≈ our Frequency_Jump. The mapping is approximate — don't force it to be exact.

## Steps

### 1. Setup VocalMat

- Clone the repo: `git clone https://github.com/ahof1704/VocalMat.git`
- Git LFS is required — the pre-trained model file is stored via LFS. If LFS isn't available, download the model manually from https://osf.io/3yc79/download and place it in `vocalmat_classifier/`.
- Verify MATLAB is available and has the required toolboxes. If MATLAB isn't available on this machine, flag it — we may need to run this on a different machine or find a Python alternative.

### 2. Prepare Input Data

- Pick ONE recording session from 5970 (the shortest one, to keep iteration fast).
- Find the raw WAV file for that session. Discover its location — look in the project directories for WAV files associated with 5970/lmt_034.
- Resample from 300 kHz to 250 kHz. This can be done with Python (librosa/scipy) or sox before feeding to VocalMat:
  ```
  # Example with sox (if available):
  sox input_300k.wav -r 250000 output_250k.wav
  
  # Example with Python:
  # scipy.signal.resample or librosa.resample
  ```
- Place the resampled WAV in VocalMat's `audios/` directory.

### 3. Run VocalMat End-to-End

- Run `VocalMat.m` in MATLAB, point it at the resampled WAV.
- VocalMat will produce:
  - An output directory with spectrogram images per detected vocalization
  - An Excel file with per-call metadata (time, frequency, duration, type classification, probability distribution across classes)
  - A `_DL.xlsx` file with the probability distribution across all 12 classes for each call

### 4. Evaluate Results

This is the critical step. We need to answer: **does VocalMat's classifier produce sensible labels on wild mouse data?**

#### 4a. Detection comparison
- Compare VocalMat's detected calls against our CNN's detections for the same session. How many calls does each find? What's the overlap? VocalMat may miss calls our CNN catches (or vice versa) because they use completely different detection methods.

#### 4b. Classification quality — visual spot check
- We already have a per-type spectrogram gallery (100 examples × 7 types = 700 PNGs). Take ~10-20 calls that VocalMat classified with high confidence in each of its categories and visually compare against our gallery. Do VocalMat's "chevron" calls look like our Chevron calls? Do its "flat" calls look like our Flat calls?

#### 4c. Classification quality — statistical check
- For calls that both pipelines detected (matched by timestamp overlap), cross-tabulate VocalMat's 12-class labels against our 7-class labels. Use the rough mapping above. A clean confusion matrix = good transfer. A scrambled matrix = the classifier doesn't generalize to wild mice.

#### 4d. Noise class utility
- How many calls does VocalMat label as "noise"? Cross-reference against our CNN's detection confidence. If VocalMat's noise class catches false positives that slipped through our CNN, that's independently useful even if the syllable classification is bad.

#### 4e. Confidence distribution
- From the `_DL.xlsx` probability distributions: are VocalMat's classifications confident (one class >> others) or uncertain (flat distribution across classes)? Systematic uncertainty = the model doesn't know what it's looking at = bad transfer.

### 5. Decision

Based on steps 4a–4e, one of three outcomes:

1. **Works well**: VocalMat's labels are sensible on wild mouse data → adopt as classifier, map to our taxonomy, validate on more sessions.
2. **Partially works**: Some types transfer, others don't. Noise class is useful → cherry-pick what works, don't use the rest.
3. **Doesn't work**: Labels are random or systematically wrong on wild mouse data → drop it, we've lost half a day, move on to the contour-mask → VAE pipeline.

Document the outcome with specific numbers and examples regardless of which path.

## Important Notes

- This is a PARALLEL experiment. It should NOT delay the contour-mask → VAE pipeline (see PLAN_contour_masked_vae_pipeline.md).
- Don't try to retrain or fine-tune VocalMat. The whole point is testing the pre-trained model as-is. If it doesn't work out of the box, it's not worth the investment.
- If MATLAB isn't readily available, check whether someone has ported VocalMat's classifier to Python, or whether the model weights can be loaded in Python via `scipy.io.loadmat`. Don't spend more than an hour on environment setup — if it's painful, skip the test.
- The resampling from 300→250 kHz slightly compresses the frequency axis (our 120 kHz upper band becomes ~100 kHz effective after resampling). This shouldn't matter much for USVs in the 30–90 kHz range but could affect very high frequency calls.
