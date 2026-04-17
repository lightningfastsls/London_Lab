# Lab Presentation — USV Detection & Analysis Pipeline

## Overarching Question
Has decades of inbreeding changed courtship behavior in lab mice? More specifically: do the USVs of lab mice and wild mice differ?

---

## Presentation Flow (from Mickey's guidance, April 15 2026)

### Act 1: "Why I built my own tool"

**1. Start with the raw signal**
- Show what a WAV file looks like — raw waveform, time-domain
- Show Audacity — the standard tool everyone knows
- Show spectrogram, point to where USVs live (30–110 kHz)

**2. The labeling problem**
- "For you it's easy to see a USV on a spectrogram — but a neural network needs thousands of labeled examples"
- Demo the custom PyQt6 labeling app: "I built my own Audacity"
- Show dual-view: 0–30 kHz audible range alongside USV range
- Walk through: raw WAV → spectrogram → labeled training data

**3. Training data journey**
- 15,444 labeled windows (5,790 USV + 9,654 non-USV)
- Three-stage negative sampling: random positions, inter-USV gaps, low-energy regions
- Hard-negative mining: 620 noise misclassifications + 144 missed USVs

### Act 2: "The CNN"

**4. Architecture walkthrough**
- Show mid-C CNN (~207K params) visually
- Accessible: "A neural network learns a function — input a spectrogram window, output a probability: how likely is this a USV?"

**5. Signal detection theory (optional deep-dive)**
- FP vs FN tradeoffs
- Precision 90.55%, recall 98.7%, ROC AUC 0.989
- Threshold tuning: sensitivity vs. specificity

**6. Cross-validation with DeepSqueak**
- Completely separate system (different architecture, training data, lab)
- 95.6% of our detections confirmed
- Agreement = strong evidence these are real USVs

### Act 3: "What the data shows"

**7. Detection results**
- Cage 5970: 7,575 detections (~32 hours)
- Cage 3452: ~456 detections (16× fewer)

**8. Classification**
- 7 Scattoni types: Flat 32%, Down 17%, Chevron 16%, Short 14%, Complex 9%, Freq Jump 7%, Up 6%
- DeepSqueak k-means (27 clusters) — likely over-split
- **UMAP + HDBSCAN**: continuous manifold, 97% in one cluster. Traditional categories = labels on a continuum.

**9. UMAP visualization (Mickey's specific requests)**
- Overlay spectrogram examples from each cluster
- Color-code by K-means cluster assignments
- **These need to be created before the talk**

**10. Temporal dynamics & sequence analysis**
- Bursty calling (peak 1,089/hour, 3 silent hours)
- Self-repetition dominant (25.7% vs 20.0% expected)
- Context reduces uncertainty by only 3.7%
- "Sticky states" model

### Act 4: "Where this is going"

**11. Hertz et al. 2020 (this lab)**
- SIS/SIM: temporal structure helps syllable labeling
- Our analysis confirms structure, but modest with 7 coarse types
- Next: autoencoder or SIM for finer categories

**12. Lab vs. wild comparison**
- Awaiting lab data from Mickey → run same pipeline → compare everything

---

## Existing Figures (from progress report)
- Syllable type distribution (n=7,921)
- Hourly call rate across 32 hours
- Syllable composition over time (stacked area)
- ICI distribution with bout threshold
- Transition probability heatmap P(B|A)
- Entropy rate convergence (n-gram 1–5)
- Mutual information at lags 1–10
- Rank-frequency (Zipf) distribution

## Figures to Create
- CNN architecture diagram (visual, accessible)
- Labeling app screenshot
- Raw WAV → spectrogram → detection visual pipeline
- UMAP with spectrogram thumbnails per cluster
- UMAP color-coded by K-means clusters
