# USV Signal Processing: Subtleties & Considerations Reference

A collection of nuances, trade-offs, and practical considerations for USV detection pipeline development.

---

## Module 1: Signal Processing Foundations

### 1.1 The Fundamental Time-Frequency Trade-off

**The Problem:**
You cannot have perfect frequency resolution AND perfect time resolution simultaneously. This is a manifestation of the uncertainty principle.

**Why It Happens:**
To know a frequency precisely, you need to observe multiple cycles of the wave. While you're observing, time passes—you're averaging over that duration.

**Practical Impact:**
| n_fft (at 250 kHz) | Frequency Resolution | Time Resolution |
|--------------------|---------------------|-----------------|
| 256                | ~977 Hz             | ~1 ms           |
| 512                | ~488 Hz             | ~2 ms           |
| 1024               | ~244 Hz             | ~4 ms           |
| 2048               | ~122 Hz             | ~8 ms           |

**Solution/Consideration:**
- For USVs, start with n_fft=512 at 250 kHz sample rate (~488 Hz frequency resolution, ~2 ms time resolution)
- This is a reasonable compromise for most USV work
- Accept that shortest calls (<15 ms) may be somewhat smeared
- Use Parameter Lab to experiment with your specific data

---

### 1.2 Window Functions and Edge Effects

**The Problem:**
STFT windows use tapering functions (Hanning, Hamming) that weight center samples more than edge samples. A call at the edge of a window contributes less than a call at the center.

**Practical Impact:**
- A short call positioned at window edge may appear weaker than it actually is
- The same call might appear with different intensities in adjacent frames

**Solution/Consideration:**
- Use sufficient overlap (hop_length = n_fft/4 gives 75% overlap)
- With good overlap, if one window catches a call poorly, adjacent windows catch it better
- Don't rely on single-frame intensity for detection decisions

---

### 1.3 Window Length vs. Call Duration

**The Problem:**
When your analysis window is longer than a call, the call gets "smeared" with surrounding silence or noise.

**Example:**
A 15 ms call in a 20 ms window gets mixed with 5 ms of whatever comes before and after.

**Practical Impact:**
- Short calls appear weaker (energy diluted by silence)
- Very short calls can disappear entirely in long windows
- Call boundaries become blurry

**Solution/Consideration:**
- Choose window length appropriate for shortest calls you want to detect
- For 10 ms minimum USV duration, window should be ≤10 ms (n_fft ≤ 2500 at 250 kHz)
- n_fft=512 gives ~2 ms window—safely shorter than shortest USVs

---

### 1.4 Hop Length and Temporal Smoothness

**The Problem:**
Hop length determines how much windows overlap. Too large = jerky time axis, may miss brief events. Too small = excessive computation, redundant data.

**Typical Setting:**
hop_length = n_fft / 4 (75% overlap)

**Solution/Consideration:**
- 75% overlap is standard starting point
- For detection (not visualization), you might use 50% overlap to reduce computation
- For very short calls, higher overlap ensures you don't miss them between frames

---

## Module 2: USV Characteristics

### 2.1 The Chevron Detection Challenge

**The Problem:**
Chevron calls are both short (need good time resolution) AND have rapid frequency changes (need good frequency resolution to see the shape). These requirements conflict.

**Practical Impact:**
- Small n_fft: You detect the call exists but frequency trajectory looks blurry/stepped
- Large n_fft: Frequency shape is clear but short calls get smeared or missed

**Solution/Consideration:**
- For binary yes/no classification, this is less critical—you just need to detect existence
- Use moderate n_fft (512) as compromise
- Accept that you may not perfectly resolve call shape for shortest chevrons
- If call-type classification becomes important later, consider multi-resolution analysis

---

### 2.2 Harmonics: One Call or Two?

**The Problem:**
Some USVs have harmonics—energy at integer multiples of fundamental frequency (e.g., 45 kHz fundamental + 90 kHz harmonic). A naive detector might count this as two calls.

**Practical Impact:**
- Double-counting calls inflates call counts
- Harmonic might be detected when fundamental is missed (or vice versa)
- Different mice/call types have different harmonic content

**Solution/Consideration (for yes/no classification):**
- Simplest: Detect based on fundamental, ignore harmonics
- If harmonic detected separately, merge nearby detections
- Define detection regions to encompass both fundamental + harmonic
- Don't need perfect harmonic handling for binary classifier—just avoid double-counting

**When Harmonics Help:**
- Clean harmonics are a sign of real vocalization (noise rarely has clean harmonics)
- Can use harmonic presence as a feature for classification

---

### 2.3 The 60 kHz (and Other Perfect Frequencies) Red Flag

**The Problem:**
Perfectly stable frequencies at "round" numbers often indicate electrical interference, not biological signals.

**Common Culprits:**
- 60 Hz (AC power in North America) and its harmonics (120, 180... 60000 Hz)
- 50 Hz (AC power in Europe) and its harmonics
- Equipment-specific frequencies

**Characteristics of Interference vs. USVs:**
| Feature | Electrical Interference | Real USV |
|---------|------------------------|----------|
| Frequency | Perfectly stable | Slight wobble/modulation |
| Duration | Very long (seconds+) | Short (10-300 ms) |
| Pattern | Continuous or regular | Irregular, clustered |

**Solution/Consideration:**
- Filter out known interference frequencies if consistent in your setup
- Flag/reject detections at exact harmonic frequencies of 50/60 Hz
- Look for unnaturally long durations as a rejection criterion
- Check your recording environment for sources of interference

---

### 2.4 Minimum Duration Filtering

**The Problem:**
Very short "detections" (3-5 ms) are almost certainly noise—transient clicks, electrical spikes, random fluctuations.

**Practical Impact:**
- High false positive rate if no duration filter
- Wastes labeling effort on obvious non-USVs

**Solution/Consideration:**
- Implement minimum duration threshold (e.g., 8-10 ms)
- USVs shorter than 10 ms are rare; rejecting them loses little true signal
- Can be adjusted based on your specific data characteristics

---

### 2.5 Microphone Frequency Response

**The Problem:**
Ultrasonic microphones have varying sensitivity across frequencies. Cheap/old mics often roll off above 70-80 kHz, making high-frequency calls appear faint or invisible.

**Practical Impact:**
- May miss high-frequency calls entirely
- Frequency distribution of detected calls may not reflect true distribution
- Comparing data across different recording setups is problematic

**Solution/Consideration:**
- Check your microphone's frequency response specification before recording
- If response rolls off, consider:
  - Different microphone
  - Adjusting detection thresholds by frequency band
  - Noting limitation in your methods
- Be cautious comparing your data to literature if equipment differs

---

### 2.6 Sample Rate and Nyquist Limit

**The Problem:**
Nyquist theorem: sample rate must be at least 2× the highest frequency of interest. For 110 kHz calls, need ≥220 kHz sample rate.

**Practical Impact:**
- Too low sample rate = aliasing (high frequencies appear as false low frequencies)
- Example: 150 kHz sample rate aliases everything above 75 kHz

**Solution/Consideration:**
- Use 250 kHz or higher for USV recording
- Check existing recordings for sample rate before analysis
- If sample rate is too low, high-frequency calls are unrecoverable

---

### 2.7 Overlapping Calls (Multi-Animal Recordings)

**The Problem:**
With 2+ mice, calls can overlap in time, creating:
- Two frequency tracks simultaneously on spectrogram
- Crossing or interleaved calls
- Ambiguity about which mouse produced which call

**Practical Impact (for your wild vs. lab mouse study):**
- Call counts may be underestimated if overlapping calls detected as one
- Or overestimated if one complex call detected as multiple
- Cannot attribute calls to specific individuals without additional methods

**Solution/Consideration (for yes/no classification):**
- For binary classifier training: overlapping calls are still USVs—label as positive
- Accept some ambiguity in call counts
- Consider:
  - Treating overlapping calls as one detection event
  - Defining "detection" as "USV present in this time window" rather than counting discrete calls
  - If individual attribution needed, requires different approach (e.g., multiple mics, source separation)

---

### 2.8 Wild vs. Lab Mouse Generalization

**The Problem:**
Wild mice might have different USV characteristics than lab strains—different frequency ranges, call type distributions, durations. A classifier trained on one population may not generalize to the other.

**Practical Impact:**
- Training only on lab mouse data, then testing on wild mice (or vice versa) may give poor results
- Detection thresholds optimized for one population may miss calls from the other

**Solution/Consideration:**
- Investigate existing literature on wild mouse USVs before starting
- Include examples from BOTH populations in training data
- Validate classifier separately on each population
- Consider population as a factor in your experimental design

---

### 2.9 Distinguishing USVs from Noise: The Feature Checklist

**Characteristics of True USVs:**
- Clean edges (distinct start and end)
- Coherent frequency structure (smooth trajectory)
- Appropriate duration (10-300 ms, most 30-100 ms)
- Within expected frequency band (30-110 kHz)
- Often has slight frequency modulation (not perfectly flat)

**Characteristics of Noise:**
- Fuzzy or irregular boundaries
- Broadband energy (vertical smear on spectrogram)
- Too regular (exactly 50 or 60 kHz)
- Too short (<10 ms) or too long (>500 ms)
- No coherent frequency structure

**Solution/Consideration:**
- Use multiple features for detection, not just energy threshold
- The more features that match USV profile, the higher confidence
- Consider these features when designing your ML classifier's input

---

## Quick Reference: Recommended Starting Parameters

For 250 kHz sample rate recordings:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_fft | 512 | ~488 Hz freq resolution, ~2 ms time resolution |
| hop_length | 128 | 75% overlap, smooth temporal coverage |
| High-pass filter | 25 kHz | Remove sub-ultrasonic noise |
| Min detection duration | 10 ms | Reject transient artifacts |
| Max detection duration | 500 ms | Reject continuous interference |
| Frequency range | 30-110 kHz | Standard mouse USV range |

These are starting points—adjust based on your specific data and needs.

---

## Questions to Answer Before Building Your Pipeline

1. What is your microphone's frequency response?
2. What is your recording sample rate?
3. What is the shortest call you need to detect?
4. Will you have single or multiple animals per recording?
5. Do you need to count individual calls, or just detect presence/absence?
6. Do you have example recordings from both wild and lab mice?

---

---

## Module 3: Detection Approaches

### 3.1 Recall vs. Precision Trade-off at Candidate Generation

**The Problem:**
At the initial detection stage (energy-based detector), you must choose between catching all USVs (high recall) vs. minimizing false positives (high precision). You can't optimize both simultaneously.

**Why It Matters:**
Missing a class of USVs at the candidate stage means your final ML model can never learn to detect them. You can't recover from data you never collected. False positives, while annoying, are correctable through labeling.

**Practical Impact:**
- Threshold too high → miss quiet/unusual USVs → biased training data → biased model
- Threshold too low → overwhelming number of candidates → labeling becomes impractical

**Solution/Consideration:**
- At candidate generation, optimize for HIGH RECALL (sensitivity)
- Accept higher false positive rate (aim for 30-50% true USVs in candidate pool)
- The labeling step is where you separate true USVs from false positives
- If false positive rate exceeds ~70%, consider adding simple feature filters (bandwidth, duration) to reduce noise while maintaining recall

---

### 3.2 Training Data Bias from Detector Thresholds

**The Problem:**
If your energy detector uses a threshold that misses certain USV types (quiet calls, unusual frequencies), your candidate pool never includes them. You label what the detector gives you. Your training data therefore lacks these examples—or worse, they're labeled as negatives.

**Practical Impact:**
- Model "learns" that quiet signals aren't USVs (because training data said so)
- Model fails on exactly the cases your detector missed
- This is invisible until you test on diverse data

**Example:**
Energy threshold misses USVs below -40 dB → training data has no quiet USVs → CNN learns "quiet = not USV" → CNN fails on quiet calls in new recordings

**Solution/Consideration:**
- Use a deliberately LOW energy threshold for candidate generation
- Manually review some below-threshold regions to verify you're not missing call types
- Include examples from different recording conditions in training data
- Test trained model on recordings with varying SNR

---

### 3.3 Class Imbalance: Random Patches vs. Detector Candidates

**The Problem:**
Random spectrogram patches from recordings are mostly silence/noise (maybe 1-5% contain USVs). Labeling random patches is inefficient—you label 500 patches but get only ~25 positive examples.

**Practical Impact:**
- Wasted labeling effort on obvious negatives
- Insufficient positive examples for training
- Class imbalance in training data

**Solution/Consideration:**
- Label detector-flagged candidates, not random patches
- Candidates are enriched for USVs (30-50% true positives)
- 500 labeled candidates → ~200 positive examples vs. ~25 from random sampling
- This is 8× more efficient use of labeling time

---

### 3.4 Class Imbalance Between Populations

**The Problem:**
In your study, you have 10 hours of lab mice but only 2 hours of wild mice. Proportional labeling creates training data that's ~83% lab mouse calls. The model learns lab mouse patterns much better.

**Practical Impact:**
- Overall accuracy looks good (driven by majority class)
- Performance on minority class (wild mice) may be poor
- If wild mice have unique call types, model may miss them entirely

**Solution/Consideration:**
1. **Oversample minority class** — Label ALL wild mouse candidates, only subset of lab mouse candidates
2. **Stratified evaluation** — Always report accuracy separately for each population
3. **Prioritize minority labeling** — Each wild mouse example is more valuable; label those first
4. **Data augmentation** — Augment underrepresented class (pitch shift, noise addition, time stretch)
5. **Collect more data** — If feasible, 2 hours may not be enough for wild mice

**Critical Question:**
Do wild mice have call types that lab mice don't produce? If yes, no amount of lab mouse data helps—you need wild mouse examples of those types. Investigate literature before heavy labeling investment.

---

### 3.5 Domain Shift Between Training and Deployment

**The Problem:**
Your model trains on data from one set of conditions but deploys on data from different conditions. Even if the model is perfect on training-like data, it may fail on shifted data.

**Common Domain Shifts:**
| Factor | Training | Deployment |
|--------|----------|------------|
| SNR | High (clean recordings) | Low (noisy environment) |
| Microphone | Mic A | Mic B (different frequency response) |
| Recording setup | Anechoic chamber | Reverberant cage |
| Animal population | Lab mice | Wild mice |

**Practical Impact:**
- 95% test accuracy but poor real-world performance
- Model fails silently (you don't know it's wrong)

**Solution/Consideration:**
- Include diverse recording conditions in training data
- Test on held-out data from each condition separately
- If deploying to new conditions, collect and label some examples from that condition
- Be suspicious of high accuracy on homogeneous test sets

---

### 3.6 Energy Detection Limitations

**The Problem:**
Pure energy-based detection (threshold on total power in frequency band) cannot distinguish between signals with similar energy but different spectral characteristics.

**What Energy Detection Cannot Distinguish:**
| Signal Type | Energy | True USV? |
|-------------|--------|-----------|
| 50 kHz tone, 50 ms | High | Yes |
| Broadband click, 5 ms | High | No |
| Continuous electrical hum | Very High | No |
| Quiet frequency sweep | Low | Yes (but missed) |

**Solution/Consideration:**
- Energy detection is a starting point, not a final solution
- Add feature filters to improve candidate quality:
  - Spectral bandwidth (USVs are narrowband)
  - Spectral flatness (USVs are tonal, not noise-like)
  - Duration (reject < 10 ms, > 500 ms)
- These additions maintain recall while improving precision

---

### 3.7 Feature Engineering Ceiling

**The Problem:**
In classical ML (SVM, Random Forest), performance is limited by your choice of features. If the distinguishing pattern isn't captured by your features, the classifier can't learn it.

**Example:**
You extract: energy, bandwidth, flatness, duration
A USV type is only distinguishable by its frequency contour shape
Contour features not included → classifier can't learn to detect that type

**Practical Impact:**
- Diminishing returns from algorithm tuning if features are inadequate
- May need domain expertise to design good features
- Easy to miss important features you didn't think of

**Solution/Consideration:**
- This is why deep learning (CNNs) became attractive: network learns its own features
- If using classical ML, iterate on features based on error analysis
- Look at misclassified examples and ask "what feature would distinguish this?"

---

### 3.8 The Training Data Quality Principle

**The Core Insight:**
> Training data quality matters more than algorithm complexity.

A simple CNN with 1000 perfectly labeled examples will outperform a complex architecture with 10,000 noisy labels.

**Practical Impact:**
- Time spent on labeling quality has higher ROI than algorithm tweaking
- Garbage in, garbage out—noisy labels create confused models
- The labeling tool (GUI) is as important as the detection algorithm

**Solution/Consideration:**
- Invest in your labeling workflow (clear interface, keyboard shortcuts, consistency)
- Consider inter-rater reliability if multiple people label
- Review a sample of labels periodically for quality control
- When in doubt, don't label (uncertain examples can be excluded)

---

### 3.9 Labeling Efficiency: The Practical Bottleneck

**The Problem:**
The ML pipeline is only as fast as your slowest step. Labeling is almost always the bottleneck—it requires human time and attention that can't be parallelized easily.

**Typical Time Costs:**
| Step | Time |
|------|------|
| Running energy detector | Minutes |
| Labeling 500 candidates | Hours |
| Training CNN | Minutes to hours |
| Inference on new data | Seconds |

**Solution/Consideration:**
- Optimize labeling interface ruthlessly (keyboard shortcuts, batch operations)
- Pre-sort candidates by confidence (review uncertain cases, auto-accept high-confidence)
- Consider active learning: train preliminary model, have it flag uncertain cases for human review
- Your GUI 10.py tool is critical infrastructure—worth investing time to make it efficient

---

### 3.10 Energy Computation: Peak vs Mean Mode

**The Problem:**
When computing energy per frame in the USV frequency band, should you use mean or max (peak) energy across frequency bins? The choice significantly affects detection of narrow-band USVs.

**Mean Energy Mode (Original Approach):**
- Averages energy across all frequency bins in the band
- Works well when USV energy is spread across multiple bins
- **Weakness:** Narrow-band USVs (energy concentrated at one frequency) get diluted by low-energy bins
- A strong 50 kHz tone gets averaged with near-zero energy at 30-45 kHz and 55-110 kHz

**Peak Energy Mode (Improved Approach):**
- Uses maximum energy in the band per frame
- Detects signals with energy concentrated at a single frequency
- Better for narrow-band USVs that are characteristic of mouse vocalizations
- **Weakness:** May be more sensitive to single-bin noise spikes (mitigated by bandwidth filter)

**Empirical Results (from 300 kHz recordings):**
| Metric | Mean Mode | Peak Mode |
|--------|-----------|-----------|
| Total candidates | 256 | 490 |
| Missed narrow-band USVs | Many | Few |
| False positives (broadband noise) | Some | Rejected by bandwidth filter |

**Solution/Consideration:**
- Use **peak energy mode** as default for USV detection
- Combine with bandwidth filter to reject broadband noise (see 3.11)
- The combination maintains high recall while improving precision

---

### 3.11 Bandwidth Filtering for Noise Rejection

**The Problem:**
Broadband noise (clicks, electrical interference, equipment noise) can have high peak energy but lacks the narrow-band signature of USVs. How do you distinguish them?

**USV Bandwidth Characteristics:**
- Real USVs typically span 5-15 kHz bandwidth at any given moment
- Even frequency-modulated sweeps maintain narrow instantaneous bandwidth
- Noise often has energy spread across 50+ kHz simultaneously

**Implementation:**
1. At the peak frame of each candidate, measure bandwidth at -10 dB from peak
2. If bandwidth exceeds threshold (e.g., 20 kHz), reject as broadband noise
3. Only check at peak frame—checking across entire segment picks up noise from non-peak frames

**Critical Detail:**
Check bandwidth at **peak frame only**, not across the entire candidate segment. A USV surrounded by noise will have narrow bandwidth at its peak frame but wide bandwidth if measured across all frames.

**Example:**
```
Frame analysis at candidate peak:
- Peak energy: +5 dB at 70 kHz
- -10 dB threshold: -5 dB
- Frequencies above -5 dB: 68-72 kHz (4 kHz bandwidth)
→ Passes filter (4 kHz < 20 kHz threshold)

Broadband noise candidate:
- Peak energy: +3 dB at 45 kHz
- Frequencies above -7 dB: 30-100 kHz (70 kHz bandwidth)
→ Rejected (70 kHz > 20 kHz threshold)
```

**Recommended Settings:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| max_bandwidth_hz | 20,000 | Rejects broadband noise, keeps most USVs |
| Bandwidth threshold | -10 dB from peak | Standard definition of occupied bandwidth |

---

## Updated Quick Reference: Detection Pipeline Parameters

| Stage | Parameter | Recommended | Rationale |
|-------|-----------|-------------|-----------|
| Energy detector | Threshold | Low (high recall) | Don't miss USV types |
| Energy detector | Energy mode | Peak | Better for narrow-band USVs |
| Energy detector | Max bandwidth | 20 kHz | Reject broadband noise |
| Energy detector | Min duration | 8-10 ms | Reject transient artifacts |
| Energy detector | Max duration | 500 ms | Reject continuous interference |
| Candidate extraction | Window size | 300-500 ms | Show call with context |
| Labeling | Target pool composition | 30-50% true USVs | Efficient but not overwhelming |
| Training data | Min examples per class | 200+ | Rough minimum for CNN |
| Training data | Class balance | Roughly equal | Or use class weights |
| Evaluation | Test set | Separate from training | Never test on training data |
| Evaluation | Stratification | By population, condition | Catch domain shift issues |

---

---

## Module 4: Training Data Preparation

### 4.1 Class Imbalance Causes Biased Learning

**The Problem:**
If your labeled data has many more negatives than positives (e.g., 350 "not USV" vs. 50 "USV"), the model learns that "not USV" is almost always the right answer.

**Why It Happens:**
- Model can achieve 87.5% accuracy by always predicting majority class
- During training, gradients are dominated by the majority class (7× more examples)
- Model becomes good at recognizing noise, mediocre at recognizing USVs

**Practical Impact:**
- High overall accuracy but poor recall on USVs
- Model misses the thing you actually care about detecting

**Solution/Consideration:**
| Strategy | How It Works |
|----------|--------------|
| Undersample majority | Randomly select 50-100 negatives to match positives |
| Oversample minority | Duplicate or augment USV examples |
| Class weights | Tell model that USV errors are 7× more costly |
| Adjust detector | Produce better-balanced candidates upstream |

**Target Ratio:** Aim for 1:1 to 1:3 (positives:negatives). Some imbalance is fine; 1:7+ is problematic.

---

### 4.2 Split by Recording, Not by Candidate

**The Problem:**
Candidates from the same recording are correlated. Adjacent candidates may show the same call or very similar noise patterns. If these appear in both training and test sets, you're testing on nearly-duplicate data.

**Example of Wrong Approach:**
```
All candidates randomly shuffled → 70/15/15 split
lab_mouse_01_00142.png (training) and lab_mouse_01_00145.png (test) 
are 3 ms apart, nearly identical
```

**Practical Impact:**
- Test accuracy is artificially inflated
- Model appears to generalize but actually memorized recording-specific patterns
- Real-world performance will be worse than test metrics suggest

**Solution/Consideration:**
```
Correct: Split at recording level first
lab_mouse_01.wav, lab_mouse_02.wav → training (all their candidates)
lab_mouse_03.wav → validation (all their candidates)
lab_mouse_04.wav → test (all their candidates)
```

No recording should appear in multiple splits.

---

### 4.3 Validation vs. Test Accuracy Discrepancy

**The Problem:**
Your validation accuracy is 94% but test accuracy is 78%. The same model performs very differently on two held-out datasets.

**Possible Causes:**

| Cause | Explanation |
|-------|-------------|
| Split by candidate, not recording | Validation has near-duplicates of training data |
| Hyperparameter tuning on validation | Each adjustment indirectly "trains" on validation |
| Non-representative validation set | Validation happened to have easier examples |
| Different population distribution | Validation is all lab mice, test includes wild mice |

**Solution/Consideration:**
- Test set accuracy is the truth—trust it over validation
- If discrepancy is large, investigate your split methodology
- Ensure validation and test have similar characteristics (stratification)
- Limit hyperparameter tuning iterations to avoid overfitting to validation

---

### 4.4 Augmentation Overfitting Trap

**The Problem:**
Heavy augmentation (e.g., 5× or 10×) creates many variants of each real example. The model sees the "same" call repeatedly with minor modifications and memorizes specific calls rather than learning general patterns.

**Symptoms:**
- Training accuracy approaches 99%
- Validation accuracy plateaus or decreases
- Large gap between training and validation performance

**Practical Impact:**
- Model appears to learn well (high training accuracy)
- Actually just memorized the augmented training set
- Won't generalize to new data

**Solution/Consideration:**
- Watch validation accuracy, not training accuracy
- Right amount of augmentation: validation accuracy improves
- Too much augmentation: validation accuracy stagnates or drops
- Typical safe range: 2-5× augmentation
- Never augment validation or test sets

---

### 4.5 Intra-Rater Consistency (Solo Labeler Drift)

**The Problem:**
Even a single labeler can be inconsistent over time. What you call a USV on day 1 might differ from day 5 after seeing hundreds of examples. Your internal criteria drift.

**Why It Happens:**
- Fatigue changes decision thresholds
- Exposure to many examples shifts your "prototype" of a USV
- Ambiguous cases get resolved differently on different days

**Practical Impact:**
- Early labels and late labels have systematic differences
- Model learns inconsistent patterns
- Harder to diagnose than multi-labeler disagreement (no one to compare with)

**Solution/Consideration:**
- **Calibration exercise:** Label 50 examples, take a break, re-label the same 50, check agreement with yourself
- **Labeling guide:** Create visual examples of clear USV, clear noise, borderline cases—refer back to it
- **Periodic review:** Every 100 labels, review a random sample of previous labels
- **Session limits:** Don't label for more than 1-2 hours continuously

---

### 4.6 High Uncertainty Rate as Diagnostic Signal

**The Problem:**
If you're marking 30-40%+ of candidates as "uncertain," something is wrong upstream. This wastes labeling effort and produces an unusable dataset.

**Possible Causes:**

| Cause | Solution |
|-------|----------|
| Detector threshold in "gray zone" | Adjust threshold or add feature filters |
| Unclear labeling criteria | Create visual guide with edge case examples |
| Poor spectrogram visualization | Improve color scale, zoom, frequency range |
| Overly cautious labeling | Accept that some errors are okay; model is robust to a few mistakes |
| Genuinely ambiguous data | May need different detection approach or better recordings |

**Solution/Consideration:**
- Stop and investigate before continuing to label
- High uncertainty means you'll exclude or re-label those candidates anyway
- Diagnose the root cause: Is it the data? The interface? Your own understanding?
- Target: <10% uncertain rate

---

### 4.7 Minority Class in Test Set Priority

**The Problem:**
With limited recordings (e.g., 4 lab mice, 2 wild mice), random splitting might put all of one population in training and none in test. You can't evaluate generalization to that population.

**Example of Bad Split:**
```
Train: A, B, C, E (3 lab, 1 wild)
Val: D (lab only)
Test: F (wild only)
→ Validation has zero wild mice, can't catch problems during development
```

**Solution/Consideration:**
- Prioritize having minority class (wild mice) in test set
- Accept imperfect balance if data is limited
- Evaluate each population separately: "How does model perform on wild mice specifically?"
- If test performance differs dramatically between populations, you have a generalization problem

**Practical Split Strategy (limited data):**
```
Train: A, B, C, E (3 lab, 1 wild) — most data for learning
Val: subset of D — quick feedback during training
Test: rest of D + F — final evaluation includes both populations
```

---

### 4.8 Label Storage: Include Metadata

**The Problem:**
Minimal labels (just filename + yes/no) make debugging impossible later. When something goes wrong, you can't trace back to understand why.

**What to Store:**

```csv
filename,label,source_wav,start_ms,end_ms,labeler,labeled_at,notes
lab_mouse_01_00142.png,usv,lab_mouse_01.wav,1420,1580,shachar,2025-01-15T10:23:00,clear sweep
```

| Field | Why It Matters |
|-------|----------------|
| source_wav | Trace back to original recording |
| start_ms, end_ms | Find exact location in recording |
| labeler | Track if multiple people label |
| labeled_at | Detect temporal drift in labeling |
| notes | Record edge cases for later review |

**Solution/Consideration:**
- Storage is cheap; missing metadata is expensive
- You will thank yourself when debugging model failures
- Enables analyses like "do my early labels differ from late labels?"

---

### 4.9 Partial and Overlapping Calls

**The Problem:**
Some candidates contain partial calls (cut off at window edge) or overlapping calls (two calls at once). What label do you assign?

**For Partial Calls:**
- If clearly a USV even when partial → label as USV
- Model should learn that partial calls are still calls
- If too ambiguous to judge → mark uncertain

**For Overlapping Calls:**
- For binary detection (your use case) → label as "USV present"
- You're detecting presence, not counting
- Model learns "is there at least one USV?" not "how many?"

**Solution/Consideration:**
- Document your decision and apply it consistently
- Note: this means your model cannot count calls, only detect presence
- If counting matters later, you'll need a different approach (instance segmentation)

---

### 4.10 Faint Calls: Include Them

**The Problem:**
You see a candidate that's definitely a USV but barely visible. Tempting to skip or mark uncertain.

**Why You Should Label as USV:**
- This is exactly what your model needs to learn from
- Excluding faint calls biases toward loud-only detection
- Same problem as detector threshold bias (Module 3)—you create systematic blind spots

**Practical Impact of Excluding Faint Calls:**
- Model learns "quiet = not USV"
- Misses quiet calls in deployment
- Particularly problematic if wild mice vocalize more quietly than lab mice

**Solution/Consideration:**
- If you can tell it's a USV, label it as USV regardless of intensity
- Faint-but-real is more valuable training data than obvious-and-loud
- If truly can't tell, then mark uncertain (but be honest about why)

---

### 4.11 Quality Control Checklist Before Training

Run through this before starting model training:

- [ ] No duplicate candidates (same timestamp appearing twice)
- [ ] All labeled files exist on disk
- [ ] Label distribution is reasonable (not worse than 1:5 imbalance)
- [ ] Both populations represented in test set (and ideally validation)
- [ ] No recording appears in multiple splits
- [ ] Uncertain labels excluded (or handled deliberately)
- [ ] Augmentation applied only to training set
- [ ] Random sample of labels visually inspected for correctness
- [ ] Metadata complete (source recording, timestamps, labeler, date)
- [ ] Labeling consistency checked (re-labeled subset matches original)

---

## Updated Quick Reference: Full Pipeline Parameters

| Stage | Parameter | Recommended | Rationale |
|-------|-----------|-------------|-----------|
| **Detection** | Energy threshold | Low (high recall) | Don't miss USV types |
| | Min duration | 8-10 ms | Reject transient artifacts |
| | Max duration | 500 ms | Reject continuous interference |
| **Candidates** | Window size | 300-500 ms | Show call with context |
| | Target composition | 30-50% true USVs | Efficient labeling |
| **Labeling** | Uncertain rate | <10% | Higher suggests upstream problem |
| | Session length | 1-2 hours max | Prevent fatigue/drift |
| | Calibration | Every few days | Re-label 50, check consistency |
| **Splits** | Split unit | Recording (not candidate) | Prevent data leakage |
| | Ratio | 70/15/15 or 80/10/10 | More training if data limited |
| | Stratification | By population, by class | Ensure representation |
| **Training Data** | Class balance | 1:1 to 1:3 | Prevent majority-class bias |
| | Min examples/class | 200+ | Rough minimum for CNN |
| | Augmentation | 2-5× on training only | More can cause overfitting |
| **Evaluation** | Test set | Untouched until final eval | Unbiased estimate |
| | Metrics | Report per-population | Catch generalization failures |

---

## Module 5: Spectrogram Extraction for Labeling/Training

### 5.1 Extraction Pipeline Overview

After candidate detection, each candidate needs a spectrogram image for:
1. **Human labeling** - Review mode with axes, labels, and candidate markers
2. **CNN training** - Raw images with consistent dimensions, no decorations

The extraction parameters must balance visual clarity, computational cost, and CNN input requirements.

---

### 5.2 STFT Parameters (Locked for This Dataset)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| sample_rate | 300,000 Hz | Actual recording sample rate (not 250 kHz) |
| n_fft | 512 | ~586 Hz frequency resolution at 300 kHz |
| hop_length | 128 | 75% overlap, smooth temporal coverage |
| window | hann | Standard choice, good spectral leakage suppression |

**Why match detection parameters?**
The extraction STFT uses the same parameters as the energy detector to ensure the spectrogram shows exactly what the detector "saw" when it flagged the candidate.

---

### 5.3 Frequency Range

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| freq_min_hz | 20,000 Hz | Below USV band (25 kHz) for context |
| freq_max_hz | 120,000 Hz | Above USV band (110 kHz) for context |

**Why wider than detection band?**
- Detection uses 25-110 kHz (strict USV range)
- Extraction uses 20-120 kHz to show surrounding frequency context
- Helps labelers see if energy extends outside expected band (noise indicator)
- Helps CNN learn frequency boundaries

---

### 5.4 Image Dimensions

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| image_height_px | 256 | Fixed height for CNN input consistency |
| pixels_per_ms | 2.0 | 100ms candidate → 200px width |
| min_width_px | 128 | Minimum width even for short candidates |
| max_width_px | 512 | Maximum width to limit memory/computation |

**Temporal resolution:**
At 2.0 pixels/ms, a typical 50ms USV spans 100 pixels - sufficient to see frequency modulation details.

**Width clamping:**
- Very short candidates (<64ms) get padded to min_width_px
- Very long candidates (>256ms) get clamped to max_width_px
- This ensures CNN receives manageable input sizes

---

### 5.5 Color Scale (dB Mapping)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| db_floor | -80.0 dB | Black level (noise floor) |
| db_ceiling | 0.0 dB | White level (maximum amplitude) |
| colormap | magma | Perceptually uniform, good for spectrograms |

Default colormap for extraction/labeling PNGs is `magma` (matches labeling app display).

**dB range considerations:**
- 80 dB dynamic range captures both faint USVs and loud ones
- USVs typically appear in -40 to -10 dB range relative to max
- Noise floor is usually around -60 to -70 dB

---

### 5.6 Render Modes

**Review Mode (for human labeling):**
- Matplotlib figure with axes and labels
- Green vertical lines marking candidate boundaries
- Title showing candidate ID and peak frequency
- Colorbar showing dB scale
- Suitable for manual inspection and labeling

**Training Mode (for CNN input):**
- Raw image array, no axes or decorations
- Exact pixel dimensions (height × width)
- Values normalized to 0-255 grayscale or colormap
- Ready for direct input to neural network

---

### 5.7 Final Parameters Used (2026-01-17)

**Detection (EnergyDetector):**
```
threshold_db: -20.0 (relative to max peak)
freq_min_hz: 25,000
freq_max_hz: 110,000
min_duration_ms: 10.0
max_duration_ms: 500.0
energy_mode: peak
max_bandwidth_hz: 20,000
merge_gap_ms: 5.0
```

**Extraction (SpectrogramExtractor):**
```
sample_rate: 300,000
n_fft: 512
hop_length: 128
freq_min_hz: 20,000
freq_max_hz: 120,000
image_height_px: 256
pixels_per_ms: 2.0
db_floor: -80.0
db_ceiling: 0.0
colormap: magma
```

**Results:**
- 490 candidates detected across 50 recordings
- 490 spectrogram images extracted in review mode
- Ready for Phase 3: Labeling Tool

---

*Document updated 2026-01-17 with Module 5 (Spectrogram Extraction).*
