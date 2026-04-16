# Deep-Read Extraction: Oren et al. 2024 — Vocal Labeling of Others by Nonhuman Primates

**Paper:** Oren, G., Shapira, A., Lifshitz, R., Vinepinsky, E., Cohen, R., Fried, T., Hadad, G.P., & Omer, D. (2024). Science, 385(6712), 996–1003.  
**DOI:** 10.1126/science.adp3757  
**Zenodo (code + data):** https://doi.org/10.5281/zenodo.12721811  
**Extraction date:** 2026-04-15  
**Purpose:** Technical methodology extraction for adapting the Omer lab vectorization technique to mouse USV analysis.

---

## ⚠️ CRITICAL FINDING: Vectorization Technique

### STATUS: **YES — THE TECHNIQUE IS IN THIS PAPER AND CONFIRMED BY SOURCE CODE**

Mickey's description matches the method exactly. The paper describes it in the main text (p.1) and refers to supplementary fig. S1A/S1B for the visual. The actual MATLAB implementation is available on Zenodo (`spectrogramming.m`).

### The Technique in Detail

**What Mickey described:**
> "Vectorize the pixels that had the highest amplitude — find in each pixel column what is the highest amplitude and take that one, but also concatenate the amplitudes to the end of the vector, and use that to cluster."

**What the paper says (main text, p.1):**
They analyze each call's time-frequency representation (spectrogram), extract the frequency modulation (FM) and amplitude modulation (AM) trajectories, normalize and resample to a standard length, embedding each call in an **80-dimensional feature space** (40 FM + 40 AM dimensions).

### Exact Implementation from `spectrogramming.m` (Zenodo)

Here is the full pipeline, reconstructed from the MATLAB source code:

#### Step 1: Spectrogram computation (STFT)

```matlab
window = calls(i).SampleRate * 0.05;   % 50ms window
noverlap = window / 2;                  % 50% overlap (25ms hop)
[tf, f, t] = stft(call(i).soundwave, calls(i).SampleRate, ...
    'Window', hanning(window), ...
    'OverlapLength', noverlap, ...
    'FrequencyRange', 'onesided');
```

**Parameters:**
- Window size: 50ms Hanning window
- Hop size: 25ms (50% overlap)
- Full one-sided STFT computed first

#### Step 2: Frequency band selection

```matlab
ii = find(f >= 6000 & f <= 9000);
tf = tf(ii, :);
f = f(ii);
```

They crop to **6–9 kHz** — the phee call frequency range. For mouse USVs, this would be ~30–110 kHz.

#### Step 3: Time-axis resampling to fixed length

```matlab
spect_dim = 40;  % target time steps
[x1, y1] = meshgrid(1:size(tf,2), 1:size(tf,1));
[x2, y2] = meshgrid(linspace(1, size(tf,2), spect_dim), 1:size(tf,1));
tf = interp2(x1, y1, tf, x2, y2);
```

This is the key normalization step: every call, regardless of duration, is **interpolated to exactly 40 time columns** using 2D interpolation. The frequency axis is preserved at original resolution; only the time axis is resampled.

#### Step 4: Ridge extraction (FM trajectory) — "find the highest amplitude per column"

```matlab
[frq(:,i), ~, lr] = tfridge(abs(tf), f, 'NumRidges', 1);
```

`tfridge` extracts the **dominant frequency ridge** from the spectrogram — this is MATLAB's built-in function that finds the frequency bin with maximum energy at each time step, with continuity constraints. This produces a **40-element vector of frequencies** (the FM trajectory / pitch contour).

This is exactly what Mickey described: "find in each pixel column what is the highest amplitude and take that one."

#### Step 5: Amplitude extraction along the ridge

```matlab
ft(:,i) = tf(lr);
```

`lr` contains the linear indices of the ridge points. `tf(lr)` extracts the **complex spectrogram values at the ridge locations**, giving a **40-element amplitude trajectory** (the AM profile). This is the "concatenate the amplitudes" part.

#### Step 6: Result — 80D vector per call

The output is saved as two matrices:
- `frq` — 40 × N matrix of frequency values (FM trajectories)
- `ft` — 40 × N matrix of complex amplitudes at ridge (AM trajectories)

These are later concatenated as `[am; fm]` (80 × N) in the classification code.

### Preprocessing Before Classification (from `RF_Generic.m`)

```matlab
am = abs(ft);                              % magnitude of complex amplitude
fm = frq;                                   % frequency values
am = smoothdata(am, 1, 'movmedian', 6);    % median smooth AM, window=6
fm = smoothdata(frq, 1, 'movmean', 5);     % mean smooth FM, window=5
p = [am; fm];                               % concatenate: 80D vector

% Per-caller normalization (rescale to [0,1])
for i = 1:length(caller_id)
    idx = find(caller == caller_id(i));
    p(1:40, idx)  = rescale(p(1:40, idx),  0, 1);  % AM normalized per caller
    p(41:end, idx) = rescale(p(41:end, idx), 0, 1); % FM normalized per caller
end
```

Key details:
- AM is the **absolute value** of the complex STFT at the ridge
- AM is median-smoothed (window=6), FM is mean-smoothed (window=5)
- AM and FM are independently rescaled to [0,1] **per caller** (not globally)
- Final vector: [AM₁...AM₄₀, FM₁...FM₄₀] = 80 dimensions

---

## 2. Machine Learning / Classification Pipeline

### Random Forest Classifiers

**Implementation:** MATLAB `TreeBagger` (ensemble of 150 decision trees per model).

**Training protocol:**
- For each caller monkey, 100 separate random-forest models are trained
- Each model uses a **Random Under-Sampling (RUS)** balanced subset: 100 calls per receiver class
- Out-of-bag (OOB) prediction used for evaluation (no separate test split — OOB is the hold-out)
- OOB predictor importance computed for feature ranking

**Classification tasks performed:**
1. **Receiver ID from individual caller** — train per-caller models to identify which receiver the call was directed at. Average AUC across all monkeys: 0.798 ± 0.065.
2. **Receiver ID from all callers pooled** — train on calls from all 9 monkeys. AUC = 0.754.
3. **Caller ID from response calls** — identify who the response is addressed to after playback.

**Session generalization:** Leave-one-session-out cross-validation confirmed that classifiers were not learning session-specific artifacts (KS test between standard and LOSO accuracy distributions: P = 0.49).

**Feature contribution:** Both AM and FM features contributed similarly, with AM features showing a slightly but significantly higher contribution (t = 5.06, P < 0.0001; see fig. S2).

### Specific Accuracy Results (diagonal of confusion matrices)

**Caller Adonis** (4 receivers): AUC = 0.939. Individual receiver accuracies: Ceto 84.67%, Dia 65.97%, Dionysus 71.53%, Ella 89.65%.

**Caller Ella** (4 receivers): AUC = 0.814. Individual receiver accuracies: Adonis 52.57%, Bhumi 69.86%, Dia 63.66%, Dionysus 52.49%.

**All monkeys pooled:** Average accuracy significantly above chance (one-tailed t = 134.08, P < 0.0001).

### Dimensionality Reduction

- **PCA:** Used to reduce the 16 acoustic features to 3 dimensions (mean explained variance: 75 ± 3%). Applied per-caller for the acoustic features analysis (fig. S5).
- **t-SNE:** Used to embed the 16-dimensional explained-variance vectors into 2D for visualizing family clustering (fig. 4J). Parameters: perplexity=3, metric=euclidean, init=pca, random_state=10 (from MATLAB code calling Python sklearn).
- **MDS (Multidimensional Scaling):** Applied to dissimilarity matrix (1 − proximity) from the random forest leaf co-occurrence to visualize family clustering (fig. 4E).

---

## 3. The 16 Acoustic Features (from `calc_acoustic_features.m`)

These are **separate from** the 80D vectorization used for classification. They were used for the explained-variance analysis (fig. 4I-J) to understand which acoustic properties encode receiver identity.

### FM Features (8)
| Feature | Description |
|---|---|
| `freq_diff` | Last FM value − first FM value (overall frequency change) |
| `freq_max_idx` | Time index of maximum frequency |
| `freq_slope1` | Frequency at peak − frequency at start |
| `freq_slope2` | Frequency at peak − frequency at end |
| `freq_max` | Maximum frequency value |
| `freq_min` | Minimum frequency value |
| `freq_mean` | Mean frequency across call |
| `freq_integ` | Sum of frequency values (integral) |

### AM Features (8)
| Feature | Description |
|---|---|
| `amp_diff` | Last AM value − first AM value |
| `amp_max` | Maximum amplitude value |
| `amp_max_idx` | Time index of maximum amplitude |
| `amp_slope1` | Amplitude at peak − amplitude at start |
| `amp_slope2` | Amplitude at end − amplitude at peak |
| `amp_integ` | Sum of amplitude values (integral) |
| `amp_mean` | Mean amplitude across call |
| `frq_max_amp` | Frequency value at the time of maximum amplitude |

**Key finding:** All 16 features contributed to some extent to explained variance across all monkeys — no specific subset solely encodes receiver identity. The features are z-scored before PCA.

---

## 4. The "Vocal Labels" Evidence

### Establishing receiver-specificity (not random, not caller-state)

1. **Random forest classification significantly above chance** for all 9 caller monkeys (P < 0.0001 for each).
2. **Leave-one-session-out control** rules out session-specific artifacts (KS test P = 0.49).
3. **Classification accuracy increases over time within a session** (fig. S7): accuracy dips around call index ~20 then monotonically increases, suggesting calls converge toward the current partner as the session progresses. This also means measured accuracy is an underestimate.
4. **Shuffle controls:** Randomly permuted caller-receiver labels destroy classification accuracy, confirming the signal is in the call-receiver mapping.

### Playback Experiment (Virtual Monkey System)

**Design:** One monkey in the small enclosure; the long enclosure replaced by a closed-loop playback system (speaker + PC + microphone). The system plays back previously recorded phee calls to initiate and maintain dialogues using simple heuristics derived from natural conversations.

**Two call types played back:**
- **Directed calls:** Previously recorded calls specifically addressed to the participating monkey
- **Nondirected calls:** Calls originally directed at other monkeys

**Results:**
- Monkeys answer directed calls with significantly higher probability (Wilcoxon signed-rank: z = 3.88, P < 10⁻⁴; paired t: t = 5.73, P < 1.6 × 10⁻⁵)
- Cumulative response probability significantly higher for directed calls from onset (Cox regression: β = 1.39, P < 2.4 × 10⁻⁹)
- 2/3 monkeys (Adonis, Baloo) responded correctly to caller identity; Bolt did not on average but showed non-random response patterns (χ² tests all P < 0.0001)

---

## 5. Family Dialect Finding

### Quantification method: Random Forest Proximity

**Proximity** is defined as the proportion of times two calls land in the same leaf of a decision tree — it is a nonlinear similarity measure that emerges naturally from the random forest.

**Protocol:**
1. Train all-monkeys classifier (100 models, 150 trees each)
2. For each pair of callers, compute average proximity only between calls directed at the same receiver
3. Construct proximity matrix (fig. 4D) → reveals family grouping
4. Apply MDS to (1 − proximity) → family clusters emerge clearly (fig. 4E)

**Statistical tests:**
- Within-family vs. across-family proximity: significantly higher within-family for all three groups (Wilcoxon rank sum; family A: z = 109.9; family B: z = 47.4; family C: z = 75.69; all P < 0.0001)
- Same-receiver vs. different-receiver proximity within families: significantly higher for same receiver (family A: z = 220.7; family B: z = 17; family C: z = 976.2; all P < 0.0001). This rules out nonspecific family convergence.

### Vocal Learning Evidence

**Not imitation of receiver's calls:** Classification accuracy drops significantly when testing a caller's model on the receiver's calls back to the caller (Wilcoxon signed-rank: z = 5.35, P < 0.0001). No difference between receiver's calls and any other monkey's calls to the caller (z = 0.49, P = 0.31). This means labels are NOT learned by imitating what the receiver sounds like.

**Genetic vs. learned:** Families A and C consist entirely of unrelated adults paired as mature adults — yet they show the same within-family similarity as family B (parents + offspring). This strongly implies vocal learning among adults, not genetic predisposition.

**Explained-variance clustering (fig. 4J):** t-SNE embedding of each monkey's 16-feature explained-variance vector shows family-level clustering in which acoustic features each family preferentially uses to encode receiver identity. Families A and B cluster tightly; family C clusters closer to B.

---

## 6. Code & Data Availability

### Zenodo Repository
**URL:** https://zenodo.org/records/12721811  
**License:** CC-BY 4.0  
**Size:** 130 MB  
**Language:** MATLAB

**Files available:**
| File | Purpose |
|---|---|
| `spectrogramming.m` | **Core vectorization** — STFT → ridge extraction → 80D vector |
| `calc_acoustic_features.m` | 16 named acoustic features (fig. 4I-J) |
| `RF_Generic.m` | Random forest training + OOB evaluation |
| `RUS.m` | Random Under-Sampling for class balancing |
| `calc_proximity.m` | RF leaf co-occurrence proximity (fig. 4D-G) |
| `calc_examplars.m` | Medoid call selection |
| `calc_immitation.m` | Imitation analysis (fig. 4H) |
| `calssify_all_monkey.m` | All-monkeys classifier (fig. 4A-C) |
| `calc_roc.m` | ROC curve computation |
| `calc_medoid.m` | Medoid computation |
| `majorityVoting.m` | Ensemble voting |
| `Fig_1.mat` – `Fig_4.mat` | Figure data (Fig_4.mat is 125.7 MB — contains proximity data) |

---

## 7. Related Omer Lab Papers

### Directly relevant to vocalization analysis:

1. **Sternberg, T., London, M., Omer, D., & Adi, Y. (2025). "GmSLM: Generative Marmoset Spoken Language Modeling." Findings of EMNLP.**
   - **This is a Mickey London + David Omer collaboration.** Uses self-supervised speech models on marmoset vocalizations. May contain additional vectorization/tokenization approaches. Worth reading for the SSL representation approach.

2. **Omer, D. (2025). "Mouse vocalization: Singing the line." Current Biology, 35(12), R611–R612.**
   - Didi wrote a commentary on singing mice territorial vocalization. He is actively thinking about mouse vocalizations — relevant context for your collaboration path.

3. **Oren, G. & Omer, D. (2025). "Vocal labeling of others by nonhuman primates: A response to Jaakkola (2025)." Learning & Behavior, 53(4), 319–320.**
   - Response to criticisms. Clarifies that labels are arbitrary (not imitations), that cross-caller models reveal family conventions, and that vocal accommodation does not account for the observed behavior.

### The vectorization technique appears to originate in this 2024 Science paper — it is not referenced as coming from an earlier publication. The supplementary materials (fig. S1A-B) provide the visual illustration.

---

## 8. Translation to Mouse USVs — Implementation Notes

### What maps directly:
- **FM trajectory extraction via ridge detection** maps perfectly to mouse USV pitch contours
- **AM trajectory along the ridge** maps to amplitude envelope of USVs
- **Resampling to fixed time steps** solves variable-length USVs (same problem as variable-length phee calls)
- **Concatenation of [AM; FM]** produces a fixed-length vector per USV

### What needs adaptation:

| Parameter | Phee calls (this paper) | Mouse USVs (your data) |
|---|---|---|
| Frequency range | 6–9 kHz | ~30–110 kHz |
| Window size | 50ms | Likely ~1–5ms (USVs are much shorter, ~10–100ms) |
| Hop size | 25ms | ~0.5–2ms |
| Target time steps | 40 | TBD — could be 40, or scale proportionally to typical USV duration |
| Ridge extraction | `tfridge` (MATLAB) | Python equivalent: `scipy.signal` peak finding per column, or `librosa` pitch tracking |
| Normalization | Per-caller rescale to [0,1] | Per-recording or per-mouse rescale |

### Python implementation sketch:

```python
import numpy as np
from scipy.signal import stft, find_peaks
from scipy.interpolate import interp2d

def vectorize_usv(audio, sr, freq_range=(30000, 110000), n_steps=40):
    """
    Omer-style vectorization: FM ridge + AM along ridge → 2*n_steps vector.
    """
    # Step 1: STFT
    nperseg = int(sr * 0.001)  # 1ms window for USVs
    noverlap = nperseg // 2
    f, t, Zxx = stft(audio, sr, nperseg=nperseg, noverlap=noverlap)
    
    # Step 2: Crop to frequency range
    freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
    Zxx = Zxx[freq_mask, :]
    f_crop = f[freq_mask]
    
    # Step 3: Resample time axis to fixed n_steps
    mag = np.abs(Zxx)
    # Interpolate to n_steps time columns
    t_new = np.linspace(0, mag.shape[1]-1, n_steps)
    mag_resampled = np.array([np.interp(t_new, np.arange(mag.shape[1]), mag[i,:]) 
                              for i in range(mag.shape[0])])
    
    # Step 4: Ridge extraction (peak frequency per time column)
    fm = np.array([f_crop[np.argmax(mag_resampled[:, j])] for j in range(n_steps)])
    
    # Step 5: Amplitude along ridge
    am = np.array([mag_resampled[np.argmax(mag_resampled[:, j]), j] for j in range(n_steps)])
    
    # Step 6: Smooth
    from scipy.ndimage import median_filter, uniform_filter1d
    am = median_filter(am, size=6)
    fm = uniform_filter1d(fm, size=5)
    
    # Step 7: Concatenate
    return np.concatenate([am, fm])  # 2*n_steps = 80D vector
```

### Key comparison opportunities:
1. Omer 80D vector representation vs. your CNN autoencoder latent space
2. Omer-style UMAP/PCA vs. your existing UMAP on classified USVs
3. Random forest receiver-ID approach → could you classify mouse identity from USVs?
4. The proximity-based similarity measure from RF is an interesting nonlinear alternative to cosine similarity for comparing USV vectors

---

## 9. Spectrogram Parameters Summary

| Parameter | Value |
|---|---|
| STFT window | Hanning, 50ms |
| Overlap | 50% (25ms hop) |
| Frequency range | 6–9 kHz (one-sided FFT, then cropped) |
| Time normalization | 2D interpolation to 40 time steps |
| Frequency resolution | Preserved at STFT native resolution within 6–9 kHz |
| Ridge extraction | MATLAB `tfridge`, single ridge, continuity-constrained |
| Smoothing (AM) | Moving median, window=6 |
| Smoothing (FM) | Moving mean, window=5 |
| Normalization | Per-caller rescale AM and FM independently to [0,1] |
| Final vector | [AM₁...AM₄₀, FM₁...FM₄₀] = 80D |

---

*Document prepared for arscontexta ingestion. Key tags: omer-lab, vocalization-vectorization, phee-calls, marmoset, random-forest, spectrogram-ridge, FM-AM-trajectory, mouse-USV-adaptation, ELSC.*
