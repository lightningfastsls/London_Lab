# Deep-Read: Hertz et al. 2020 — Temporal Structure of Mouse Courtship Vocalizations Facilitates Syllable Labeling

**Citation:** Hertz, S., Weiner, B., Perets, N. & London, M. (2020). Communications Biology, 3, 333.  
**DOI:** https://doi.org/10.1038/s42003-020-1053-7  
**Code:** https://github.com/london-lab/MouseUSVs  
**Data:** mouseTube, group label "London Lab"  
**Language/Framework:** MATLAB (all three adapted algorithms and the SIS/SIM code)

---

## 1. Syntax Information Score (SIS)

### 1.1 Definition

SIS is defined as the **mutual information** between the next syllable $X_n$ and its prefix (suffix of depth $D$): $Y = (X_{n-D}, \ldots, X_{n-1})$.

$$\text{SIS} = I(X; Y) = \sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} p(x, y) \log \frac{p(x, y)}{p(x) \, p(y)}$$

### 1.2 Equivalent formulations

1. **Entropy difference form:**  
   $I(X; Y) = H(X) - H(X|Y)$  
   where $H(X) = H(X_n)$ is the 0th-order entropy and $H(X|Y) = H(X_n | X_{n-1}, \ldots, X_{n-D})$ is the entropy rate of the $D$th-order Markov model.  
   **In practice: SIS = (entropy rate at depth 0) − (entropy rate at depth D).**

2. **KL-divergence form:**  
   $I(X; Y) = D_{KL}(p(x, y) \| p(x) p(y))$

### 1.3 Inputs

The three probability mass functions needed:

- **$p(y)$** = $p(X_{n-1}, \ldots, X_{n-D})$ — the probability of each suffix (stored in suffix tree leaves as visit counts / total)
- **$p(x, y)$** = $p(X_n, X_{n-1}, \ldots, X_{n-D})$ — computed by multiplying conditional probability $p(X_n | \text{suffix})$ by the suffix probability $p(y)$, via law of total probability
- **$p(x)$** = $p(X_n)$ — the 0th-order marginal distribution of labels

All are estimated from labeled sequences stored in a suffix tree data structure.

### 1.4 Entropy rate formula

The entropy rate of the $m$th-order Markov model (used as the inner component):

$$H_m = -\sum_{i,j} \mu_i P_{ij} \log P_{ij}$$

where $\mu_i$ = probability of visiting the $i$th leaf (suffix) and $P_{ij}$ = conditional probability of label $j$ given suffix $i$. Reference: Cover & Thomas (2005), Theorem 4.2.4.

### 1.5 Interpretation

| SIS value | Meaning |
|-----------|---------|
| 0 bits/symbol | No temporal structure captured (either data has no structure, OR algorithm assigns random/uniform labels, OR algorithm assigns all USVs the same label) |
| Higher positive values | Labeling captures more sequential regularity — knowing the prefix reduces uncertainty about the next syllable |
| Upper bound = $H(X_n)$ | The 0th-order entropy; a labeling with very biased distribution has a lower ceiling for SIS |

**Key design property:** SIS is insensitive to the 0th-order distribution itself. An algorithm that assigns all USVs the same label gets SIS = 0 (even though entropy rate = 0, which would look "predictable"). An algorithm that assigns random labels also gets SIS = 0 (high entropy at all orders, no drop). This makes SIS superior to raw entropy rate for comparing algorithms.

### 1.6 Individual pair contributions

SIS can be decomposed into contributions from each $(D+1)$-tuple. For depth 1 (pairs), the contribution of pair $(x_{n-1}, x_n)$ is:

$$p(x_n, x_{n-1}) \log \frac{p(x_n, x_{n-1})}{p(x_n) \cdot p(x_{n-1})}$$

This is equivalent to the pointwise KL divergence between the observed joint and the independence-assumed product. Pairs where $P = Q$ contribute 0. The dominant contributors in their data were **self-repetitions** (same syllable following itself).

### 1.7 Statistical significance / variance estimation

- Entropy rates and SIS computed over **25 repetitions**, each using a **60% random subsample** of sequences to build the suffix tree.
- Error bars = 2 standard deviations across the 25 repetitions.
- Validated against **synthetic Markov models** where analytical entropy rate and SIS can be computed from the transition matrix (via eigenvalue decomposition for stationary distribution). Supplementary Fig. 4 shows estimated values converge to analytical values.
- No explicit shuffle test / p-value reported for SIS itself, but the convergence to analytical values on synthetic data serves as validation.

### 1.8 Suffix tree validity criterion

For higher $N_c$ or deeper trees: they required that the fraction of conditional probabilities with value 0 (i.e., $(D+1)$-tuples never observed) be **less than 10%**. This limited them to $N_c \leq 64$ for depth 1 and $N_c \leq 16$ for depth 2.

### 1.9 Normalized SIS

For comparing across different numbers of clusters: $\text{SIS}_{\text{norm}} = \text{SIS} / \log_2(N_c)$. They found the normalized SIS showed no clear dependency on $N_c$, suggesting the raw SIS increase with more clusters is roughly proportional to the bits needed to encode them.

---

## 2. Syntax Information Maximization (SIM) Algorithm

### 2.1 Goal

Given an initial set of cluster centroids (from iMUPET), find a new set of centroids that **maximizes SIS** on a test set — i.e., the clustering should respect both acoustic similarity (proximity to centroid) and sequential predictiveness.

### 2.2 Initialization

- Uses **iMUPET centroids** as the starting point (chosen because iMUPET had the second-best SIS, giving room for improvement toward iMSA's score).
- All syllables preprocessed through MUPET's gammatone filter pipeline (16 filters), converting each syllable to a **vector of length 2016**.
- Initial centroids = mean of all syllable vectors in each cluster.

### 2.3 Step-by-step procedure (pseudocode)

```
INPUT: labeled_sequences (split 50/50 into train/test), initial centroids C[1..K]
PREPROCESS: all syllables → 2016-dim vectors via gammatone filter (16 filters)

failure_counter = 0

REPEAT:
    # Step 2: Generate random perturbation vector
    V = uniform_random(0.9, 1.1) for each dimension  # length 2016
    
    best_delta_SIS = -inf
    best_centroid_idx = None
    
    FOR k = 1 to K:  # perturb each centroid in turn
        C_perturbed[k] = C[k] ⊙ V   # element-wise (dot) product
        
        # Reassign ALL training syllables to nearest centroid
        # (all centroids unchanged except C[k] → C_perturbed[k])
        new_labels = assign_to_nearest(train_syllables, centroids_with_perturbation_k)
        
        # Compute SIS on training set with new labels
        delta_SIS[k] = SIS(new_labels) - SIS(current_labels)
        
        IF delta_SIS[k] > best_delta_SIS:
            best_delta_SIS = delta_SIS[k]
            best_centroid_idx = k
    
    # Step 3: Accept or reject
    IF best_delta_SIS > 0:
        APPLY perturbation to C[best_centroid_idx]
        Reassign all training syllables
        failure_counter = 0
    ELSE:
        failure_counter += 1
        IF failure_counter >= 5:
            APPLY perturbation anyway (forced accept)
            failure_counter = 0
        ELSE:
            DO NOT apply perturbation

UNTIL convergence (SIS on training set stabilizes)

# Replay perturbation chain on test set
FOR each accepted perturbation in order:
    Apply same centroid change to test set
    Reassign test syllables, compute SIS
```

### 2.4 Key details

- **Perturbation type:** multiplicative — each dimension of the centroid is scaled by a factor drawn uniformly from [0.9, 1.1]. This means the perturbation vector $V$ is the same for whichever centroid is being perturbed in a given iteration, but a new $V$ is drawn each iteration.
- **One centroid at a time:** in each iteration, each of the $K$ centroids is perturbed independently (with the same $V$), ΔSIS is evaluated for each, and only the best one is accepted.
- **Forced accept after 5 failures:** prevents getting stuck in local optima. If 5 consecutive iterations fail to improve SIS, the best perturbation (even if negative) is applied anyway.
- **Train/test split:** 50/50 by sequences. SIM optimizes on train only; test SIS is evaluated post-hoc by replaying the exact perturbation chain.
- **Convergence:** not formally defined beyond "SIS stabilizes." From Fig. 7a, convergence appears around ~16,000–24,000 iterations.

### 2.5 Hyperparameters

| Parameter | Value |
|-----------|-------|
| Number of clusters ($K$) | 8 (inherited from iMUPET) |
| Perturbation range | Uniform[0.9, 1.1] per dimension |
| Failure threshold | 5 consecutive non-improving iterations |
| Train/test split | 50% / 50% of sequences |
| Suffix tree depth for SIS | 1 (primary) and 2 (secondary) |
| Feature vector dimensionality | 2016 (from 16 gammatone filters) |

### 2.6 Computational cost

The authors acknowledge SIM is "computationally suboptimal" — each iteration requires:
1. Perturbing one centroid
2. Reassigning all training syllables to nearest centroid (K distance computations × N syllables)
3. Rebuilding suffix tree and computing SIS

With ~173K training syllables (half of 346K), 8 centroids, and ~24K iterations, this is expensive. No runtime reported. The authors state their motivation was proof-of-concept, not computational efficiency.

### 2.7 Results

- **Depth 1:** SIM surpassed iMSA's SIS (the previous best).
- **Depth 2:** SIM approached but remained slightly below iMSA, though significantly above iMUPET's starting point.
- The improvement generalized to the test set, confirming the new centroids capture real structure.

---

## 3. USV Feature Extraction / Vectorization

### 3.1 Per-algorithm representations

The paper does NOT define a single universal feature vector. Each algorithm uses its own representation:

**iMSA (Mouse Song Analyzer v1.3):**
- Rule-based: detects **pitch jumps** (frequency discontinuities) in the spectrogram.
- Categories: Simple (no jump), Up, Down, Multiple — then each split by median duration → 8 labels.
- No vectorization per se; it's a deterministic rule system.
- Preprocessing: gap removal (short silence gaps within a syllable are bridged).

**iVoICE:**
- No explicit preprocessing mentioned.
- Hierarchical clustering on a training subset of 4000 syllables.
- Similarity measure: **spectral similarity** (details in VoICE paper, Burkett et al. 2015).
- Results in 8 centroids; remaining syllables assigned by nearest centroid using same spectral similarity metric.

**iMUPET:**
- Preprocessing: **gammatone filter bank** (16 filters) applied to spectrogram representation.
- This converts each syllable into a **vector of length 2016**.
- K-means clustering on training subset of 5000 syllables.
- Distance metric: **cosine distance** between filtered syllable vector and centroid.
- 8 centroids; remaining syllables assigned to nearest.

### 3.2 Spectrogram parameters

- Recording: UltraSoundGate system (Avisoft), **sampling rate 250 kHz**, 16-bit.
- Online monitoring: 256-point FFT spectrogram.
- Frequency range for USV detection: >20 kHz (ultrasonic range).
- The paper does not specify the exact FFT/hop parameters used for the analysis spectrograms (as opposed to online monitoring). The MUPET gammatone representation is the most detailed: 16 gammatone filters → 2016-dim vector.

### 3.3 Parsing

Custom USV parser developed (code available at GitHub repo). Detects start/end times of each syllable. ISI threshold for sequence boundary: **160 ms** (ISI > 160 ms = new sequence).

### 3.4 Basic features stored in database

For each syllable: start time, end time, duration, ISI to next syllable, mean frequency (strongest frequency at each time point, averaged).

⚠️ **Uncertainty flag:** The exact spectrogram parameters for the feature extraction pipeline (FFT window size, hop size, frequency resolution for the analysis — as opposed to the 256-point online monitoring FFT) are not explicitly stated in the main text. The gammatone filter details for iMUPET reference Van Segbroeck et al. 2017.

---

## 4. Labeling Algorithms Compared

### 4.1 The three algorithms

| Algorithm | Approach | Preprocessing | Training data | Clustering | Distance metric | Labels |
|-----------|----------|---------------|---------------|------------|-----------------|--------|
| **iMSA** (Mouse Song Analyzer v1.3, Chabout et al. 2015) | Rule-based | Gap removal | None | None — deterministic rules | N/A | 4 natural (Simple/Up/Down/Multiple) → 8 via median duration split |
| **iVoICE** (Burkett et al. 2015) | Unsupervised | None | 4000 syllables | Hierarchical clustering | Spectral similarity | 8 (natural parameter) |
| **iMUPET** (Van Segbroeck et al. 2017) | Unsupervised | Gammatone filter (16 filters) | 5000 syllables | K-means | Cosine distance | 8 (user-chosen $K$) |

The "i" prefix denotes that Hertz et al. adapted/modified each algorithm for consistency (same syllable parser, same database).

### 4.2 Disagreements between algorithms

- **No one-to-one mapping** between labels across algorithms. The joint distribution of iMSA × iMUPET labels (Fig. 2d) shows that for most iMSA labels, iMUPET labels are distributed fairly uniformly (no dominant diagonal).
- Same finding for iMSA × iVoICE (Supplementary Fig. 3).
- The mapping is also **not independent** — the observed joint distribution differs from the product of marginals. So there's partial overlap but substantial disagreement.
- iMSA produces the **most non-uniform** label distribution (>50% in the two "Simple" labels). iMUPET produces the **most uniform** distribution.

---

## 5. Evaluation

### 5.1 Dataset

| Property | Value |
|----------|-------|
| Mouse strain | C57BL/6 |
| Context | Male-female courtship interaction |
| Recording sessions | 385 (349 from London lab + 36 from mouseTube) |
| Total recording time | ~78 hours |
| Total syllables | 346,632 |
| Total sequences | 33,481 |
| ISI threshold for sequences | 160 ms |
| Males | Multiple males, varying sexual experience, recorded multiple times |
| Estrous control | Not controlled (sessions scheduled independently) |

### 5.2 Metrics

1. **Entropy rate** at depths 0–4 (bits/symbol)
2. **SIS** at depths 1 and 2 (bits/symbol) — the primary comparison metric
3. **Normalized SIS** = SIS / log₂(Nc) — for cross-Nc comparisons
4. **Individual pair/triplet SIS contributions** — decomposition showing which motifs drive the score
5. **Train/test generalization** for SIM

### 5.3 Quantitative results (8 labels, from Figures 4a–d and 7d)

**Entropy rates (depth 0 → depth 2):**

| Algorithm | H₀ (bits/sym) | H₁ (bits/sym) | H₂ (bits/sym) |
|-----------|---------------|---------------|---------------|
| iMSA | ~2.45 | ~2.25 | ~2.20 |
| iVoICE | ~2.75 | ~2.65 | ~2.62 |
| iMUPET | ~2.90 | ~2.78 | ~2.73 |

(Approximate values read from Fig. 4a; iMUPET's H₀ ≈ 2.9 is close to log₂(8) = 3.0, reflecting its near-uniform distribution.)

**SIS values (from Fig. 4b):**

| Algorithm | SIS depth 1 (bits/sym) | SIS depth 2 (bits/sym) |
|-----------|----------------------|----------------------|
| iMSA | ~0.22 | ~0.27 |
| iVoICE | ~0.10 | ~0.14 |
| iMUPET | ~0.13 | ~0.17 |
| **SIM** | ~0.23 | ~0.25 |

**Key findings:**
- iMSA achieves the highest SIS despite having the lowest 0th-order entropy — meaning pitch jumps are a particularly informative feature for sequential structure.
- SIM (starting from iMUPET) reaches or exceeds iMSA at depth 1, confirming that sequence information can improve purely acoustic clustering.
- iMUPET with 32 clusters surpasses iMSA with 8 clusters at depth 1 — but at 4× model complexity.

### 5.4 Relevance to your data

Your current numbers for comparison:
- 7-type Scattoni: conditional entropy reduction 3.7%, MI at lag 1 = 0.093 bits
- UMAP+HDBSCAN: 3 clusters (97% in one) — essentially no useful labeling for syntax

The SIS framework gives you a principled way to compare these. Your 0.093 bits MI is directly comparable to SIS depth 1. Hertz et al. got ~0.13–0.22 bits with 8 labels on C57BL/6 courtship data. Your lower value could reflect: fewer categories (7 vs 8), coarser categories (Scattoni rules vs. pitch-jump or k-means), wild-derived mice having different syntax, or smaller dataset effects.

---

## 6. Architecture / Implementation Details

- **Language:** MATLAB (all code — parser, adapted algorithms, SIS computation, SIM, suffix tree construction).
- **Repository:** https://github.com/london-lab/MouseUSVs
- **Recording hardware:** Avisoft UltraSoundGate (CM16/CMPA mic, 116H interface, USGH recorder), 250 kHz, 16-bit.
- **Data sharing:** WAV files on mouseTube (group "London Lab"); supplementary data for all figures in Supplementary Data 1.
- **Computational cost:** Not reported quantitatively. SIM is acknowledged as "computationally suboptimal." Each iteration involves full reassignment + suffix tree rebuild. ~24K iterations visible in Fig. 7a.

---

## 7. Key Figures

### Fig. 1 — Parsing and basic statistics
Shows spectrogram examples, ISI distribution (bimodal: peaks at ~20 ms and ~70 ms), syllable duration (exponential), mean frequency (~40–120 kHz), sequence length (power-law). Critically: **Fig. 1g** shows duration correlation between adjacent syllables (r = 0.44, p < 0.001) — short follows short, long follows long. This pre-labeling correlation is the first evidence of temporal structure.

### Fig. 2 — Algorithm comparison
Side-by-side: label distributions (iMSA skewed, iMUPET uniform), pair distributions vs. independence baseline (red line), and the joint iMSA×iMUPET confusion matrix showing no one-to-one mapping.

### Fig. 3 — Suffix tree and entropy rate framework
Explains how suffix trees encode Markov models. **Fig. 3c** is critical: shows entropy rate vs. depth for four toy labelings (2, 4, 4, 5 labels). Demonstrates that entropy rate alone is insufficient (biased labeling gets low entropy rate "for free") and motivates the SIS.

### Fig. 4 — SIS comparison of the three algorithms
**The central result.** Fig. 4a: entropy rates; Fig. 4b: SIS values showing iMSA > iMUPET > iVoICE. **Fig. 4c**: individual pair SIS contributions — self-repetitions dominate. The "Simple-long / Simple-short" split in iMSA shows same-duration pairs appearing more than expected, cross-duration pairs appearing less — suggesting duration is an important sub-feature within the "Simple" category.

### Fig. 5 — Effect of increasing Nc for iMUPET
SIS increases with Nc (up to 64 for depth 1), but normalized SIS is roughly flat. At Nc = 32, iMUPET surpasses iMSA's SIS at depth 1.

### Fig. 6 — SIM algorithm illustration
Schematic: shows how a syllable with feature-based similarity 0.31 to S and 0.24 to T might be relabeled T if T gives +0.04 bits SIS vs. S's +0.01 bits. The "handwriting/Q→U" analogy is very intuitive.

### Fig. 7 — SIM results
**Fig. 7a**: SIS convergence curves for train and test sets at depth 1 and 2. Both improve, confirming generalization. **Fig. 7d**: Final comparison — SIM matches or exceeds iMSA.

---

## 8. Limitations and Open Questions (Author-Acknowledged)

1. **Female vocalizations not controlled for.** Male-female interaction setup yields many USVs but includes unattributed female calls.

2. **Behavioral phase not accounted for.** Courtship sessions have stereotypic phases (approach, sniffing, mounting, intromission) with different syllable distributions. Syntax structure may relate to mating context — not addressed.

3. **Number of true syllable classes unknown.** Most analysis uses 8 labels (a balance of richness vs. statistical validity). The true number could be higher. Increasing Nc improves SIS but requires exponentially more data.

4. **SIM is computationally suboptimal.** Proof-of-concept; not designed for efficiency. Each iteration requires full reassignment and suffix tree rebuild.

5. **Motif analysis limited to length 2–3.** Longer motifs require exponentially more data. Statistical analyses in prior work also limited to order ≤3.

6. **SIS doesn't identify motifs directly** — it quantifies total sequential information. Motif identification requires examining individual tuple contributions.

7. **HMM connection acknowledged but not pursued.** The labeling-as-noisy-channel framework (true Markov source → noisy observation via biased algorithm) is a Hidden Markov Model. Combining SIS with HMM estimation could improve labeling further.

8. **Stationarity assumption.** The Markov model assumes underlying transition probabilities don't change over time or between sessions. This is likely violated given behavioral phase effects.

---

## 9. Implementation Notes for Your Pipeline

### 9.1 Computing SIS on your data

You already have MI at lag 1 = 0.093 bits with 7-type Scattoni. This IS essentially SIS at depth 1 (if computed as $H_0 - H_1$). To fully replicate:

1. Build a suffix tree from your labeled sequences (7 labels, depth 1 and 2).
2. Compute entropy rate at each depth using $H_m = -\sum_{ij} \mu_i P_{ij} \log P_{ij}$.
3. SIS = $H_0 - H_D$.
4. Decompose into per-pair contributions to find which transitions drive the score.
5. Use 25× bootstrap (60% of sequences) for error bars.

### 9.2 Comparing your three labelings

Apply the same SIS computation to:
- 7-type Scattoni (your rule-based)
- DeepSqueak k-means (27 clusters) — note: may need the <10% zero-probability check for depth 2
- UMAP+HDBSCAN (3 clusters, 97% in one) — SIS will almost certainly be ~0 given the degenerate distribution

This directly tells you which labeling captures the most sequential structure.

### 9.3 Running SIM

Starting from DeepSqueak's 27 centroids (or re-running k-means with 8 clusters for comparability):
1. You need each USV represented as a fixed-length vector (DeepSqueak's representation, or your own spectrogram features).
2. Implement the perturbation loop: multiplicative noise in [0.9, 1.1], perturb each centroid, pick best ΔSIS.
3. Train/test split by sequences (50/50).
4. Monitor convergence on both sets.

### 9.4 Key consideration: your dataset size

Hertz et al. had 346K syllables / 33K sequences. You have ~7,575 + ~456 = ~8,031 detections. This is **~43× smaller**, which will severely limit:
- Maximum viable depth (probably depth 1 only, maybe depth 2 with few labels)
- Maximum viable Nc (the <10% zero criterion will bite hard)
- SIM convergence (less data for each centroid adjustment to reflect real structure)

You may need to aggregate across more recordings before SIM becomes viable, or keep Nc low (≤8).

---

## 10. Connections to Your Current Analysis

| Your finding | Hertz et al. parallel |
|---|---|
| MI at lag 1 = 0.093 bits (7 Scattoni types) | SIS depth 1 ≈ 0.10–0.22 bits (8 labels, C57BL/6). Your lower value is expected with fewer labels and wild-derived mice. |
| Self-repetition dominant | Same — self-repetition pairs are the largest SIS contributors (Fig. 4c). |
| Conditional entropy reduction only 3.7% | Hertz et al. see reductions of ~7–10% from depth 0 to depth 1 with 8 labels. Your coarser categories explain the smaller reduction. |
| UMAP+HDBSCAN finds continuous manifold (3 clusters, 97% in one) | Consistent with the paper's finding that USV feature space lacks clear separability — clusters overlap, which is exactly why SIM helps. |
| DeepSqueak k-means (27 clusters) | Directly analogous to iMUPET. You could compute SIS on these 27 clusters and compare to your 7-type Scattoni — the Hertz framework predicts the comparison will be informative. |

---

*Document prepared for arscontexta ingestion. Source: direct reading of the full paper text (12 pages + methods). Formulas verified against Methods section. Approximate quantitative values read from figures — exact values available in Supplementary Data 1.*

*⚠️ Fact-check reminders: (1) The 2016-dimensional vector from gammatone filters — verify exact construction in Van Segbroeck et al. 2017 (MUPET paper). (2) Spectrogram parameters for the analysis pipeline (as opposed to 256-pt online monitoring FFT) are not specified in this paper. (3) The [0.9, 1.1] perturbation range and failure threshold of 5 are from the Methods section but could be tunable.*
