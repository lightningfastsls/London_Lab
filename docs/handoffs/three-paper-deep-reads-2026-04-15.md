# Handoff: Three Paper Deep-Reads (2026-04-15)

**Purpose:** Summary of three papers ingested into the knowledge graph in a single session, what each contributes to the project, and how they connect to each other and to active work.

**Source files (all archived to `archive/inbox/`):**
- `amvoc-stoumpou-2022-deep-read-2026-04-15.md`
- `hertz_2020_deep_read.md` (root copy still exists; inbox copy archived)
- `oren-2024-vocal-labeling-deep-read-2026-04-15.md`

---

## Paper 1: Stoumpou et al. 2022 — AMVOC (Analysis of Mouse Vocal Communication)

**Citation:** Stoumpou et al. (2022). *Bioacoustics*, 32(2), 199-229.
**What it is:** The best available open-source Python tool for unsupervised USV analysis. MIT-licensed, PyTorch-based convolutional autoencoder.

### What the paper does

AMVOC detects USVs via dynamic spectral thresholding (Event F1 = 90.5%, highest in their benchmark), then encodes each USV spectrogram through a 3-layer convolutional autoencoder (input 64x160 -> bottleneck 8x8x20 = 1,280 features). Post-processing reduces this to ~320 features via variance thresholding, then PCA, then clustering (k-means, GMM, or agglomerative) in PCA space. Deep features scored 37% higher than 4-feature handcrafted baselines in blinded human evaluation.

### Notes created (14 total)

Key notes include architecture spec (`AMVOC autoencoder encodes 64x160 spectrogram patches...`), training philosophy (`trains for only 2 epochs deliberately...`), feature pipeline (`4-stage feature pipeline reduces 1280 bottleneck features...`), gap analysis (`lacks batch normalization dropout validation monitoring...`), and the semi-supervised retraining loop.

### How it relates to the repo

| Connection | Details |
|------------|---------|
| **Unsupervised classification baseline** | AMVOC is the strongest unsupervised Python tool we could use instead of (or alongside) DeepSqueak's MATLAB-locked clustering. Our 7,518 classified USVs could be re-processed through AMVOC's autoencoder for comparison. |
| **Architecture template** | AMVOC's 3-conv + MaxPool encoder is the natural starting point for our own autoencoder or VQ-VAE. The input size (64x160 = 128ms x 160 freq bins) is well-justified and matches our detection windows. |
| **Feature extraction for analysis** | The 1,280D bottleneck embeddings could feed our UMAP/HDBSCAN pipeline, addressing the open question of whether [[raw acoustic features versus learned embeddings may yield different clustering structure for mouse USVs]]. Our current HDBSCAN (3 clusters, 97% in one) used raw features; autoencoder features might reveal sub-structure. |
| **Detection comparison** | AMVOC's spectral thresholding (Event F1 90.5%) is a useful reference point for our CNN detector (F1 91.7%). Different detection philosophy (classical DSP vs learned), similar performance. |
| **Wild mouse gap** | AMVOC was trained on lab strains (B6D2F1/J + C57BL/6J). Our wild-mouse data is a different domain. The gap analysis note flags batch normalization, dropout, and VAE variant as high-value improvements for our noisier wild-mouse recordings. |

**Active analysis roadmap connections:**
- Phase A3 (acoustic feature deep-dive): AMVOC embeddings could augment the PCA/UMAP feature exploration
- Phase B2 (cross-animal comparison): autoencoder features provide a richer representation for comparing 5970 vs 3452 repertoires
- Open Question #1 (is the continuous manifold truly continuous?): learned embeddings may reveal density ridges that raw features miss

---

## Paper 2: Hertz et al. 2020 — Syntax Information Score (SIS)

**Citation:** Hertz, S., Weiner, B., Perets, N. & London, M. (2020). *Communications Biology*, 3, 333.
**What it is:** **This is a Mickey London lab paper.** It defines SIS -- the gold-standard metric for evaluating USV classification schemes by their sequential predictive power. Also introduces SIM (Syntax Information Maximization), an algorithm that iteratively improves clustering by maximizing SIS.

### What the paper does

Hertz et al. compared three USV labeling algorithms (iMSA, iVoICE, iMUPET) on 346K syllables from 385 C57BL/6 courtship sessions. The key insight: different algorithms produce completely different labelings (no one-to-one mapping between their categories), yet SIS can objectively rank them. iMSA (rule-based pitch-jump detection) achieved the highest SIS (0.22 bits at depth 1) despite having the most skewed label distribution. SIM then demonstrated that starting from iMUPET's acoustic clusters and optimizing for sequential predictiveness could match or exceed iMSA's score.

The dominant SIS contributors were **self-repetitions** (same syllable following itself), suggesting that the most informative sequential pattern in mouse courtship is call-type persistence, not complex motif sequences.

### Notes created (10 total)

Key notes include the SIS formula (`SIS equals entropy rate at depth zero minus entropy rate at depth D...`), iMSA as the SIS champion (`iMSA rule-based pitch-jump classification produces the highest SIS...`), SIM algorithm details, the 160ms ISI threshold, suffix tree validity criteria, self-repetition dominance, and the dataset scale comparison (346K vs our 8K).

### How it relates to the repo

| Connection | Details |
|------------|---------|
| **The metric for evaluating our labelings** | SIS is the principled way to compare our three classification schemes: 7-type Scattoni (MI at lag 1 = 0.093 bits), DeepSqueak 27-cluster k-means, and UMAP/HDBSCAN (3 clusters). Our Phase A2 (sequential structure) already computed MI = 0.093 bits, which IS essentially SIS at depth 1. |
| **Mickey's own methodology** | This is from Mickey London's lab. Using SIS on our data is directly continuous with the lab's published methodology. It's the natural evaluation framework Mickey would expect. |
| **Scale constraint** | Our 8K dataset is 43x smaller than Hertz's 346K. This limits us to depth 1 (maybe depth 2 with few labels) and constrains max Nc due to the <10% zero-probability criterion. SIM may not converge with our data volume. |
| **Self-repetition finding** | Our Phase A2 results should check whether self-repetition also dominates SIS contributions in our wild-mouse data. If it does, it confirms cross-strain generality; if not, wild mice may have richer syntax. |
| **iMSA connection to ridge extraction** | The reflect pass found that iMSA's pitch-jump rules *implicitly perform ridge extraction* — they detect frequency discontinuities along the dominant frequency contour. This means iMSA's top SIS score (0.22 bits) is empirical evidence that ridge-based features capture sequential structure, even before implementing Omer's vectorization. |

**Active analysis roadmap connections:**
- Phase A2 (sequential structure): SIS is the evaluation metric; our 0.093 bits MI is directly comparable to Hertz's 0.10-0.22 range
- Phase B2 (cross-animal comparison): compare SIS between 5970 and 3452 to test whether sequential structure differs between individuals
- Open Question #2 (do bouts have signatures?): bout-level SIS could reveal within-bout vs between-bout sequential structure

---

## Paper 3: Oren et al. 2024 — Vocal Labeling of Others by Nonhuman Primates

**Citation:** Oren, G. et al. (2024). *Science*, 385(6712), 996-1003.
**What it is:** Demonstrated that marmosets encode receiver identity in their phee calls (the "vocal labels" finding). The paper includes the exact vectorization technique Mickey described ("find the highest amplitude per column and concatenate").

### What the paper does

Oren et al. vectorize each marmoset call as an 80D feature vector: 40 FM dimensions (frequency ridge trajectory) + 40 AM dimensions (amplitude along the ridge). All calls are time-normalized to 40 steps via 2D interpolation. Random forest classifiers trained on these vectors achieved AUC 0.798 for receiver identity classification across 9 callers. A playback experiment confirmed behavioral relevance: monkeys responded more to calls directed at them (Cox regression beta = 1.39, P < 2.4x10^-9). Family-level vocal conventions were discovered via RF proximity (leaf co-occurrence), with unrelated adults showing the same within-family similarity as parent-offspring groups, implying vocal learning.

### Notes created (11 total)

Key notes include the core 80D vectorization method, ridge extraction, time-axis resampling, per-caller normalization, the Zenodo code availability, RF classification results, GmSLM collaboration, RF proximity, leave-one-session-out validation, the 16-feature parallel with DeepSqueak, and the open question about Omer vs AMVOC clustering.

### How it relates to the repo

| Connection | Details |
|------------|---------|
| **The vectorization technique Mickey described** | This paper contains exactly the method Mickey explained. The source code is available on Zenodo (CC-BY 4.0). A Python implementation (~50 lines) is sketched in the deep read and could be added to our pipeline. |
| **Superset of AMVOC mode 3** | AMVOC's feature mode 3 extracts a 90D resampled frequency contour (FM only). Omer vectorization adds the 40D amplitude trajectory, making it a strict superset. Whether the AM component reveals substructure in our USV manifold is an empirically testable question. |
| **Identity classification from USVs** | Oren showed AUC 0.798 for receiver identity from vocalizations. Our dyad design (5970, 3452, 9252 = three separate animals) naturally supports the analogous experiment: can we classify mouse identity from USV acoustic features? |
| **RF proximity as similarity measure** | RF leaf co-occurrence provides a nonlinear, task-adapted similarity measure. This is an alternative to cosine/euclidean distance or JSD for comparing USV sets between animals — potentially more sensitive for the N=3 cross-animal comparison in Phase B2. |
| **LOSO validation template** | Leave-one-session-out CV with KS test is directly applicable to our 5-session design (USV1-USV5). If we build any classifier, this validates cross-session generalization. |
| **London-Omer collaboration** | GmSLM (Sternberg et al. 2025, EMNLP Findings) is co-authored by Mickey London and David Omer. The vectorization technique exists within a broader methodological collaboration that also includes SSL approaches to primate vocalizations. |

**Active analysis roadmap connections:**
- Phase A3 (acoustic feature deep-dive): Omer vectorization is a new feature set to project onto the UMAP manifold
- Phase B2 (cross-animal comparison): RF proximity could quantify similarity between 5970 and 3452 repertoires
- Phase D (synthesis): the vectorization technique could appear in a methods section alongside DeepSqueak features

---

## How the Three Papers Connect to Each Other

```
                    AMVOC (Stoumpou 2022)
                    Unsupervised autoencoder
                    1,280D learned embeddings
                         |
                         | Feature mode 3 (90D FM contour)
                         | is architecturally similar to...
                         v
                    Oren et al. 2024 ---------> Hertz et al. 2020
                    80D ridge vectorization      SIS evaluation metric
                    (40 FM + 40 AM)              (0.22 bits for iMSA)
                         |                            |
                         | iMSA pitch-jump rules      |
                         | are implicit ridge          |
                         | extraction, achieving       |
                         | top SIS score               |
                         +----------------------------+
                                    |
                                    v
                          Our USV pipeline
                          7,518 calls classified
                          MI at lag 1 = 0.093 bits
```

The three papers form a triangle:

1. **AMVOC -> Oren:** AMVOC's frequency contour (mode 3) and Oren's FM ridge are the same algorithmic idea (peak frequency per column, resampled to fixed length). Oren adds the AM component, making it a superset. The open question is whether AM reveals substructure that FM-only misses.

2. **Oren -> Hertz:** iMSA's pitch-jump rules -- which achieve the highest SIS in Hertz's benchmark -- are implicitly performing ridge extraction. This means ridge-based features (which Oren formalizes) already have empirical validation as sequentially informative features. If we implement Omer vectorization and cluster from it, we can compute SIS on those labels and benchmark against Hertz's numbers.

3. **AMVOC -> Hertz:** AMVOC's autoencoder embeddings could generate cluster labels evaluable by SIS. This would benchmark learned representations against the handcrafted ones Hertz compared (iMSA, iMUPET, iVoICE). The prediction: autoencoder-derived labels should achieve SIS between iMUPET (0.13 bits, also unsupervised clustering) and iMSA (0.22 bits, captures pitch-jump information that the autoencoder likely learns).

### The synthesis for our project

All three papers converge on the same pipeline for our data:

```
WAV recordings
  -> CNN detection (done, 7,518 calls)
  -> Feature extraction (three options now available):
       a. DeepSqueak 16 metrics (done)
       b. AMVOC autoencoder 1,280D embeddings (implementable)
       c. Omer 80D ridge vectorization (implementable, ~50 lines Python)
  -> Classification (two done, more possible):
       a. Traditional taxonomy, 7 types (done)
       b. UMAP+HDBSCAN (done, 3 clusters)
       c. k-means on Omer 80D (new option)
       d. k-means on AMVOC embeddings (new option)
  -> Evaluation:
       SIS (Hertz framework) ranks all classification schemes
  -> Cross-animal comparison:
       RF proximity (Oren) or JSD/PERMANOVA (existing tools)
```

The immediate actionable step: implement Omer 80D vectorization in Python (small effort, ~50 lines), generate labels via k-means, compute SIS, and compare against our existing 0.093 bits. This is a self-contained experiment that connects all three papers.

---

## Vault Statistics

| Paper | Notes created | Notes enriched | Topic maps updated |
|-------|-------------|----------------|-------------------|
| AMVOC (Stoumpou 2022) | ~14 | ~3 | 3 (unsupervised-usv-discovery, classification-tools, signal-processing) |
| Hertz et al. 2020 | ~10 | ~4 | 2 (unsupervised-usv-discovery, classification-methodology) |
| Oren et al. 2024 | 11 | 4 | 5 (unsupervised-usv-discovery, classification-tools, wild-lab-vocal-comparison, bioacoustic-ssl, signal-processing) |
| **Total** | **~35** | **~11** | **6 unique topic maps** |

Reflect pass found 10 additional cross-note connections (bidirectional links added). One dangling link fixed (Hann window note name mismatch).

---

## Unprocessed Source Files

The root copies of two source files still exist in the project root:
- `AMVOC_deep_read_extraction.md` — inbox copy archived, root copy remains
- `hertz_2020_deep_read.md` — not yet moved to archive

These can be cleaned up or kept as reference.
