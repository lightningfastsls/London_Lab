---
description: "Vacation plan v2 for ~1hr/day autonomous progress via Claude Code + RustDesk: three workstreams (info theory + null models, probing, LMT integration) with merged Gemini vs. original plan comparison, detailed implementation specs, and session scheduling."
source_type: conversation
date_accessed: "2026-02-23"
status: unprocessed
---

# Vacation Master Plan v2 — Source Capture

## Key Points

- Three workstreams: (1) information theory + null models, (2) probing experiments, (3) LMT integration
- Gemini suggestions merged into original plan: adds burstiness_coefficient, ngram_idioms (idiom detection), and zipf_via_shannon_entropy as cross-validation; keeps original null model framework (Gemini had no equivalent)
- Null models are the most important part: without them, information-theoretic metrics (Zipf, entropy rate, excess entropy) are uninterpretable
- Burstiness by context directly bridges information theory and LMT behavioral analysis
- LMT already has a USV Toolbox (LMT-USV-Toolbox on GitHub) that handles synchronization — don't build from scratch
- Probing framework: train linear/MLP probes on frozen transformer hidden states to identify which layer encodes which acoustic property — guides VQ-VAE layer selection
- Remote workflow: morning 15 min spawn sessions, phone check-ins 25 min, optional evening commit + /seed

## Raw Notes

### Workstream 1: Information Theory + Null Models

Merged spec produces 3 modules:

**`usv_language/analysis/information_theory.py`**
- `zipf_exponent_mle` — Clauset et al. 2009 MLE power law fit (alpha, xmin, KS p-value)
- `zipf_via_shannon_entropy` — entropy-based Zipf estimate, cross-validates MLE on small datasets
- `entropy_rate` — H(X_n|X_{n-1},...,X_{n-k}) for k=0..max_order with Miller-Madow correction
- `excess_entropy` — I(past;future) via block entropy extrapolation (Crutchfield & Feldman 2003)
- `mutual_information_rate` — I(X_t; X_{t+lag}) at varying lags
- `transition_matrix` — K×K bigram transition probability matrix
- `conditional_entropy_by_lag` — H(X_t|X_{t-lag}) for single token at varying distance (vs. contiguous history in entropy_rate)
- `bigram_productivity` — unique observed bigrams / K², with bootstrap CI
- `ngram_idioms` — detects n-grams exceeding chance frequency (z-score>3 or FDR p<0.01), candidates for compositional phrases
- `burstiness_coefficient` — CV of inter-event intervals (CV=1 Poisson, >1 bursty, <1 regular); plus Kleinberg burst detection
- `burstiness_by_context` — burstiness broken down by behavioral context labels

**`usv_language/analysis/null_models.py`** (NullModelGenerator)
- `shuffled` — random permutation, preserves frequencies, destroys all structure
- `markov_order_k` — fit k-th order Markov, generate surrogates
- `renewal_process` — fit inter-event interval distribution
- `hmm_surrogate` — fit HMM (hidden behavioral states hypothesis)
- `phase_randomized` — preserve autocorrelation/power spectrum, destroy higher-order structure

**`usv_language/analysis/statistical_tests.py`** (NullModelComparison)
- `compare` — z-scores, rank-based p-values, effect sizes
- `full_analysis` — all null models × all metrics, main publishable table

Analytically verifiable tests (key subset):
- Shuffled uniform K=64 → entropy rate = log2(64) = 6.0 at all orders
- Markov-1 → entropy rate at order ≥ 2 same as order 1
- Poisson events → burstiness CV ≈ 1
- Planted idiom [5,12,33] at 10x expected rate → detected by ngram_idioms

### Workstream 2: Probing Experiments

**`usv_language/analysis/acoustic_properties.py`** (AcousticPropertyExtractor)
Extracts ground-truth labels from spectrogram data:
- `peak_frequency`, `spectral_centroid`, `energy` — continuous regression targets
- `is_voiced`, `frequency_direction` — classification targets
- `bout_position`, `time_since_last_usv` — temporal continuous targets

**`usv_language/analysis/probing.py`** (ProbingExperiment, ProbingAnalysisPipeline)
- Load hidden states (memory-mapped from extract_hidden_states.py output)
- Pooling: mean/max/first/last over time dimension
- Probes: linear (LogisticRegression/Ridge), mlp_1layer
- 5-fold CV, outputs accuracy/R² + selectivity (probe acc - majority baseline)
- Key output: (layer × property) heatmap showing where information lives

### Workstream 3: LMT Integration

Key discovery: LMT-USV-Toolbox already handles WAV-behavioral synchronization.
Pre-code questions: SQLite file location, AviSoft integration method, wild vs. lab labels, LMT version.

Three analysis tiers:
- Tier 1: Event-triggered USV rate — PETH in ±2s window per event type (sanity check + wild vs. lab coupling)
- Tier 2: Behavioral context × vocal repertoire — MANOVA on CNN features or chi-squared on VQ-VAE codes
- Tier 3: Vocal sequence → behavioral prediction — mutual information I(vocal_sequence; next_behavior)

Code: `src/usv_spectrogram/lmt/` — db_loader.py, synchronizer.py, event_triggered.py, context_analysis.py

### Where Workstreams Converge

- Null models prove structure is real (not statistical artifact)
- Probing shows what transformer learned + guides VQ-VAE layer selection
- Burstiness_by_context bridges information theory and behavioral analysis
- LMT integration grounds everything in biology
- Combined answer: "Does transformer encode behaviorally meaningful vocal categories that differ between wild and lab populations?"

### Session Schedule (5 sessions)

- Session 1: info theory — Zipf (both methods), entropy rate, excess entropy, transition matrix
- Session 2: null models + statistical comparison framework
- Session 3: burstiness, compositionality (bigram productivity + idiom detection), conditional entropy by lag
- Session 4: probing framework (acoustic properties + probing experiments)
- Session 5: LMT research (explore SQLite schema, design doc)

## Processing Notes

{After /reduce: what was extracted, what was skipped, what needs follow-up}
