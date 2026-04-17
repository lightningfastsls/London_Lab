# Lab Data Pipeline — Future Planning

## Context
Mickey will provide lab mouse (standard inbred strain) recording data. When it arrives, the full detection + analysis pipeline needs to be run on it. This document ensures continuity — whoever picks this up knows what exists, what to run, and what the comparison goals are.

## The Research Question
Has decades of inbreeding changed courtship behavior in lab mice? Compare wild-derived mouse USVs (current data) vs. lab mouse USVs (incoming data).

## Current Data (wild-derived mice)
- **Cage 5970** (usv_lmt_034): 6,400 WAV files → 7,575 detections. Fully classified + analyzed.
- **Cage 3452** (usv_lmt_035): ~5,400 WAV files → ~456 detections. Detection done, classification pending.
- 16× difference in detection rate between cages — may reflect individual differences.

## What Exists (the pipeline)

**Detection**: CNN (207K params), full post-processing (hysteresis, FP filter, temperature-calibrated confidence, triage). 346 tests.

**Classification** (three approaches):
1. Rule-based taxonomy — 7 Scattoni types
2. DeepSqueak bridge — Python↔MATLAB k-means (27 clusters on 5970)
3. UMAP + HDBSCAN — data-driven density clustering

**Analysis modules**:
- Repertoire statistics (Shannon entropy, PERMANOVA, JSD, transitions)
- Information theory (Zipf, entropy rates, idiom detection, burstiness)
- Temporal dynamics
- LMT integration (PETH, DB loader, synchronizer — ready, awaiting behavioral data)

## When Lab Data Arrives — Run Order

**Phase 1: Detection**
- Run existing CNN pipeline, same parameters as wild-derived runs
- Review a sample manually — lab recordings may have different noise profiles
- If performance differs, may need threshold adjustment or FP filter retuning

**Phase 2: Classification**
- Apply all three classification approaches
- Generate same outputs: type distributions, confidence scores, UMAP embeddings

**Phase 3: Comparison** (the science)
1. Repertoire: same syllable types at same rates?
2. Acoustic space: project both populations into same UMAP — overlap or separate?
3. Temporal dynamics: calling rates, bout structures, ICI distributions
4. Sequential structure: same self-repetition dominance? same transitions?
5. Entropy: is one population more stereotyped? (inbreeding hypothesis → lab mice more stereotyped)

**Phase 4: Advanced** (after basic comparison)
- SIM (Syntax Information Maximization, from Hertz et al. 2020) on both populations
- Autoencoder trained on combined data, compare latent representations
- Vectorization approach (peak-frequency-per-column + amplitudes)
- Cross-population PCA/UMAP on autoencoder bottleneck features

## Blockers
- **Lab data**: not yet received. Format/size/structure unknown.
- **LMT behavioral data**: pending. Needed for behavioral correlation (PETH, event coupling).
- **Bout threshold**: under review (0.6s vs. 0.14–0.25s from mixture model). Affects all sequential analyses. See `docs/questions-for-mickey.md` Q1.
- **Cage 3452 classification**: pending. Needed before cross-animal comparison (priority B1 in progress report).
