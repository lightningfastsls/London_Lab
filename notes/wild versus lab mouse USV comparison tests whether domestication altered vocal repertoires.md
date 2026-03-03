---
description: Core research question: do wild-caught and lab-bred mice differ in USV repertoires? Two complementary analyses use CNN feature space statistics and VQ-VAE code sequences.
type: hypothesis
confidence: speculative
topics:
  - "[[experimental-methods]]"
  - "[[classification]]"
---

# wild versus lab mouse USV comparison tests whether domestication altered vocal repertoires

Domestication involves selection pressure on behavior, including acoustic communication. Wild-caught mice and laboratory-bred mice have experienced divergent evolutionary and developmental histories, and it is an open question whether this divergence is reflected in their ultrasonic vocalizations. Demonstrating differences (or their absence) would have implications for the use of lab mice as behavioral models and for understanding how domestication shapes communication systems.

The comparison uses two complementary analytical approaches applied to the same CNN-classified and VQ-VAE-encoded USV dataset. In the CNN feature space analysis, per-recording feature vectors are extracted from the CNN penultimate layer and reduced via PCA. Population-level differences are then assessed using MANOVA on PCA components (testing whether wild and lab mice occupy distinct regions of feature space), permutation tests on population centroids (robust to distributional assumptions), chi-squared tests on cluster usage frequencies (whether the two populations preferentially emit different call types), and Mann-Whitney tests on call characteristics such as duration, peak frequency, and frequency modulation depth.

The VQ-VAE code sequence approach treats each bout as a discrete sequence of learned codebook tokens. Analyses compare code frequency distributions between populations, transition probability matrices (whether certain code transitions are more common in one population), and population-unique n-grams (sequences of tokens that appear exclusively or predominantly in one population). Population labels are derived from the WAV directory structure or from a metadata CSV mapping filenames to population identities.

Specific statistical methods for the comparison include: (1) MANOVA on first 10 PCA components of CNN features, (2) chi-squared test on VQ-VAE code frequency distributions per population, (3) comparison of transition matrices via Frobenius norm with permutation test, (4) identification of population-unique n-grams. The behavioral context dimension adds: compare burstiness_by_context between populations -- if wild mice show stronger context-dependent variation in vocal timing patterns than lab mice, that is evidence for courtship degradation at the temporal dynamics level.

Both analysis streams build on the learned representations produced by [[three convolutional blocks with global average pooling suffice for USV classification on small datasets]] and [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]]. A critical prerequisite is [[whether population-level metadata is available for context-dependent VQ-VAE analysis]] -- without reliable mouse ID, sex, strain, and population labels associated with each recording, the cross-population comparison cannot be executed. The confidence is speculative because neither analysis has been run; the hypothesis that domestication alters USV repertoires is biologically plausible but unconfirmed in the current dataset.

---

## Statistical Framework for Repertoire Comparison

The standard statistical methods for comparing USV repertoire distributions between populations, in order of common use:

- **PERMANOVA** on Bray-Curtis distance matrices of per-animal syllable proportions (most widely used; available via `scikit-bio` in Python)
- **Chi-square or Fisher's exact tests** for comparing overall syllable type proportions between groups
- **KL divergence / Jensen-Shannon divergence** for quantifying distributional distance between populations (Bhattacherjee et al. used KL divergence at 95% CIs)
- **Markov chain transition analysis** comparing syllable-to-syllable transition probability matrices between groups, tested with chi-square or permutation tests
- **Shannon entropy** H = -sum(p_i * log2(p_i)) for quantifying repertoire diversity per individual or group. Prediction: wild mice should show higher H (more diverse repertoires). See [[Shannon entropy quantifies USV repertoire diversity with higher values indicating more evenly distributed syllable usage]].
- **Chi-squared test** on pooled syllable counts as a simpler alternative to PERMANOVA when sample sizes are sufficient

For latent-space approaches: [[distributional comparisons in VAE latent space using Earth Mover Distance or Jensen-Shannon divergence may be more biologically meaningful than categorical repertoire comparison]].

Wild mouse vocalization literature is primarily from Michael London's own research group. Most published USV research uses inbred lab strains (e.g., C57BL/6), making this wild-lab comparison particularly novel — there is little existing literature on wild mouse USVs for comparison. However, [[Zala et al 2020 showed wild-derived mice modulate USVs with social context producing 9 types during interaction versus 6 during introduction]] provides one of the few published wild-mouse USV studies, showing context-dependent repertoire modulation that must be controlled for in cross-population comparisons.

The specific causal hypothesis is directional: [[inbreeding and absence of courtship selection pressure in captivity caused lab mice to degrade courtship vocal competence]] — not just that wild and lab mice differ, but that lab mice have specifically degraded their courtship vocal repertoire. Preliminary evidence supports this: [[wild mice show more diverse USV repertoires than lab mice as preliminary evidence for courtship vocal degradation]]. The significance of the finding depends on [[combined cross-modal evidence from USV repertoire and MiceCraft movement data builds a stronger case for courtship degradation]].

---

Source:
- [ROADMAP](../ROADMAP.md), Phase 5
- inbox/deepsqueak-usv-syllable-classification-practical-guide.md (Compass artifact, 2026-02-23) -- statistical framework, Zala et al. evidence
- inbox/raven-deepsqueak-classification-bridge-plan.md (2026-02-23) -- Shannon entropy prediction, chi-squared alternative
- vacation-master-plan-v2 (archived to archive/inbox/)

Relevant Notes:
- [[DeepSqueak built-in classification enables pre-VQ-VAE repertoire comparison between wild and lab populations]] -- immediate science strategy using existing tools
- [[Chabout et al 2015 established that male mice change syllable syntax with social context]] -- establishes that USV syntax carries behavioral information, supporting the hypothesis that repertoire differences reflect behavioral divergence
- [[no published work has applied VQ-VAE to animal vocalizations making this a genuine research gap]] -- the VQ-VAE approach to this comparison is novel
- [[temporal alignment between USV detections and LMT behavioral events enables USV-behavior correlation analysis]] -- method for connecting USV differences to behavioral differences
- [[Zala et al 2020 showed wild-derived mice modulate USVs with social context producing 9 types during interaction versus 6 during introduction]] -- wild mouse repertoire modulation evidence
- [[distributional comparisons in VAE latent space using Earth Mover Distance or Jensen-Shannon divergence may be more biologically meaningful than categorical repertoire comparison]] -- latent space comparison alternative
- [[classifiers trained on lab mice generalize poorly to wild mice requiring population-specific training data]] -- generalization challenge for cross-population work
- [[BootSnap snapshot ensemble CNN on gammatone spectrograms outperformed DeepSqueak classification with F1 67 percent on wild mice]] -- best available wild-mouse classification baseline
- [[burstiness by behavioral context bridges information theory and LMT behavioral analysis]] -- temporal emission patterns per behavioral context differ between populations
- [[PERMANOVA on Bray-Curtis dissimilarity is the standard ecological method for testing whether syllable repertoire compositions differ between populations]] -- primary multivariate comparison method
- [[Jensen-Shannon divergence on categorical syllable proportions provides a symmetric bounded measure for comparing repertoire distributions between populations]] -- pairwise distributional distance
- [[row-stochastic transition matrices capture sequential structure in syllable sequences testable between populations via Frobenius norm with permutation test]] -- sequential syntax comparison method

Topics:
- [[experimental-methods]]
- [[classification]]
