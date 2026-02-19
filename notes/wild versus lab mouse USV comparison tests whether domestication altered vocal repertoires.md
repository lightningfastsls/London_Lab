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

Both analysis streams build on the learned representations produced by [[three convolutional blocks with global average pooling suffice for USV classification on small datasets]] and [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]]. A critical prerequisite is [[whether population-level metadata is available for context-dependent VQ-VAE analysis]] -- without reliable mouse ID, sex, strain, and population labels associated with each recording, the cross-population comparison cannot be executed. The confidence is speculative because neither analysis has been run; the hypothesis that domestication alters USV repertoires is biologically plausible but unconfirmed in the current dataset.

---

Wild mouse vocalization literature is primarily from Michael London's own research group. Most published USV research uses inbred lab strains (e.g., C57BL/6), making this wild-lab comparison particularly novel — there is little existing literature on wild mouse USVs for comparison.

The specific causal hypothesis is directional: [[inbreeding and absence of courtship selection pressure in captivity caused lab mice to degrade courtship vocal competence]] — not just that wild and lab mice differ, but that lab mice have specifically degraded their courtship vocal repertoire. Preliminary evidence supports this: [[wild mice show more diverse USV repertoires than lab mice as preliminary evidence for courtship vocal degradation]]. The significance of the finding depends on [[combined cross-modal evidence from USV repertoire and MiceCraft movement data builds a stronger case for courtship degradation]].

---

Source: [[ROADMAP.md]], Phase 5

Relevant Notes:
- [[DeepSqueak built-in classification enables pre-VQ-VAE repertoire comparison between wild and lab populations]] -- immediate science strategy using existing tools
- [[Chabout et al 2015 established that male mice change syllable syntax with social context]] -- establishes that USV syntax carries behavioral information, supporting the hypothesis that repertoire differences reflect behavioral divergence
- [[no published work has applied VQ-VAE to animal vocalizations making this a genuine research gap]] -- the VQ-VAE approach to this comparison is novel
- [[temporal alignment between USV detections and LMT behavioral events enables USV-behavior correlation analysis]] -- method for connecting USV differences to behavioral differences

Topics:
- [[experimental-methods]]
- [[classification]]
