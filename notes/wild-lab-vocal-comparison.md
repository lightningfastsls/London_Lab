---
description: Courtship degradation hypotheses, wild mouse vocal behavior, repertoire statistical methods, and the converging research question
type: moc
topics: "[[experimental-methods]]"
---

# wild-lab-vocal-comparison

The central biological question: did domestication alter mouse vocal repertoires? Wild mice show more diverse USV repertoires, and inbreeding plus absence of courtship selection pressure may have degraded lab mouse vocal competence. This map covers the hypotheses, supporting evidence, and statistical frameworks for testing them.

## Core Research Question
- [[wild versus lab mouse USV comparison tests whether domestication altered vocal repertoires]] -- the core research question driving the project
- [[classifiers trained on lab mice generalize poorly to wild mice requiring population-specific training data]] -- BootSnap's key cross-population finding implies genuine vocal differences
- [[distributional comparisons in VAE latent space using Earth Mover Distance or Jensen-Shannon divergence may be more biologically meaningful than categorical repertoire comparison]] -- continuous alternative to categorical repertoire comparison

## Courtship Degradation Hypotheses
- [[inbreeding and absence of courtship selection pressure in captivity caused lab mice to degrade courtship vocal competence]] -- the directional degradation hypothesis
- [[wild mice show more diverse USV repertoires than lab mice as preliminary evidence for courtship vocal degradation]] -- preliminary supporting evidence
- [[USVs are one component of a multimodal courtship behavior suite including mounting approach and movement]] -- multimodal courtship framing
- [[combined cross-modal evidence from USV repertoire and MiceCraft movement data builds a stronger case for courtship degradation]] -- cross-modal evidence strategy

## Wild Mouse Vocal Behavior
- [[Zala et al 2020 showed wild-derived mice modulate USVs with social context producing 9 types during interaction versus 6 during introduction]] -- context-dependent repertoire modulation in wild mice
- [[recording groups 5970 3452 2379 are all wild mouse dyads not different strains]] -- all current data is wild mice; numeric IDs are dyad identifiers

## Research Strategy
- [[DeepSqueak built-in classification enables pre-VQ-VAE repertoire comparison between wild and lab populations]] -- immediate science: classify first, then model
- [[VQ-VAE investigation of language-like sequential structure in USVs is a separate deeper question from courtship degradation]] -- two-tier strategy separates tractable from ambitious
- [[temporal alignment between USV detections and LMT behavioral events enables USV-behavior correlation analysis]] -- USV-behavior correlation method
- [[whether specific USV call types predict specific courtship outcomes like female receptivity to mounting]] -- functional specificity question

## Repertoire Statistical Methods
- [[PERMANOVA on Bray-Curtis dissimilarity is the standard ecological method for testing whether syllable repertoire compositions differ between populations]] -- non-parametric test from community ecology (Anderson 2001)
- [[Shannon entropy quantifies USV repertoire diversity with higher values indicating more evenly distributed syllable usage]] -- diversity metric with wild > lab prediction
- [[Jensen-Shannon divergence on categorical syllable proportions provides a symmetric bounded measure for comparing repertoire distributions between populations]] -- symmetric [0,1] distributional distance
- [[row-stochastic transition matrices capture sequential structure in syllable sequences testable between populations via Frobenius norm with permutation test]] -- sequential syntax comparison method

## Methodological Tensions
- [[forcing USVs into discrete categories may obscure the continuous variation that distinguishes populations]] -- categorical comparison may miss continuous differences

## Converging Hypothesis
- [[the converging research question asks whether transformer encodes behaviorally meaningful vocal categories differing between wild and lab populations]] -- where information theory, probing, and LMT integration converge

## Open Questions
- [[whether population-level metadata is available for context-dependent VQ-VAE analysis]]

## Related Areas
- [[experimental-methods]] -- parent hub for all experimental methodology
- [[behavioral-integration]] -- LMT methods for testing USV-behavior correlations
- [[classification]] -- repertoire comparison depends on syllable classification
- [[representation-learning]] -- VQ-VAE discretization and probing for the deeper question

---

Topics:
- [[experimental-methods]]
