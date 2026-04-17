---
description: The numbered recording groups (5970, 3452, 2379) each represent a different pair of wild mice, not different strains or populations
type: claim
confidence: proven
topics: "[[experimental-methods]], [[wild-lab-vocal-comparison]], [[training-methodology]]"
---

# recording groups 5970 3452 2379 are all wild mouse dyads not different strains

All recording groups in the current dataset are **wild mice**. The different numeric identifiers (5970, 3452, 2379, etc.) represent different mouse dyads (pairs), not different strains, populations, or experimental conditions.

This means:
- All labeled training data comes from wild mouse recordings
- The CNN model is trained exclusively on wild mouse USVs
- Cross-group variation reflects individual/pair differences, not population differences
- [[classifiers trained on lab mice generalize poorly to wild mice requiring population-specific training data]] -- our model is already population-specific to wild mice
- [[recording-level splits prevent data leakage in USV classification]] -- splits should respect dyad boundaries
- Since [[Zala et al 2020 showed wild-derived mice modulate USVs with social context producing 9 types during interaction versus 6 during introduction]], our dyad recordings may contain context-dependent repertoire variation within the wild population
- This clarifies [[whether population-level metadata is available for context-dependent VQ-VAE analysis]] -- all groups are wild mouse dyads, so population identity is known even without a metadata CSV

When lab mouse recordings become available, they will need to be treated as a distinct population for comparison purposes per [[wild versus lab mouse USV comparison tests whether domestication altered vocal repertoires]]. Our wild mice provide the baseline against which [[inbreeding and absence of courtship selection pressure in captivity caused lab mice to degrade courtship vocal competence]] predicts degradation.
