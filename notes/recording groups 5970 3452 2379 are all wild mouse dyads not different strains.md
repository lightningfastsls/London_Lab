---
description: The numbered recording groups (5970, 3452, 2379) each represent a different pair of wild mice, not different strains or populations
type: claim
confidence: confirmed
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

When lab mouse recordings become available, they will need to be treated as a distinct population for comparison purposes per [[wild versus lab mouse USV comparison tests whether domestication altered vocal repertoires]].
