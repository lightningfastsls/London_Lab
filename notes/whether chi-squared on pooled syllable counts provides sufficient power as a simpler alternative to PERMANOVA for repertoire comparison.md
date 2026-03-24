---
description: Chi-squared is computationally trivial but assumes discrete well-separated categories — PERMANOVA captures multivariate distributional structure via Bray-Curtis but needs permutation testing
type: open-question
confidence: speculative
topics: "[[classification-methodology]]"
---

# whether chi-squared on pooled syllable counts provides sufficient power as a simpler alternative to PERMANOVA for repertoire comparison

PERMANOVA on Bray-Curtis dissimilarity matrices is the standard approach for testing whether vocal repertoires differ between groups (e.g., wild vs. lab mice). It captures multivariate distributional structure but requires permutation testing and careful interpretation of dispersion effects.

**The simpler alternative:** A chi-squared goodness-of-fit test on pooled syllable type counts would test whether the frequency distribution of syllable categories differs between groups. This is computationally trivial and statistically well-understood.

**When chi-squared might suffice:**
- Syllable types are well-separated discrete categories (not a continuum)
- The question is "do groups differ in syllable usage?" rather than "how do they differ?"
- Sample sizes are large enough that pooling across individuals is defensible

**When PERMANOVA is necessary:**
- Syllable type boundaries are fuzzy or overlapping
- Individual-level variation matters (chi-squared on pooled counts erases individual structure)
- The research question involves subtle distributional shifts, not gross category differences
- Recording-level structure needs to be preserved — [[recording-level splits prevent data leakage in USV classification]]

**Unresolved:** No direct power comparison exists for USV repertoire data. The key unknown is whether pooling across individuals within a group destroys meaningful variance or whether group-level patterns are robust enough to survive aggregation.
