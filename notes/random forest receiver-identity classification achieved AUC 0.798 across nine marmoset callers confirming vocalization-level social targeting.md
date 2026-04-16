---
description: "Oren 2024 showed individual marmosets encode receiver identity in phee calls — RF classification AUC 0.798, confirmed by playback experiment and LOSO cross-validation"
type: finding
confidence: proven
conditions:
  - marmoset phee calls, not directly tested on mouse USVs
meta_state: current
source: "inbox/oren-2024-vocal-labeling-deep-read-2026-04-15.md"
topics:
  - "[[classification]]"
  - "[[experimental-methods]]"
---

# random forest receiver-identity classification achieved AUC 0.798 across nine marmoset callers confirming vocalization-level social targeting

Oren et al. (2024, Science) trained 100 random forest models (150 trees each) per caller monkey to classify which receiver a phee call was directed at, using the [[Omer lab 80-dimensional FM plus AM ridge vectorization embeds each vocalization call in a fixed-length feature space|80D FM+AM vector representation]]. Results:

- **Average AUC across all 9 callers:** 0.798 +/- 0.065
- **All-monkeys pooled classifier:** AUC = 0.754
- **All 9 callers significantly above chance** (one-tailed t = 134.08, P < 0.0001)
- **Leave-one-session-out** validation confirmed no session-specific artifact learning (KS test P = 0.49)

The finding was validated by a **playback experiment** (virtual monkey system): monkeys answered directed calls with significantly higher probability than nondirected calls (Wilcoxon z = 3.88, P < 10^-4; Cox regression beta = 1.39, P < 2.4 x 10^-9).

Both AM and FM features contributed to classification, with AM features showing a slightly but significantly higher contribution (t = 5.06, P < 0.0001).

For mouse USV analysis, this result establishes that vocalization-level social targeting is biologically real and classifiable from acoustic features alone. The analogous question for mice: can individual identity or social target be classified from USV acoustic features? This would require the labeled metadata on caller identity that we have from the dyad recording design (5970, 3452, 9252 are separate dyads).

---

Source:
- inbox/oren-2024-vocal-labeling-deep-read-2026-04-15.md (deep read, 2026-04-15)
- Oren, G. et al. (2024). Science, 385(6712), 996-1003.

Relevant Notes:
- [[Omer lab 80-dimensional FM plus AM ridge vectorization embeds each vocalization call in a fixed-length feature space]] -- the representation used for classification
- [[PERMANOVA on Bray-Curtis dissimilarity is the standard ecological method for testing whether syllable repertoire compositions differ between populations]] -- statistical framework for our repertoire comparison
- [[leave-one-session-out cross-validation rules out session-specific artifacts in vocalization-based social identity classifiers]] -- the validation methodology
- [[Ivanenko et al 2020 showed DNNs achieve 77-84 percent accuracy classifying emitter sex from spectrograms]] -- convergent evidence: emitter sex (mouse) and receiver identity (marmoset) both classifiable from vocalization acoustics
- [[VQ token sequences discriminate call types but lose individual identity information during discretization]] -- VQ discretization loses the identity signal this classifier detects (AUC 0.798); sharpens codebook size tradeoff
- [[Zala et al 2020 showed wild-derived mice modulate USVs with social context producing 9 types during interaction versus 6 during introduction]] -- cross-species convergence: mice modulate by context, marmosets encode receiver identity

Topics:
- [[classification]]
- [[experimental-methods]]
