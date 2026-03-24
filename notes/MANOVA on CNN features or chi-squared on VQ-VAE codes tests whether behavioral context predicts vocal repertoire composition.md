---
description: "Tier 2 LMT analysis — tests whether syllable type distribution differs across behavioral contexts, bridging simple PETH correlation (Tier 1) and vocal prediction of behavior (Tier 3)"
type: method
confidence: speculative
meta_state: current
topics:
  - "[[experimental-methods]]"
  - "[[representation-learning]]"
---

# MANOVA on CNN features or chi-squared on VQ-VAE codes tests whether behavioral context predicts vocal repertoire composition

This Tier 2 analysis asks a fundamentally different question than the other two LMT analysis tiers. Where [[event-triggered USV rate via PETH in plus-minus 2 second windows per event type serves as LMT integration sanity check]] tests whether mice vocalize *more* during certain behaviors (a rate question), and [[mutual information between vocal sequence and next behavior quantifies vocal prediction of behavioral transitions]] tests whether *what mice say* predicts what they'll *do next* (a prediction question), Tier 2 tests whether behavioral context shapes *what mice say* — whether the distribution of syllable types changes depending on what the mouse is doing.

The statistical approach depends on the representation. For CNN-based syllable classifications, which produce continuous feature vectors, MANOVA (multivariate analysis of variance) tests whether the centroid of the feature distribution differs across behavioral contexts. For VQ-VAE discrete codes, which produce categorical syllable assignments, chi-squared tests whether the frequency distribution of code usage differs across contexts. Both answer the same biological question — does behavior predict repertoire — but operate on different data types.

This matters because a positive result would mean that mice adjust their vocal repertoire to match their behavioral state, which is a stronger claim than simply vocalizing more during social interactions (Tier 1). It would suggest context-dependent vocal production, a hallmark of communicative sophistication. However, it is weaker than Tier 3's claim that vocal sequences predict *future* behavior, because context-dependent production could arise from simple arousal modulation rather than genuine communicative intent.

The three tiers together form an escalating evidence chain: correlation (do USVs co-occur with behavior?), composition (does behavior shape what is said?), prediction (does what is said predict what happens next?). Each positive result raises the bar for the communicative interpretation, with Tier 3 providing the most compelling evidence.

---

Source:
- vacation-master-plan-v2 (archived to archive/inbox/)

Relevant Notes:
- [[event-triggered USV rate via PETH in plus-minus 2 second windows per event type serves as LMT integration sanity check]] — Tier 1: simpler rate-based test that must pass before Tier 2 is meaningful
- [[mutual information between vocal sequence and next behavior quantifies vocal prediction of behavioral transitions]] — Tier 3: stronger predictive test that builds on Tier 2's finding
- [[burstiness by behavioral context bridges information theory and LMT behavioral analysis]] — related context-dependent analysis using temporal rather than repertoire statistics

Topics:
- [[classification-methodology]]
- [[experimental-methods]]
- [[representation-learning]]
