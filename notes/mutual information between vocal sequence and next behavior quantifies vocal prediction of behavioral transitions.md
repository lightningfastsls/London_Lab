---
description: "I(vocal_sequence; next_behavior) measures whether USV code sequences predict upcoming behavioral events — the strongest test of communicative function"
type: method
confidence: speculative
meta_state: current
topics:
  - "[[representation-learning]]"
  - "[[experimental-methods]]"
---

# mutual information between vocal sequence and next behavior quantifies vocal prediction of behavioral transitions

This Tier 3 analysis represents the strongest test of communicative function in USV sequences, because it asks whether the vocalizations carry information about what the mouse is about to do — not just what it is currently doing. The core metric is I(vocal_sequence; next_behavior), the mutual information between a window of VQ-VAE code sequences preceding a behavioral transition and the identity of the next behavioral event. If this quantity is significantly above zero after controlling for behavioral autocorrelation, it means the vocal stream contains predictive information about upcoming behavior that is not redundant with behavioral history alone.

The method works as follows. For each behavioral transition (e.g., approach followed by oral-oral contact), extract the VQ-VAE code sequence in a window preceding the transition onset. Use this code sequence as the feature vector to predict the next behavioral event via a classification task. The critical comparison is against a baseline model that predicts the next behavior from behavioral history alone — the sequence of preceding behavioral events without any vocal information. If the vocal-informed model outperforms the behavioral-history-only model, the difference in predictive accuracy directly reflects the additional information carried by the vocal channel.

This design controls for a major confound, because behavioral sequences are themselves temporally autocorrelated — if a mouse is approaching, it is more likely to make contact next regardless of what it vocalizes. Therefore, the baseline must include behavioral history to ensure that any improvement from vocal information reflects genuine vocal prediction rather than behavioral prediction that happens to correlate with vocal patterns. The mutual information estimate uses the classification-based approach: I(V; B) is approximated by the difference in cross-entropy loss between the vocal-informed and behavioral-history-only classifiers.

The analysis also introduces two additional probing targets that connect to the transformer representation learning framework. **Behavioral state** tests whether the transformer hidden states encode the current behavioral context of the mouse — if so, the transformer has learned cross-modal information despite being trained only on acoustic input. **Time to next event** tests whether the transformer representations predict when the next behavioral transition will occur, which would indicate temporal prediction capacity extending beyond the acoustic domain.

This analysis is speculative because it depends on multiple upstream components working correctly: accurate USV detection, meaningful VQ-VAE codebook, reliable LMT behavioral annotations, and precise temporal alignment. But if the result is positive, it provides the most compelling evidence that USV sequences function communicatively, because prediction of future behavior is a hallmark of signals that carry genuine semantic content rather than merely accompanying ongoing actions.

---

Source:
- vacation-master-plan-v2 (archived to archive/inbox/)

Relevant Notes:
- [[Hertz et al 2020 demonstrated that USV sequence statistics carry predictive information]] -- prior evidence that USV sequences are predictive, supporting the motivation for this analysis
- [[the converging research question asks whether transformer encodes behaviorally meaningful vocal categories differing between wild and lab populations]] -- this analysis directly tests the converging hypothesis by measuring vocal-behavioral information flow

Topics:
- [[representation-learning]]
- [[experimental-methods]]
