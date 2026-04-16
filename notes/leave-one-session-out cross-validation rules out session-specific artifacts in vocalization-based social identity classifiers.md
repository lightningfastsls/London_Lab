---
description: "LOSO CV with KS test (P=0.49) confirmed Oren 2024 classifiers were not learning session-specific artifacts — a validation design applicable to cross-session USV analysis"
type: method
confidence: proven
conditions:
  - requires multiple recording sessions per animal; our 5-session USV design qualifies
meta_state: current
source: "inbox/oren-2024-vocal-labeling-deep-read-2026-04-15.md"
topics:
  - "[[experimental-methods]]"
---

# leave-one-session-out cross-validation rules out session-specific artifacts in vocalization-based social identity classifiers

Oren et al. (2024) used leave-one-session-out (LOSO) cross-validation as a critical control: train on all sessions except one, test on the held-out session, repeat for each session. The distribution of LOSO accuracies was compared to standard (within-session) accuracies using a Kolmogorov-Smirnov test (P = 0.49), confirming no significant difference — the classifiers were not exploiting session-specific artifacts like room acoustics, microphone position, or time-of-day effects.

This validation design is directly applicable to our USV analysis:
- **Our data:** 5 USV sessions per dyad (USV1-USV5), recorded across different days. Session-specific variation (mic placement, background noise) could confound any classifier.
- **Application:** If we build a classifier to distinguish individual mice by their USVs, LOSO CV across USV1-USV5 would confirm the classifier generalizes across recording sessions.
- **Implementation:** For each fold, hold out one session entirely; train on the remaining 4; test on the held-out session. Compare LOSO accuracy distribution to standard CV accuracy using KS test.

An additional finding from Oren 2024: classification accuracy **increases over time within a session** (dipping around call index ~20 then monotonically increasing), suggesting calls converge toward the current partner. This implies measured accuracy is an underestimate of the true receiver-encoding signal — early calls in a session are noisier.

---

Source:
- inbox/oren-2024-vocal-labeling-deep-read-2026-04-15.md (deep read, 2026-04-15)
- Oren, G. et al. (2024). Science, 385(6712), 996-1003.

Relevant Notes:
- [[random forest receiver-identity classification achieved AUC 0.798 across nine marmoset callers confirming vocalization-level social targeting]] -- the classifier validated by this method
- [[classifiers trained on lab mice generalize poorly to wild mice requiring population-specific training data]] -- cross-population generalization challenge that LOSO addresses at the session level

Topics:
- [[experimental-methods]]
