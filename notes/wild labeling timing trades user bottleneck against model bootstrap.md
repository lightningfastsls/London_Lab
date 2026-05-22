---
description: "Labeling the wild OOD set in parallel with training keeps the project moving but burns 100+ user minutes against an unfocused task; labeling after v2 lets the model bootstrap your queue but adds wall-time"
type: tension
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[experimental-methods]]"
  - "[[classification-methodology]]"
---

# Wild labeling timing trades user bottleneck against model bootstrap

The Phase 1.4 wild transfer evaluation requires ~200 user-labeled wild calls. Producing these labels takes ~30 seconds per call ≈ 100 minutes of focused human effort — not a trivial cost, especially for a single rater who is also the project lead. The plan asks when this labeling should happen, and there are two viable schedules with opposing trade-offs.

**Pole A — Label in parallel with Phase 1.1–1.3 training.** The model and the labels are both prepared simultaneously. When 1.3 finishes, 1.4 starts immediately because labels exist. Total project wall-time minimized. Risk: the user has to label "blind" — picking calls from the 5970 corpus without any model confidence scores to stratify by. The 200 calls may be unbalanced toward common syllable types because that's what the user encounters first, leaving minority classes severely undersampled. Also, the user labels during weeks when the project's main work is model training, which they may not want to do (motivation mismatch).

**Pole B — Label sequentially, after v2 is trained.** Project wall-time longer by however long labeling takes. But the user gets to bootstrap from v2's confidence scores: the labeling tool can present calls stratified by v2's prediction (5 from each predicted class, prioritizing low-confidence ones), giving an OOD evaluation set that has at least one call per Grimsley class. The user labels with model context, which both reduces cognitive load (no need to type-recall, just verify/correct) and produces a more useful evaluation distribution.

### When Each Pole Wins

| Situation | Pick |
|---|---|
| User has 100 free minutes during Phase 1.1–1.3 | A |
| User wants to label only once and not have it block 1.4 | A |
| 200-call distribution should reflect rare classes | B |
| User finds labeling-with-suggestions less cognitively demanding | B |
| Project deadline matters more than per-class coverage | A |

### Dissolution Attempts

The plan adopts pole B as the default — the model bootstrap argument wins. Wall-time penalty is small relative to total project duration (3–4 weeks), and per-class stratification is high-value because Phase 1.4 success criterion involves per-class precision/recall comparisons that depend on having calls in each class. Pole A's "label whatever you encounter" mode produces evaluation noise that compounds the small-sample-size problem ([[200 call OOD evaluation gives descriptive not inferential statistics at 12 classes]]).

The tension does dissolve under that lens: the user's labeling effort is the same in both poles, but pole B uses it better. Pole A only wins under a specific constraint (free time during 1.1–1.3 that won't be free later) that didn't apply here.

---

Source: [[lab_cnn_classifier_plan_2026-05-20]]

Relevant Notes:
- [[200 call OOD evaluation gives descriptive not inferential statistics at 12 classes]] — why per-class coverage matters
- [[held-out dual-rater verdicts serve as independent acceptance test for single-rater-trained classifiers]] — related labeling-protocol decision

Topics:
- [[experimental-methods]]
- [[classification-methodology]]
