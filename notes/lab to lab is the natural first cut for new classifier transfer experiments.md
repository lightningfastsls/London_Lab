---
description: "When designing a transfer experiment with limited curated data, training on lab data and deploying on lab data minimizes the dimensions of domain gap — call duration, cage acoustics, age range are all matched — leaving model architecture as the only failure surface"
type: methodology
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[classification-methodology]]"
  - "[[wild-lab-vocal-comparison]]"
---

# Lab to lab is the natural first cut for new classifier transfer experiments

Every transfer learning experiment crosses one or more domain gaps. The recording-environment gap (cage acoustics), the population gap (lab vs wild mice), the age gap (neonatal vs adult), the protocol gap (microphone, sample rate, preprocessing), and the labeling gap (single-rater vs dual-rater) compound multiplicatively in their failure modes — a transfer failure could be caused by any one of them, and you can't tell which without controlled experiments to isolate the gaps.

The natural design discipline is to test the smallest-gap configuration first. For the lab CNN classifier, that's lab-to-lab: train on VocalMat's lab-recorded neonatal mouse data, evaluate on our own lab 131204 adult mouse data. The cage-acoustics gap is small (both are controlled environments with similar microphone hardware), the call-duration gap is small (both 30–200 ms range), the labeling gap is small (both are experimenter-labeled), and the population gap is intermediate (both are lab strains, ages differ).

If lab-to-lab fails, the architecture is the problem — the model literally cannot learn what we asked it to learn. Fix the model.
If lab-to-lab succeeds and wild deployment fails, one or more of (cage acoustics, call duration, age) is the cause. Each is independently testable: DANN addresses cage acoustics, patch-size variants address call duration, age-filtered training subsets address age. The triage is mechanical once lab-to-lab works.

The alternative — jumping straight to wild deployment — collapses all sources of failure into one number (κ=0.13) and gives no signal for which is the cause. The kappa-0.13 VocalMat transfer test (memory `project_vocalmat_transfer_test`) was an instance of this: it told us "transfer failed" without telling us why. Six months of mechanism investigation followed to identify patch-duration mismatch as the leading cause. A lab-to-lab experiment first would have shown the model worked fine on lab data and pointed at wild-specific features as the gap — much faster diagnosis.

The plan's three independent constraints (Apache 2.0 license + 845 dual-rater lab verdicts + soft-notch only calibrated for lab 131204) all happen to align with lab-first sequencing. The methodological reason is more fundamental than any of them: with limited curated data, minimize the domain gaps you're testing simultaneously.

---

Source: [[lab_cnn_classifier_plan_2026-05-20]]

Relevant Notes:
- [[classifiers trained on lab mice generalize poorly to wild mice requiring population-specific training data]] — the eventual generalization concern this defers
- [[patch-duration mismatch was the mechanistic cause of VocalMat-on-wild kappa 0.13 failure]] — the failure case that justified this sequencing
- [[strain monoculture is not the cause of lab to wild transfer failure when training spans multiple strains]] — eliminates one candidate domain gap

Topics:
- [[classification-methodology]]
- [[wild-lab-vocal-comparison]]
