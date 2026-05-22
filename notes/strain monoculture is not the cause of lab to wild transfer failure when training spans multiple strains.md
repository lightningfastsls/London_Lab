---
description: "A lab-trained classifier that fails on wild mice is often blamed on training data using only one strain — but if training spans 5+ strains the strain hypothesis is mechanically ruled out, redirecting attention to cage acoustics and call-duration distribution"
type: claim
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[wild-lab-vocal-comparison]]"
  - "[[classification-tools]]"
---

# Strain monoculture is not the cause of lab to wild transfer failure when training spans multiple strains

A common first hypothesis for why a lab-trained mouse USV classifier fails on wild data is "the training set only used one strain (typically C57BL/6J), so the model overfit to that strain's vocal repertoire." This is a reasonable concern for many lab-bred-only models — but it is mechanically false for the VocalMat dataset and any other multi-strain training corpus. VocalMat's 12,954 training images come from 5 strains (C57BL/6J, NZO/HlLtJ, 129S1/SvImJ, NOD/ShiLtJ, PWK/PhJ) across 3 neonatal ages × both sexes — approximately 30 sub-cohorts. The training distribution is already heterogeneous in strain.

If a model trained on 5 strains fails on a 6th strain, the failure mode cannot be "the model only knows the training strain." There aren't enough degrees of freedom for that explanation to be correct — the model has demonstrably learned features that generalize across at least 5 strain boundaries. The failure must instead be in a dimension orthogonal to strain: cage acoustics (the recording environment differs from VocalMat's labs), call-duration distribution (wild calls are 30–50 ms; lab range is 30–200 ms; see [[patch-duration mismatch was the mechanistic cause of VocalMat-on-wild kappa 0.13 failure]]), age (VocalMat is neonatal P5–P15; wild + most lab data is adult), or sample rate differences (the model trained on 250 kHz audio; our pipeline natively records at 300 kHz and must resample).

The lesson is methodological. When diagnosing transfer failure, count the dimensions of training-data variation explicitly before reaching for a "monoculture" explanation. A 5-strain training set rules out strain. A 3-age training set rules out age (within the trained range). A multi-cage training set rules out cage. What's left is the actual failure mode. The Phase 1.4 plan explicitly names this: "strain monoculture is NOT the issue (VocalMat already spans 5 strains); cage acoustics + call-duration distribution are the real domain gaps."

---

Source: [[lab_cnn_classifier_plan_2026-05-20]]

Relevant Notes:
- [[classifiers trained on lab mice generalize poorly to wild mice requiring population-specific training data]] — the broader claim this constrains
- [[patch-duration mismatch was the mechanistic cause of VocalMat-on-wild kappa 0.13 failure]] — the actual mechanism for one specific case
- [[cage acoustics drive between-cohort spectrogram separation more than biology]] — the second-most-likely real mechanism for transfer failure

Topics:
- [[wild-lab-vocal-comparison]]
- [[classification-tools]]
