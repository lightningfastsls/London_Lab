---
type: enrichment
target_note: "[[VocalMat provides 12954 labeled USV spectrograms freely available as training data]]"
source_task: lab_cnn_classifier_plan
addition: "(1) Training labels are single-rater; expect 5-15% noise in minority classes. (2) Test set of 4441 USVs uses dual-rater consensus with arbitration (gold-standard). (3) Class imbalance ratio is 24.5x between Step-up (n=1814) and Multi-steps (n=74). (4) Training data spans 5 strains (C57BL/6J, NZO/HlLtJ, 129S1/SvImJ, NOD/ShiLtJ, PWK/PhJ) x 3 neonatal ages x both sexes -- ~30 sub-cohorts."
source_lines: "30-55"
created: 2026-05-21
---

# Enrichment 001: [[VocalMat provides 12954 labeled USV spectrograms freely available as training data]]

Source: [[lab_cnn_classifier_plan_2026-05-20]] (lines 30-55)

## Reduce Notes

The existing note describes VocalMat as a 12,954-image labeled training dataset. The plan adds four classes of detail not currently captured:

1. **Labeling protocol asymmetry** (lines 32-33): Training set is single-rater (one experienced experimenter), test set is dual-rater + consensus arbitration. The two subsets have different label quality and should not be treated as interchangeable evaluations. This is methodologically significant for any downstream use — see [[VocalMat test set quality dual-rater consensus exceeds training set quality single-rater]].

2. **Expected label noise rate** (line 32): 5–15% in minority classes is the standard literature expectation for single-rater protocols. The existing note does not mention noise expectations; consumers of the dataset may implicitly assume gold-standard labels.

3. **Class imbalance** (lines 40-55): 24.5× imbalance ratio between Step-up and Multi-steps. The existing note's "12954 labeled images" framing makes the dataset sound uniform; the per-class table reveals which classes have ~1,800 examples vs ~74. This drives the binding minority-class strategy question (D5 in the plan) — see [[class imbalance 24x breaks naive cross-entropy and requires class-weighted plus focal plus oversampling]].

4. **Strain/age/sex heterogeneity** (line 34): 5 strains × 3 neonatal ages × both sexes = ~30 sub-cohorts. This is significant for cage-invariance discussions — it means strain monoculture is NOT the cause of any downstream transfer failure (see [[strain monoculture is not the cause of lab to wild transfer failure when training spans multiple strains]]).

Rationale: enrichment rather than new note because the central claim "VocalMat provides 12,954 labeled spectrograms" remains the same — the four additions are properties of that resource, not different resources. They make the existing note more useful without fragmenting the graph.

---

## Enrich (pending)
## Reflect (pending)
## Reweave (pending)
## Verify (pending)
