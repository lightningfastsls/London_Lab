---
type: enrichment
target_note: "[[recording-level splits prevent data leakage in USV classification]]"
source_task: lab_cnn_classifier_plan
addition: "Phase 1.1 of the lab CNN classifier plan specifies 80/10/10 stratified split with recording-level grouping. The cage-confound mechanism (cohorts share recording-environment signatures across all calls in a recording) is the specific failure mode being prevented, not just generic train/test correlation."
source_lines: "84-88"
created: 2026-05-21
---

# Enrichment 002: [[recording-level splits prevent data leakage in USV classification]]

Source: [[lab_cnn_classifier_plan_2026-05-20]] (lines 84-88)

## Reduce Notes

The existing note describes recording-level splits as a generic data-leakage prevention mechanism. The plan adds two concrete properties not currently captured:

1. **Specific split ratio**: 80/10/10 stratified train/val/test. The existing note mentions splits exist but doesn't specify proportions. Phase 1.1 binds these ratios to the lab CNN classifier pipeline.

2. **The cage-confound mechanism**: recording-level splits don't just prevent generic memorization of training samples in the test set. They specifically prevent the model from learning recording-environment ("cage") features that would otherwise appear as cohort identity. The existing note's framing ("model can cheat by memorizing recording-specific noise patterns") is correct but understates the *systematic* nature of the leakage — it's not random memorization, it's a discoverable shortcut that any model will find unless splits actively prevent it. See [[cage acoustics drive between-cohort spectrogram separation more than biology]] for the empirical demonstration that this leakage IS the dominant signal in unprepared data.

Rationale: enrichment rather than new note because the core claim "recording-level splits prevent data leakage" remains the same — the addition is mechanistic (why the leakage is severe enough to require the discipline) and specific (which ratios + which protocol). Without enrichment, the existing note reads as advice that applies in general; with enrichment, it's grounded in the lab CNN classifier's binding implementation.

---

## Enrich (pending)
## Reflect (pending)
## Reweave (pending)
## Verify (pending)
