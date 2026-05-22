---
type: enrichment
target_note: "[[per-recording normalization compensates for varying noise floors across recording sessions]]"
source_task: lab_cnn_classifier_plan
addition: "The per-recording Z-score implementation exists in production at src/usv_spectrogram/postprocessing/normalization.py but is currently DORMANT — not wired into any active pipeline. The Phase 1.0 cleaning validation gate activates it for the first time outside production, as one of 4 cleaning layers. Its role in the classifier pipeline is cage-invariance (one of 3 defenses against the cage-acoustics confound)."
source_lines: "64, 122"
created: 2026-05-21
---

# Enrichment 005: [[per-recording normalization compensates for varying noise floors across recording sessions]]

Source: [[lab_cnn_classifier_plan_2026-05-20]] (lines 64, 122)

## Reduce Notes

The existing note describes per-recording normalization as a noise-floor compensation mechanism. The plan adds two operational facts not currently captured:

1. **Dormant production status**: The per-recording Z-score implementation EXISTS at `src/usv_spectrogram/postprocessing/normalization.py` but is NOT wired into any active pipeline. The plan documents this as part of the "Existing Cleaning Infrastructure" inventory — labeled "Implemented but dormant — wire in" (line 64). This is methodologically important: future contributors looking at the cleaning stack should know this layer is implemented-but-unused, not missing. The lab CNN classifier (Module 18.1) activates it for the first time outside production, as one of 4 layers in the cleaning_pipeline.py orchestrator.

2. **Role in cage-invariance defense-in-depth**: The plan positions per-recording Z-score as one of THREE independent defenses against the cage-acoustics confound — alongside soft-notch (cage-specific) and DANN gradient-reversal (architectural). The existing note frames per-recording normalization as a noise-floor-compensation technique; the plan reveals it's also a cage-invariance technique because per-recording statistics absorb between-cage differences in absolute power. This dual purpose is worth recording — see [[cage acoustics drive between-cohort spectrogram separation more than biology]] for the broader confound this addresses.

Rationale: enrichment rather than new note because the core claim about per-recording normalization stands. The additions are status (dormant in production) and scope (cage-invariance as a second purpose beyond noise floor) — both make the existing note more accurate without contradicting it.

---

## Enrich (pending)
## Reflect (pending)
## Reweave (pending)
## Verify (pending)
