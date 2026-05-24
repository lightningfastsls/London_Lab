---
description: "VocalMat's 0.22s spectrogram patch assumes lab-range call durations of 30-200ms; wild-mouse calls of 30-50ms get embedded in 82 percent silence and the CNN sees noise, not signal"
type: claim
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[wild-lab-vocal-comparison]]"
  - "[[classification-tools]]"
  - "[[detection]]"
---

# Patch-duration mismatch was the mechanistic cause of VocalMat-on-wild kappa 0.13 failure

The 2026-05-20 VocalMat transfer test (memory `project_vocalmat_transfer_test`) found κ=0.13 agreement between VocalMat's CNN classifier applied to our wild 5970 data and a human reference. That looks like "the model doesn't transfer" — a high-level conclusion that hides the actual mechanism. The mechanism is more specific: VocalMat's spectrogram input format uses a 0.22-second patch, designed around lab mouse call durations of 30–200 ms. For lab data, the call fills most of the patch and the surrounding silence is small. For wild 5970 calls (median 30–50 ms), the call occupies less than 20% of the patch — the other 82% is whatever ambient signal exists around the call. The CNN, trained to discriminate syllable types from spectrograms where most pixels carry call energy, sees mostly noise pixels at inference time and produces near-random predictions weighted by class prior.

This is a mechanistic explanation, not a domain-gap one. The classifier "knows" the syllable types perfectly well — it just isn't being shown them. A 0.08-second variant patch with the same training data and same architecture would plausibly transfer better, because the wild call would occupy a larger fraction of the input. This is open question 1 in the lab CNN classifier plan: build the short-patch variant during Phase 1.4 if the standard-patch version doesn't generalize.

The general pattern: when a model fails to transfer between two data regimes, distinguish the **representation gap** (model can't see the same feature in both) from the **learning gap** (model never learned the relevant feature). The former is fixed by re-engineering inputs; the latter requires re-engineering training. Misdiagnosing the patch-mismatch case as a learning gap would have justified retraining on wild data — expensive and unnecessary when the actual fix is changing one preprocessing parameter.

---

Source: [[lab_cnn_classifier_plan_2026-05-20]]

Relevant Notes:
- [[classifiers trained on lab mice generalize poorly to wild mice requiring population-specific training data]] — the original conclusion this refines; the cause is patch geometry, not the training distribution
- [[strain monoculture is not the cause of lab to wild transfer failure when training spans multiple strains]] — another candidate explanation that's ruled out by the data

Topics:
- [[wild-lab-vocal-comparison]]
- [[classification-tools]]
- [[detection]]
