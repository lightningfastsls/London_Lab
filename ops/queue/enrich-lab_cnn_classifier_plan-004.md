---
type: enrichment
target_note: "[[ResNets outperform Vision Transformers for USV classification on neonatal mouse data]]"
source_task: lab_cnn_classifier_plan
addition: "ResNet-18 is the right capacity for ~13k labeled examples specifically; with smaller datasets ViTs underfit, with larger datasets EfficientNet-B0 (5.3M params, lower overfit risk) is the documented fallback. The capacity reasoning is explicit, not just a SOTA-on-benchmark argument."
source_lines: "24, 144"
created: 2026-05-21
---

# Enrichment 004: [[ResNets outperform Vision Transformers for USV classification on neonatal mouse data]]

Source: [[lab_cnn_classifier_plan_2026-05-20]] (lines 24, 144)

## Reduce Notes

The existing note describes ResNets outperforming ViTs on neonatal mouse USV classification — a benchmark-grounded claim. The plan adds two layers of reasoning not currently captured:

1. **Capacity reasoning by dataset size**: The plan justifies ResNet-18 specifically as "right capacity for ~13k examples" (line 24). This is an explicit mid-sized-dataset argument that wouldn't apply at 1k or 1M examples. The existing note's framing makes ResNet-vs-ViT sound like a categorical superiority — the plan reveals it's a dataset-size-conditional choice. At 13k examples, ViTs would underfit (too many parameters, insufficient data). At 1M examples, ViTs might win. The plan's choice is tied to the specific size of VocalMat's training set.

2. **Documented fallback (EfficientNet-B0)**: The plan names EfficientNet-B0 (5.3M params, lower overfit risk) as the fallback if ResNet-18 overfits. The existing note doesn't mention fallback architectures or under what conditions to swap. This is methodologically important — overfitting on ~13k examples is a real possibility, and having a pre-named fallback prevents bikeshedding when it happens.

Rationale: enrichment rather than new note because the central architectural comparison stands. The additions are scoping conditions (dataset size that justifies the choice) and contingency (what to do if it doesn't work) — both make the existing note more usable for future projects without contradicting it.

---

## Enrich (pending)
## Reflect (pending)
## Reweave (pending)
## Verify (pending)
