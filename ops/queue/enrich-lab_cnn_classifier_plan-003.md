---
type: enrichment
target_note: "[[Perch 2.0 trained on 14795 species achieves state of the art bioacoustic embeddings that transfer across taxa]]"
source_task: lab_cnn_classifier_plan
addition: "Recent (2025) evidence suggests Perch 2.0 bioacoustic-pretrained embeddings often outperform ImageNet-pretrained CNNs on cross-domain bioacoustic tasks; this is the rationale for the Phase 1.2 Perch 2.0 sidequest (linear probe in parallel with ResNet-18 baseline)."
source_lines: "138, 167, 300"
created: 2026-05-21
---

# Enrichment 003: [[Perch 2.0 trained on 14795 species achieves state of the art bioacoustic embeddings that transfer across taxa]]

Source: [[lab_cnn_classifier_plan_2026-05-20]] (lines 138, 167, 300)

## Reduce Notes

The existing note describes Perch 2.0 as a state-of-the-art bioacoustic embedding model trained on 14,795 species. The plan adds a comparative claim not currently captured:

1. **Cross-domain transfer superiority over ImageNet-pretrained CNNs**: The plan cites recent 2025 evidence (arXiv:2512.03219) that bioacoustic-pretrained embeddings often beat ImageNet-pretrained CNNs on cross-domain bioacoustic tasks. The existing note's claim "state of the art bioacoustic embeddings" is true but doesn't address the practical question: is Perch better than the standard ImageNet-ResNet-18 baseline that bioacoustic projects default to? The 2025 evidence says: often yes.

2. **Operationalized as a 1-day sidequest** (D3 in the plan): The plan budgets a parallel linear-probe evaluation in Phase 1.2 — embed every training/val/test patch with frozen Perch, train a 12-class linear classifier on those embeddings, compare macro F1 vs the ResNet-18 baseline. This is the cheapest possible test of the Perch-beats-ImageNet hypothesis on our specific data. The existing note doesn't mention this practical application path.

Rationale: enrichment rather than new note because the central claim about Perch 2.0's quality remains. The addition is application context — when and why to use Perch over alternatives — which is exactly the kind of practical detail that makes the existing note actionable.

---

## Enrich (pending)
## Reflect (pending)
## Reweave (pending)
## Verify (pending)
