---
description: Conflict between ADR-004 specifying 80/10/10 splits and ROADMAP Phase 9 specifying 70/15/15, both using recording-level splitting but with different ratios.
type: open-question
confidence: likely
topics:
  - "[[experimental-methods]]"
---

# split ratio inconsistency between DECISIONS.md 80-10-10 and ROADMAP Phase 9 70-15-15 needs resolution

DECISIONS.md ADR-004 specifies 80/10/10 train/validation/test splits, while ROADMAP Phase 9 (DatasetAssembler) specifies 70/15/15. Both documents correctly require recording-level splitting per [[recording-level splits prevent data leakage in USV classification]], but the actual proportions disagree. This is an unresolved inconsistency that will require an explicit decision before DatasetAssembler is implemented.

The trade-off is straightforward: larger validation and test sets (15% each at 70/15/15) provide more reliable metric estimates, which matters most when the dataset is small. With ~840 currently labeled candidates, 10% test = ~84 samples — a very small number from which to draw reliable F1 estimates. Moving to 15% test = ~126 samples, improving statistical reliability at the cost of 126 fewer training examples. As the dataset scales to 30K labels, 10% test = 3K samples, which is more than sufficient, making the 80/10/10 ratio more practical at scale.

A sensible resolution is to adopt 70/15/15 for the current small-data regime and plan to switch to 80/10/10 once the dataset exceeds ~15K labels, formalizing this as a conditional in ADR-004 rather than treating it as a fixed ratio. This would need to be documented explicitly so that DatasetAssembler implements the correct ratio for the current label count and so that metric comparisons across training runs use the same split consistently.

This question intersects with [[recording-level splits reduce effective training set size but prevent data leakage]] (which explains why recording-level splitting makes the effective training set smaller than the label count suggests) and [[model size should scale with labeled dataset size to balance underfitting and overfitting]] (where split ratio affects the training set size that determines model capacity).

---

Source: [[ROADMAP.md]]
