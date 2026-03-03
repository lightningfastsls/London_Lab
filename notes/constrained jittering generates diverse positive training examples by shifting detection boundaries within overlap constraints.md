---
description: Five jittered copies per positive detection with 40ms window and minimum 50% overlap with original; multiplies positive samples while preserving USV signal alignment.
type: method
confidence: proven
topics:
  - "[[experimental-methods]]"
---

# constrained jittering generates diverse positive training examples by shifting detection boundaries within overlap constraints

Positive training examples for the USV CNN come from manually labeled detections. Because labeling is time-intensive, the raw count of positive samples is always smaller than the count of negative samples, creating class imbalance. One approach to address this asymmetry is data augmentation that generates synthetic variations of confirmed positive examples without introducing false labels.

Constrained jittering produces N=5 new positive examples per original detection by randomly shifting the extraction window within a 40ms range, subject to the constraint that the shifted window must overlap the original detection by at least 50%. This constraint is critical: an unconstrained shift could produce a window that contains mostly silence or noise rather than the actual USV call, converting a positive example into a mislabeled negative. The 50% overlap threshold ensures the core USV signal remains present in each jittered variant. A context padding of 20ms is added on each side before the window is passed to the CNN.

The practical effect is a 5x multiplication of positive sample counts before training. Combined with [[3x class weight boost compensates for USV class imbalance in CNN training]], this addresses class imbalance from two angles: increasing the raw count of positive examples and increasing the loss weight assigned to each positive example during training. These two mechanisms are complementary rather than redundant.

Jittering applies only to positive examples. Negatives are sampled fresh from unannotated regions at each training cycle, providing variety through sampling diversity rather than synthetic augmentation. The technique is chosen over alternatives such as random cropping because the constraint mechanism preserves temporal alignment with the actual USV event, which matters given that [[recording-level splits reduce effective training set size but prevent data leakage]] already reduces the effective positive pool.

---

Source: [ROADMAP](../ROADMAP.md), Phase 2
