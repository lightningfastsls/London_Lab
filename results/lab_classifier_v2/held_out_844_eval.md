# Held-out 844 — real patch-loading evaluation (Module 18.4 bonus)

**Checkpoint:** `results/lab_classifier_v2/best.pt`
**Set:** 844 patches (705 usv / 139 noise = 83.5% usv)
**Collapse rule:** 12-class argmax == 'Noise' (index 0) → noise, else usv.

> Pooled accuracy is reported but **misleading** on an 83.5%-usv set: a trivial 'always usv' classifier already scores 0.8353. Read the **per-class** rows.

## Per-class metrics

| Class | Recall | Precision | F1 |
|---|---|---|---|
| **noise** (recall = specificity) | 0.1583 | 0.8148 | 0.2651 |
| **usv** (recall = sensitivity) | 0.9929 | 0.8568 | 0.9198 |

- **Balanced accuracy:** 0.5756
- **Macro-F1 (binary):** 0.5925
- Pooled accuracy (misleading): 0.8555 vs always-usv baseline 0.8353

## Confusion (rows = true, cols = predicted)

| true \ pred | noise | usv |
|---|---|---|
| **noise** | 22 | 117 |
| **usv** | 5 | 700 |

## 12-class argmax breakdown by ground-truth verdict

| Syllable class | true_usv | true_noise |
|---|---|---|
| Noise | 5 | 22 |
| Step up | 160 | 11 |
| Down-FM | 72 | 4 |
| Short | 66 | 41 |
| Chevron | 9 | 4 |
| Up-FM | 62 | 5 |
| Flat | 89 | 39 |
| Two steps | 11 | 2 |
| Step down | 112 | 5 |
| Complex | 17 | 1 |
| Reverse Chevron | 22 | 4 |
| Multi-steps | 80 | 1 |
