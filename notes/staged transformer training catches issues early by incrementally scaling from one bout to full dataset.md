---
description: Four training stages (1 bout, 10 bouts, 100 bouts, full dataset) with explicit verification criteria at each stage prevent wasted compute on broken training runs.
type: method
confidence: likely
topics:
  - "[[representation-learning]]"
  - "[[experimental-methods]]"
---

# staged transformer training catches issues early by incrementally scaling from one bout to full dataset

The four-stage training protocol structures compute investment to catch failure modes as cheaply as possible. Stage A: overfit a single bout (~1 hour of compute) — verify loss decreases monotonically, no NaN/Inf in weights or activations, and that the model can memorize the training data. Stage B: overfit 10 bouts (~hours) — check that training loss still converges and that validation loss shows slight improvement over random initialization, indicating some generalization. Stage C: train 100 bouts (~half day) — validation loss should track training loss with a modest gap, no severe overfitting or underfitting. Stage D: full dataset (~1-2 days on A100) — early stopping based on validation loss plateau.

Each stage has explicit verification criteria rather than relying on general intuition. Stage A failure (cannot overfit one bout) indicates a bug in the training loop, data loading, or architecture — not a hyperparameter problem. Stage B failure (cannot overfit 10 bouts) suggests capacity issues. Stage C failure (validation loss diverges from training) suggests overfitting or learning rate schedule problems. This staging principle applies general machine learning practice to the specific context of 300kHz spectrogram data where a single "full run" is expensive.

Monitoring is designed for interpretability throughout: loss curves after every epoch, per-frequency-bin error (to identify which frequency ranges are hardest to predict), predicted vs actual spectrogram visualizations every 10 epochs (critical for catching MSE blurriness early — see [[MSE loss for next-column prediction may produce blurry spectrograms requiring a mixture density output head]]), and attention pattern visualizations. The staged approach connects directly to [[pre-norm transformer architecture improves training stability for spectrogram prediction]] — stability monitoring begins at Stage A, and architectural issues are resolved before scaling to expensive later stages.

---

Source: [ROADMAP](../ROADMAP.md), Phase 8
