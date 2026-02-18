---
description: Train-set mean/std applied to val/test sets; computing statistics on val/test data leaks future information and inflates performance metrics.
type: decision
confidence: proven
topics:
  - "[[experimental-methods]]"
---

# normalization statistics must be computed on training set only to prevent data leakage

Data leakage in machine learning occurs when information from the evaluation set influences the model or its preprocessing pipeline. For spectrogram normalization, the risk is subtle: if mean and std vectors are computed across the full dataset rather than the training split alone, the normalization transform has "seen" the validation and test distributions before evaluation. This inflates reported metrics by making the model's input distribution at test time slightly more favorable than it would be in genuine deployment.

The correct procedure is to compute per-bin mean and std exclusively on training set spectrograms, then apply those fixed statistics to validation and test spectrograms as a frozen transform. This mirrors the deployment scenario, where the model encounters audio from recordings it has never seen, normalized using statistics derived from whatever training data was available at the time of training.

This principle is the preprocessing analog of [[recording-level splits prevent data leakage in USV classification]], which applies at the sample assignment level. That decision ensures no recording contributes samples to both training and evaluation splits. The current decision ensures the preprocessing statistics themselves do not encode evaluation-set information. Both constraints address the same underlying concern: that the measured performance should reflect generalization to genuinely unseen data.

In practice, the training-set normalization statistics are saved to an npz file alongside the model checkpoint. See [[per-frequency-bin normalization removes frequency-dependent energy bias in spectrogram input]] for the formula and implementation details. Without the saved statistics file, a loaded model cannot be applied correctly to new data.

---

Source: [[ROADMAP.md]], Phase 2
