---
description: "Single parameter T fits in seconds on held-out validation data — our fitted T=0.905 indicates mild overconfidence, consistent with Guo et al findings for well-trained small models"
type: method
confidence: proven
meta_state: current
created: 2026-03-29
topics:
  - "[[detection]]"
  - "[[training-methodology]]"
---

# Temperature scaling is the simplest effective calibration — one scalar divides logits before sigmoid

Temperature scaling works by dividing the model's logit output by a single learned scalar T before applying the sigmoid activation. When T > 1, the sigmoid curve is flattened, spreading probabilities away from 0 and 1 (reducing overconfidence). When T < 1, probabilities are sharpened toward the extremes (reducing underconfidence). The parameter T is fitted by minimizing negative log-likelihood on a held-out validation set, which typically converges in seconds because it is a one-dimensional convex optimization problem.

Our calibration pipeline (scripts/calibrate_temperature.py) fitted T = 0.905 on the matched-windows CNN. A temperature below 1.0 but close to it indicates mild overconfidence — the model's raw probabilities are slightly too sharp but not dramatically miscalibrated. This is consistent with Guo et al.'s findings that well-trained smaller models (our CNN is a compact architecture) tend to be less miscalibrated than large-scale models trained on ImageNet. The calibration improved Expected Calibration Error (ECE) from 0.024 to 0.014, cutting it nearly in half.

The near-unity temperature is actually good news for our pipeline, because it means the hysteresis thresholds optimized on raw probabilities should transfer reasonably well to calibrated probabilities without major re-tuning. Had T been far from 1.0 (say, 0.3 or 3.0), all downstream threshold values would have needed significant adjustment.

If future model iterations produce an abnormally low T (below 0.5), this would signal severe miscalibration that temperature scaling alone may not adequately address. In that case, Attended Temperature Scaling (Mozafari et al., 2019) offers a more flexible approach that learns attention-weighted temperatures across different input features, at the cost of more parameters and a larger validation set requirement.

---

Source:
- archive/inbox/post-processing-pipeline-research.md (2026-03-27)

Relevant Notes:
- [[modern CNNs are systematically miscalibrated — confidence does not match accuracy]] -- the foundational finding that motivates temperature scaling
- [[ROC AUC is invariant to temperature scaling but threshold interpretability improves]] -- what calibration does and does not change about model evaluation
- [[isotonic regression overfits on small validation sets — prefer temperature scaling]] -- why the simpler method is preferred for our sample size

Topics:
- [[detection]]
- [[training-methodology]]
