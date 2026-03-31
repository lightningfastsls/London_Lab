---
description: "Isotonic regression has O(n) effective parameters and requires thousands of samples — our 2139 validation samples are borderline, making the 1-parameter temperature approach safer"
type: method
confidence: likely
meta_state: current
created: 2026-03-29
topics:
  - "[[detection]]"
  - "[[training-methodology]]"
---

# Isotonic regression overfits on small validation sets — prefer temperature scaling

Calibration methods sit on a spectrum of complexity. Temperature scaling uses one parameter (T), Platt scaling uses two (slope and intercept of a logistic regression on logits), and isotonic regression uses up to O(n) effective parameters — it fits a piecewise-constant non-decreasing function through the validation data with as many segments as needed. This flexibility is isotonic regression's strength on large datasets, because it can correct arbitrarily complex miscalibration patterns that a single temperature cannot capture. But on small datasets, that flexibility becomes a liability: the fitted function overfits to the validation set's idiosyncrasies rather than capturing the model's true calibration curve.

Our validation set contains 2139 samples. This is technically enough for isotonic regression to run without crashing, but it is borderline for reliable generalization. Empirical studies (including Guo et al., 2017) show that temperature scaling matches or outperforms isotonic regression on datasets below roughly 5000 validation samples, precisely because the single parameter cannot overfit. With 2139 samples and a binary classification task, the effective sample sizes for the two classes are even smaller (split by class prevalence), which further limits the data available for fitting a complex calibration curve.

Platt scaling occupies a middle ground with two parameters (sigmoid fit on logits), offering slightly more flexibility than temperature scaling without isotonic regression's risk of overfitting. However, for our case where T = 0.905 (near unity), Platt scaling's extra parameter would likely converge to a near-identity transformation anyway, providing no practical benefit.

The recommendation is therefore clear: use temperature scaling for our current pipeline and revisit isotonic regression only if the validation set grows substantially (to 5000+ samples) or if temperature scaling produces poor ECE results on a future model iteration.

---

Source:
- archive/inbox/post-processing-pipeline-research.md (2026-03-27)

Relevant Notes:
- [[temperature scaling is the simplest effective calibration — one scalar divides logits before sigmoid]] -- the preferred method that this note argues in favor of
- [[modern CNNs are systematically miscalibrated — confidence does not match accuracy]] -- the foundational finding motivating calibration method selection

Topics:
- [[detection]]
- [[training-methodology]]
