---
description: "2024 neonatal study detection algorithm uses entropy rather than energy thresholding — high precision without neural networks; paired with ResNet classifier for 86.79% classification accuracy"
type: method
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[detection-landscape]]"
  - "[[signal-processing]]"
---

# Entropy-based USV detection achieves 94.9 percent recall and 99.3 percent precision as a classical signal processing alternative

A 2024 neonatal USV analysis pipeline (JASA) used an entropy-based detection algorithm achieving 94.9% recall and 99.3% precision — metrics that exceed our CNN baseline of 89.7% precision / 93.8% recall. Unlike energy thresholding, entropy-based detection measures the spectral complexity of each frame: USVs have low entropy (concentrated energy in narrow frequency bands) while noise has high entropy (energy spread across many bins).

This approach is conceptually similar to our bandwidth filter but more principled: rather than rejecting candidates exceeding a bandwidth threshold, entropy directly measures how "peaked" versus "spread" the spectral energy distribution is. Since [[peak energy mode detects narrow-band USVs better than mean energy across the frequency band]], entropy-based detection extends this principle from peak detection to full distributional analysis.

The combination of entropy detection (94.9% recall, 99.3% precision) with ResNet classification (86.79% accuracy) in the neonatal study represents an alternative two-stage architecture to our energy detector + CNN approach. The higher precision of entropy detection (99.3% vs our energy detector's permissive threshold) means fewer false positives reach the classifier, reducing the class imbalance problem.

---

Source: usv-detection-methods-landscape-2024-2026-research-2026-02-28 (archived to archive/inbox/)

Relevant Notes:
- [[peak energy mode detects narrow-band USVs better than mean energy across the frequency band]] -- entropy extends the "narrow-band detection" concept
- [[energy threshold at negative 60 dB is deliberately low to maximize recall in the first stage]] -- our approach prioritizes recall; entropy-based could offer better precision
- [[maximum bandwidth filter of 20 kHz rejects broadband noise in energy detection]] -- bandwidth filter is a simple approximation of what entropy captures rigorously

Topics:
- [[detection]]
- [[signal-processing]]
