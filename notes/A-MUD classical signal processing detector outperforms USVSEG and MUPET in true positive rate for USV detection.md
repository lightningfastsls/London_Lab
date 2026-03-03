---
description: "Runs on STx acoustic software and is 4-12x faster than manual segmentation — outperforms other classical methods when both TPR and false detection rates are considered"
type: finding
confidence: likely
meta_state: current
topics:
  - "[[detection]]"
  - "[[signal-processing]]"
---

# A-MUD classical signal processing detector outperforms USVSEG and MUPET in true positive rate for USV detection

A-MUD (Automatic Mouse Ultrasound Detector, 2017) is a classical signal-processing algorithm for USV detection that runs on STx acoustic software. In benchmarks comparing classical (non-deep-learning) detection methods, A-MUD outperformed both USVSEG and MUPET when both true positive rate and false detection rate are jointly considered. The method achieves 4-12x speedup over manual segmentation, making it practical for large recording datasets.

In the systematic comparison by Ivanenko et al. (2023), A-MUD achieved 90.6% precision / 80.0% recall, placing it behind U-Net (91.1%/92.1%) but ahead of DeepSqueak (66.4%/63.7%). This makes A-MUD the best-performing classical signal processing method in the benchmark, though deep learning approaches now surpass it.

However, A-MUD requires the proprietary STx software platform, limiting its accessibility compared to open-source alternatives like USVSEG (which has a Python port). The STx dependency means A-MUD cannot be integrated into automated Python pipelines, unlike our energy detector approach which uses similar signal-processing principles but is fully Python-native.

A-MUD's performance represents the ceiling of what classical signal processing can achieve for USV detection. Deep learning methods (DAS at 98%/99% precision/recall, SqueakOut at 90.22 Dice) now surpass it, confirming that the field's shift toward learned representations is warranted for maximum detection accuracy.

---

Source:
- usv-detection-methods-landscape-2024-2026-research-2026-02-28 (archived to archive/inbox/)

Relevant Notes:
- [[energy threshold at negative 60 dB is deliberately low to maximize recall in the first stage]] — our approach uses similar energy-based detection principles
- [[two-stage coarse-to-fine filtering is effective for imbalanced detection tasks]] — A-MUD is a single-stage approach, contrasting with our two-stage pipeline

Topics:
- [[detection]]
- [[signal-processing]]
