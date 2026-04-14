---
description: "Ulloa et al 2021 open-source Python library — potential reference implementation for validating our hysteresis logic against an established ecological acoustics tool"
type: baseline
confidence: proven
meta_state: current
created: 2026-03-29
topics:
  - "[[detection]]"
  - "[[classification-tools]]"
---

# Scikit-maad implements double-threshold hysteresis binarization for ecological acoustics

Scikit-maad (Ulloa et al., 2021) is an open-source Python library for quantitative soundscape analysis that includes a double-threshold hysteresis binarization function. The library is designed for ecological acoustics — analyzing biodiversity through sound recordings — and provides spectral, temporal, and spatial indices for characterizing soundscapes. The hysteresis binarization is one component of its broader detection and segmentation toolkit.

The existence of this implementation matters for two reasons. First, it serves as independent validation that dual-threshold hysteresis is a recognized technique in bioacoustic signal processing, not an ad hoc invention. While our note on the absence of hysteresis in mouse USV tools ([[no existing mouse USV tool uses explicit hysteresis for event detection]]) establishes a gap in that specific subfield, scikit-maad confirms the approach is established in the broader ecological acoustics community. Second, it provides a potential reference implementation for cross-validation — if our hysteresis module produces different results from scikit-maad on the same input, that discrepancy would warrant investigation.

However, scikit-maad operates on spectrogram-level binary masks rather than on CNN probability streams, so direct comparison requires adaptation. Our hysteresis operates on a 1D time series of calibrated CNN probabilities (calibrated via [[temperature scaling is the simplest effective calibration — one scalar divides logits before sigmoid]]), whereas scikit-maad's version operates on 2D time-frequency representations. The core logic (onset threshold to start, sustain threshold to maintain) is the same, but the dimensionality differs. Therefore, scikit-maad is better used as a conceptual reference and sanity check than as a drop-in replacement or direct benchmark.

---

Source:
- archive/inbox/post-processing-pipeline-research.md (2026-03-27)

Relevant Notes:
- [[no existing mouse USV tool uses explicit hysteresis for event detection]] -- the gap in mouse USV tools that scikit-maad's broader ecological acoustics work helps contextualize
- [[hysteresis subsumes gap-filling and minimum duration as special cases of dual-threshold logic]] -- our implementation that could be cross-validated against scikit-maad's approach
- [[DCASE class-dependent post-processing parameters improved F1 from 37 to 44 percent]] -- validates that careful post-processing optimization yields large gains; scikit-maad provides the established implementation pattern for one such technique
- [[two-stage coarse-to-fine filtering is effective for imbalanced detection tasks]] -- scikit-maad's hysteresis binarization is one instantiation of the coarse-to-fine pattern applied at the spectrogram level
- [[modern CNNs are systematically miscalibrated — confidence does not match accuracy]] -- scikit-maad operates on spectrogram masks rather than probability streams, sidestepping the calibration problem that affects our CNN-probability-based hysteresis

Topics:
- [[detection]]
- [[classification-tools]]
