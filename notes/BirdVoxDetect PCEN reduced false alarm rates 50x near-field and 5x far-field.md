---
description: "Dramatic improvement in bird flight call detection validates PCEN for variable-distance bioacoustic recording — the near-field vs far-field asymmetry suggests distance-dependent noise is the primary confounder"
type: finding
confidence: proven
conditions: []
meta_state: current
created: 2026-03-29
topics:
  - "[[detection]]"
  - "[[signal-processing]]"
---

# BirdVoxDetect PCEN reduced false alarm rates 50x near-field and 5x far-field

The BirdVoxDetect system for bird flight call detection provides the most compelling empirical evidence for PCEN's effectiveness in bioacoustic applications. When replacing log-magnitude spectrograms with PCEN, false alarm rates dropped by a factor of 50 for near-field recordings and by a factor of 5 for far-field recordings. These numbers are striking because they suggest that log-magnitude spectrograms are severely suboptimal when signal levels vary across recordings — the 50x near-field improvement indicates that strong signals were generating abundant false positives under log-magnitude representation, likely because high-energy non-vocal transients were being amplified rather than normalized. This directly relates to [[CNN false positives cluster in noisy regions where energy patterns superficially resemble USV structure]] — PCEN's adaptive normalization would suppress exactly the kind of high-energy non-vocal transients that trigger structural noise mimics in our CNN.

The asymmetry between near-field and far-field improvements is itself informative. Near-field recordings have higher SNR but also higher absolute energy variation, because the source-to-microphone distance matters more when the source is close. PCEN's adaptive gain normalization therefore has more room to improve near-field performance. Far-field recordings already have low SNR, so the adaptive normalization helps less — there is simply less signal to separate from noise regardless of the normalization scheme. This suggests that distance-dependent energy variation is the primary confounder that PCEN addresses, rather than frequency-dependent noise patterns.

For our USV recordings, mouse-to-microphone distance varies continuously within a cage as animals move, creating exactly the kind of variable-distance scenario where PCEN excels. This variable distance also means [[low-amplitude and short-duration USVs are the primary source of false negatives and training bias]] — the same mouse call recorded from far away looks like a low-amplitude USV, and PCEN's adaptive gain boost for quiet signals could improve detection of these distance-attenuated calls. Unlike bird monitoring stations where microphone placement is fixed and distance variation comes from different bird positions, our cage recordings have the additional complication that the same mouse produces calls at different distances within a single recording session. This makes the case for PCEN even stronger in our application, but the requirement for model retraining means it must wait for the next training iteration as described in [[PCEN is the gold standard adaptive normalization in bioacoustic literature]].

---

Source:
- archive/inbox/post-processing-pipeline-research.md (2026-03-27)

Relevant Notes:
- [[PCEN is the gold standard adaptive normalization in bioacoustic literature]] -- the method description and implementation constraints for our pipeline
- [[PCEN normalization is more robust than log-mel spectrograms for few-shot bioacoustic scenarios]] -- convergent evidence from DCASE 2022: PCEN's adaptive gain also improves few-shot generalization across recording conditions, not just false alarm rates
- [[per-recording normalization compensates for varying noise floors across recording sessions]] -- our current approach that partially addresses the same problem
- [[per-frequency-bin normalization removes frequency-dependent energy bias in spectrogram input]] -- our current per-bin z-score normalization is simpler but static; PCEN would replace this with adaptive per-channel normalization that responds to local energy dynamics
- [[shared lab space without sound attenuation explains why noise robustness is a primary design constraint]] -- the environmental context that makes variable-distance normalization critical
- [[CNN false positives cluster in noisy regions where energy patterns superficially resemble USV structure]] -- PCEN would suppress the high-energy transients that trigger structural noise mimics, addressing the 50x near-field improvement mechanism
- [[low-amplitude and short-duration USVs are the primary source of false negatives and training bias]] -- PCEN's adaptive gain boost for quiet signals could improve detection of distance-attenuated calls
- [[noisy USVs are valid positive training samples because the model must learn detection in degraded conditions]] -- BirdVoxDetect's 50x near-field improvement validates that variable-energy signals are exactly what adaptive normalization handles; PCEN would make noisy USVs in training data easier for the model to distinguish from non-USV noise

Topics:
- [[detection]]
- [[signal-processing]]
