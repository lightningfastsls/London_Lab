---
description: Two-stage detect-then-classify architecture — permissive energy detector maximizes recall, CNN classifier filters for precision, with temporal grouping into bouts
type: moc
topics: "[[index]]"
---

# detection

How we find USVs in raw audio. The pipeline uses a two-stage architecture: a permissive energy detector generates candidates, then a CNN classifier filters for precision. Segment continuity and bout extraction add temporal structure.

## Core Ideas
- [[two-stage detection uses permissive energy detector followed by CNN precision filter]] -- the architectural pattern: energy for recall, CNN for precision
- [[20-120 kHz detection range pads the mouse USV band to avoid clipping edge-case calls]] -- padded 30-110 kHz band defining the frequency domain for detection
- [[energy threshold at negative 60 dB is deliberately low to maximize recall in the first stage]] -- permissive first stage by design
- [[peak energy mode detects narrow-band USVs better than mean energy across the frequency band]] -- max energy per frame avoids signal dilution
- [[maximum bandwidth filter of 20 kHz rejects broadband noise in energy detection]] -- rejects candidates spanning more than 20 kHz
- [[segment continuity bridges brief amplitude dips that fragment single USVs]] -- 5 ms gap bridging with frequency/energy tolerance
- [[bout-level spectrograms preserve inter-USV timing context for transformer training]] -- grouping USVs into behavioral episodes
- [[bout gap threshold of 500 ms groups temporally clustered USVs while separating distinct episodes]] -- the specific grouping parameter
- [[two-stage coarse-to-fine filtering is effective for imbalanced detection tasks]] -- the general pattern behind our pipeline
- [[recall versus precision tradeoff in two-stage USV detection]] -- the designed tradeoff between detection stages
- [[batch detection with skip-existing enables incremental processing of large WAV collections]] -- headless processing of ~6,500 WAV files with error recovery
- [[CNN baseline of 89.7 percent precision and 93.8 percent recall at threshold 0.05 validates the two-stage detection approach]] -- F1 91.7% baseline on ~840 labels
- [[noise-interrupted long USVs get split into two detections by the CNN sliding window]] -- CNN-level splitting of long calls with noisy gaps, distinct from energy detector fragmentation
- [[harmonics of a USV are treated as one call not multiple detections]] -- labeling convention: fundamental + harmonic = one call
- [[overlapping calls from multiple mice are labeled positive because USV presence is the classification target not individual identity]] -- detection target is presence, not source attribution
- [[implicit two-tier labeling emerges from CNN probability scores versus human binary overrides]] -- CNN scores + human overrides create two confidence tiers
- [[low-amplitude and short-duration USVs are the primary source of false negatives and training bias]] -- faint/short calls are the hardest to detect
- [[CNN false positives cluster in noisy regions where energy patterns superficially resemble USV structure]] -- noise structural mimicry triggers false positives

## Post-Processing Pipeline
- [[hysteresis subsumes gap-filling and minimum duration as special cases of dual-threshold logic]] -- dual-threshold detection unifies three post-hoc filters into one principled mechanism
- [[no existing mouse USV tool uses explicit hysteresis for event detection]] -- landscape gap: DeepSqueak, DAS, VocalMat, USVSEG, MUPET all use single threshold + gap-fill
- [[scikit-maad implements double-threshold hysteresis binarization for ecological acoustics]] -- independent validation of hysteresis in broader bioacoustics
- [[DCASE class-dependent post-processing parameters improved F1 from 37 to 44 percent]] -- post-processing parameter optimization yields large gains without model changes

## Calibration
- [[modern CNNs are systematically miscalibrated — confidence does not match accuracy]] -- Guo et al 2017: overconfidence distorts threshold semantics
- [[temperature scaling is the simplest effective calibration — one scalar divides logits before sigmoid]] -- our T=0.905 indicates mild overconfidence; ECE halved
- [[isotonic regression overfits on small validation sets — prefer temperature scaling]] -- 2139 validation samples are borderline for isotonic; 1-param approach safer
- [[ROC AUC is invariant to temperature scaling but threshold interpretability improves]] -- calibration improves threshold portability, not discrimination

## Evaluation Methodology
- [[F2 score weights recall approximately 4x more than precision — standard for bioacoustic detection where missed calls bias statistics]] -- recall-weighted metric aligns with scientific use case
- [[collar-based evaluation with tolerance windows suits bioacoustics better than IoU-based overlap matching]] -- ±200ms tolerance accommodates STFT boundary uncertainty

## Two-Stage FP Filtering (Literature)
- [[Clarfeld 2025 secondary logistic regression on primary detections achieved 85-90 percent FP filtering accuracy]] -- validates two-stage pattern across taxa
- [[VocalMat two-stage morphological filtering plus CNN noise classification achieves over 98 percent detection rate]] -- hand-engineered first stage versus our model-derived approach
- [[BootSnap includes an explicit false-positive class alongside 11 USV syllable categories]] -- unified classification with explicit noise class as alternative to two-stage filtering
- [[BirdVoxDetect PCEN reduced false alarm rates 50x near-field and 5x far-field]] -- PCEN normalization as preprocessing-stage FP reduction
- [[PCEN is the gold standard adaptive normalization in bioacoustic literature]] -- next-iteration improvement requiring model retraining

## CNN Architecture
- [[mid-c-cnn-balances-capacity-and-inference-speed-for-14k-samples]] -- [32,96,192] filter config with ~207K params for matched-windows retrain

## Detection App Save State
- [[saved-previous ghost detections current editable and saved-current form three aligned detection state tiers in the app]] -- three-tier model: editable current, matched saved-current (blue), historical ghost (gray)

## Label Persistence
- [[JSON label files provide human-readable version-controllable persistence for detection labels and metadata]] -- one JSON per WAV stores detections, user labels, probability curves; git-friendly and inspectable
- [[timestamp proximity matching with configurable tolerance bridges detection systems that use different internal time representations]] -- engineering pattern for re-associating detections across tools when exact timestamps differ due to STFT framing

## Sub-Maps
- [[detection-landscape]] -- architectural taxonomy, alternative tools, source separation, annotation ecosystem (28 notes)

## Open Questions
- [[optimal bout gap threshold may vary across behavioral contexts and recording conditions]]
- [[whether very short USV signals near the 8-10 ms boundary should be included or excluded from training]]
- Whether segment continuity parameters need per-recording tuning

## Related Areas
- [[classification]] -- the CNN precision filter that follows energy detection
- [[signal-processing]] -- STFT parameters that feed the energy detector
- [[experimental-methods]] -- evaluation methodology for detection performance
- [[representation-learning]] -- detection quality directly shapes what the VQ-VAE codebook can learn, since the transformer trains on detected bouts
- [[generative-modeling]] -- bounded gain stability principle from diffusion analysis applies to two-stage detection: error amplification between stages is the general risk pattern

---

Topics:
- [[index]]
