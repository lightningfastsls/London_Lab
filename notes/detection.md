---
description: USV detection pipeline -- energy detection, candidate generation, segment continuity, bout extraction
type: moc
---

# detection

How we find USVs in raw audio. The pipeline uses a two-stage architecture: a permissive energy detector generates candidates, then a CNN classifier filters for precision. Segment continuity and bout extraction add temporal structure.

## Core Ideas
- [[two-stage detection uses permissive energy detector followed by CNN precision filter]] -- the architectural pattern: energy for recall, CNN for precision
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

## Open Questions
- [[optimal bout gap threshold may vary across behavioral contexts and recording conditions]]
- [[whether very short USV signals near the 8-10 ms boundary should be included or excluded from training]]
- Whether segment continuity parameters need per-recording tuning

## Related Areas
- [[classification]] -- the CNN precision filter that follows energy detection
- [[signal-processing]] -- STFT parameters that feed the energy detector
- [[experimental-methods]] -- evaluation methodology for detection performance

---

Topics:
- [[index]]
