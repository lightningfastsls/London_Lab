---
description: LMT synchronization architecture, recording infrastructure, and methods for correlating USV sequences with behavioral events
type: moc
---

# behavioral-integration

Connecting USV analysis to behavioral context via the Live Mouse Tracker system. Recording infrastructure at Institut Pasteur provides synchronized USV + behavior data. LMT integration follows a tiered approach from simple rate correlation to mutual information between vocal sequences and behavioral transitions.

## Recording Infrastructure
- [[AviSoft Recorder captures synchronized USV recordings within the LMT behavioral tracking system]] -- recording software
- [[Live Mouse Tracker from Institut Pasteur synchronizes vocalization recordings with social behavior events]] -- behavioral tracking + USV synchronization
- [[shared lab space without sound attenuation explains why noise robustness is a primary design constraint]] -- acoustic environment constraining detection design
- [[Pasteur USV cloud platform enables online testing of detection methods without local infrastructure]] -- external testing platform
- [[LMT USV Toolbox provides Python-based offline USV processing as a reference implementation]] -- reference implementation from the LMT team

## LMT Integration Architecture
- [[LMT integration code belongs in dedicated src-usv_spectrogram-lmt subpackage]] -- architectural boundary: db_loader, synchronizer, event_triggered, context_analysis
- [[information theory and null model foundation must precede probing and LMT integration]] -- ordering constraint: validated analytical tools before biological interpretation
- [[whether LMT SQLite schema supports the required temporal resolution for USV-behavior synchronization]] -- 30 fps video vs sub-ms acoustic precision constrains analysis granularity

## LMT Integration Methods
- [[event-triggered USV rate via PETH in plus-minus 2 second windows per event type serves as LMT integration sanity check]] -- Tier 1: peri-event time histograms as basic correlation check
- [[mutual information between vocal sequence and next behavior quantifies vocal prediction of behavioral transitions]] -- Tier 3: I(vocal_sequence; next_behavior) as strongest test of communicative function
- [[MANOVA on CNN features or chi-squared on VQ-VAE codes tests whether behavioral context predicts vocal repertoire composition]] -- Tier 2: does behavior shape what is said?
- [[burstiness by behavioral context bridges information theory and LMT behavioral analysis]] -- context-dependent burstiness connects temporal statistics to behavioral analysis

## Related Areas
- [[experimental-methods]] -- parent hub for all experimental methodology
- [[wild-lab-vocal-comparison]] -- the biological questions these methods test
- [[representation-learning]] -- null models and information-theoretic tools feed integration analysis
- [[detection]] -- detection pipeline produces the USV events that get correlated with behavior

---

Topics:
- [[experimental-methods]]
