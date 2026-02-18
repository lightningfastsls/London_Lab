---
description: Open question on whether mouse ID, sex, strain, and social context metadata are available to enable population-level comparison of VQ-VAE codebook usage.
type: open-question
confidence: speculative
topics:
  - "[[experimental-methods]]"
---

# whether population-level metadata is available for context-dependent VQ-VAE analysis

The scientific value of the VQ-VAE codebook analysis depends critically on the availability of population-level metadata. Comparing codebook usage frequency distributions across mouse groups — wild versus lab, male versus female, isolated versus pair-housed — requires knowing which recording belongs to which group. This metadata (mouse ID, sex, strain, social context, population origin) may be encoded in the WAV directory structure, embedded in filenames, or stored in a separate metadata CSV that has not yet been confirmed to exist.

If metadata is unavailable, context-dependent analysis must be skipped entirely. The codebook can still be interpreted by exemplar galleries and decoder visualization, but the cross-population comparison that motivates the VQ-VAE approach — testing whether [[wild versus lab mouse USV comparison tests whether domestication altered vocal repertoires]] — cannot be executed. This would reduce the VQ-VAE from a scientific instrument to a visualization tool.

Resolving this question requires explicit investigation before committing to the VQ-VAE training phase. The recommended action is to audit the 5970 USV recording directory structure and any accompanying experimental logs to determine what metadata is reliably associated with each WAV file. If metadata exists in any parseable form, a metadata manifest (CSV mapping filename to experimental conditions) should be created before VQ-VAE training begins, so that analysis scripts can join on it at inference time.

---

Source: [[ROADMAP.md]]
