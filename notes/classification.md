---
description: CNN-based USV classification, model architecture, training strategies, and planned transformer exploration
type: moc
---

# classification

How we classify detected candidates as USV or noise. Current approach uses a small CNN (~101K params) with three convolutional blocks. Future work explores a transformer + VQ-VAE architecture for discovering discrete USV vocabulary.

## Core Ideas
- [[three convolutional blocks with global average pooling suffice for USV classification on small datasets]] -- small CNN (~101K params) with variable-size input
- [[3x class weight boost compensates for USV class imbalance in CNN training]] -- extreme recall bias (pos_weight ~35.4)
- [[three-source negative sampling teaches the CNN the full spectrum of non-USV audio]] -- random chunks, inter-USV gaps, low-energy regions
- [[CNN trained only on energy-detector candidates classifies everything as USV because it never sees normal audio]] -- the selection bias that motivated multi-source negatives
- [[transformer-first then VQ-VAE avoids forcing premature discretization]] -- two-phase architecture for discovering USV vocabulary
- [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] -- K=64 vs traditional ~10-15 types
- [[middle-layer hidden states capture mid-level concepts better than early or late layers for VQ-VAE input]] -- layer 4 of 8 as default extraction point
- [[codebook collapse prevention requires simultaneous EMA updates plus dead code reset plus k-means init plus L2 normalization]] -- four mechanisms for VQ-VAE stability
- [[separating representation learning from discretization enables richer feature discovery]] -- the general principle behind two-phase training
- [[class weight boosting biases toward recall at the cost of precision]] -- the tradeoff from extreme pos_weight
- [[multi-source negative sampling is necessary when the training pipeline pre-filters candidates]] -- general pattern for pre-filtered pipelines
- [[causal attention in autoregressive transformer matches the scientific question of predicting what comes next in USV streams]] -- causal mask enforces temporal prediction, matching the research question
- [[pre-norm transformer architecture improves training stability for spectrogram prediction]] -- LayerNorm before attention/FFN for ~25-30M param model
- [[MSE loss for next-column prediction may produce blurry spectrograms requiring a mixture density output head]] -- MSE averages multimodal futures; GMM fallback planned
- [[staged transformer training catches issues early by incrementally scaling from one bout to full dataset]] -- 4-stage protocol: 1→10→100→full bouts
- [[comparing VQ-VAE across transformer layers reveals which abstraction level yields the most interpretable codebook]] -- layers 2,4,6,8 comparison by perplexity/utilization/recon
- [[concept injection decodes what each codebook entry predicts as acoustic continuation]] -- mapping discrete symbols back to interpretable spectrograms
- [[Zipf's law exponent reveals whether VQ-VAE code sequences have language-like frequency distribution]] -- alpha ~1.0 for natural language; test on USV codes
- [[entropy rate decreasing with context length indicates sequential predictability in USV code streams]] -- n-gram conditional entropy from 1 to 8
- [[excess entropy measures long-range structure complexity in discrete code sequences]] -- mutual information between past and future halves
- [[bigram productivity ratio measures compositionality of USV code sequences]] -- unique bigrams / K^2 measures combinatorial freedom
- [[model size should scale with labeled dataset size to balance underfitting and overfitting]] -- small <5K, medium 5-15K, large 15K+ labels
- [[VQ-VAE codebook visualization decodes entries through the full pipeline back to spectrogram space]] -- decode entries through transformer output head
- [[exemplar galleries ground abstract codebook entries in concrete acoustic examples]] -- N=10 nearest encoder outputs with ±50 frame context
- [[CNN baseline of 89.7 percent precision and 93.8 percent recall at threshold 0.05 validates the two-stage detection approach]] -- F1 91.7% baseline
- [[noisy USVs are valid positive training samples because the model must learn detection in degraded conditions]] -- labeling policy: noise-embedded USVs are valid positives
- [[good negative training samples must be unambiguously not USV to prevent label noise]] -- quality criterion for negative samples
- [[overlapping calls from multiple mice are labeled positive because USV presence is the classification target not individual identity]] -- classification target is presence, not source identity
- [[low-amplitude and short-duration USVs are the primary source of false negatives and training bias]] -- key failure mode and bias source
- [[CNN false positives cluster in noisy regions where energy patterns superficially resemble USV structure]] -- noise structural mimicry as FP source
- [[MSE loss simplicity versus GMM output head expressiveness for spectrogram prediction]] -- pragmatic "try simple first" tension
- [[HPC dependency for transformer training versus local-only development capability]] -- ~25-30M params needs A100; code testable locally

## Literature Context
- [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]] -- the key finding that motivated our VQ-VAE approach
- [[traditional Holy and Guo 2005 USV taxonomy defines discrete types but Goffinet 2021 showed USVs form a continuum]] -- the paradigm shift from discrete to continuous
- [[DeepSqueak uses monolithic Faster R-CNN detection whereas our two-stage pipeline allows independent tuning of recall and precision]] -- competitive positioning
- [[VocalMat represents supervised classification with predefined categories achieving 86 percent accuracy on 11 USV types]] -- traditional supervised approach baseline
- [[MUPET uses gammatone filterbank and unsupervised k-means to discover 100-140 data-driven USV types]] -- unsupervised predecessor to our approach
- [[Tjandra et al 2020 applied transformer VQ-VAE for unsupervised unit discovery in human speech with K equals 128]] -- closest architectural analog
- [[no published work has applied VQ-VAE to animal vocalizations making this a genuine research gap]] -- novelty claim
- [[Chabout et al 2015 established that male mice change syllable syntax with social context]] -- motivates sequence analysis
- [[Hertz et al 2020 demonstrated that USV sequence statistics carry predictive information]] -- supports sequence modeling
- [[Ivanenko et al 2020 showed DNNs achieve 77-84 percent accuracy classifying emitter sex from spectrograms]] -- identity information in spectrograms

## Open Questions
- [[whether FSQ provides more stable discretization than VQ-VAE for USV codebook learning]] -- FSQ achieves 100% utilization by design
- [[whether attention patterns in the trained transformer attend beyond the immediately preceding frame]] -- purely local attention would mean no long-range learning
- [[whether 50 percent overlap in chunk windowing loses critical bout-boundary information]] -- same event gets different positional encodings
- Scaling behavior of small CNN as labeled dataset grows from 2K to 30K

## Related Areas
- [[detection]] -- upstream pipeline that generates candidates for classification
- [[signal-processing]] -- STFT parameters that produce the spectrogram input
- [[experimental-methods]] -- dataset splits, class weighting, negative sampling

---

Topics:
- [[index]]
