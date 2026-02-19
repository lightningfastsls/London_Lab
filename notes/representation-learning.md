---
description: VQ-VAE and transformer architecture for unsupervised discovery of discrete USV vocabulary and sequential structure
type: moc
---

# representation-learning

Unsupervised discovery of structure in USV vocalizations. A transformer predicts next spectrogram columns autoregressively, then VQ-VAE discretizes internal representations into a learned codebook. Information-theoretic measures test whether the resulting code sequences have language-like properties.

## Core Architecture
- [[transformer-first then VQ-VAE avoids forcing premature discretization]] -- two-phase training: learn representations first, quantize second
- [[separating representation learning from discretization enables richer feature discovery]] -- the general principle behind two-phase training
- [[causal attention in autoregressive transformer matches the scientific question of predicting what comes next in USV streams]] -- causal mask enforces temporal prediction
- [[pre-norm transformer architecture improves training stability for spectrogram prediction]] -- LayerNorm before attention/FFN for ~25-30M param model
- [[MSE loss for next-column prediction may produce blurry spectrograms requiring a mixture density output head]] -- MSE averages multimodal futures; GMM fallback planned
- [[MSE loss simplicity versus GMM output head expressiveness for spectrogram prediction]] -- pragmatic "try simple first" tension
- [[staged transformer training catches issues early by incrementally scaling from one bout to full dataset]] -- 4-stage protocol: 1->10->100->full bouts
- [[HPC dependency for transformer training versus local-only development capability]] -- ~25-30M params needs A100; code testable locally

## VQ-VAE Codebook
- [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] -- K=64 vs traditional ~10-15 types
- [[middle-layer hidden states capture mid-level concepts better than early or late layers for VQ-VAE input]] -- layer 4 of 8 as default extraction point
- [[codebook collapse prevention requires simultaneous EMA updates plus dead code reset plus k-means init plus L2 normalization]] -- four mechanisms for VQ-VAE stability
- [[comparing VQ-VAE across transformer layers reveals which abstraction level yields the most interpretable codebook]] -- layers 2,4,6,8 comparison by perplexity/utilization/recon

## Codebook Interpretation
- [[concept injection decodes what each codebook entry predicts as acoustic continuation]] -- mapping discrete symbols back to interpretable spectrograms
- [[VQ-VAE codebook visualization decodes entries through the full pipeline back to spectrogram space]] -- decode entries through transformer output head
- [[exemplar galleries ground abstract codebook entries in concrete acoustic examples]] -- N=10 nearest encoder outputs with +/-50 frame context

## Information-Theoretic Analysis
- [[Zipf's law exponent reveals whether VQ-VAE code sequences have language-like frequency distribution]] -- alpha ~1.0 for natural language; test on USV codes
- [[entropy rate decreasing with context length indicates sequential predictability in USV code streams]] -- n-gram conditional entropy from 1 to 8
- [[excess entropy measures long-range structure complexity in discrete code sequences]] -- mutual information between past and future halves
- [[bigram productivity ratio measures compositionality of USV code sequences]] -- unique bigrams / K^2 measures combinatorial freedom

## Literature Context
- [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]] -- the key finding that motivated the VQ-VAE approach
- [[Tjandra et al 2020 applied transformer VQ-VAE for unsupervised unit discovery in human speech with K equals 128]] -- closest architectural analog from speech domain
- [[no published work has applied VQ-VAE to animal vocalizations making this a genuine research gap]] -- novelty claim
- [[MUPET uses gammatone filterbank and unsupervised k-means to discover 100-140 data-driven USV types]] -- unsupervised predecessor using handcrafted features

## Open Questions
- [[whether FSQ provides more stable discretization than VQ-VAE for USV codebook learning]] -- FSQ achieves 100% utilization by design
- [[whether attention patterns in the trained transformer attend beyond the immediately preceding frame]] -- purely local attention would mean no long-range learning
- [[whether 50 percent overlap in chunk windowing loses critical bout-boundary information]] -- same event gets different positional encodings

## Related Areas
- [[classification]] -- CNN operational pipeline that produces labeled data feeding this research
- [[detection]] -- upstream detection pipeline
- [[signal-processing]] -- STFT parameters that produce the spectrogram input

---

Topics:
- [[index]]
