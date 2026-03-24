---
description: VQ-VAE pipeline for unsupervised discovery of discrete USV vocabulary -- transformer architecture, codebook design, information-theoretic analysis, null models, and probing
type: moc
topics: "[[index]]"
---

# representation-learning

The core pipeline for unsupervised discovery of structure in USV vocalizations. A transformer predicts next spectrogram columns autoregressively, then VQ-VAE discretizes internal representations into a learned codebook. Information-theoretic measures test whether the resulting code sequences have language-like properties. For the broader clustering landscape see [[unsupervised-usv-discovery]], for SSL/foundation models see [[bioacoustic-ssl]], for LoRA/PEFT see [[model-adaptation]].

## Core Architecture
- [[transformer-first then VQ-VAE avoids forcing premature discretization]] -- two-phase training: learn representations first, quantize second
- [[separating representation learning from discretization enables richer feature discovery]] -- the general principle behind two-phase training
- [[causal attention in autoregressive transformer matches the scientific question of predicting what comes next in USV streams]] -- causal mask enforces temporal prediction
- [[pre-norm transformer architecture improves training stability for spectrogram prediction]] -- LayerNorm before attention/FFN for ~25-30M param model
- [[MSE loss for next-column prediction may produce blurry spectrograms requiring a mixture density output head]] -- MSE averages multimodal futures; GMM fallback planned
- [[MSE loss simplicity versus GMM output head expressiveness for spectrogram prediction]] -- pragmatic "try simple first" tension
- [[staged transformer training catches issues early by incrementally scaling from one bout to full dataset]] -- 4-stage protocol: 1->10->100->full bouts
- [[HPC dependency for transformer training versus local-only development capability]] -- ~25-30M params needs A100; code testable locally

## Model Artifacts
- [[PyTorch pt format is the standard model artifact format giving native save-load with no extra dependencies]] -- .pt checkpoints with state_dict, optimizer state, training metadata

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
- [[mutual information rate at varying lags measures temporal dependency strength within USV code sequences]] -- I(X_t; X_{t+lag}) at varying lags complements entropy rate and conditional entropy
- [[burstiness coefficient via coefficient of variation of inter-event intervals distinguishes Poisson from bursty temporal patterns]] -- CV=1 Poisson, >1 bursty, <1 regular; plus Kleinberg burst detection

## Quantization Methods
- [[whether FSQ provides more stable discretization than VQ-VAE for USV codebook learning]] -- FSQ achieves 100% utilization by design
- [[FSQ eliminates codebook collapse by construction achieving 100 percent utilization through fixed scalar quantization]] -- ICLR 2024 evidence for FSQ at 400-700 bps speech codecs
- [[discrete audio token taxonomy from 2025 survey covers quantization methods beyond simple VQ]] -- comprehensive taxonomy: RVQ, SVQ, GVQ, FSQ, PQ and more

## Null Model Hierarchy
- [[shuffled null model preserves code frequencies but destroys all sequential structure]] -- simplest baseline: tests whether metrics exceed independent-code expectation
- [[Markov order-k null model generates surrogates preserving k-step transition dependencies]] -- tests whether structure exceeds k-step local dependencies
- [[HMM surrogate null model tests whether USV sequences arise from hidden behavioral state switching]] -- tests the Chabout et al hidden behavioral state switching hypothesis
- [[phase randomized null model preserves autocorrelation spectrum while destroying higher-order structure]] -- tests whether linear temporal correlations alone explain patterns
- [[renewal process null model fits inter-event interval distribution for temporal structure testing]] -- preserves IEI distribution and code frequencies but destroys code-to-code identity dependencies
- [[analytically verifiable test cases validate information-theoretic metric implementations]] -- ground-truth sanity checks with known analytical solutions
- [[null model comparison framework produces z-scores rank-based p-values and effect sizes as the publishable statistical output]] -- the statistical machinery that turns surrogates into publishable z-scores, p-values, and effect sizes

## Workstream Ordering
- [[information theory and null model foundation must precede probing and LMT integration]] -- metrics and null models validated before probing experiments or behavioral analysis

## Probing & Interpretability
- [[linear and MLP probes on frozen transformer hidden states identify which layer encodes which acoustic property]] -- standard NLP interpretability technique adapted for USV transformer
- [[probe selectivity measured as accuracy minus majority baseline distinguishes genuine encoding from trivial prediction]] -- corrects for class imbalance in probe accuracy
- [[layer-property heatmap is the key output showing where acoustic information lives across transformer depth]] -- primary deliverable of probing experiments, guides VQ-VAE layer selection
- [[acoustic property extraction from spectrogram data produces ground truth targets for probing experiments]] -- seven properties (peak freq, centroid, energy, is_voiced, freq direction, bout position, time since last USV) as probe labels
- [[pooling strategy choice over the time dimension determines what information probing experiments can access from hidden states]] -- mean/max/first/last pooling each emphasize different temporal information, a necessary control variable

## Methodological Tensions
- [[forcing USVs into discrete categories may obscure the continuous variation that distinguishes populations]] -- motivates treating VQ-VAE codebook entries as "reference points along a continuum" rather than natural categories

## Converging Hypothesis
- [[the converging research question asks whether transformer encodes behaviorally meaningful vocal categories differing between wild and lab populations]] -- integration point where information theory, probing, and LMT workstreams converge

## Open Questions
- [[whether attention patterns in the trained transformer attend beyond the immediately preceding frame]] -- purely local attention would mean no long-range learning
- [[whether 50 percent overlap in chunk windowing loses critical bout-boundary information]] -- same event gets different positional encodings
- [[whether flow matching could replace VQ-VAE for unsupervised USV representation learning]] -- continuous paths vs discrete tokens; flow matching avoids codebook collapse but loses information-theoretic analysis framework

## Related Areas
- [[unsupervised-usv-discovery]] -- clustering landscape and literature context motivating the VQ-VAE approach
- [[bioacoustic-ssl]] -- SSL and foundation models that could provide input features or alternative backbones
- [[model-adaptation]] -- LoRA/PEFT for efficient adaptation; ICL-LoRA theoretical bridge
- [[transformer-architecture]] -- the attention and MLP mechanics underlying the spectrogram prediction model
- [[classification]] -- CNN operational pipeline that produces labeled data feeding this research
- [[detection]] -- upstream detection pipeline
- [[signal-processing]] -- STFT parameters that produce the spectrogram input
- [[generative-modeling]] -- diffusion/flow matching as potential alternative generative framework; bounded gain stability principle transfers

---

Topics:
- [[index]]
