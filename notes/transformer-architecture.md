---
description: Self-attention mechanics, positional encoding, MLP function, residual streams, normalization, and in-context learning theory -- the mechanistic internals of transformer models
type: moc
---

# transformer-architecture

How transformers work at the mechanistic level. Covers the core computation primitives (attention, MLP, residual stream), architectural choices (positional encoding, normalization), complexity tradeoffs, and the theoretical framework for in-context learning as implicit optimization. These notes explain *why* transformer internals produce the behaviors observed in downstream applications.

## Self-Attention Mechanics
- [[Q-K-V separation enables asymmetric context-dependent relevance matching through three independently specialized projections]] -- the foundational query/key/value architecture enabling context-dependent retrieval
- [[a single attention matrix computes fixed embedding similarity that cannot learn task-specific notions of similarity unlike Q-K-V projections]] -- why raw dot-product attention is insufficient without learned projections
- [[using identical Q and K projections makes pre-softmax scores symmetric reducing the model's ability to learn directed relationships though row-wise softmax partially breaks this symmetry]] -- why separate Q and K matrices matter for asymmetric relationships
- [[multi-head attention splits computation into parallel specialized subspaces without increasing total computation]] -- parallel sub-spaces at constant cost via d_model/h dimensionality per head
- [[attention heads empirically specialize into positional syntactic semantic and rare-word roles with most encoder information concentrated in few heads]] -- emergent head specialization with 38/48 heads prunable at 0.15 BLEU loss
- [[dividing by sqrt(d_k) prevents softmax saturation by rescaling dot products to unit variance regardless of dimension]] -- the scaling factor that keeps gradients flowing
- [[unscaled attention dot products grow with dimension causing softmax collapse to one-hot distributions with vanishing gradients]] -- the failure mode that sqrt(d_k) scaling prevents

## Self-Attention vs Alternatives
- [[self-attention has O(n²d) time complexity while recurrence has O(nd²) making attention faster when sequence length is shorter than model dimension]] -- the fundamental complexity tradeoff between attention and recurrence
- [[self-attention requires only O(1) sequential operations enabling full parallelization versus O(n) for RNNs]] -- why transformer training is dramatically faster than recurrent models
- [[self-attention provides O(1)-path global context from layer 1 while CNNs require many stacked layers to aggregate distant information]] -- direct pairwise access eliminates information bottlenecks
- [[self-attention lacks inductive bias for local structure leading to hybrid architectures for domains where locality matters]] -- drives Conformer and CoAtNet designs for audio/vision where local patterns dominate

## Positional Encoding
- [[positional encoding diversified from additive sinusoidal and learned embeddings to multiplicative RoPE and score-bias ALiBi with RoPE becoming dominant in practice]] -- the evolutionary landscape of position representation approaches
- [[sinusoidal positional encoding is added not concatenated to token embeddings preserving dimension while forcing position-content interaction]] -- Vaswani 2017 original design and its tradeoffs
- [[RoPE encodes relative position through rotation of Q and K vectors where the dot product naturally incorporates the position difference between tokens]] -- dominant approach used by LLaMA and Mistral
- [[ALiBi adds linear distance penalties to attention scores enabling train-short-test-long extrapolation with equivalent perplexity at 2x and reasonable degradation at longer ranges]] -- position-free alternative adopted by BLOOM and MPT

## Feed-Forward Networks & MLP
- [[MLP layers store factual associations as distributed key-value memories where first-layer weights match patterns and second-layer weights output associated information]] -- Geva et al 2021: FFN as distributed key-value storage
- [[attention alone cannot compute nonlinear features and MLP alone cannot communicate across positions making both necessary for the complete transformer compute primitive]] -- why both components are irreducible
- [[transformers implement a gather-then-transform cycle where attention moves information between positions and MLP transforms it independently]] -- the two-phase computation pattern within each block

## Architecture & Layer Stacking
- [[the residual stream architecture lets transformer components read from and write to a shared information stream enabling additive accumulation]] -- Elhage et al 2021: the shared bus that enables component analysis
- [[stacking transformer blocks creates hierarchical abstraction from syntax in lower layers through structure in middle layers to semantics in upper layers]] -- probing evidence for layer-wise specialization
- [[pre-norm leaves the residual path untouched enabling stable gradient flow while OLMo 2 and Gemma 3 adopted hybrid peri-normalization combining pre-norm and output-norm with QK-Norm]] -- normalization strategies from GPT-2 pre-norm through 2024 peri-LN
- [[pre-norm transformer architecture improves training stability for spectrogram prediction]] -- practical application: our USV transformer uses pre-norm for 8-block 512-dim model

## In-Context Learning Theory
- [[ICL is mathematically equivalent to query-dependent low-rank weight modifications of the MLP where each context token contributes a rank-1 update]] -- Dherin et al 2025: ICL as virtual weight modification, query-dependent
- [[sequential ICL context processing follows dynamics resembling online stochastic gradient descent with learning rate determined by attention magnitude]] -- the SGD analogy for how context tokens are processed
- [[the attention plus MLP stacking structure enables ICL because contextual layers modify input representations that naturally transfer to weight-space modifications]] -- architectural explanation for why ICL emerges
- [[Von Oswald et al showed a single linear self-attention layer can implement a gradient descent step with trained transformers on regression tasks matching this construction]] -- constructive proof of ICL-as-GD for linear case
- [[the ICL-as-implicit-weight-update analysis covers only single transformer blocks and final token output not full autoregressive generation]] -- important scope limitation of the theoretical framework
- [[function vectors are compact single vectors encoding ICL task representations that can be transplanted across contexts and composed algebraically]] -- Todd et al 2024: modular task representations extracted via causal mediation

## Induction Heads
- [[induction heads implement pattern completion via a two-layer circuit where previous-token heads write context and induction heads read it to predict continuations]] -- Olsson et al 2022: the [A][B]...[A]→[B] two-layer mechanism
- [[induction heads emerge in a sharp phase transition during training that coincides with the onset of in-context learning ability supported by six causal lines of evidence]] -- discrete phase transition, not gradual improvement
- [[induction heads in larger models generalize from exact-match token copying to fuzzy pattern completion]] -- hypothesis that larger-model induction heads perform approximate matching

## ICL Boundaries
- [[ICL fails on specification-heavy tasks reaching less than half of fine-tuned performance due to inadequate schema comprehension]] -- the ceiling where attention cannot replace weight-based learning
- [[ICL performance degrades with excessive context because the issue is attention quality not token capacity]] -- more tokens hurt when attention distributes over noise

## Emergent Behavior Parallels
- [[DeepSeek-R1-Zero trained purely with GRPO produced emergent reasoning behaviors including self-reflection and verification without explicit training]] -- emergent reasoning from RL training parallels induction head phase transition: both show complex behaviors appearing suddenly from simple optimization
- [[induction heads emerge in a sharp phase transition during training that coincides with the onset of in-context learning ability supported by six causal lines of evidence]] -- the ICL emergence this parallels

## Open Questions
- How do multi-layer ICL dynamics differ from the single-block theoretical analysis?
- Whether induction heads in decoder-only models show the same specialization patterns as encoder models
- How ICL-as-implicit-optimization theory connects to scaling laws

## Related Areas
- [[representation-learning]] -- applies transformer architecture to USV spectrogram prediction and VQ-VAE; LoRA/PEFT variants exploit low-rank structure described here
- [[agent-cognition]] -- behavioral consequences of ICL mechanisms in multi-turn settings; ICL-to-LoRA knowledge internalization spectrum
- [[context-management]] -- attention degradation and context window limits derive from the attention mechanics described here
- [[signal-processing]] -- STFT parameters that produce the spectrogram input to our transformer model
- [[rl-alignment]] -- RL training produces emergent behaviors (reasoning, self-reflection) that parallel induction head emergence
- [[generative-modeling]] -- pre-norm training stability is the same bounded gain principle that determines diffusion model stability; diffusion/flow matching provides alternative generation approaches for spectrogram prediction

---

Topics:
- [[index]]
