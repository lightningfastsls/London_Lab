---
description: "Probing classifiers (linear Ridge/LogReg and shallow MLP) trained on frozen hidden states reveal whether and how the transformer encodes acoustic properties at each layer"
type: method
confidence: likely
meta_state: current
topics:
  - "[[representation-learning]]"
  - "[[experimental-methods]]"
---

# linear and MLP probes on frozen transformer hidden states identify which layer encodes which acoustic property

Probing is a standard interpretability technique from NLP (Belinkov 2022) adapted here for USV transformer analysis. The core idea is simple: freeze the transformer weights, extract hidden state vectors from each layer for each input frame, and train simple classifiers to predict ground-truth acoustic properties from those vectors. If a simple classifier achieves high accuracy, the information must be explicitly present in the hidden state representation, because a simple model lacks the capacity to compute complex transformations — it can only read what is already there.

The method uses two probe architectures that test different encoding hypotheses. A linear probe (Ridge regression for continuous targets, logistic regression for categorical targets) tests whether the property is linearly encoded — directly readable by a linear readout, which means the representation explicitly separates that property along some axis in the embedding space. A shallow MLP probe (one hidden layer with ReLU activation) tests whether the property is nonlinearly encoded — present but entangled with other properties in a way that requires nonlinear extraction. If MLP R-squared significantly exceeds linear R-squared for a given layer-property pair, the information requires nonlinear transformation to access, which suggests it is implicitly rather than explicitly represented.

The experimental setup uses 5-fold cross-validation stratified by recording to prevent data leakage. Target properties span the acoustic feature space: continuous targets include peak_frequency, spectral_centroid, bandwidth, energy, and duration; categorical targets include is_voiced, frequency_direction (up/down/flat), and bout_position (early/middle/late). Each probe is trained independently for each (layer, property) combination, producing the layer-property heatmap that is the primary deliverable.

This analysis directly guides VQ-VAE layer selection, because the layer with the richest acoustic encoding is the best candidate for the VQ-VAE extraction point. If [[middle-layer hidden states capture mid-level concepts better than early or late layers for VQ-VAE input]] holds for our USV transformer, the probing heatmap should show a peak in the middle layers — but this is an empirical question that the probing experiment can definitively answer rather than assume.

---

Source:
- vacation-master-plan-v2 (archived to archive/inbox/)

Relevant Notes:
- [[middle-layer hidden states capture mid-level concepts better than early or late layers for VQ-VAE input]] -- the hypothesis that probing experiments will empirically test
- [[comparing VQ-VAE across transformer layers reveals which abstraction level yields the most interpretable codebook]] -- probing results inform which layers to compare in VQ-VAE experiments

Topics:
- [[representation-learning]]
- [[experimental-methods]]
