---
description: "A (layers x properties) heatmap with R-squared and selectivity scores is the primary deliverable of probing experiments, directly informing VQ-VAE layer selection"
type: finding
confidence: likely
meta_state: current
topics:
  - "[[representation-learning]]"
---

# layer-property heatmap is the key output showing where acoustic information lives across transformer depth

The layer-property heatmap is the central deliverable of the probing experiment pipeline, because it compresses the results of dozens of individual probe trainings into a single visualization that answers the question: "Where does the transformer store what it knows about USV acoustics?" Each cell in the heatmap contains one number — R-squared for continuous targets or selectivity for categorical targets — representing how well a given layer's hidden states predict a given acoustic property.

The rows of the heatmap correspond to transformer layers (e.g., layers 2, 4, 6, 8 in the current architecture), while the columns correspond to target acoustic properties: peak_frequency, spectral_centroid, bandwidth, energy, duration (continuous), and is_voiced, frequency_direction, bout_position, time_since_last_usv (categorical). Each cell thus represents one probing experiment — a 5-fold cross-validated probe trained on frozen hidden states from that specific layer to predict that specific property.

The interpretation follows a clear logic. If early layers score highest on low-level properties (energy, peak_frequency) while middle layers score highest on mid-level properties (frequency_direction, bout_position), this confirms the NLP finding that [[middle-layer hidden states capture mid-level concepts better than early or late layers for VQ-VAE input]] — representations become increasingly abstract with depth. But if a particular layer uniformly scores highest across most properties, that layer is the optimal VQ-VAE extraction point because it contains the richest acoustic information in the most accessible form.

The heatmap also reveals what the transformer does not encode. If bout_position has near-zero selectivity at all layers, the transformer has not learned to represent where a USV falls within its bout — which would suggest that bout-level context is not captured in the current architecture and might require explicit positional encoding. Similarly, if time_since_last_usv shows low R-squared, the temporal spacing between USVs is not represented, which matters because temporal patterns are a key dimension of USV syntax.

The column-wise average (mean score across all properties for each layer) provides the simplest criterion for VQ-VAE layer selection: choose the layer with the highest average score. But the row-wise pattern (which properties are encoded where) provides richer insight into the transformer's learned representation hierarchy and may motivate more sophisticated extraction strategies, such as multi-layer VQ-VAE or layer-specific codebooks for different property types.

---

Source:
- vacation-master-plan-v2 (archived to archive/inbox/)

Relevant Notes:
- [[comparing VQ-VAE across transformer layers reveals which abstraction level yields the most interpretable codebook]] -- the heatmap provides the empirical basis for choosing which layers to compare
- [[middle-layer hidden states capture mid-level concepts better than early or late layers for VQ-VAE input]] -- the hypothesis that the heatmap will empirically test

Topics:
- [[representation-learning]]
