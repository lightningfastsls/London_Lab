---
description: "Mean, max, first, and last pooling over variable-length hidden state sequences each emphasize different temporal information, affecting what probing experiments can detect"
type: method
confidence: likely
meta_state: current
topics:
  - "[[representation-learning]]"
  - "[[experimental-methods]]"
---

# pooling strategy choice over the time dimension determines what information probing experiments can access from hidden states

Probing experiments require converting variable-length hidden state sequences (one vector per input frame) into a fixed-size vector that a probe classifier can consume. The choice of pooling strategy over the time dimension is not neutral — it determines which temporal information the probe can access, and therefore which aspects of the representation it can test.

Four standard strategies exist, each with distinct information-theoretic properties:

- **Mean pooling** averages all frame-level hidden states, giving equal weight to every time step. This preserves information about the average activation pattern across the entire sequence but washes out transient or localized features. Best for testing whether a property is diffusely encoded across the full sequence.

- **Max pooling** takes the element-wise maximum across all frames, emphasizing peak activations regardless of when they occur. This is sensitive to the strongest signal in any frame and is useful for detecting whether a property is present anywhere in the sequence, even briefly.

- **First-frame pooling** uses only the initial hidden state. This tests what information the model encodes at sequence onset — relevant for properties determined by the opening of a vocalization (e.g., onset frequency direction).

- **Last-frame pooling** uses only the final hidden state. In an autoregressive transformer with causal attention, the last frame has attended to all preceding frames, making it the most information-rich single frame. This tests what the model has accumulated about the entire sequence by the time it reaches the end.

The practical implication: a probe might fail not because the transformer lacks the information, but because the pooling strategy discards it. Running probes with multiple pooling strategies and comparing results reveals whether information is localized (max/first/last outperform mean) or distributed (mean competitive with others). This is a standard control in NLP probing but easy to overlook when adapting the technique for spectrogram-domain models.

---

Source:
- [[vacation-master-plan-v2]]

Relevant Notes:
- [[linear and MLP probes on frozen transformer hidden states identify which layer encodes which acoustic property]] -- the probing method whose results depend on this pooling choice
- [[layer-property heatmap is the key output showing where acoustic information lives across transformer depth]] -- heatmap entries change depending on pooling strategy used
- [[acoustic property extraction from spectrogram data produces ground truth targets for probing experiments]] -- the target properties whose detectability is pooling-dependent

Topics:
- [[representation-learning]]
- [[experimental-methods]]
