---
description: "Computing burstiness CV separately per behavioral event type reveals whether different social contexts produce fundamentally different temporal emission patterns"
type: finding
confidence: speculative
meta_state: current
topics:
  - "[[representation-learning]]"
  - "[[experimental-methods]]"
---

# burstiness by behavioral context bridges information theory and LMT behavioral analysis

When burstiness analysis is performed globally on all USV emissions, it produces a single characterization of the temporal pattern — a single CV value that averages across all behavioral contexts, recording conditions, and interaction phases. But mouse vocalizations occur in specific behavioral contexts: during approach, contact, following, side-by-side proximity, and other social interactions tracked by the Live Mouse Tracker system. This global averaging may obscure fundamentally different temporal dynamics that characterize distinct behavioral states.

By computing burstiness separately per behavioral context — using LMT behavioral annotations to label each USV emission with its concurrent behavioral event — we can test whether different contexts produce fundamentally different temporal dynamics. If approach behaviors produce CV greater than 2 (highly bursty, with intense call flurries followed by silence), while side-by-side contact produces CV approximately equal to 1 (random Poisson-like timing), that would suggest distinct vocalization "modes" tied to behavioral states. This would mean that the temporal structure of USV emission is not a fixed property of the animal but rather a context-dependent signal that varies with social situation.

This analysis directly bridges Workstream 1 (information theory and sequential structure) and Workstream 3 (LMT behavioral integration), because it requires both the temporal analysis tools from information theory and the behavioral annotations from LMT. The connection is made possible by [[temporal alignment between USV detections and LMT behavioral events enables USV-behavior correlation analysis]], which provides the temporal correspondence between acoustic events and behavioral labels. The burstiness_by_context function would be the computational meeting point of these two research threads, taking as input the timestamped USV detections and the behavioral event timeline, and producing per-context CV values that can be compared statistically. If the per-context CVs differ significantly (e.g., via Kruskal-Wallis test), this would constitute evidence that vocalization temporal patterns encode behavioral state information — complementing the sequential structure evidence from code-level analyses.

---

Source:
- vacation-master-plan-v2 (archived to archive/inbox/)

Relevant Notes:
- [[temporal alignment between USV detections and LMT behavioral events enables USV-behavior correlation analysis]] -- provides the temporal correspondence infrastructure that makes per-context burstiness computation possible
- [[Live Mouse Tracker from Institut Pasteur synchronizes vocalization recordings with social behavior events]] -- the behavioral tracking system that provides the context labels for stratified analysis
- [[burstiness coefficient via coefficient of variation of inter-event intervals distinguishes Poisson from bursty temporal patterns]] -- the global burstiness method that this note extends to per-context analysis
- [[Hertz et al 2020 demonstrated that USV sequence statistics carry predictive information]] -- if burstiness varies by context, temporal statistics carry behavioral information
- [[USVs are one component of a multimodal courtship behavior suite including mounting approach and movement]] -- vocalizations occur within a broader behavioral context that burstiness-by-context can help characterize

Topics:
- [[representation-learning]]
- [[experimental-methods]]
