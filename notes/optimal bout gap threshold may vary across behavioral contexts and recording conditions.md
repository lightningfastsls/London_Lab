---
description: "500 ms bout gap is a reasonable default but may need per-experiment adjustment for different behavioral paradigms"
type: open-question
confidence: speculative
conditions: []
meta_state: current
topics:
  - "[[detection]]"
---

# optimal bout gap threshold may vary across behavioral contexts and recording conditions

The current [[bout gap threshold of 500 ms groups temporally clustered USVs while separating distinct episodes]] was chosen as a reasonable default. However, the optimal threshold likely varies across experimental contexts: courtship vocalizations may have different inter-call timing than distress calls, and different mouse strains may vocalize at different rates. Recording conditions also matter -- if recordings are made in high-noise environments, the energy detector may miss some USVs entirely, creating apparent gaps where the mouse was actually vocalizing. Future work should validate the 500 ms threshold against manual annotations across multiple behavioral paradigms and consider making it a per-experiment configurable parameter.

### ROADMAP Context

ROADMAP specifies 500 ms as the default bout gap but makes it configurable via BoutExtractionConfig, signaling that the pipeline anticipates needing per-experiment adjustment. Different behavioral contexts produce characteristically different inter-USV timing: courtship vocalizations tend to cluster tightly, distress calls may be more episodic, and exploratory vocalizations can be sparse. Cross-species transfer is also uncertain — the 500 ms threshold was tuned specifically for mouse USVs and may not generalize. See [[bout gap threshold of 500 ms groups temporally clustered USVs while separating distinct episodes]] for the rationale behind the default value.

---

Source:
- DECISIONS.md (ADR-014) -- USV Spectrogram project architecture decisions

Relevant Notes:
- [[bout gap threshold of 500 ms groups temporally clustered USVs while separating distinct episodes]] -- the current parameter choice
- [[bout-level spectrograms preserve inter-USV timing context for transformer training]] -- the pipeline this feeds
- [[segment continuity bridges brief amplitude dips that fragment single USVs]] -- segment continuity parameters face a similar per-context variation challenge at the within-USV level

Topics:
- [[detection]]
