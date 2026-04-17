---
_schema:
  entity_type: "research-note"
  applies_to: "notes/*.md (excluding topic maps with type: moc and tensions with type: tension — those use separate schemas in templates/topic-map.md and templates/tension.md)"
  required:
    - description
    - topics
  optional:
    - confidence
    - conditions
    - meta_state
    - source
  optional_by_type:
    open-question:
      - confidence        # uncertainty is carried by the type itself; omit confidence
  enums:
    type:
      - finding            # empirical: observed in data / experiment
      - claim              # argued from principle; methodology or conceptual claim
      - decision           # project decision with rationale
      - method             # procedure / recipe / how-to
      - hypothesis         # testable prediction awaiting evidence
      - baseline           # reference point for comparison
      - open-question      # genuinely open research question, no answer yet
      - pattern            # recurring structure observed across cases
      - source             # pointer to external literature + derived-claim index
    confidence:
      - proven             # robust evidence; multiple observations or ground truth
      - likely             # single observation or strong argument; hedged
      - experimental       # preliminary; may be revised
      - speculative        # intuition / wild guess; low support
    meta_state:
      - current
      - outdated
      - superseded

# Template fields
description: ""
type: ""
confidence: ""
conditions: []
meta_state: current
topics: []
---

# {prose-as-title — one claim, readable as a sentence}

{Content — evidence, reasoning, context. Transform source material into your own understanding.}

---

Source:
- {origin of this insight — paper, experiment, observation}

Relevant Notes:
- [[related note]] -- relationship context

Topics:
- [[relevant-topic-map]]
