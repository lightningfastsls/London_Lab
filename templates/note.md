---
_schema:
  entity_type: "research-note"
  applies_to: "notes/*.md"
  required:
    - description
    - topics
  optional:
    - confidence
    - conditions
    - meta_state
    - source
  enums:
    type:
      - finding
      - decision
      - method
      - hypothesis
      - baseline
      - open-question
      - pattern
    confidence:
      - proven
      - likely
      - experimental
      - speculative
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
