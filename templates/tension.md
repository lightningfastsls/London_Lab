---
_schema:
  entity_type: "tension"
  applies_to: "notes/*.md WITH type: tension (argued contradictions between defensible positions; distinct shape from research-notes)"
  required:
    - description
    - type         # must be "tension"
    - topics
    - status       # pending | resolved | dormant
  optional:
    - confidence   # may be used when one pole has stronger evidence
    - created
    - meta_state
  enums:
    type:
      - tension    # only valid value for tensions
    status:
      - pending    # unresolved, both poles remain defensible
      - resolved   # synthesis or dissolution found; retain note as historical record
      - dormant    # tension still exists but no longer load-bearing for current work

# Template fields
description: ""
type: tension
status: pending
topics: []
---

# {prose-as-title — name the tension as a sentence}

## The Tension

{Describe both poles. Each should be defensible by someone serious. If one pole is obviously wrong, this is not a tension — it's a finding.}

## Quick Test

{One question a reader can ask to decide which pole applies in their situation.}

## When Each Pole Wins

**Pole A wins when:** {conditions}

**Pole B wins when:** {conditions}

## Dissolution Attempts

{Known attempts to resolve or reframe the tension. Note which have succeeded partially, which have failed, and why.}

## Practical Applications

{How this tension shows up in the current work and what it constrains.}

---

Source:
- {origin — paper, debate, observation}

Relevant Notes:
- [[related note]] -- relationship context

Topics:
- [[relevant-topic-map]]
