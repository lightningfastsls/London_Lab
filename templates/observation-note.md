---
_schema:
  entity_type: "observation"
  applies_to: "ops/observations/*.md"
  required:
    - description
    - category
  optional:
    - trigger
    - frequency
  enums:
    category:
      - friction
      - surprise
      - process-gap
      - methodology
    status:
      - pending
      - promoted
      - implemented
      - archived

# Template fields
description: ""
category: ""
trigger: ""
status: pending
---

# {prose-sentence describing what was observed}

{Context — what happened, when, what triggered this observation.}

## Proposed Action
{What should change? Note: observations accumulate. When 10+ pending observations exist, /rethink triages them.}
