---
_schema:
  entity_type: "source-capture"
  applies_to: "inbox/*.md"
  required:
    - description
    - source_type
  optional:
    - url
    - author
    - date_accessed
    - research_tool
    - research_query
  enums:
    source_type:
      - paper
      - article
      - documentation
      - experiment-log
      - codebase
      - conversation
      - recording-analysis
    status:
      - unprocessed
      - in-progress
      - processed

# Template fields
description: ""
source_type: ""
url: ""
author: ""
date_accessed: ""
status: unprocessed
---

# {source title or brief description}

## Key Points
- {main takeaway 1}
- {main takeaway 2}

## Raw Notes
{Capture reactions, quotes, observations. This gets processed into atomic notes.}

## Processing Notes
{After /reduce: what was extracted, what was skipped, what needs follow-up}
