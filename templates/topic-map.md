---
_schema:
  entity_type: "topic-map"
  applies_to: "notes/*.md WITH type: moc (topic maps / navigation hubs; distinct from research-notes in templates/note.md)"
  required:
    - description
    - type         # must be "moc"
    - topics       # self-referential (topic map links to itself) or links to parent topic map
  optional:
    - parent_map
    - meta_state
  enums:
    type:
      - moc        # only valid value for topic maps

# Template fields
description: ""
type: moc
---

# {topic area name}

{Brief overview of this topic area and why it matters to the research.}

## Core Ideas
- [[note-title]] -- context phrase explaining why to follow this link

## Open Questions
- {What remains unresolved in this area?}

## Related Areas
- [[other-topic-map]] -- how these areas connect

---

Topics:
- [[index]]
