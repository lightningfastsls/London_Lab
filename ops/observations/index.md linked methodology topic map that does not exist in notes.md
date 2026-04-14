---
description: index.md had a `methodology` link pointing to notes/methodology.md which was never created; methodology content lives in ops/methodology/ not notes/
category: process-gap
trigger: /graph skill detected dangling link during graph analysis
status: archived
archived: 2026-03-02
archived_by: rethink-2026-03-02
resolved: 2026-02-19
---

# index.md linked methodology topic map that does not exist in notes

During graph analysis via the /graph skill, a dangling `methodology` link was found in `notes/index.md`. The link pointed to `notes/methodology.md` which does not exist. Methodology self-knowledge lives in `ops/methodology/` (operational space), not in the `notes/` topic map system, per the CLAUDE.md knowledge graph design.

The setup process that generated `notes/index.md` during /setup added a `methodology` topic map entry without creating the corresponding file. This is a setup-time omission.

## Proposed Action

The dangling link has been removed from `notes/index.md`. If a `methodology` topic map in `notes/` is ever needed (e.g., notes about the vault's processing methodology itself), it should be created following the topic-map template and populated before being linked from index.md.

Setup scripts or templates that generate index.md should either always create the methodology topic map file, or omit the link if methodology notes are routed entirely through ops/methodology/.
