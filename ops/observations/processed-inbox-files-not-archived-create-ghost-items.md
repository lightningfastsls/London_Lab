---
description: "Files marked status:processed by /seed remain in inbox/ inflating item count and triggering false inbox-pressure alerts"
category: process-gap
status: archived
archived: 2026-03-20
archived_by: rethink-2026-03-20
resolution: "Added Source Archival section to /reduce SKILL.md (2026-03-07) — moves source from inbox/ to archive/inbox/ after successful extraction"
first_observed: 2026-03-01
occurrences: 11
---

# Processed inbox files not archived create ghost items that inflate inbox pressure

During inbox backlog clearing, `deepsqueak-usv-syllable-classification-practical-guide.md` was found in inbox/ despite having `status: processed`, `processed_date: "2026-02-23"`, and `processed_task: seed-002` in its frontmatter. All 33 major claims from this source already existed as notes in the vault, many directly citing it as their source.

The file inflated the inbox count (11 items, above the threshold of 3) and was flagged by the session orient hook as needing processing — but it had already been fully reduced. This created wasted work: a full `/reduce` scan that found zero new extractions.

**Root cause:** The `/seed` workflow marks the source frontmatter as `status: processed` but does not move it to `archive/inbox/`. The `/reduce` workflow creates notes but does not archive the source either. No step in the pipeline handles the archive move automatically.

**Proposed fix:** Either:
1. `/reduce` should move the source to `archive/inbox/` after successful extraction (at the end of the workflow)
2. `/seed` should move the source to `archive/inbox/` immediately after queuing (since the original is preserved in the queue task file)
3. The session orient hook should check frontmatter `status: processed` and exclude those from inbox counts

Option 1 is cleanest — the source stays accessible during the full reduce cycle, then gets archived when extraction is complete.

**Impact:** Each ghost item wastes ~5 minutes of agent time on orientation + duplicate scanning before discovering nothing to extract. With multiple ghost items, this compounds.
