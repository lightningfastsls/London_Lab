---
description: Complete reference for every available command
type: manual
generated_from: "arscontexta-0.8.0"
---
# Skills

## Processing Skills
- **/reduce** -- Extract atomic notes from source material in inbox/. Reads source, identifies claims, creates notes with proper schema.
- **/reflect** -- Find connections between notes. Reads existing notes, discovers relationships, adds wiki links.
- **/reweave** -- Update older notes with new connections and context. Revisits notes in light of recent additions.
- **/verify** -- Quality-check notes for description quality, schema compliance, and composability.
- **/validate** -- Run schema validation across all notes against template definitions.

## Orchestration Skills
- **/seed** -- Research a topic and deposit sources in inbox/. Combines web research with structured capture.
- **/ralph** -- Orchestrated processing: runs the full pipeline on a batch of inbox items.
- **/pipeline** -- End-to-end pipeline execution with phase tracking and quality gates.
- **/tasks** -- Manage the processing queue. View pending, in-progress, and completed tasks.

## Navigation Skills
- **/stats** -- Vault metrics: note counts, link density, topic map sizes, processing throughput.
- **/graph** -- Interactive graph analysis: triangles, clusters, bridges, influence flow.
- **/next** -- Get intelligent next-action recommendations based on queue state and conditions.

## Growth Skills
- **/learn** -- Deep research on a topic with provenance tracking. Deposits structured sources in inbox/.
- **/remember** -- Capture operational friction as observations in ops/observations/.

## Evolution Skills
- **/rethink** -- Triage accumulated observations and tensions. Promotes, implements, or archives.
- **/refactor** -- Restructure notes: merge, split, or reorganize topic maps.

## Plugin Commands (always available)
- **/arscontexta:health** -- Vault health diagnostics (quick, full, or three-space mode)
- **/arscontexta:ask** -- Query the bundled research knowledge base
- **/arscontexta:architect** -- Get research-backed evolution advice
- **/arscontexta:help** -- Contextual guidance and command discovery
- **/arscontexta:tutorial** -- Interactive walkthrough for new users
- **/arscontexta:add-domain** -- Add a new knowledge domain
- **/arscontexta:recommend** -- Architecture advice for new use cases
- **/arscontexta:reseed** -- Re-derive system from first principles
- **/arscontexta:upgrade** -- Apply plugin knowledge base updates

See [[workflows]] for how skills chain together.
