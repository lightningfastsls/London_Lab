---
description: First session guide -- creating your first note and building connections
type: manual
generated_from: "arscontexta-0.8.0"
---
# Getting Started

## Your First Session

When you start a new session, the orient hook loads your current state:
- ops/goals.md shows active threads
- ops/reminders.md shows pending commitments
- Condition triggers surface maintenance needs

## Creating Your First Note

1. Capture raw material in inbox/ (a paper finding, experiment result, design decision)
2. Run /reduce to extract atomic notes from the source
3. Each note gets a prose-sentence title that works as a claim

Example: Instead of "STFT parameters", write "512-point FFT at 300kHz provides 585Hz frequency resolution suitable for USV detection"

## How Connections Work

- Wiki links: `[[note title]]` creates a graph edge between notes
- Topic maps: hub pages that organize notes by research area
- Semantic search: qmd finds related notes by content similarity

## The Session Rhythm

1. **Orient** -- Read goals, check triggers
2. **Work** -- Do your task, capture insights
3. **Persist** -- Update goals, save session state

## Next Steps

- Read [[workflows]] for the full processing pipeline
- Read [[skills]] to see all available commands
- Try /arscontexta:tutorial for an interactive walkthrough
