---
description: Processing pipeline, maintenance cycle, and session rhythm
type: manual
generated_from: "arscontexta-0.8.0"
---
# Workflows

## The Processing Pipeline

The full pipeline for turning raw material into connected knowledge:

1. **/seed** -- Research and capture sources in inbox/
2. **/reduce** -- Extract atomic notes from sources (inbox/ -> notes/)
3. **/reflect** -- Find connections between new and existing notes
4. **/reweave** -- Update older notes with new context
5. **/verify** -- Quality-check everything

Use /ralph or /pipeline for orchestrated end-to-end processing.

## Session Rhythm

Every session follows Orient -> Work -> Persist:

### Orient
- Read ops/goals.md for active threads
- Check ops/reminders.md for pending commitments
- SessionStart hook shows vault stats and condition triggers

### Work
- Do your primary task
- Capture insights as you discover them
- Route through inbox/ -> /reduce -> notes/

### Persist
- Update ops/goals.md with current state
- Stop hook captures session data to ops/sessions/

## Maintenance Cycle

Maintenance is condition-based. Run /arscontexta:health to check vault state, then address triggered conditions:

| Condition | Action |
|-----------|--------|
| Inbox >= 3 items | /reduce or /pipeline |
| Orphan notes > 7 days | /reflect on orphans |
| Observations >= 10 | /rethink |
| Tensions >= 5 | /rethink |
| Stale notes (30d + few links) | /reweave |
| Topic map > 40 notes | Split the topic map |

## Batch Processing

For large batches: /ralph handles orchestrated processing with phase tracking and fresh context per phase.

See [[skills]] for individual command details.
See [[configuration]] for adjusting pipeline settings.
