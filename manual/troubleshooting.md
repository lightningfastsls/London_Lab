---
description: Common issues and resolution patterns
type: manual
generated_from: "arscontexta-0.8.0"
---
# Troubleshooting

## Orphan Notes
**Symptom:** Notes with no incoming links -- invisible to traversal.
**Fix:** Run /reflect on orphaned notes. Check with /arscontexta:health quick.

## Dangling Links
**Symptom:** Wiki links pointing to non-existent notes.
**Fix:** Create the missing note or update the link. Run /arscontexta:health quick to find all.

## Stale Content
**Symptom:** Notes not updated in 30+ days with sparse connections.
**Fix:** Run /reweave on stale notes to find new connections.

## Methodology Drift
**Symptom:** System behavior diverging from what CLAUDE.md specifies.
**Fix:** Run /rethink to detect drift. Review ops/observations/ for patterns.

## Inbox Overflow
**Symptom:** Too many items accumulating in inbox/.
**Fix:** Run /reduce or /pipeline to process. Don't capture more until inbox < 3.

## Pipeline Stalls
**Symptom:** Tasks stuck in queue with no progress.
**Fix:** Check with /next for intelligent recommendations. Review ops/queue/queue.json.

## Common Mistakes

| Mistake | Correction |
|---------|-----------|
| Writing directly to notes/ | Route through inbox/ -> /reduce |
| Bare links in topic maps | Add context phrases: `[[note]] -- why follow this` |
| Generic note titles | Use prose claims: "X causes Y" not "Notes on X" |
| Ignoring health triggers | Run /arscontexta:health regularly |
| Creating topic maps too early | Wait until 5+ notes cluster naturally |

See [[meta-skills]] for /rethink and /remember.
See [[configuration]] for threshold adjustments.
