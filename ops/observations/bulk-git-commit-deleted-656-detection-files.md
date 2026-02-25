---
description: "A cleanup commit (78d1c70) staged and committed the deletion of 656 USV_Detections files — data was recoverable from git history but appeared lost"
category: friction
trigger: "Raven export dry-run showed only 2/36 detection directories; git log revealed mass deletion in 'clean git status' commit"
status: resolved
resolution: "Restored via git checkout 046c035 -- USV_Detections/; added Git Data Safety guardrails to CLAUDE.md"
---

# Bulk git commit to clean status accidentally deleted 656 detection files

On 2026-02-23, running `export_raven_tables.py --dry-run` showed only 2 matched WAV files instead of the expected ~36. Investigation revealed:

1. Commit `046c035` ("Commit code updates and USV_Detections") added all 38 detection subdirectories (656 files total — JSONs, PNGs, tracking files)
2. Commit `78d1c70` ("chore: commit working tree to clean git status") deleted all but 2 subdirectories — 656 file deletions committed in one shot

The root cause was bulk-staging deletions to achieve a clean `git status` without reviewing what was being staged.

**Recovery:** `git checkout 046c035 -- "USV_Detections/"` restored all files.

**Prevention:** Added to CLAUDE.md > Common Mistakes to Avoid > Git Data Safety:
- Never `git add -A` without reviewing status
- Check `git diff --cached --stat` before cleanup commits
- Stage data directories by specific file name
- Use `git log -- <path>` to find data in history
