---
description: "PROJECTS.md had outdated status for PyQt6 app and pipeline description, causing agent to question correct user claims"
category: process-gap
trigger: "Agent cited stale docs to contradict user about energy detector and PyQt6 app status"
status: archived
archived: 2026-03-03
archived_by: rethink-2026-03-03
resolution: "CLAUDE.md guardrail added (line 190): 'Don't cite documentation status to contradict user claims without verifying CODE first'. PROJECTS.md Section 4 updated to 'DONE and operational'."
---

# Stale documentation caused agent to distrust user's correct description of the current pipeline

During the DeepSqueak integration session (2026-02-25), the user correctly stated that:
1. The PyQt6 detection app is built and operational (`src/usv_spectrogram/app/`)
2. The energy detector is NOT part of the current detection pipeline

PROJECTS.md was wrong on both counts:
- Section 4 said the PyQt6 app was "Not yet started" — it's fully built (67KB main_window.py)
- Line 29 described the pipeline as "energy detection → CNN classification" — the app uses CNN directly with no energy detector (`grep` for `EnergyDetector` in the app directory returned zero matches)

The energy detector (`src/usv_spectrogram/detection/energy_detector.py`) was used historically to generate training data for the CNN, but is not part of the current production detection workflow.

**Current production pipeline:** WAV → PyQt6 app (CNN sliding window detection) → detections → Raven export

## Fixes applied
- Updated PROJECTS.md Section 4 status to "DONE and operational"

## Proposed Action
1. Keep documentation in sync with actual code state — when a major feature ships, update PROJECTS.md immediately
2. When agent finds docs contradicting user claims, verify the CODE (not just docs) before drawing conclusions
3. Update PROJECTS.md line 29 pipeline description to remove energy detector from the current flow
