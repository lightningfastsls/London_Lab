---
name: implementor-stage-gate
description: Enforce stage-by-stage implementation for multi-step task briefs. Use when acting as the implementor to deliver one stage at a time, pausing for explicit user confirmation before proceeding unless the user requests all stages at once.
---

# Implementor Stage Gate

## Overview
Apply a strict stage gate: implement only the current stage of the task brief, then stop and ask for confirmation before moving on.

## Workflow
1) Read the task brief and identify the staged plan.
2) Implement only the first incomplete stage.
3) Update `tasks/<date>_<slug>/10_impl_notes.md` with what changed and why.
4) Report progress and explicitly ask whether to proceed to the next stage.
5) Proceed to the next stage only after the user confirms or explicitly requests all stages.

## Exceptions
- If the user explicitly asks for all stages at once, note that in `10_impl_notes.md` and proceed without gating.
- If a stage is blocked, stop and ask for direction before continuing.
