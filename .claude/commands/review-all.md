---
name: review-all
description: Run comprehensive review — master review + DSP check + documentation audit. Use after completing a module and writing the handoff.
disable-model-invocation: true
---

Run a full review of the module: $ARGUMENTS

**Prerequisites:** The implementor must have written `docs/reviews/<module>-handoff.md` first.
If the handoff doesn't exist, write it now before proceeding.

## Review Steps

### 1. Determine Review Tier
Check the module's **Review Tier** in `ROADMAP.md`:
- **Tier 1 (Housekeeping):** 10 tool calls, quick pass/fail
- **Tier 2 (Standard):** 30 tool calls, full structured review
- **Tier 3 (Critical):** 60 tool calls, deep analysis

### 2. Spawn master-reviewer subagent
Use the **master-reviewer** subagent with the tier-appropriate prompt from `docs/reviews/REVIEW-TEMPLATE.md`.
- Insert the module name into the template
- Set max_turns based on tier (10/30/60)
- Model: sonnet

### 3. If DSP/signal processing changes are involved
Also use the **dsp-reviewer** subagent to verify:
- STFT parameters match ADR-002
- Frequency ranges are correct
- dB scaling is correct
- Sample rate is explicit (ADR-001)

### 4. If detection algorithm changes are involved
Also use the **detection-validator** subagent to verify detection correctness.

### 5. Write the review file
**The main session writes the review file** (not the subagent — avoids escaping issues).
Save to: `docs/reviews/<module>-review.md`
Use the Review Output Format from `docs/reviews/REVIEW-TEMPLATE.md`.

### 6. Report
Provide a unified summary:
- Total issues found by severity (Blockers / Warnings / Suggestions)
- Verdict: APPROVED or CHANGES NEEDED
- If CHANGES NEEDED: list fixes in priority order
- Documentation gaps found

### 7. Fix issues
Fix any blockers. Update the Fix Log in the review file as you go.
Re-run tests after fixes.
