# Review System

This directory contains module reviews and the templates that guide them.

## Core Principle: Implementor Writes the Handoff

The implementor (the session that built the module) writes an **Implementation Handoff** before requesting a review. This gives the reviewer pre-digested context instead of forcing it to explore the entire codebase. The handoff is the reviewer's primary input.

---

## Review Tiers

Every module in ROADMAP.md should be tagged with a tier. Match review depth to module complexity.

### Tier 1 — Housekeeping
**For:** Cleanup, config changes, dependency updates, small bug fixes, documentation-only changes.
**Tool call budget:** 10 max.
**Model:** Sonnet.
**What to check:** Nothing broke, tests pass, no orphaned references.
**Output:** Short pass/fail summary (no full structured review needed).

### Tier 2 — Standard
**For:** New modules, new scripts, new pipeline stages, dataset changes, refactors.
**Tool call budget:** 30 max.
**Model:** Sonnet.
**What to check:** Pattern adherence, test coverage, DSP parameter consistency, ROADMAP alignment.
**Output:** Full structured review (Blockers/Warnings/Suggestions).

### Tier 3 — Critical
**For:** ML model changes, detection algorithm changes, STFT/DSP modifications, training pipeline changes, anything touching `energy_detector.py`.
**Tool call budget:** 60 max.
**Model:** Sonnet (Opus only for complex DSP debugging).
**What to check:** Full checklist — algorithmic correctness, DSP parameter consistency, data leakage prevention, test anti-greenwashing, cross-module impact, documentation completeness.
**Output:** Full structured review with detailed analysis.

---

## Tier Selection Guide

| Change Type | Tier | Rationale |
|-------------|------|-----------|
| Documentation only | 1 | No runtime impact |
| Test additions (no code change) | 1 | Can't break existing behavior |
| Config value tweak | 1-2 | Depends on which config |
| New utility function/script | 2 | New code path |
| Streamlit UI change | 2 | User-facing but isolated |
| Detection parameter change | 2-3 | Affects detection results |
| New dataset pipeline stage | 2 | Data integrity risk |
| STFT parameter change | 3 | Core DSP, affects everything downstream |
| Energy detector logic | 3 | Core algorithm |
| CNN architecture change | 3 | Model behavior |
| New training pipeline | 3 | Data integrity + model quality risk |
| VQ-VAE model changes | 3 | Research-critical code |

---

## Implementation Handoff Template

The implementor writes this file AFTER completing a module and BEFORE requesting a review.
Save as: `docs/reviews/<module>-handoff.md`

```
# Implementation Handoff: [Module Name]

**Module:** [e.g., Hard Negative Mining Pipeline]
**Review Tier:** [1 | 2 | 3]
**Date:** [YYYY-MM-DD]
**Branch:** [branch name, or "main" if committing directly]

## What Changed

Summary of what was built/changed (3-5 bullet points).

## Files Changed

List of files created, modified, or deleted:
- `src/usv_spectrogram/detection/hard_negatives.py` (NEW) — hard negative mining logic
- `scripts/mine_hard_negatives.py` (NEW) — CLI entry point
- `src/usv_spectrogram/detection/config.py` (MODIFIED) — added mining config fields
- `tests/test_energy_detector.py` (MODIFIED) — added 5 mining tests

## Key Decisions Made

Non-obvious choices the reviewer should scrutinize:
- Chose inter-USV gap sampling over random-time sampling because [reason]
- Set minimum gap duration to 50ms based on [evidence]
- [etc.]

## What I'm Unsure About

Areas where I'd like extra scrutiny:
- The frequency band filtering during negative extraction — edge cases near 25kHz?
- Whether the jittering bounds are tight enough to avoid overlapping real USVs

## Test Results

```
.\.venv\Scripts\python.exe -m pytest tests/ -v
295 passed, 0 failed
```

## ROADMAP Exit Criteria Status

- [x] Criterion 1
- [x] Criterion 2
- [ ] Criterion 3 (reason it's not done yet)

## Docs Written/Updated

- `docs/modules/hard-negative-mining.md` — created
- `docs/architecture/patterns.md` — added Pattern N (Negative Sampling)
- `DECISIONS.md` — no new ADRs needed
```

---

## Reviewer Prompt Templates

### Tier 1 — Housekeeping Review

Copy-paste this prompt when spawning a master-reviewer for a Tier 1 review:

```
Review module [MODULE NAME] in the USV Detection project.
This is a TIER 1 (housekeeping) review. Budget: 10 tool calls max.

Read the handoff first:
1. docs/reviews/[module]-handoff.md

Then verify:
1. Tests still pass (check pytest output in handoff)
2. py_compile passes on all changed files
3. No orphaned references to removed code (grep for deleted identifiers)
4. Docs mentioned in handoff actually exist
5. IMPLEMENTATION_PROGRESS.md updated if applicable

Output: Short pass/fail summary with any issues found.
Return your findings as text — the main session will write the review file.
```

### Tier 2 — Standard Review

Copy-paste this prompt when spawning a master-reviewer for a Tier 2 review:

```
Review module [MODULE NAME] in the USV Detection project.
This is a TIER 2 (standard) review. Budget: 30 tool calls max.

Read these first (in order):
1. docs/reviews/[module]-handoff.md (PRIMARY INPUT — start here)
2. ROADMAP.md — the module's section only
3. DECISIONS.md — relevant ADRs
4. docs/architecture/patterns.md

Then review the code files listed in the handoff. Check each item:

1. ROADMAP alignment — does the implementation match the /implement spec?
2. Pattern adherence — follows established patterns from patterns.md
3. DSP parameter consistency — any STFT, frequency, or threshold values must match DECISIONS.md (ADR-001: sr=300000, ADR-002: n_fft=512/hop=128, ADR-003: threshold=0.05)
4. Data leakage prevention — if dataset splits are involved, verify they respect recording boundaries per ADR-004
5. Test coverage — new behavior has tests, no test expectations modified to pass (check git diff if available)
6. Test anti-greenwashing — are expected values calculated from first principles, not copy-pasted from failing output?
7. "What I'm unsure about" section — give extra scrutiny to these flagged areas
8. Documentation accuracy — module doc matches actual code signatures and behavior

Output: Full structured review using the Blockers/Warnings/Suggestions format below.
Return your findings as text — the main session will write the review file.
```

### Tier 3 — Critical Review

Copy-paste this prompt when spawning a master-reviewer for a Tier 3 review:

```
Review module [MODULE NAME] in the USV Detection project.
This is a TIER 3 (critical) review. Budget: 60 tool calls max.

Read these first (in order):
1. docs/reviews/[module]-handoff.md (PRIMARY INPUT — start here)
2. ROADMAP.md — the module's section
3. DECISIONS.md — all relevant ADRs
4. docs/architecture/patterns.md
5. usv_signal_processing_reference.md (if DSP changes involved)
6. docs/modules/[module].md (if exists)
7. docs/modules/*.md for any modules this one interacts with

Then review ALL code files listed in the handoff. Full checklist — check every item:

1. ROADMAP alignment — does implementation match the /implement spec?
2. Pattern adherence — follows established patterns from patterns.md
3. DSP correctness — STFT params match ADR-002 (n_fft=512, hop=128, Hann), sample rate is sr=300000 explicit per ADR-001, frequency bounds 20-120kHz, dB scaling correct
4. ML rigor — class balance maintained (ADR-005: 3.0x USV weight), evaluation on held-out test set, no training on validation data
5. Data leakage prevention — splits respect recording boundaries per ADR-004, no temporal correlation leakage
6. Test coverage — happy-path + edge cases + DSP-specific tests where applicable
7. Test anti-greenwashing — NO test expectations modified to make tests pass (verify via git diff if available). Expected values must be derived from first principles, not copied from failing output
8. Cross-module impact — does this break the detection app, training pipeline, labeling tools, or VQ-VAE pipeline?
9. Signal processing conventions — sr=300000 always explicit, Hann window, frequency bounds checked, units in variable names (_hz, _ms, _db)
10. Backward compatibility — does this break existing saved labels, trained models, or detection exports?
11. Documentation — module doc, patterns.md, DECISIONS.md all accurate and updated
12. "What I'm unsure about" — deep scrutiny on every flagged area

Required specialized agents (invoke if relevant):
- dsp-reviewer: MUST invoke for any signal processing changes
- detection-validator: MUST invoke for any detection logic changes

Output: Full structured review with Documentation Status table.
Return your findings as text — the main session will write the review file.
```

---

## Review Output Format

All reviews (Tier 2 and 3) use this structure. Tier 1 can use a shortened version.

```
# [Module Name] Module Review

**Reviewed by:** Master Reviewer
**Date:** [date]
**Module:** [module name]
**Tier:** [1 | 2 | 3]
**Verdict:** [APPROVED | APPROVED WITH WARNINGS | CHANGES NEEDED]

---

## BLOCKER (must fix before next module)

### B[n]. [Short title]

**Files:** [file paths with line numbers]
**Problem:** [What's wrong and WHY it matters — e.g., "sr=250000 used instead of 300000, which shifts all frequency bins by 17%"]
**Fix:** [Concrete fix instructions — exact code change if possible]

---

## WARNINGS (fix soon)

### W[n]. [Short title]

**File:** [file path with line number]
**Problem:** [description]
**Fix:** [actionable fix]

---

## SUGGESTIONS (nice to have)

| # | Issue | File | Fix |
|---|-------|------|-----|
| S1 | ... | ... | ... |

---

## Summary

| Severity | Count | Items |
|----------|-------|-------|
| BLOCKER | N | B1, B2, ... |
| WARNING | N | W1, W2, ... |
| SUGGESTION | N | S1, S2, ... |

---

## Verdict

**[APPROVED / APPROVED WITH WARNINGS / CHANGES NEEDED]**

[If CHANGES NEEDED: what must be fixed, in priority order.]

---

## Documentation Status (Tier 2-3 only)

| Doc | Status | Issues |
|-----|--------|--------|
| Module doc (`docs/modules/`) | EXISTS / MISSING / STALE | [details] |
| `docs/architecture/patterns.md` | UP TO DATE / NEEDS UPDATE | [what's missing] |
| `DECISIONS.md` | UP TO DATE / NEEDS UPDATE | [new ADR needed?] |
| `IMPLEMENTATION_PROGRESS.md` | UPDATED / NOT UPDATED | [details] |

---

## Fix Log

Track resolution of findings here. Implementor updates this section after fixing issues.

| Item | Status | Fixed in | Date | Notes |
|------|--------|----------|------|-------|
| B1 | OPEN | | | |
| W1 | OPEN | | | |
```

---

## Severity Definitions

- **BLOCKER**: Correctness risk, DSP parameter error, data leakage, test anti-greenwashing violation, or silent data corruption that WILL cause problems. Must fix before next module.
- **WARNING**: Doesn't break today but creates risk or deviates from spec. Missing test coverage, pattern deviation without justification, stale docs. Fix soon.
- **SUGGESTION**: Code quality, naming, documentation improvements. Fix when convenient.

## Verdict Rules

| Blockers | Warnings | Verdict |
|----------|----------|---------|
| 0 | 0 | APPROVED |
| 0 | 1-2 | APPROVED WITH WARNINGS |
| 0 | 3+ | CHANGES NEEDED (cumulative risk) |
| 1+ | any | CHANGES NEEDED |

A CHANGES NEEDED verdict means the implementor must fix all blockers (and high-priority warnings) and update the Fix Log. Re-review is required only if blockers were found.

---

## Workflow

1. Implementor completes module (follows `docs/workflow/completion-sequence.md`)
2. Implementor writes handoff (`docs/reviews/<module>-handoff.md`) using the template above
3. Main session spawns master-reviewer subagent with the tier-appropriate prompt above (Sonnet model)
4. **Main session writes the review file** (not the subagent — avoids shell escaping issues with special characters)
5. Implementor fixes issues, updates Fix Log in the review file
6. If blockers existed: re-review after fixes (can be Tier 1 spot check on the fixes)
