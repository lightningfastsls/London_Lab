# Workflow Migration Plan: Structured Review Loop for USV Project

**Source workflow:** Tevel ERP project (`D:\we_do_this\tevel-erp`)
**Target project:** Mickey London Lab — USV Detection & Analysis Pipeline
**Created:** 2026-02-16
**Purpose:** Bring the `/implement` → handoff → master-reviewer → fix-log workflow to this project

---

## Why This Migration

The tevel-erp project has a proven implementation workflow:
1. **ROADMAP.md** defines modules as executable `/implement` commands — copy-paste into Claude Code
2. **Implementation Completion Sequence** ensures every module produces a handoff document
3. **Master-reviewer subagent** reviews code against the handoff, producing structured findings
4. **Fix log** tracks resolution of blockers/warnings

This project already has strong foundations (CLAUDE.md state machine, approval requests, 295 tests) but lacks the **structured review loop**. Implementation quality is tracked in a journal (`IMPLEMENTATION_PROGRESS.md`) but there's no formal handoff→review→fix cycle.

The migration adds the review loop without replacing what already works.

---

## Session Breakdown

Execute these in order. Each session is self-contained.

| Session | What | Est. Effort | Key Files |
|---------|------|-------------|-----------|
| **1. Foundation** | Directory structure, DECISIONS.md, review template, completion sequence doc | Medium | 4 new files, 2 new dirs |
| **2. CLAUDE.md Update** | Lean additions + remove Dual-AI section + slim optional sections | Small | 1 file modified |
| **3. ROADMAP.md** | Convert existing plans to `/implement` format | Large | 1 large new file, read 4-5 existing plans |
| **4. Patterns + Module Docs** | Architecture patterns, retroactive module docs | Medium | 5-7 new files |

---

## Session 1: Foundation

### Goal
Create the directory structure and foundational files that all future sessions depend on.

### Files to Create

#### 1.1 Directory Structure

Create these directories:
```
docs/reviews/          (handoff and review files go here)
docs/modules/          (module interface docs go here)
docs/architecture/     (patterns and architecture docs go here)
docs/workflow/         (workflow process docs go here)
```

#### 1.2 `DECISIONS.md` (project root)

This file collects architectural decisions that are currently scattered across planning docs. Use the ADR format below. Read these files to extract decisions:
- `usv_signal_processing_reference.md`
- `vq_vae_transformer_plan.md`
- `IMPLEMENTATION_PROGRESS.md`
- `CNN_RETRAINING_EXPERIMENT_PLAN.md`
- `USV_TRAINING_PIPELINE_PLAN.md`
- `SCALING_TO_30K_ROADMAP.md`

**Format to follow:**

```markdown
# USV Detection & Analysis — Architecture Decision Reference

> Claude Code: read this file when making architectural, data, or design decisions.
> This is the source of truth for "how should I build this?" questions.

---

## ADR-001: Sample Rate — 300 kHz

**Context:** Mouse USVs range from 25-110 kHz. Nyquist requires sampling at least 2x the highest frequency of interest. Our recordings come from the London Lab hardware.

**Decision:** Sample rate is 300,000 Hz (300 kHz). This captures up to 150 kHz, well above the 110 kHz USV ceiling.

**Rule:** Always specify `sr=300000` explicitly. Never rely on library defaults (librosa defaults to 22050 Hz). Some older docs reference 250 kHz — ignore those, the actual hardware records at 300 kHz.

---

## ADR-002: STFT Parameters

**Context:** STFT is the core transform for all spectrogram and detection work. Parameters control the time/frequency resolution tradeoff.

**Decision:**
- `n_fft = 512` (~1.7ms window at 300 kHz) — good time resolution for short USVs
- `hop_length = 128` (75% overlap) — smooth spectrogram, 4 frames per window
- Window function: Hann
- Frequency range: 20-120 kHz (captures full mouse USV range 25-110 kHz with margin)

**Rule:** Any change to STFT parameters requires DSP review (dsp-reviewer agent). These parameters affect every downstream module.

---

## ADR-003: Detection Threshold — 0.05

**Context:** CNN classifier outputs probability 0-1. Threshold determines USV vs. Not USV classification.

**Decision:** Optimal threshold is 0.05, discovered through systematic sweep. Detection app uses 0.04 high / 0.03 low for progressive labeling presets.

**Rule:** Threshold changes require evaluation on the full test set with precision/recall/F1 reported. Never change threshold without baseline comparison.

---

## ADR-004: Dataset Splitting — By Recording, Not By Candidate

**Context:** USV candidates extracted from the same recording are temporally correlated. Splitting randomly would leak temporal patterns into validation/test sets.

**Decision:** Split by recording file, not by individual candidate. All candidates from one recording go into the same split.

**Rule:** Never shuffle candidates across splits without grouping by source recording first. The `splits.py` module enforces this.

---

## ADR-005: Class Weighting — 3.0x for USV Class

**Context:** Dataset is somewhat imbalanced, and false negatives (missed USVs) are more costly than false positives (noise classified as USV) for research purposes.

**Decision:** Class weight 3.0x for USV class during training. This protects recall at a small precision cost.

**Rule:** When retraining, always use class weighting. The specific weight (3.0) can be tuned but the asymmetry must be preserved — recall matters more than precision for detection.

---

## ADR-006: CNN Architecture — 3 Conv Blocks + GlobalAvgPool

**Context:** USV spectrograms are small fixed-size patches. Model must be lightweight enough for real-time inference in the detection app.

**Decision:**
- 3 convolutional blocks: 32 → 64 → 128 filters
- GlobalAveragePooling (not Flatten) — reduces overfitting, parameter-efficient
- ~101,000 parameters total
- Input: spectrogram patches (grayscale)

**Rule:** Architecture changes require retraining + evaluation on held-out test set. Model scaling configs (small → medium → large) are defined in `models/config.py` for when dataset grows to 10K+ labels.

---

## ADR-007: VQ-VAE Codebook Approach for Language Structure

**Context:** Investigating whether mouse USVs have compositional (language-like) structure. Need to discover discrete "phoneme-like" units.

**Decision:** Use VQ-VAE (Vector Quantized Variational Autoencoder) with causal Transformer to:
1. Encode spectrogram columns into discrete codebook entries
2. Learn sequential structure via causal Transformer
3. Analyze codebook usage, transitions, n-grams for compositional evidence

**Rule:** VQ-VAE is a separate subproject (`usv_language/`) with its own configs, tests, and training pipeline. It consumes USV spectrograms but does not modify the detection pipeline.

---

## ADR-008: Negative Sample Strategy

**Context:** Original CNN was trained only on energy-detector candidates, so it couldn't recognize "no USV" cases. This created a critical blind spot.

**Decision:** Comprehensive negative sampling from three sources:
1. Random time slices (no energy peak)
2. Inter-USV gaps (temporal gaps between detected USVs)
3. Low-energy regions (below detection threshold)

**Rule:** Every training dataset must include comprehensive negatives. Never train on detector-positive-only data.

---

## ADR-009: Model Artifacts — PyTorch .pt Files

**Context:** Models need to be saved, versioned, and loaded for inference.

**Decision:**
- Trained models saved as PyTorch state dicts (`.pt` files)
- Production model at `models/full_retrained_cnn/best_model.pt`
- Experimental models in `models/experiment_*/`
- `models/` directory is tracked in git (models are small, ~400KB)

**Rule:** Never overwrite `best_model.pt` without first copying it to a dated backup. The training pipeline handles this automatically.

---

## ADR-010: Label Storage Format — JSON

**Context:** Labels from the detection app need to be stored alongside detections for the training pipeline.

**Decision:** Labels stored as JSON files with candidate metadata, CNN probability, user label, and boundary adjustments. The `LabelStorage` class handles serialization.

**Rule:** Label format changes must be backward-compatible. Old labels must still load correctly after format updates.
```

**IMPORTANT:** The above is a starting template. The session executing this should:
1. Read the reference docs listed above to find any decisions I missed
2. Verify each decision is still accurate (especially ADR-001 — check if sample rate is truly 300k or 250k, there's a conflict in existing docs)
3. Add any additional ADRs discovered

---

#### 1.3 `docs/reviews/REVIEW-TEMPLATE.md`

This is the master template for handoffs and reviews. Create it with this exact content:

```markdown
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
**Model:** Sonnet (fast, cheap).
**What to check:** Nothing broke, tests pass, no orphaned references.
**Output:** Short pass/fail summary (no full structured review needed).

### Tier 2 — Standard
**For:** New modules, new scripts, new pipeline stages, dataset changes.
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
```

---

## Reviewer Prompt Templates

### Tier 1 — Housekeeping Review

```
Review module [MODULE NAME] in the USV Detection project.
This is a TIER 1 (housekeeping) review. Budget: 10 tool calls max.

Read the handoff first:
1. docs/reviews/[module]-handoff.md

Then verify:
1. Tests still pass (check pytest output in handoff)
2. py_compile passes on all changed files
3. No orphaned references to removed code
4. IMPLEMENTATION_PROGRESS.md updated if applicable

Output: Short pass/fail with any issues found. Keep it concise.

Return your findings as text — the main session will write the review file.
```

### Tier 2 — Standard Review

```
Review module [MODULE NAME] in the USV Detection project.
This is a TIER 2 (standard) review. Budget: 30 tool calls max.

Read these first (in order):
1. docs/reviews/[module]-handoff.md (PRIMARY INPUT — start here)
2. ROADMAP.md — the module's section only
3. DECISIONS.md — relevant ADRs
4. docs/architecture/patterns.md

Then review the code files listed in the handoff. Focus on:
1. ROADMAP alignment — does the implementation match the spec?
2. Pattern adherence — follows established patterns from patterns.md
3. DSP parameter consistency — any STFT/frequency/threshold values match DECISIONS.md
4. Test coverage — new behavior has tests, existing tests not modified to pass
5. "What I'm unsure about" section — give extra scrutiny here
6. No data leakage — splits respect recording boundaries (ADR-004)
7. Documentation accuracy — module doc matches actual code

Output: Full structured review (Blockers/Warnings/Suggestions format).
Return your findings as text — the main session will write the review file.
```

### Tier 3 — Critical Review

```
Review module [MODULE NAME] in the USV Detection project.
This is a TIER 3 (critical) review. Budget: 60 tool calls max.

Read these first (in order):
1. docs/reviews/[module]-handoff.md (PRIMARY INPUT — start here)
2. ROADMAP.md — the module's section
3. DECISIONS.md — all relevant ADRs
4. docs/architecture/patterns.md
5. docs/modules/[module].md (if exists)
6. docs/modules/*.md for any modules this one interacts with
7. usv_signal_processing_reference.md (if DSP changes involved)

Then review all code files listed in the handoff. Full checklist:
1. ROADMAP alignment — spec compliance
2. Pattern adherence — established project patterns
3. DSP correctness — STFT params, frequency ranges, dB scaling, threshold values
4. ML rigor — class balance, data leakage prevention, evaluation methodology
5. Test coverage — happy-path + edge cases + DSP-specific tests
6. Test anti-greenwashing — no test expectations modified to pass (check git diff)
7. Cross-module impact — does this break detection app, training pipeline, etc.?
8. Signal processing conventions — sr=300000 explicit, Hann window, frequency bounds
9. Documentation — module doc, patterns doc, DECISIONS.md all accurate
10. "What I'm unsure about" — deep scrutiny on flagged areas

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
**Verdict:** [APPROVED | CHANGES NEEDED]

---

## BLOCKER (must fix before next module)

### B[n]. [Short title]

**Files:** [file paths with line numbers]
**Problem:** [What's wrong and WHY it matters]
**Fix:** [Concrete fix instructions]

---

## WARNINGS (fix soon)

### W[n]. [Short title]

**File:** [file path]
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

**[APPROVED or CHANGES NEEDED]**

[If CHANGES NEEDED: what must be fixed, in priority order.]

---

## Documentation Status (Tier 2-3 only)

| Doc | Status | Issues |
|-----|--------|--------|
| Module doc | EXISTS / MISSING / STALE | [details] |
| patterns.md | UP TO DATE / NEEDS UPDATE | [what's missing] |
| DECISIONS.md | UP TO DATE / NEEDS UPDATE | [new ADR needed?] |
| IMPLEMENTATION_PROGRESS.md | UPDATED / NOT UPDATED | [details] |

---

## Fix Log

| Item | Status | Fixed by | Date | Notes |
|------|--------|----------|------|-------|
| B1 | OPEN | | | |
```

---

## Severity Definitions

- **BLOCKER**: Correctness risk, DSP parameter error, data leakage, or test anti-greenwashing violation that WILL cause problems. Must fix before next module.
- **WARNING**: Doesn't break today but creates risk or deviates from spec. Fix soon.
- **SUGGESTION**: Code quality or minor improvements. Fix when convenient.

## Verdict Rules

- **APPROVED**: Zero blockers, warnings are minor and documented.
- **CHANGES NEEDED**: Any blockers, OR 3+ warnings that together represent significant risk.

## Workflow

1. Implementor completes module
2. Implementor writes handoff (`docs/reviews/<module>-handoff.md`)
3. Main session spawns master-reviewer subagent with tier-appropriate prompt (Sonnet model)
4. **Main session writes the review file** (not the subagent — avoids shell escaping issues with special characters in findings)
5. Implementor fixes issues, updates Fix Log in the review file
```

---

#### 1.4 `docs/workflow/completion-sequence.md`

This is the detailed workflow that CLAUDE.md will point to. Create with this exact content:

```markdown
# Implementation Completion Sequence

**Non-negotiable.** Follow this for ANY task that creates or modifies 2+ files.

## The 7-Step Sequence

### Step 1: Create Tasks (Including Handoff)

When breaking work into tasks at the start, ALWAYS include this as the LAST task:
> "Write implementation handoff (`docs/reviews/<module>-handoff.md`)"

This ensures the handoff appears in the task list and won't be forgotten after testing output bloats the context window.

**Why this matters:** Without the handoff task in the list, sessions forget to write it. Test debugging can consume 20+ tool calls and push the handoff out of mind. The task list is a persistent reminder that survives context bloat. This lesson was learned the hard way — a review without a handoff took 106 tool calls and 26 minutes because the reviewer had to explore the entire codebase blind.

### Step 2: Write Code

Implement all files per the approved plan. Follow:
- Established patterns (`docs/architecture/patterns.md`)
- Architecture decisions (`DECISIONS.md`)
- ROADMAP spec (if this is a ROADMAP module)

### Step 3: Run Module Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_<module>.py -v
```

### Step 4: Run Full Test Suite

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

Check for regressions in existing tests. If VQ-VAE tests are relevant:
```powershell
.\.venv\Scripts\python.exe -m pytest usv_language/tests/ -v
```

### Step 5: Fix Any Failures

Iterate until all tests pass. **NEVER modify test expectations to make tests pass** — fix the code or discuss with the user.

### Step 6: Write the Handoff

The handoff task should be sitting in your task list as `pending`. Write it now:
- Save to `docs/reviews/<module>-handoff.md`
- Follow the template in `docs/reviews/REVIEW-TEMPLATE.md`
- Be honest in "What I'm Unsure About" — this directs the reviewer's attention

Mark the handoff task as `completed`.

### Step 7: Report

Summarize results to user:
- What was built
- Test results (pass counts)
- Any known limitations
- Mention that the handoff is ready for review

---

## When to Skip This Sequence

- Single-line fixes (typo, config tweak)
- Tasks that only modify 1 file
- Documentation-only changes
- Exploratory/research tasks

For these, just do the work, run tests, and report. No handoff needed.

---

## Requesting a Review

After the handoff is written, the user (or main session) triggers a review:

1. Determine the review tier (see `docs/reviews/REVIEW-TEMPLATE.md`)
2. Spawn a master-reviewer subagent with the tier-appropriate prompt from the template
3. The main session writes the review file based on the subagent's findings
4. Fix any blockers, update the Fix Log in the review file

---

## Relationship to Existing Workflow

This sequence **extends** the existing CLAUDE.md workflow, it doesn't replace it:

| Existing CLAUDE.md | This Sequence |
|-------------------|---------------|
| Approval Request before code | Still required (Step 2 assumes approval was given) |
| py_compile after every edit | Still required during Step 2 |
| Update IMPLEMENTATION_PROGRESS.md | Still required in Step 7 report |
| Run specialized agents (dsp-reviewer, etc.) | Still required — run DURING implementation, not as replacement for master review |
| Don't modify test expectations | Reinforced in Step 5 |
```

---

### Session 1 Verification Checklist

After creating all files, verify:
- [ ] `docs/reviews/` directory exists
- [ ] `docs/modules/` directory exists
- [ ] `docs/architecture/` directory exists
- [ ] `docs/workflow/` directory exists
- [ ] `DECISIONS.md` exists in project root, has at least ADR-001 through ADR-010
- [ ] `docs/reviews/REVIEW-TEMPLATE.md` exists with all three tier templates
- [ ] `docs/workflow/completion-sequence.md` exists with the 7-step sequence
- [ ] Each ADR in DECISIONS.md was verified against actual code/docs (especially sample rate)
- [ ] No existing files were modified in this session

---

## Session 2: CLAUDE.md Update

### Goal
Add lean workflow references to CLAUDE.md. Remove the Dual-AI section. Optionally slim other sections.

### Changes to Make

#### 2.1 REMOVE: Dual-AI Workflow Section (lines ~279-298)

Delete the entire section:
```markdown
## DUAL-AI WORKFLOW: Claude Code + Codex
[... everything through the end of that section ...]
```

This section is no longer relevant — the user is using Claude Code exclusively.

#### 2.2 ADD: Implementation Completion Sequence (after "Session Workflow" section, before "Model Selection Guide")

Add this lean section (~15 lines):

```markdown
## Implementation Completion Sequence

**Non-negotiable** for any task creating/modifying 2+ files. Full details: `docs/workflow/completion-sequence.md`

**Key steps:** Create tasks (including handoff task) → Write code → Run module tests → Run full suite → Fix failures → Write handoff (`docs/reviews/<module>-handoff.md`) → Report

**Critical rule:** The handoff task must be created at the START as a TaskCreate item. It persists in the task list as a visible reminder even after 50+ tool calls of test debugging.

**When to skip:** Single-file changes, documentation-only, exploratory tasks.
```

#### 2.3 ADD: Module Reviews (after the new section above)

```markdown
## Module Reviews

Reviews use a **tiered system** matched to module complexity. Full templates: `docs/reviews/REVIEW-TEMPLATE.md`

| Tier | For | Budget | Model |
|------|-----|--------|-------|
| 1 — Housekeeping | Config, cleanup, small fixes | 10 calls | Sonnet |
| 2 — Standard | New modules, scripts, pipelines | 30 calls | Sonnet |
| 3 — Critical | ML models, DSP changes, detection algorithm | 60 calls | Sonnet |

**Workflow:** Implementor writes handoff → main session spawns master-reviewer → main session writes review file → implementor fixes issues.

**Rule:** Handoff is mandatory input for review. Never skip it.
```

#### 2.4 ADD: Documentation Requirements (after the reviews section)

```markdown
## Documentation: Document As You Build

### Required Docs Per Module
1. **Module doc** (`docs/modules/<module>.md`) — purpose, public interface, usage, decisions
2. **Architecture patterns** (`docs/architecture/patterns.md`) — update if you establish a new pattern
3. **ADR** (`DECISIONS.md`) — add if you make a non-obvious architectural decision

### Before Building a New Module
1. Read `ROADMAP.md` for the module spec and dependencies
2. Read `DECISIONS.md` for architectural constraints
3. Read `docs/architecture/patterns.md` for established patterns
4. Read `docs/modules/*.md` for any module you'll interact with
```

#### 2.5 ADD: Key Reference Documents Update

Update the existing "Key Reference Documents" table to include new files:

```markdown
| Document | When to Read |
|----------|--------------|
| `IMPLEMENTATION_PROGRESS.md` | **Start of every session** |
| `ROADMAP.md` | **Before implementing any module** |
| `DECISIONS.md` | **Before any architectural/design choice** |
| `docs/architecture/patterns.md` | Before implementing (follow established patterns) |
| `docs/workflow/completion-sequence.md` | When implementing 2+ file changes |
| `docs/reviews/REVIEW-TEMPLATE.md` | When writing handoff or requesting review |
| `USV_TRAINING_PIPELINE_PLAN.md` | Building training data generation pipeline |
| `USV_DETECTION_APP_IMPLEMENTATION.md` | Building PyQt6 desktop app for detection |
| `usv_signal_processing_reference.md` | Any signal processing work |
```

#### 2.6 OPTIONAL: Slim Signal Processing Conventions

The signal processing conventions table (lines ~300-313) could move to `docs/architecture/dsp-conventions.md` with a one-line reference in CLAUDE.md:
```markdown
## Signal Processing Conventions
See `docs/architecture/dsp-conventions.md`. Key rule: always specify `sr=300000` explicitly.
```

This saves ~15 lines. Only do this if CLAUDE.md is still over 300 lines after the other changes.

#### 2.7 OPTIONAL: Move Token Usage Optimization

The "Token Usage Optimization" section (lines ~233-260) is useful but long. Could move to `docs/workflow/token-optimization.md` with a brief reference:
```markdown
## Token Usage
See `docs/workflow/token-optimization.md`. Key: use haiku for exploration, sonnet for implementation.
```

This saves ~25 lines. Again, only if CLAUDE.md is still heavy.

### Session 2 Verification Checklist

After making changes:
- [ ] Dual-AI Workflow section is completely removed
- [ ] Implementation Completion Sequence section added (pointing to `docs/workflow/completion-sequence.md`)
- [ ] Module Reviews section added (pointing to `docs/reviews/REVIEW-TEMPLATE.md`)
- [ ] Documentation requirements section added
- [ ] Key Reference Documents table updated with new files
- [ ] CLAUDE.md is under 320 lines (target: ~280-300)
- [ ] py_compile still works on all Python files (sanity check — CLAUDE.md changes shouldn't break anything)
- [ ] No existing behavioral contract rules were accidentally removed

---

## Session 3: ROADMAP.md

### Goal
Create a single executable ROADMAP.md that defines upcoming work as `/implement` commands.

### Prerequisites
Read these files thoroughly first:
1. `SCALING_TO_30K_ROADMAP.md` — active learning loop, milestones 1-5
2. `vq_vae_transformer_plan.md` — VQ-VAE phases 0-6 (0-6 are DONE, execution pending)
3. `USV_CLUSTERING_EXPLORATION_PLAN.md` — unsupervised clustering approach
4. `USV_DETECTION_APP_IMPLEMENTATION.md` — detection app features/roadmap
5. `USV_TRAINING_PIPELINE_PLAN.md` — training pipeline architecture
6. `IMPLEMENTATION_PROGRESS.md` — what's already done
7. `DECISIONS.md` — architectural constraints to reference

### ROADMAP.md Format

Follow this structure exactly:

```markdown
# USV Detection & Analysis — Implementation Roadmap

> This is the executable implementation plan. Each module has a `/implement` command
> that can be copy-pasted into Claude Code. Read DECISIONS.md before implementing.

## Status Key
- **DONE** — Implemented and tested
- **READY** — Dependencies met, can start
- **BLOCKED** — Waiting on dependency or external input
- **FUTURE** — Not yet prioritized

---

## Phase [N]: [Phase Name]

### [N.M] [Module Name]

**What:** [1-2 sentence description]
**Status:** [DONE | READY | BLOCKED | FUTURE]
**Review Tier:** [1 | 2 | 3]
**Depends on:** [prerequisite modules]

/implement [Module Name]

[Multi-line detailed spec. Include:
- Exact file paths to create/modify
- Data structures with field types
- Algorithm description (for ML/DSP modules)
- Integration points with existing code
- Edge cases to handle
- Config parameters with values]

**Test plan:**
1. [Specific test case with expected behavior]
2. [Specific test case with expected behavior]
3. [...]

**Exit criteria:**
- [ ] Criterion with observable outcome
- [ ] Criterion with observable outcome
- [ ] [...]
```

### Content to Convert

Based on the existing planning docs, here are the approximate phases to define. The session executing this should read the full plans and fill in the detailed `/implement` specs:

**Already Done (mark as DONE — still document for reference):**
- Phase 1: Detection Pipeline (EnergyDetector, DetectionConfig, Candidate)
- Phase 2: Spectrogram Extraction (SpectrogramExtractor)
- Phase 3: Labeling Tool (Streamlit labeling app)
- Phase 4: Dataset Preparation (splits, quality checks, metadata)
- Phase 5: CNN Classifier (architecture, training, evaluation)
- Phase 6: Detection App (PyQt6 desktop app with progressive labeling)
- Phase 7: VQ-VAE + Transformer (usv_language/ module, phases 0-6)

**Upcoming Work (needs full `/implement` specs):**
- Scaling Phase 1: Hard Negative Mining at Scale
- Scaling Phase 2: Active Learning Loop Automation
- Scaling Phase 3: Model Architecture Scaling (small → medium → large)
- Scaling Phase 4: Comprehensive QC Pipeline
- Clustering Exploration: Feature extraction, clustering, visualization
- VQ-VAE Execution: Preprocess real data, train on HPC, analyze results

### Key Rules for Writing the ROADMAP

1. **`/implement` commands must be self-contained** — a developer should be able to copy-paste the command block and have everything needed to build the module
2. **Include exact file paths** — not "create a new file in the detection module" but "create `src/usv_spectrogram/detection/hard_negatives.py`"
3. **Include data structures** — show class fields, config parameters, function signatures
4. **Test plans are specific** — not "test edge cases" but "test that a 5ms candidate is rejected by min_duration filter"
5. **Exit criteria are observable** — not "works correctly" but "precision >= 89% on held-out test set"
6. **Reference ADRs** — when a decision constrains the implementation, cite it (e.g., "per ADR-004, split by recording")

### Session 3 Verification Checklist

After creating ROADMAP.md:
- [ ] All existing completed work is listed as DONE
- [ ] Upcoming modules have full `/implement` command blocks
- [ ] Each module has: What, Status, Review Tier, Depends on, implement block, Test plan, Exit criteria
- [ ] `/implement` blocks include exact file paths
- [ ] Test plans have specific assertions, not vague descriptions
- [ ] Exit criteria are measurable/observable
- [ ] ADRs are referenced where relevant
- [ ] No conflicts with existing DECISIONS.md rules

---

## Session 4: Patterns + Module Docs

### Goal
Document established code patterns and create retroactive module docs for key existing modules.

### 4.1 `docs/architecture/patterns.md`

Read the source code to extract patterns. Expected patterns to document:

```markdown
# Architecture Patterns

Reusable patterns established across the USV Detection codebase.
Reference these when building new modules.

## 1. Config Dataclass Pattern

All configurable modules use frozen dataclasses with defaults:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class DetectionConfig:
    sample_rate: int = 300_000
    n_fft: int = 512
    hop_length: int = 128
    min_freq_hz: float = 20_000
    max_freq_hz: float = 120_000
    # ...
```

Key rules:
- frozen=True (immutable after creation)
- Sensible defaults for every field
- Numeric params have units in the name (_hz, _ms, _db)

## 2. Candidate Data Flow

Candidates flow through the pipeline as frozen dataclasses:
```
WAV → EnergyDetector.detect() → list[Candidate]
Candidate → SpectrogramExtractor.extract() → numpy array
numpy array → CNN.predict() → probability float
probability + label → LabelStorage → JSON file
```

## 3. Test Fixture Pattern

Tests use synthetic WAV data, never real recordings:
```python
@pytest.fixture
def synthetic_wav(tmp_path):
    """Generate a WAV with known frequency content."""
    # ... generate sine wave at known frequency ...
    return wav_path
```

## 4. Script CLI Pattern

Scripts in `scripts/` follow this pattern:
```python
import argparse

def main():
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument(...)
    args = parser.parse_args()
    # ... do work ...

if __name__ == "__main__":
    main()
```

## 5. PyQt6 Widget Structure

Detection app follows Model-View separation:
- `core/` — business logic (no Qt imports)
- `widgets/` — Qt widgets (no business logic)
- `main_window.py` — orchestration (connects core to widgets)

## 6. [Add more patterns as discovered during this session]
```

### 4.2 Module Docs

Create docs for the most critical existing modules. Prioritize:

1. **`docs/modules/energy-detector.md`** — the core detection algorithm
2. **`docs/modules/cnn-classifier.md`** — the ML model
3. **`docs/modules/detection-app.md`** — the PyQt6 app
4. **`docs/modules/dataset-pipeline.md`** — splits, quality checks, metadata
5. **`docs/modules/vq-vae-pipeline.md`** — the usv_language/ subproject

Each module doc follows this format:

```markdown
# [Module Name]

**Phase:** [which phase built this]
**ADRs:** [relevant ADR numbers]
**Tests:** [test file path and count]

## Purpose

[1-2 sentences]

## Public Interface

[Function signatures, class APIs, CLI commands]

## Data Model

[Key classes, dataclasses, data flow]

## Usage Examples

[Code snippets showing how to use the module]

## Key Decisions

[Numbered list with rationale — reference ADRs]

## Integration Points

[What consumes this module, what this module consumes]
```

**Don't write all 5 in one session if context is getting heavy.** Prioritize energy-detector and cnn-classifier as they're the most referenced by downstream work. The rest can be done in a follow-up.

### Session 4 Verification Checklist

After creating pattern and module docs:
- [ ] `docs/architecture/patterns.md` exists with at least 5 patterns
- [ ] Each pattern has a code example
- [ ] At least 2 module docs created in `docs/modules/`
- [ ] Module docs have: Purpose, Public Interface, Usage Examples
- [ ] No contradictions with DECISIONS.md
- [ ] IMPLEMENTATION_PROGRESS.md updated to note doc creation

---

## Post-Migration: How the Workflow Works Day-to-Day

Once all 4 sessions are complete, this is the workflow for new features:

### Implementing a ROADMAP Module

1. **Read** ROADMAP.md, find the module, copy the `/implement` command
2. **Paste** into Claude Code — it enters plan mode, presents approach
3. **Approve** — Claude creates tasks (including handoff task)
4. **Implement** — Claude writes code, runs tests, fixes failures
5. **Handoff** — Claude writes `docs/reviews/<module>-handoff.md`
6. **Review** — You spawn master-reviewer: paste the tier-appropriate prompt from `docs/reviews/REVIEW-TEMPLATE.md`, inserting the module name
7. **Fix** — Claude fixes blockers/warnings, updates Fix Log
8. **Done** — Module is complete, documented, reviewed

### Implementing Ad-Hoc Work (not in ROADMAP)

Same as above, but skip step 1. The existing CLAUDE.md approval request still applies. If the work modifies 2+ files, follow the completion sequence.

### When to Add to ROADMAP vs Just Do It

- **Add to ROADMAP:** New capabilities, new pipeline stages, major refactors
- **Just do it:** Bug fixes, small enhancements, config tweaks

---

## Files Reference

After all 4 sessions, these new files will exist:

```
mickey_london_lab/
├── DECISIONS.md                              (NEW — session 1)
├── ROADMAP.md                                (NEW — session 3)
├── CLAUDE.md                                 (MODIFIED — session 2)
├── docs/
│   ├── reviews/
│   │   └── REVIEW-TEMPLATE.md                (NEW — session 1)
│   ├── workflow/
│   │   └── completion-sequence.md            (NEW — session 1)
│   ├── architecture/
│   │   └── patterns.md                       (NEW — session 4)
│   └── modules/
│       ├── energy-detector.md                (NEW — session 4)
│       ├── cnn-classifier.md                 (NEW — session 4)
│       ├── detection-app.md                  (NEW — session 4, optional)
│       ├── dataset-pipeline.md               (NEW — session 4, optional)
│       └── vq-vae-pipeline.md                (NEW — session 4, optional)
```

Existing files preserved as-is:
- `IMPLEMENTATION_PROGRESS.md` — continues as session journal
- `USV_TRAINING_PIPELINE_PLAN.md` — reference (superseded by ROADMAP for new work)
- `vq_vae_transformer_plan.md` — reference (superseded by ROADMAP for new work)
- `SCALING_TO_30K_ROADMAP.md` — reference (superseded by ROADMAP for new work)
- All other existing planning docs — kept for historical context
