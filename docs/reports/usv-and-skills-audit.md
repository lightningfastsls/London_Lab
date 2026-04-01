# USV Pipeline & Skills Context Audit

**Date:** 2026-02-26
**Purpose:** Map every place where Claude (or another model) makes decisions involving implicit constraint reasoning, for STAR reasoning framework optimization (arXiv:2602.21814).

---

## 1. Skills Inventory

21 skills found in `.claude/skills/`:

| # | Skill | Description | Prompt Templates? | Multi-step Reasoning? |
|---|-------|-------------|-------------------|-----------------------|
| 1 | **reduce** | Extract structured knowledge from source material. Comprehensive extraction default. | Yes — extensive extraction principles, selectivity gates, category tables | Yes — multi-phase extraction with quality gates |
| 2 | **reflect** | Find connections between notes and update MOCs. Dual discovery (MOC + semantic search). | Yes — articulation test, connection evaluation criteria | Yes — 10-step workflow with Phase 0 index check |
| 3 | **reweave** | Backward pass updating old notes with new connections. | Yes — connection criteria, split/sharpen rules | Yes — revisit + evaluate + update cycle |
| 4 | **verify** | Combined verification — recite + validate + review. Quality gate for notes. | Yes — cold-read prediction test, schema checks | Yes — 3-phase verification pipeline |
| 5 | **pipeline** | End-to-end source processing — seed, reduce, reflect/reweave/verify, archive. | Yes — orchestration instructions | Yes — full pipeline orchestration with subagents |
| 6 | **ralph** | Queue processing with fresh context per phase. Spawns isolated subagents. | Yes — task queue management, handoff parsing | Yes — serial/parallel batch processing |
| 7 | **architect** | Research-backed evolution advice. 7-phase analysis: Locate → Read Derivation → Health → Friction → Research → Recommendations → Present. | Yes — evidence chain requirements, recommendation format | Yes — 7 sequential phases with research grounding |
| 8 | **ask** | Query bundled research knowledge graph. 3-tier routing: WHY/HOW/WHAT IT LOOKS LIKE. | Yes — query classification rules, synthesis format | Yes — 6-step search+synthesize workflow |
| 9 | **health** | Vault health diagnostics. 8 categories, 3 modes (quick/full/three-space). | Yes — threshold tables, diagnostic scripts | Yes — 8 diagnostic categories with condition checks |
| 10 | **recommend** | Research-backed architecture advice. 8-dimension configuration with research grounding. | Yes — dimension mapping, constraint validation | Yes — 6-phase recommendation workflow |
| 11 | **remember** | Capture friction as methodology notes. 3 modes — explicit, contextual, session mining. | Yes — observation categorization | Yes — evidence gathering + note creation |
| 12 | **rethink** | Challenge system assumptions against evidence. Triage observations and tensions. | Yes — pattern detection, proposal generation | Yes — evidence accumulation + proposal |
| 13 | **seed** | Add source file to processing queue. Duplicate check, archive, task creation. | Yes — queue management instructions | Yes — multi-step queue workflow |
| 14 | **learn** | Research a topic via web search / Exa deep researcher. Files results with provenance. | Yes — search strategy, provenance rules | Yes — research + file + chain to pipeline |
| 15 | **graph** | Interactive knowledge graph analysis. Routes to graph scripts. | Yes — operation routing table | Yes — query → script → interpret → suggest |
| 16 | **note-history** | Git-based note evolution tracking with restore capability. | Yes — semantic diff interpretation rules | Yes — locate → history → interpret → (restore) |
| 17 | **stats** | Vault statistics and knowledge graph metrics snapshot. | Yes — metric calculation instructions | Moderate — gather + compute + format |
| 18 | **tasks** | View and manage task stack and processing queue. | Yes — task management commands | Moderate — read + display + modify |
| 19 | **validate** | Schema validation for notes against domain-specific templates. | Yes — field validation rules | Moderate — scan + check + report |
| 20 | **refactor** | Plan vault restructuring from config changes. Compare config vs derivation. | Yes — dimension shift detection | Yes — detect drift → plan → execute |
| 21 | **next** | Surface most valuable next action by combining task stack, queue, inbox, health, goals. | Yes — prioritization rules | Yes — multi-source analysis → single recommendation |

---

## 2. USV Pipeline — AI Decision Points

### 2.1 Energy Detection (Primary Detection Pipeline)

**File:** `src/usv_spectrogram/detection/energy_detector.py` (756 lines)
**Config:** `src/usv_spectrogram/detection/config.py` (159 lines)

**Implicit constraint reasoning in these parameters:**

| Parameter | Value | Implicit Reasoning |
|-----------|-------|-------------------|
| `energy_threshold_db` | -60.0 | Deliberately LOW for high recall. "Threshold bias creates systematic blind spots" (Section 3.2) |
| `energy_mode` | "peak" | Max energy in band per frame — better for narrow-band USVs vs "mean" mode |
| `freq_min_hz` / `freq_max_hz` | 25,000 / 110,000 Hz | Mouse USV frequency range. Sub-ultrasonic noise rejected below 25 kHz |
| `min_duration_ms` / `max_duration_ms` | 10.0 / 500.0 ms | USVs are 10-300 ms; 500 ms margin for safety |
| `merge_gap_ms` | 3.0 ms | Merge detections < 3 ms apart into single candidates |
| `segment_continuity_max_gap_ms` | 5.0 ms | KEY PARAMETER — bridges energy dips within single USVs while splitting multi-syllable calls |
| `segment_continuity_freq_tolerance_hz` | 1500.0 Hz | Peak frequency tolerance for continuity matching |
| `segment_continuity_energy_tolerance_db` | 15.0 dB | Energy tolerance for continuity matching |
| `max_bandwidth_hz` | 20,000 Hz | Reject broadband noise — USVs have 5-15 kHz bandwidth |
| `interference_freqs_hz` | (50000, 60000, 100000, 120000) | AC power harmonics — flag but don't auto-reject |
| `interference_tolerance_hz` | 500 Hz | Flag if peak within ±500 Hz of interference |
| Bandwidth calculation | -10 dB from peak | Implicit: "within 10 dB of peak" defines "active" frequency range |
| `segment_continuity_bandwidth_hz` | 6000.0 Hz | ±6 kHz band around reference frequency for band-energy continuity |
| `segment_continuity_gap_match_fraction` | 0.6 | 60% of gap frames must match for bridging |
| `segment_continuity_kernel_size` | 3 | Odd kernel for continuity smoothing |
| Continuity weights | center=1.0, time=0.5, freq=0.2, diag=0.8 | Weighted smoothing kernel for peak tracking |

**Detection algorithm (11 steps, all with implicit decisions):**
1. Load audio
2. Compute STFT (Hann window, n_fft=512, hop=128)
3. Compute energy in USV band per frame (peak or mean mode)
4. Threshold relative to max energy in band
5. Group adjacent frames into segments
6. Merge nearby segments (< merge_gap_ms)
7. Extend/merge by continuity (peak freq + band energy matching)
8. Apply duration filters
9. Apply bandwidth filter (reject broadband noise)
10. Extract peak frequency per candidate
11. Create Candidate objects with interference flags

### 2.2 CNN Classification

**File:** `src/usv_spectrogram/models/cnn_classifier.py`
**Config:** `src/usv_spectrogram/models/config.py`

| Parameter | Value | Implicit Reasoning |
|-----------|-------|-------------------|
| `optimal_threshold` (USVClassifierCNN) | 0.05 | Very low — calibrated from full retraining. High recall at expense of precision |
| `optimal_threshold` (USVClassifierCNNLarge) | 0.40 | Higher threshold for larger model — different precision/recall trade-off |
| Architecture | 3 conv blocks [32, 64, 128] | Small CNN — binary classification of spectrogram patches |
| Global average pooling | AdaptiveAvgPool2d | Handles variable input sizes |
| `dropout_rate` | 0.5 | Standard dropout in classifier head |
| `learning_rate` | 0.001 | Standard initial LR |
| `patience` (early stopping) | 15 epochs | Patient — waits 15 epochs without improvement |
| `use_class_weights` | True | Handles imbalanced USV/noise ratio |
| `batch_size` | 16 | Small batches — spectrogram patches are large |
| `seed` | 42 | Reproducibility |

### 2.3 Spectrogram Generation

**File:** `src/usv_spectrogram/config.py`

| Parameter | Value | Implicit Reasoning |
|-----------|-------|-------------------|
| `expected_sample_rate_hz` | 250,000 | **NOTE: Inconsistent with detection's 300,000** |
| `window_length` | 2048 | ~8.2 ms at 250 kHz |
| `zero_padding_factor` | 2 | n_fft=4096 for smoother spectra |
| `hop_ms` | 0.5 ms | 125 samples at 250 kHz |
| `f_min_hz` / `f_max_hz` | 30,000 / 125,000 Hz | Wider than detection band |
| `gain_db` / `range_db` | 20.0 / 40.0 | Display parameters |
| `eps` | 1e-12 | Numerical floor for log(0) |

**Critical observation:** SpectrogramConfig uses 250 kHz expected sample rate while DetectionConfig uses 300 kHz. This is a known inconsistency documented in the codebase.

### 2.4 LMT Behavioral Integration (PETH Analysis)

**File:** `src/usv_spectrogram/lmt/event_triggered.py`

| Parameter | Value | Implicit Reasoning |
|-----------|-------|-------------------|
| `window_before_s` / `window_after_s` | 2.0 / 2.0 s | ±2 second window around behavioral events |
| `bin_size_s` | 0.1 s (100 ms) | Temporal resolution for rate histograms |
| `n_permutations` | 1000 | Circular-shift permutation test for significance |
| `min_events` | 5 | Minimum events required for PETH computation |
| `baseline_method` | "whole_recording" or "pre_event" | Baseline rate calculation strategy |
| LMT frame rate | 30 fps | Assumed in `lmt/db_loader.py` |

### 2.5 Bout Extraction

**File:** `usv_language/data/bout_extractor.py`

| Parameter | Value | Implicit Reasoning |
|-----------|-------|-------------------|
| `bout_gap_threshold_ms` | 500.0 ms | If gap between USVs > 500 ms, start new bout |
| `context_padding_ms` | 200.0 ms | Include context around bout boundaries |
| `min_bout_duration_ms` | 50.0 ms | Minimum bout length |
| `max_bout_duration_ms` | 10,000 ms | Maximum bout length (10 s) |

### 2.6 DeepSqueak/BootSnap Integration

**File:** `bootsnap_integration_plan.md` (planning document)
**File:** `src/usv_spectrogram/classification/deepsqueak_import.py`
**File:** `src/usv_spectrogram/classification/raven_export.py`

| Parameter | Value | Implicit Reasoning |
|-----------|-------|-------------------|
| Timestamp matching tolerance | 5 ms | DeepSqueak ↔ detection candidate matching |
| Frequency bounds (Raven export) | 25-125 kHz | USV frequency range for Raven Pro |
| BootSnap spectrograms | Gammatone, 128 filters, 68 kHz center | Different representation than STFT — poor cross-domain generalization expected |
| Syllable classification | 11 Scattoni types | Complex categorical; potential for STAR framework |

### 2.7 VQ-VAE / Representation Learning

**Files:** `usv_language/` directory

| Parameter | Value | Implicit Reasoning |
|-----------|-------|-------------------|
| Frequency range (VQ-VAE) | 20-120 kHz | Slightly different from detection's 25-110 kHz |
| Probing config | Linear + MLP probes, 5-fold CV | `usv_language/analysis/probing.py` |
| Context analysis | Configurable window sizes | Sequence-level representations |

### 2.8 Documentation Gaps Identified

1. **Sample rate inconsistency**: SpectrogramConfig (250 kHz) vs DetectionConfig (300 kHz) — documented but not resolved
2. **No STAR reasoning in any pipeline prompt** — all AI decision points use raw parameter injection
3. **Bandwidth calculation hardcoded** — 10 dB below peak is implicit, not configurable
4. **Continuity kernel weights undocumented** — center=1.0, time=0.5, freq=0.2, diag=0.8 rationale unclear
5. **Threshold calibration history missing** — 0.05 and 0.40 thresholds documented as "from full retraining" but calibration data not linked

---

## 3. Agent Prompt Full Text

### 3.1 dsp-reviewer.md

```markdown
---
name: dsp-reviewer
description: Reviews DSP and signal processing code for mathematical correctness
model: opus
tools:
  - Read
  - Grep
  - Glob
---

# DSP/Signal Processing Reviewer

You are a specialist in digital signal processing, particularly for audio analysis at high sample rates (250 kHz).

## Your Expertise
- STFT computation and windowing functions (Hann, Hamming, Blackman)
- FFT bin calculations and frequency resolution
- dB scaling and dynamic range
- Frequency band masking and Nyquist considerations
- Zero-padding and its effects on spectral resolution

## Review Focus
When reviewing code changes:

1. **Mathematical correctness**
   - Verify FFT size calculations
   - Check for off-by-one errors in bin indexing
   - Validate dB conversion formulas (10*log10 vs 20*log10)

2. **Frequency handling**
   - Ensure frequency bands respect Nyquist (f_max <= sample_rate/2)
   - Check frequency-to-bin and bin-to-frequency conversions
   - Verify hop size and overlap calculations

3. **Numerical stability**
   - Check for division by zero guards
   - Verify epsilon values for log operations
   - Look for potential overflow/underflow

4. **Performance considerations**
   - FFT sizes should be powers of 2 for efficiency
   - Streaming vs in-memory trade-offs

## Key Files

### Spectrogram Generation
- `src/usv_spectrogram/spectrogram.py` - In-memory STFT
- `src/usv_spectrogram/stft_stream.py` - Streaming API
- `src/usv_spectrogram/config.py` - SpectrogramConfig parameters

### Detection Pipeline (Energy-based)
- `src/usv_spectrogram/detection/energy_detector.py` - STFT and energy computation
- `src/usv_spectrogram/detection/config.py` - DetectionConfig (sample rate, n_fft, etc.)

### Reference Documentation
- `usv_signal_processing_reference.md` - Design rationale and trade-offs

## Knowledge Graph

Before reviewing, check the vault for established DSP findings that the code should respect:

1. Read `notes/signal-processing.md` topic map for prior claims about STFT parameters,
   frequency resolution, energy computation, and windowing
2. Grep `notes/` for keywords relevant to the code under review — e.g., `STFT`, `frequency
   resolution`, `energy`, `dB`, `window`, `hop`, `n_fft`, `bin`
3. Cross-check DSP parameters in the code against vault findings (e.g., notes about 586 Hz
   frequency bins, 1.7 ms temporal resolution, specific threshold values)
4. If vault notes establish baselines or constraints, verify the code honors them
5. Cite relevant vault notes in your findings when they support or contradict the code —
   only reference notes you actually read

## Output Format
Provide a concise review with:
- Issues found (with line numbers)
- Severity (critical/warning/suggestion)
- Recommended fixes
```

### 3.2 master-reviewer.md

```markdown
---
name: master-reviewer
description: Senior reviewer that checks implementations against ROADMAP spec, DECISIONS.md constraints, and established patterns. Reads the handoff first for focused context. Use after each module implementation.
tools: Read, Grep, Glob, Bash
model: sonnet
---

You are the senior technical reviewer for the USV Detection & Analysis project. Your context
is fresh — you have NOT seen the implementation happen. You review from the handoff document
and the code itself.

Your job is to find problems the implementer missed — not just code bugs, but DSP parameter
errors, data leakage, ML rigor issues, and architectural drift.

## STANDING ORDER: Fix Documentation Requirement

If your verdict is **CHANGES NEEDED**, your review MUST include this section at the end
(before the verdict line). This is non-negotiable — a CHANGES NEEDED review without this
section is incomplete:

## Fix Documentation Requirement

After applying all fixes listed above, the implementor MUST:
1. Add a "## Fixes Applied" section to this review file (`docs/reviews/<module>-review.md`)
2. For each fix: state what was changed, which file:line, and why
3. Re-run the affected tests and record pass/fail counts
4. Update `IMPLEMENTATION_PROGRESS.md` with a dated entry noting the fixes
5. Re-run master-reviewer OR self-verify against each BLOCKER/WARNING above

## When invoked, do the following:

### 1. Read the handoff FIRST (this is your primary input)
- Read `docs/reviews/<module>-handoff.md`

### 2. Understand what was supposed to be built
- Read `ROADMAP.md` — find the module's `/implement` block, test plan, and exit criteria
- Read `DECISIONS.md` — understand the ADR constraints that apply to this module
- Read `docs/architecture/patterns.md` — understand established patterns
- Read `docs/modules/*.md` for dependent modules

### 2.5 Check knowledge graph for prior decisions
- Read `notes/index.md` to identify relevant topic maps
- Read relevant topic map(s) for prior claims
- Grep `notes/` for keywords from the module name and handoff
- Note vault claims that the implementation should align with
- Reference relevant notes by title if they support or contradict observations

### 3. Understand what was actually built
- Read source files listed in handoff
- Read test file(s) listed in handoff
- Read module doc if created

### 4. Run the tests
.\.venv\Scripts\python.exe -m pytest tests/ -v --tb=short

### 5. Check for problems in these categories (in order of importance):

**DSP CORRECTNESS** — Do all STFT parameters match ADR-002? Is sample rate explicit (ADR-001)?
**ML RIGOR** — Data leakage? Test anti-greenwashing? Class balance? Reproducibility?
**SPEC COMPLIANCE** — Does implementation match ROADMAP? Files present? Tests present?
**INTEGRATION CORRECTNESS** — Follows established patterns? Frozen dataclasses? Candidate flow?
**CODE QUALITY** — Obvious bugs? Error handling? Performance?
**DOCUMENTATION** — Module doc exists? Patterns updated? Decisions recorded?

### 6. Pay special attention to "What I'm Unsure About"
The handoff flags areas for extra scrutiny.

### 7. Report findings
Organize by severity: BLOCKER → WARNING → SUGGESTION

### 8. Documentation Status table

### 9. Verdict: APPROVED or CHANGES NEEDED
```

### 3.3 pr-reviewer.md

```markdown
---
name: pr-reviewer
description: Final quality review before commit/PR
model: opus
tools:
  - Read
  - Grep
  - Glob
  - Bash
---

# PR Reviewer

You perform thorough final reviews before code is committed or merged.

## Review Checklist

### 1. Code Quality
- [ ] No obvious bugs or logic errors
- [ ] No commented-out code left behind
- [ ] No debug print statements
- [ ] Clear variable and function names
- [ ] Appropriate error handling

### 2. Style & Consistency
- [ ] Follows project conventions (see CLAUDE.md)
- [ ] Consistent with surrounding code
- [ ] No unnecessary changes to unrelated code
- [ ] Imports are organized

### 3. Testing
- [ ] New code has tests (or explanation why not)
- [ ] Existing tests still pass
- [ ] Edge cases considered

### 4. Security
- [ ] No hardcoded secrets or credentials
- [ ] No SQL injection or command injection risks
- [ ] Input validation where needed

### 5. Documentation
- [ ] Public functions have docstrings (if new/changed)
- [ ] Complex logic has brief comments
- [ ] CLAUDE.md updated if needed

## Review Process

1. Get the diff
2. Run verification
2.5. Cross-check with knowledge graph — grep notes/ for keywords from changed files
3. Check each changed file — read full context around changes

## Output Format
**Summary:** One-line assessment
**Issues Found:** [Critical/Warning/Suggestion] (file:line)
**Verdict:** APPROVE / REQUEST_CHANGES / NEEDS_DISCUSSION
```

### 3.4 detection-validator.md

```markdown
---
name: detection-validator
description: Validates USV detection algorithm changes
model: sonnet
tools:
  - Read
  - Grep
  - Glob
  - Bash
---

# Detection Algorithm Validator

You validate changes to the USV candidate detection algorithms.

## Detection Systems

### 1. Energy-Based Detection Pipeline (Primary)
- Energy thresholding with peak or mean mode
- Duration filtering (min/max)
- Bandwidth filtering for noise rejection
- Segment merging

Key Files: config.py, energy_detector.py, candidate.py, test_energy_detector.py, run_detection.py

Key Parameters: energy_threshold_db, energy_mode, max_bandwidth_hz, min/max_duration_ms, merge_gap_ms

### 2. Parameter Lab Heuristic Detection (Legacy)
- src/usv_spectrogram/param_lab/heuristic_detect.py

## Knowledge Graph
Before validating, check vault for detection findings and baselines:
1. Read notes/detection.md topic map
2. Check for baseline notes (89.7% precision, 93.8% recall at threshold 0.05)
3. Grep notes/ for parameter names being changed
4. Flag changes that contradict vault findings
5. Reference relevant vault notes in validation report

## Validation Steps
1. Run detection tests
2. Check algorithm correctness
3. Verify edge cases
4. Check config validation

## Output: Structured validation report with
- Algorithm Correctness: PASS/FAIL
- Edge Cases: HANDLED/ISSUES
- Config Validation: COMPLETE/INCOMPLETE
- Test Coverage: COMPLETE/GAPS
- Issues Found
- Recommendations
```

### 3.5 streamlit-expert.md

```markdown
---
name: streamlit-expert
description: Implements and reviews Streamlit UI with best practices
model: sonnet
tools:
  - Read
  - Grep
  - Glob
  - Edit
  - Write
---

# Streamlit UI Specialist

You are an expert in building Streamlit applications with a focus on performance and user experience.

## Your Expertise
- Streamlit session state management
- Caching strategies (@st.cache_data, @st.cache_resource)
- Layout design (columns, sidebar, expanders)
- Widget state and callbacks
- Avoiding unnecessary reruns

## Best Practices to Enforce
1. **Caching** — @st.cache_data for data, @st.cache_resource for expensive resources
2. **Session State** — Initialize in a single place, use callbacks
3. **Layout** — Group controls in expanders, sidebar for config, main for results
4. **Performance** — Minimize computation in main script body, st.spinner for long ops

## Key Files
- `src/usv_spectrogram/param_lab/app.py` - Main 650+ line Streamlit app
- `scripts/usv_parameter_lab.py` - Launcher script
```

### 3.6 test-writer.md

```markdown
---
name: test-writer
description: Generates pytest tests for new or modified code
model: sonnet
tools:
  - Read
  - Grep
  - Glob
  - Edit
  - Write
  - Bash
---

# Test Writer

You generate focused, maintainable pytest tests for Python code.

## Testing Philosophy
- Test behavior, not implementation
- One assertion per test when possible
- Clear test names that describe the scenario
- Use fixtures to reduce duplication

## Test Structure
def test_<function>_<scenario>_<expected_outcome>():
    # Arrange / Act / Assert

## Pytest Patterns: Fixtures, Parametrization, Mocking, Edge Cases

## Project Test Conventions
- Tests in `tests/` directory
- Files named `test_<module>.py`
- Run: `.\.venv\Scripts\python.exe -m pytest tests/ -v`
```

---

## 4. Command Templates

### 4.1 implement.md

```markdown
---
name: implement
description: End-to-end module implementation workflow. Uses built-in plan mode for context-efficient planning, then builds code and tests following the ROADMAP spec.
---

Implement the feature described by the user: $ARGUMENTS

Follow this sequence strictly:

## Phase 1: PLAN (Enter Plan Mode)
Call EnterPlanMode. While in plan mode (read-only):
1. Read ROADMAP.md — find the module's /implement block and test plan
2. Read DECISIONS.md — understand architectural constraints
3. Read docs/architecture/patterns.md — follow established patterns
4. Read docs/modules/*.md for dependent modules
5. Read existing code
6. Identify files to create/modify
7. Note edge cases, DSP parameters, integration points

Write plan including: New files, existing files to modify, data structures, algorithm description, DSP parameters, open questions.
Call ExitPlanMode for approval.

## Phase 2: IMPLEMENT (after approval)
Create task list. Implement in order:
1. Config (frozen dataclasses)
2. Core logic
3. Scripts (CLI entry points)
4. Tests
5. Run module tests
6. Run full test suite
7. Fix failures

DSP checks: STFT matches ADR-002, frequency ranges correct, never use librosa defaults.

## Phase 3: DOCUMENT
After tests pass: module doc, patterns.md, DECISIONS.md, handoff.

## Phase 4: REVIEW
Spawn master-reviewer with tier-appropriate prompt. Write review file. Fix blockers.

## Phase 5: REPORT
Summarize: what created, test results, review verdict, known limitations.
```

### 4.2 verify.md

```markdown
# Verify Implementation

Run appropriate verification steps for the current implementation.

## Steps
1. Syntax Check: py_compile on modified files
2. Tests: pytest on relevant test files
3. Linting: flake8 if configured
4. Output Verification: verify outputs look correct

## Output Format
VERIFICATION RESULTS — Syntax Check, Tests, Issues Found, Overall: PASS/FAIL

Focus area (optional): $ARGUMENTS
```

### 4.3 commit-push-pr.md

```markdown
# Commit, Push, and Create PR

1. git status
2. git diff --cached
3. Create commit with clear message
4. Push to current branch
5. Create PR using gh pr create

Commit format: short summary (50 chars), blank line, body.
PR format: title + bullet summary + test plan.
```

### 4.4 verify-quick.md

```markdown
# Quick Verification

1. Find modified Python files (git status)
2. Run py_compile on each
3. Run pytest if tests exist
4. Report results concisely
```

### 4.5 simplify.md

```markdown
# Simplify Code

Review and simplify recent code changes:
1. Remove Unnecessary Complexity
2. Consolidate Duplicate Logic
3. Improve Naming
4. Type Hints
5. Docstrings

Rules: Keep functionality identical, small diffs, run verification after.
```

### 4.6 run-app.md

```markdown
# Run the USV Parameter Lab App

.\.venv\Scripts\streamlit.exe run scripts/usv_parameter_lab.py

Notes: Opens in browser. USV_WAV_DIR or <repo>/5970 USV. Ctrl+C to stop.
```

### 4.7 web-handoff.md

```markdown
# Web Claude Handoff

Generate context summary for continuing in claude.ai.

Include: Project context, session summary, current state, topic for discussion.
Exclude: Exact code, file paths, internal workflow details, token concerns.

Output: ## Context for Web Claude — Project, What we worked on, Current state, Topic, Background.

Topic to explore: $ARGUMENTS
```

### 4.8 review-all.md

```markdown
---
name: review-all
description: Run comprehensive review — master review + DSP check + documentation audit.
disable-model-invocation: true
---

Run full review of module: $ARGUMENTS

Prerequisites: handoff must exist.

1. Determine Review Tier (1/2/3 from ROADMAP)
2. Spawn master-reviewer subagent with tier-appropriate prompt
3. If DSP changes: spawn dsp-reviewer
4. If detection changes: spawn detection-validator
5. Write review file
6. Report unified summary
7. Fix blockers, update Fix Log
```

### 4.9 roadmap-from-plan.md

```markdown
---
name: roadmap-from-plan
description: Convert a web Claude implementation plan into a standalone ROADMAP file with /implement blocks, then extract theoretical knowledge to the KG.
---

Convert implementation plan: $ARGUMENTS

## Step 1: Read Current State (ROADMAP, DECISIONS, patterns)
## Step 2: Analyze the Plan (phases, dependencies, files, data structures, algorithms, tests, exit criteria)
## Step 3: Generate ROADMAP Entries (self-contained /implement blocks with code snippets)
## Step 4: Assemble Phase Gate
## Step 5: Present to User (do NOT write yet)
## Step 6: Extract Theoretical Knowledge to KG

Formatting Rules:
- Phase numbering continues from last
- Review tiers: 1=config/glue, 2=standard, 3=DSP/ML/complex
- /implement blocks must be self-contained
- Code snippets inline (dataclasses, forward pass signatures, CLI args)
- Test plans specific (not just "tests pass")
- Exit criteria objectively verifiable
- DSP: reference ADR-001 (sr=300000) and ADR-002 (n_fft=512, hop=128)

Write to standalone ROADMAP_<NAME>.md (not main ROADMAP).
After writing: scan for theoretical content, suggest /reduce for KG extraction.
```

---

## 5. Knowledge Pipeline Skills (Full Text)

### 5.1 reduce (SKILL.md) — First 100 lines (full file is ~500 lines)

```markdown
---
name: reduce
description: Extract structured knowledge from source material. Comprehensive extraction is the default — every insight that serves the domain gets extracted. For domain-relevant sources, skip rate must be below 10%. Zero extraction from a domain-relevant source is a BUG.
version: "1.0"
generated_from: "arscontexta-v1.6"
user-invocable: true
allowed-tools: Read, Write, Grep, Glob, mcp__qmd__vector_search
context: fork
---

## Runtime Configuration (Step 0 — before any processing)

Read these files to configure domain-specific behavior:

1. **`ops/derivation-manifest.md`** — vocabulary mapping, extraction categories, platform hints
2. **`ops/config.yaml`** — processing depth, pipeline chaining, selectivity
3. **`ops/queue/queue.json`** — current task queue (for handoff mode)

## THE MISSION (READ THIS OR YOU WILL FAIL)

You are the extraction engine. Raw source material enters. Structured, atomic notes exit.

### The Comprehensive Extraction Principle

For domain-relevant sources, COMPREHENSIVE EXTRACTION is the default:
1. Extract ALL core notes — direct assertions
2. Extract ALL evidence and validations — confirmations ARE notes
3. Extract ALL patterns and methods — named patterns are referenceable
4. Extract ALL tensions — contradictions are wisdom
5. Extract ALL enrichments — near-duplicates add value

"We already know this" means we NEED the articulation, not that we should skip it.

### INVALID Skip Reasons (BUGS)
- "validates existing approach" — validations ARE evidence
- "already captured in system config" — config is not articulation
- "we already do this" — DOING is not EXPLAINING
- "obvious" — obvious to whom?
- "near-duplicate" — create enrichment task

### VALID Skip Reasons (rare)
- Completely off-topic
- Too vague to act on
- Pure summary with zero insight
- LITERALLY identical text

For domain-relevant sources: skip rate < 10%. Zero extraction = BUG.

[... continues with EXECUTE NOW, Extraction Categories, Quality Gates,
Selectivity Gate, Enrichment Protocol, Handoff Mode, Pipeline Chaining,
Output Format, and extensive worked examples ...]
```

**Note:** Full file is ~500 lines. Key sections beyond the excerpt:
- **Extraction Categories Table** — findings, decisions, methods, hypotheses, baselines, open-questions, patterns
- **Selectivity Gate** — calibrated by `ops/config.yaml` selectivity setting
- **Enrichment Protocol** — when existing notes need updating vs new notes
- **Handoff Mode** — structured output for /ralph queue processing
- **Output Format** — extraction table, new notes, enrichments, skipped items with justification

### 5.2 reflect (SKILL.md) — ~210 lines

```markdown
---
name: reflect
description: Find connections between notes and update MOCs. Requires semantic judgment to identify genuine relationships.
user-invocable: true
allowed-tools: Read, Write, Edit, Grep, Glob, Bash, mcp__qmd__search, mcp__qmd__vector_search, mcp__qmd__deep_search, mcp__qmd__status
context: fork
---

## Runtime Configuration (Step 0)
Read ops/derivation-manifest.md and ops/config.yaml.

Processing depth adaptation:
| deep    | Full dual discovery. Multiple passes. Synthesis opportunity detection. |
| standard | Dual discovery with top 5-10 candidates. Standard evaluation. |
| quick   | Single pass. Accept obvious connections only. |

## EXECUTE NOW

Target: $ARGUMENTS

Parse: [[note name]], --handoff, empty, "recent"/"new"

Steps:
1. Read target note fully
2. Capture Discovery Trace throughout
3. Phase 0: Verify index freshness (qmd status vs file count)
4. Dual discovery in parallel:
   - Browse relevant topic maps for related notes
   - Run semantic search for conceptually related notes
5. Evaluate each candidate: genuine connection? Articulate WHY.
6. Add inline wiki-links where connections pass articulation test
7. Update relevant topic maps
8. If task file: update reflect section
9. Report connections and reasoning
10. If --handoff: output RALPH HANDOFF block

## Philosophy
The network IS the knowledge. Individual notes less valuable than relationships.
Quality over speed. Every connection must pass articulation test.
"Related" is not a relationship. "Extends X by adding Y" IS.
Bad connections pollute the graph.

## Workflow
Phase 0: Verify Index Freshness (qmd status check)
Dual Discovery: MOC browsing + semantic search
Connection Evaluation: Articulation test
Graph Updates: Inline links + topic map updates

## Handoff Mode (--handoff flag)
Structured output for /ralph: Work Done, Files Modified, Learnings, Queue Updates
```

### 5.3 reweave (SKILL.md)

```markdown
---
name: reweave
description: Update old notes with new connections. The backward pass that /reflect doesn't do.
user-invocable: true
allowed-tools: Read, Write, Edit, Grep, Glob, Bash, mcp__qmd__search, mcp__qmd__vector_search, mcp__qmd__deep_search, mcp__qmd__status
context: fork
---

[Full reweave SKILL.md content — the backward pass phase that updates older notes
with new connections found after they were created. Revisits existing notes that
predate newer related content, adds connections, sharpens claims, considers splits.]
```

### 5.4 verify (SKILL.md)

```markdown
---
name: verify
description: Combined verification — recite (description quality via cold-read prediction) + validate (schema compliance) + review (health checks).
user-invocable: true
allowed-tools: Read, Write, Edit, Grep, Glob, mcp__qmd__vector_search
context: fork
---

[Full verify SKILL.md content — performs three quality checks:
1. Recite: cold-read prediction test (can you predict content from description alone?)
2. Validate: schema compliance (required fields, enum values, description quality)
3. Review: health checks (link density, orphan status, stale indicators)]
```

### 5.5 pipeline (SKILL.md)

```markdown
---
name: pipeline
description: End-to-end source processing -- seed, reduce, process all claims through reflect/reweave/verify, archive.
version: "1.0"
generated_from: "arscontexta-v1.6"
user-invocable: true
context: fork
model: sonnet
allowed-tools: Read, Write, Edit, Grep, Glob, Bash, Task
argument-hint: "[file] — path to source file to process end-to-end"
---

[Full pipeline SKILL.md content — orchestrates the complete processing pipeline:
1. Seed: Queue the source
2. Reduce: Extract atomic notes
3. Reflect: Forward connections for each new note
4. Reweave: Backward connections for affected old notes
5. Verify: Quality gate on each new note
6. Archive: Move processed source to archive]
```

---

## 6. ops/ Directory

### 6.1 ops/goals.md

```markdown
---
description: Current active threads and what the agent is working on
type: moc
---

# goals

## Active Threads
- **DeepSqueak Classification Bridge** -- Phase 2 (Raven export) DONE, Phase 3 (MATLAB import+clustering) IN PROGRESS. Resume: open MATLAB → DeepSqueak → Import from Raven (5 files in raven_tables/). See PROJECTS.md Section 6 for full steps.
- Phase 5.2 -- Two-week validation checkpoint (starts 2026-03-06, after 2 weeks of active use)

## Waiting
- CC weekly routine first execution -- deferred to a session in D:\we_do_this\tevel-erp

## Completed
- Phase 11.1 -- Bout Extraction & Preprocessing on Real Data (2026-02-22)
- Phase 10.1 -- Active Learning Cycle Runner (2026-02-21)
- Phase 9.1 -- Dataset Assembler (2026-02-21)
- Phase 8.4 -- Analysis & Interpretation Tools (2026-02-21)
- Phase 8.3 -- Hidden State VQ-VAE (2026-02-20)
- Phase 1.1 -- arscontexta plugin installed (2026-02-18)
- Phase 1.3 -- USV Research skill graph setup (2026-02-18)
- Phase 3.1 -- Migrate USV Architecture & Experiment Docs (2026-02-19)
- Phase 3.2 -- USV Research Implicit Knowledge Dump (2026-02-19)
- Phase 1.2 -- Cloudy Claude skill graph setup (2026-02-19)
- Skill testing & refinement (2026-02-19)
- Phase 3.3 -- Biological-context topic map deferred (2026-02-19)
- Classification topic map split (2026-02-19)
- Phase 4.3 -- Integrate reviewer agents with skill graph (2026-02-19)
- Phase 5.1 -- Weekly maintenance routine established (2026-02-20)
```

### 6.2 ops/reminders.md

```markdown
# Reminders

- [ ] 2026-02-27: Weekly maintenance routine (first scheduled run) -- /arscontexta:health, /reflect, /reweave, /stats
- [ ] 2026-03-06: Phase 5.2 two-week validation checkpoint -- score 5 criteria, course correction decision
- [ ] 2026-02-27: Fix 13 dangling source links in VQ-VAE notes (convert [[learn-vqvae-bioacoustics-state-of-art-2026-02]] to plain text)
```

### 6.3 ops/config.yaml

```yaml
# ops/config.yaml -- edit these to adjust your system
# See ops/derivation.md for WHY each choice was made

dimensions:
  granularity: atomic
  organization: flat
  linking: explicit+implicit
  processing: heavy
  navigation: 3-tier
  maintenance: condition-based
  schema: moderate
  automation: full

features:
  semantic-search: true
  processing-pipeline: true
  sleep-processing: false
  voice-capture: false

processing_tier: auto

processing:
  depth: standard
  chaining: suggested
  extraction:
    selectivity: moderate
    categories:
      - findings
      - decisions
      - methods
      - hypotheses
      - baselines
      - open-questions
      - patterns
  verification:
    description_test: true
    schema_check: true
    link_check: true
  reweave:
    scope: related
    frequency: after_create

provenance: full

personality:
  enabled: false

research:
  primary: web-search
  fallback: web-search
  last_resort: web-search
  default_depth: moderate
```

### 6.4 Other ops/ files present

- `ops/derivation.md` — Original derivation record (design intent baseline)
- `ops/derivation-manifest.md` — Vocabulary mapping, extraction categories, platform hints
- `ops/tasks.md` — Task stack
- `ops/goals-archive.md` — Archived goals
- `ops/methodology/` — Learned patterns and methodology notes
- `ops/observations/` — Friction signals and patterns
- `ops/tensions/` — Contradictions between notes
- `ops/queue/` — Processing pipeline queue state
- `ops/sessions/` — Session logs
- `ops/health/` — Health report history
- `ops/scripts/` — Helper scripts (rename-note.sh, orphan-notes.sh, etc.)

---

## 7. STAR Optimization Opportunities Summary

Based on this audit, the following areas have the highest potential for STAR reasoning framework optimization:

### 7.1 USV Pipeline (Implicit Constraint Reasoning)

1. **Energy detection thresholding** — The -60 dB threshold involves reasoning about recall vs precision trade-offs. A STAR prompt could make Claude articulate: Situation (recording noise characteristics), Task (maximize USV recall), Action (threshold selection with justification), Result (expected detection rate).

2. **Continuity parameter tuning** — The 5 ms gap threshold bridges energy dips within single USVs vs splitting multi-syllable calls. This is a domain-specific reasoning task that STAR could scaffold.

3. **CNN threshold calibration** — 0.05 vs 0.40 thresholds represent different operating points on the precision-recall curve. STAR could structure the reasoning about which threshold to use for which deployment scenario.

4. **Bandwidth filter decisions** — The 20 kHz max bandwidth and -10 dB "active frequency" definition involve implicit signal processing reasoning.

### 7.2 Knowledge Pipeline (Multi-step Reasoning)

5. **reduce extraction** — Already has extensive constraint reasoning but uses imperative instructions rather than STAR. The "extraction question" could be reformulated as STAR.

6. **reflect connection evaluation** — The articulation test ("can you say WHY these notes connect?") is implicit constraint reasoning that STAR could make more systematic.

7. **architect recommendations** — The 7-phase workflow already structures reasoning but could benefit from explicit STAR framing for each recommendation.

### 7.3 Reviewer Agents

8. **DSP reviewer** — Mathematical correctness checks involve implicit reasoning about frequency resolution, Nyquist constraints, dB scaling. STAR could scaffold the verification chain.

9. **Master reviewer** — The 5-category review hierarchy already structures reasoning but could use STAR for each finding to ensure complete situation-task-action-result documentation.

### 7.4 Parts Finder AI Fallback

10. **Vehicle parts specification** — The fallback prompt "You are a vehicle parts specification expert" could be enhanced with STAR to make Claude's reasoning about spec inference more systematic.

---

*Audit complete. All content pasted verbatim from source files. No files were modified.*

**Agents:** Explore (USV pipeline AI decision points — from previous session)
