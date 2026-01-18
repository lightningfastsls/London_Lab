# CLAUDE.md

## ⛔ STOP - READ BEFORE DOING ANYTHING

**TWO MANDATORY RULES - NO EXCEPTIONS:**

1. **PLAN MODE FIRST** - For ANY task that adds features, modifies logic, or touches multiple files:
   → Use `EnterPlanMode` tool BEFORE writing any code

2. **END EVERY RESPONSE** with:
   ```
   **Agents:** [list agents used, or "None"]
   ```

Failure to follow these rules wastes user time and requires rework.

---

## Project Overview

USV Spectrogram Generator - Python tools for analyzing ultrasonic vocalization (USV) recordings at 250 kHz. Includes spectrogram generation, tiled PNG rendering, Zarr storage, USV detection pipeline, Streamlit-based Parameter Lab, and candidate labeling tool.

## Environment Setup

```powershell
.\.venv\Scripts\python.exe <script>
.\.venv\Scripts\python.exe -m pytest tests/ -v
.\.venv\Scripts\python.exe -m py_compile <file.py>
```

WAV files: `$env:USV_WAV_DIR` or fallback `<repo>/5970 USV`

## Project Structure

```
src/usv_spectrogram/       # Core library
  config.py                # SpectrogramConfig dataclass
  io_wav.py                # WAV loading utilities
  spectrogram.py           # STFT computation
  detection/               # USV detection pipeline
    config.py              # DetectionConfig dataclass
    candidate.py           # Candidate dataclass
    energy_detector.py     # EnergyDetector class
  param_lab/               # Streamlit app modules
  labeling/                # USV labeling tool
    labeling_app.py        # Streamlit labeling UI

scripts/                   # Entry points
  usv_labeling_tool.py     # Labeling tool launcher
tests/                     # Test files
```

## Code Style Rules

- Keep diffs small and focused
- No verbose comments - prefer self-documenting code
- Run py_compile after every edit
- Don't add dependencies without asking first

---

## Mandatory Session Workflow

### 0. Plan Mode First (REQUIRED for non-trivial tasks)
**Before writing any code**, enter plan mode for tasks that:
- Add new features or functionality
- Modify existing algorithms or logic
- Touch multiple files
- Have unclear requirements

### 1. Before Implementation
- Read `IMPLEMENTATION_PROGRESS.md` to understand current state
- If working on detection pipeline, read `USV_DETECTION_IMPLEMENTATION_PLAN.md`

### 2. During Implementation
- Keep diffs small and focused
- Run `py_compile` after every edit
- Use subagents for their specialties (see table below) - this is NOT optional

### 3. After Implementation
- Update `IMPLEMENTATION_PROGRESS.md` with what was changed
- Run tests to verify no regressions

### 4. Before Considering Done
- Run `detection-validator` for any detection algorithm changes
- Run `dsp-reviewer` for any STFT/signal processing changes
- Run `pr-reviewer` for final quality check

---

## Model Selection Guide

| Task Type | Model | Rationale |
|-----------|-------|-----------|
| Planning & Architecture | `opus` | Complex reasoning, design decisions |
| Algorithm Implementation | `sonnet` | Good balance of capability and speed |
| Code Reviews | `sonnet` | Thorough analysis needed |
| Documentation Writing | `haiku` | Fast, straightforward task |
| Simple Edits/Fixes | `haiku` | Quick, low complexity |
| Codebase Exploration | `haiku` | Fast searches, no complex reasoning |

---

## Project-Specific Agents

| Task | Agent | When to Use |
|------|-------|-------------|
| Review STFT/DSP/math changes | `dsp-reviewer` | ANY change to energy computation, FFT, dB scaling |
| Implement Streamlit UI | `streamlit-expert` | ANY Streamlit UI work |
| Write tests for code | `test-writer` | After implementing new features |
| Validate detection changes | `detection-validator` | ANY change to detection logic |
| Final review before commit | `pr-reviewer` | Before telling user "done" |

**These are not suggestions - they are requirements for this project.**

---

## Key Reference Documents

| Document | When to Read |
|----------|--------------|
| `IMPLEMENTATION_PROGRESS.md` | **Start of every session** |
| `USV_DETECTION_IMPLEMENTATION_PLAN.md` | Working on detection pipeline |
| `usv_signal_processing_reference.md` | Any signal processing work |

---

## Token Usage Optimization

**To reduce context/cost:**
- Don't re-read files already in conversation context
- Use `haiku` for exploration and simple tasks
- Reference docs above should only be read when needed for the task

**When to suggest starting a new chat:**
- After completing a major feature or phase
- After extensive file exploration (many reads)
- When switching to unrelated work
- If conversation becomes very long (50+ exchanges)

---

## DUAL-AI WORKFLOW: Claude Code + Codex

This project uses two AI assistants to balance quality and token usage.

### Tasks for Claude Code (THIS AI)

Handle these tasks - they require deep reasoning:

1. **Architecture & Design**
   - Designing class hierarchies and module structure
   - Data flow decisions
   - API design

2. **Complex Algorithm Implementation**
   - Energy detector with all signal processing nuances
   - Spectrogram extraction with STFT parameter handling
   - Detection threshold tuning
   - Stratified dataset splitting

3. **Debugging & Problem Solving**
   - When tests fail unexpectedly
   - Performance issues
   - Integration problems

4. **Refactoring**
   - Restructuring code while preserving functionality
   - Simplifying complex functions
   - Improving code organization

5. **Code Review**
   - Reviewing implementations for correctness
   - Checking alignment with reference documents
   - Verifying signal processing decisions

### Tasks to DEFER to Codex

When the user mentions these tasks, remind them these are good candidates for Codex:

1. **Writing Tests**
   - Unit tests for existing functions
   - Test fixtures and mocks
   - Parameterized test cases

2. **Documentation**
   - Adding docstrings to functions
   - Updating README files
   - Writing usage examples

3. **Type Hints**
   - Adding type annotations
   - Creating type stubs

4. **Boilerplate & Scaffolding**
   - `__init__.py` files
   - Basic class structures
   - Config file templates

5. **Simple Utilities**
   - File I/O helpers
   - Path manipulation
   - CSV reading/writing
   - Basic data validation

6. **Repetitive Edits**
   - Same change across multiple files
   - Renaming variables project-wide
   - Import reorganization

### Workflow Reminder

When starting a task, briefly consider:
- Is this a "reasoning" task? → Claude Code handles it
- Is this a "mechanical" task? → Suggest deferring to Codex

Example response when user asks for tests:
> "Writing tests for existing functions is a good candidate for Codex to save tokens. Would you like me to:
> (A) Handle it anyway since I'm already here, or
> (B) You can ask Codex to write tests for [function_name] with these requirements: [brief spec]"

---

## Signal Processing Conventions

- Sample rate: 250,000 Hz (unless specified otherwise)
- Default n_fft: 512
- Default hop_length: 128
- Frequency range: 25-110 kHz
- Minimum USV duration: 10 ms
- Maximum USV duration: 500 ms

---

## Common Mistakes to Avoid

- Don't use librosa's default sample rate - always specify sr=250000
- Don't forget to handle edge cases for short audio segments
- Always verify FFT parameters match expected frequency resolution

