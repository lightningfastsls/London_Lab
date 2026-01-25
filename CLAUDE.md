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

### Proactive Agent Usage Strategy

**BEFORE implementing signal processing:**
1. Run `dsp-reviewer` on reference code or similar existing code
2. Learn patterns, conventions, edge case handling
3. Implement following those patterns
4. Run `dsp-reviewer` again on your implementation

**BEFORE implementing UI:**
1. Run `streamlit-expert` on existing param_lab code
2. Learn UI patterns, state management, layout conventions
3. Implement following those patterns

**AFTER implementing (always):**
- `detection-validator` for detection logic changes
- `dsp-reviewer` for signal processing changes
- `pr-reviewer` before marking task complete

This "review → learn → implement → review" cycle prevents issues rather than fixing them after, saving tokens and time.

---

## Key Reference Documents

| Document | When to Read |
|----------|--------------|
| `IMPLEMENTATION_PROGRESS.md` | **Start of every session** |
| `USV_DETECTION_IMPLEMENTATION_PLAN.md` | Working on detection pipeline |
| `usv_signal_processing_reference.md` | Any signal processing work |

---

## Token Usage Optimization

### Proactive Strategies (Prevent Token Waste)

1. **Use specialized agents to learn first**
   - Before implementing signal processing, run `dsp-reviewer` on reference code
   - Learn patterns once, implement correctly first time
   - Avoids: implement (15K tokens) → debug (10K) → fix (8K) = 33K
   - Cost: review (5K) → implement correctly (15K) = 20K
   - **Savings: 40% reduction**

2. **Use Task tool with model="haiku" for exploration**
   - Codebase searches, file discoveries, pattern finding
   - 10x cheaper than sonnet for these tasks
   - Reserve sonnet for implementation and review

3. **Don't re-read files already in context**
   - Check conversation history before using Read tool
   - Exception: If file changed since last read

4. **Read targeted, not speculatively**
   - Use Grep to find specific patterns, then Read only matches
   - Don't "read 5 files to see which is relevant"

### When to Suggest New Session

Suggest starting fresh conversation when:
- Completing major feature or phase (clean break point)
- After extensive exploration (20+ file reads)
- Switching to unrelated work area
- Conversation >50 exchanges (context getting large)

**How to suggest:** "We've completed [X]. This is a good time to start a new session for [Y] to optimize token usage."

---

## DUAL-AI WORKFLOW: Claude Code + Codex

This project uses manual Codex handoff to save tokens and extend session length.

### Delegation Decision Tree (Claude Code Uses This)

When user requests a task, evaluate:

**✅ PROACTIVELY SUGGEST CODEX for:**
1. **Writing tests for existing code**
   - Function already implemented and tested manually
   - Just need pytest test cases
   - Example: "Write tests for energy_detector.py"

2. **Adding documentation**
   - Docstrings for existing functions
   - README updates
   - Type annotations

3. **Boilerplate & scaffolding**
   - `__init__.py` files
   - Config file templates
   - Basic class structures

4. **Repetitive edits**
   - Same change across multiple files
   - Import reorganization
   - Variable renaming project-wide

5. **Long-running background work**
   - Model training (>30 min)
   - Dataset generation
   - Batch processing

**❌ CLAUDE CODE HANDLES (don't delegate):**
- Architecture & design decisions
- Complex algorithm implementation
- Debugging unexpected behavior
- Performance optimization (requires profiling judgment)
- Signal processing (requires domain knowledge)
- Any task requiring real-time oversight

### How to Suggest Delegation

When task matches criteria above, say:
> "This is a good candidate for Codex to save tokens. I can generate a detailed spec using `/codex-task <description>`, or I can handle it now if you prefer. Which would you like?"

**Be proactive:** Don't wait for user to remember - identify opportunities and suggest them.

### Generating Specifications

When user says "use Codex" or runs `/codex-task <description>`:
1. Generate detailed specification (see AGENTS.md format)
2. Include: exact file paths, mathematical formulas, test cases, edge cases
3. Provide copy-paste ready spec for user to hand to Codex
4. Make specs detailed enough that Codex succeeds first try

### Token Savings Math

Example - Writing 20 tests:
- Claude Code: ~12K tokens
- Codex spec: ~3K tokens (Claude generates spec)
- User copies to Codex: ~0 tokens (runs locally/different API)
- **Savings: 9K tokens (75% reduction)**

For 5 such tasks per session: **45K tokens saved** = multiple extra hours of session time

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

