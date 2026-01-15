# CLAUDE.md

This file is read by Claude Code at the start of every session.

## Project Overview

USV Spectrogram Generator - Python tools for analyzing ultrasonic vocalization (USV) recordings at 250 kHz. Includes spectrogram generation, tiled PNG rendering, Zarr storage, and a Streamlit-based Parameter Lab.

## Environment Setup

```powershell
# Activate virtual environment (required for all commands)
.\.venv\Scripts\Activate.ps1

# Or use the venv Python directly
.\.venv\Scripts\python.exe <script>
```

## Data Location

WAV files location is determined by:
1. `$env:USV_WAV_DIR` environment variable (preferred)
2. Fallback: `<repo>/5970 USV`

## Commands to Run

### Verification (run after every change)
```powershell
# Syntax check changed files
.\.venv\Scripts\python.exe -m py_compile <file.py>

# Run tests
.\.venv\Scripts\python.exe -m pytest tests/ -v

# Run the Streamlit app
.\.venv\Scripts\streamlit.exe run scripts/usv_parameter_lab.py
```

### Quick sanity check
```powershell
.\.venv\Scripts\python.exe -m py_compile src/usv_spectrogram/*.py src/usv_spectrogram/**/*.py scripts/*.py
```

## Code Style Rules

- Keep diffs small and focused
- Add docstrings only for NEW or behavior-changed public functions
- No verbose comments - prefer self-documenting code
- Run py_compile after every edit to catch syntax errors immediately

## Project Structure

```
src/usv_spectrogram/       # Core library
  config.py                # SpectrogramConfig dataclass
  io_wav.py                # WAV loading utilities
  spectrogram.py           # STFT computation
  stft_stream.py           # Streaming API
  storage_zarr.py          # Zarr storage
  render_tiles.py          # PNG rendering
  param_lab/               # Streamlit app modules
    app.py                 # Main Streamlit UI
    heuristic_detect.py    # USV detection
    sweep.py               # Parameter sweeps

scripts/                   # Entry points
  make_spectrogram.py      # CLI spectrogram generator
  usv_parameter_lab.py     # Streamlit launcher

tasks/                     # Task handoff folders
tools/                     # Helper scripts
tests/                     # Test files
```

## Task Workflow

Tasks live in `tasks/YYYY-MM-DD_slug/` with three files:
- `00_task_brief.md` - Goal, scope, acceptance criteria, staged plan
- `10_impl_notes.md` - Implementation decisions and file changes
- `20_verification.md` - Test/check transcript

Create new tasks with: `python tools/new_task.py "Task Title"`

## Common Mistakes to Avoid

- Don't forget to activate .venv or use .venv\Scripts\python.exe
- Don't use hardcoded user paths - use USV_WAV_DIR env var
- Don't make large changes without verification between steps
- Don't skip py_compile checks after edits
- Don't add dependencies without asking first

## Git Practices

- Commit messages: short summary line, then details if needed
- Keep commits focused on one change
- Run verification before committing

## When to Use Subagents

Delegate to specialized agents for these tasks:

| Task | Agent | Invoke |
|------|-------|--------|
| Review STFT/DSP/math changes | dsp-reviewer | `@dsp-reviewer` |
| Implement Streamlit UI features | streamlit-expert | `@streamlit-expert` |
| Write tests for code | test-writer | `@test-writer` |
| Validate detection algorithm changes | detection-validator | `@detection-validator` |
| Final review before commit/PR | pr-reviewer | `@pr-reviewer` |

Use agents proactively when the task matches their specialty.
