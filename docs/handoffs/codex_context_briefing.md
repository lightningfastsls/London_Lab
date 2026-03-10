# Context Briefing for Codex: USV Research Pipeline

This document gives you (Codex) everything you need to set up a collaboration workflow with Claude Code in this repository. Read it fully before planning.

---

## 1. What This Repo Is

A Python toolkit for analyzing ultrasonic vocalizations (USVs) from mouse recordings at 300 kHz sample rate. It includes:

- **Spectrogram generation** from WAV files (STFT-based)
- **Energy-based USV detection** pipeline
- **CNN classifier** for USV vs noise
- **PyQt6 desktop app** for interactive detection review
- **Streamlit tools**: Parameter Lab, Labeling tool, Noise Review tool
- **VQ-VAE + Transformer** for learning USV compositional structure
- **Clustering/classification** pipeline (Raven export, DeepSqueak bridge)
- **Dataset assembly** for training data preparation

Environment: Windows 11, Python 3.12.1, venv at `.venv/`.

Run Python: `.venv\Scripts\python.exe <script>`
Run tests: `.venv\Scripts\python.exe -m pytest tests/ -v`
Compile check: `.venv\Scripts\python.exe -m py_compile <file>`

---

## 2. Project Structure

```
src/usv_spectrogram/          # Core USV library
  config.py                   # SpectrogramConfig dataclass
  io_wav.py                   # WAV loading (always specify sr=300000)
  spectrogram.py              # STFT computation
  _stft_core.py               # Low-level STFT (extract_frames, compute_stft_frames_db)
  stft_stream.py              # Streaming STFT for long files
  render_tiles.py             # Tiled PNG rendering
  storage_zarr.py             # Zarr incremental storage
  detection/                  # Energy-based USV candidate detection
    energy_detector.py        # EnergyDetector class
    candidate.py              # Candidate dataclass
    config.py                 # DetectionConfig
    extraction_config.py      # ExtractionConfig
    spectrogram_extractor.py  # Extract candidate spectrograms
  models/                     # CNN classifier
    cnn_classifier.py         # USVClassifier architecture
    trainer.py                # Training loop
    data_loader.py            # Dataset/DataLoader
    evaluate.py               # Evaluation metrics
    config.py                 # TrainingConfig
  dataset/                    # Training data assembly
    assembler.py              # DatasetAssembler pipeline
    splits.py                 # Train/val/test splitting
  app/                        # PyQt6 detection desktop app
    main_window.py            # Main application window
    core/                     # Detection logic, audio, inference (no Qt imports)
    widgets/                  # SpectrogramView, ProbabilityView (Qt rendering only)
  clustering/                 # USV repertoire clustering
  classification/             # Raven export, DeepSqueak import, repertoire stats
  lmt/                        # Live Mouse Tracker integration
  param_lab/                  # Streamlit parameter lab
  labeling/                   # Streamlit labeling + noise review tools

usv_language/                 # Transformer + VQ-VAE for USV compositional structure
  models/                     # transformer.py, vqvae.py
  data/                       # Bout extraction, normalization, spectrogram prep
  training/                   # train_vqvae.py, train_transformer.py
  analysis/                   # Codebook viz, sequence analysis, compositionality

scripts/                      # ~76 entry points (see docs/scripts-index.md)
tests/                        # pytest test suite (~430+ tests)
```

---

## 3. Architecture Patterns You Must Follow

### 3.1 Config Dataclass Pattern
All configurable modules use `@dataclass(frozen=True)` with defaults, `__post_init__` validation, and unit-suffixed field names (`_hz`, `_ms`, `_db`, `_px`, `_s`).

### 3.2 Candidate Data Flow
```
WAV -> load_wav_mono() -> numpy array
  -> EnergyDetector.detect() -> list[Candidate]
  -> SpectrogramExtractor -> PNG patches
  -> CNN.predict_proba() -> probability [0,1]
  -> LabelStorage.save() -> JSON
```

### 3.3 Test Fixtures
- Synthetic WAV data only (never real recordings in tests)
- `yield` fixtures with cleanup
- Factory fixtures for parameterized creation
- See `tests/conftest.py` for examples

### 3.4 Script CLI Pattern
- Path bootstrap: `REPO_ROOT = Path(__file__).resolve().parents[1]`
- Separate `parse_args()` function
- Return exit codes (0 success, 1 error)
- Usage examples in epilog

### 3.5 PyQt6: Model-View Separation
- `core/` = business logic (no Qt imports)
- `widgets/` = rendering (no business logic)
- `main_window.py` = orchestration connecting them via `pyqtSignal`

### 3.6 STFT Core
- Shared functions in `_stft_core.py`
- `np.fft.rfft` for real signals
- Output shape: `(n_freq_bins, n_frames)` — frequency on rows
- `band_mask` for frequency filtering

### 3.7 Import Bootstrap
All scripts add `src/` to `sys.path` before project imports:
```python
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
```

Full details: `docs/architecture/patterns.md`

---

## 4. Hard Constraints (Non-Negotiable)

### 4.1 Sample Rate
**Always specify `sr=300000` explicitly.** Never rely on library defaults. Mouse USVs go up to ~120 kHz, requiring 300 kHz for Nyquist headroom.

### 4.2 STFT Parameters
- `n_fft=512` gives ~586 Hz frequency resolution at 300 kHz
- `hop_length=128` gives 75% overlap, ~0.43 ms hop
- These are validated design decisions (see ADR-001, ADR-002 in DECISIONS.md)

### 4.3 Test Protocol (Anti-Greenwashing)
**NEVER modify test expected values to make tests pass.** If tests fail:
- If code is buggy → fix the code
- If test expectations are wrong → DISCUSS with the user first
- If unclear which is wrong → STOP, do not guess

### 4.4 Git Safety
- **Never use `git add -A` or `git add .`** without reviewing status first
- Stage specific files by name
- Check `git diff --cached --stat` before commits — hundreds of deletions = red flag

### 4.5 Validation Before Completion
Every change must be validated:
1. `py_compile` on changed files
2. `pytest` on relevant tests
3. Never claim "done" without running both

---

## 5. The Collaboration Model

### 5.1 Two-Tool Division
- **Claude Code** = orchestrator + memory + governance (owns knowledge graph, project state, architectural decisions)
- **Codex** = focused implementer (owns code, tests, scripts, debugging)

### 5.2 Ownership Boundaries

**Claude Code owns (READ-ONLY for Codex):**
- `.claude/` — skills, hooks, settings
- `ops/` — goals, reminders, tasks, queue, methodology
- `notes/` — knowledge graph (~500 atomic notes)
- `methodology/` — 249 arscontexta research claims
- `reference/` — structured reference docs
- `templates/` — note/topic-map templates
- `inbox/` — processing queue

**Codex owns (free to modify):**
- `src/` — all source code
- `tests/` — all tests
- `scripts/` — all entry-point scripts
- `usv_language/` — transformer/VQ-VAE code
- `docs/handoffs/` — handoff files for Claude Code

**Shared (modify with care):**
- `docs/` (except `docs/handoffs/` which is Codex's)
- Root config files

### 5.3 Handoff Protocol

After completing a task, write a handoff file to `docs/handoffs/` with this structure:

```markdown
# Handoff: [Task Title]
Date: YYYY-MM-DD

## Task
What was requested.

## Files Changed
- `path/to/file.py` — what changed and why

## Reasoning
Why this approach was chosen over alternatives.

## Validation
- py_compile: PASS/FAIL
- pytest results: X passed, Y failed
- Manual testing notes if applicable

## Known Risks
- Any edge cases, assumptions, or potential issues

## Worth Remembering
- Architectural decisions, patterns discovered, gotchas
- Suggested vault destinations (which topic map, what type of note)
```

### 5.4 Branch Strategy
- Codex works on feature branches (e.g., `codex/feature-name`)
- Claude Code reviews and merges to `main`
- Never force-push to `main`

---

## 6. What Codex Should Build

You are being asked to create three files for this collaboration:

### 6.1 `AGENTS.md` (repo root)
Codex's behavioral contract. Should include:
- The ownership boundaries from Section 5.2
- The hard constraints from Section 4
- The handoff protocol from Section 5.3
- The architecture patterns summary from Section 3
- Environment setup instructions
- What Codex should NOT do (modify vault, skip validation, change test expectations)

### 6.2 `docs/codex_index.md`
A compact map so Codex doesn't need to traverse the full repo every time. Should include:
- Module → file path → one-line purpose → relevant docs
- Active ADRs and their implications
- Key scripts and what they do (reference `docs/scripts-index.md`)
- Test locations and how to run them
- Current project status and active work areas

### 6.3 `docs/handoffs/` directory
Already created. Just needs a `README.md` explaining the handoff format and conventions.

---

## 7. Key Reference Documents

| Document | Purpose |
|----------|---------|
| `CLAUDE.md` | Claude Code's behavioral contract (do NOT modify) |
| `docs/architecture/patterns.md` | Established code patterns (follow these) |
| `docs/scripts-index.md` | All ~76 scripts categorized |
| `docs/modules/*.md` | Module-level documentation |
| `docs/plans/*.md` | Implementation plans |
| `docs/human/PROJECTS.md` | Human-readable project dashboard |
| `docs/human/DECISIONS.md` | Architecture Decision Records |
| `IMPLEMENTATION_PROGRESS.md` | Session archive (append-only) |

---

## 8. Current Project State

Active work areas (as of 2026-03-06):
- **DeepSqueak Classification Bridge** — Phase 3 (MATLAB import+clustering) in progress
- **Knowledge graph maintenance** — ongoing vault health, topic map management
- **Two-week validation checkpoint** — arscontexta system validation

The test suite has ~430+ tests. One known flaky test: `test_long_continuous_tone_rejected` in `test_energy_detector.py`.

---

## 9. What To Do Now

1. Read this document fully
2. Read `docs/architecture/patterns.md` for code pattern details
3. Read `docs/scripts-index.md` for the script landscape
4. Optionally skim `docs/human/PROJECTS.md` for project context
5. Create the three files described in Section 6
6. Write a handoff to `docs/handoffs/` describing what you built
