# Testing Infrastructure Audit

*Gathered 2026-03-28 — raw context for test-writing agent team design.*

**Do NOT design the testing agent — this document gathers and organizes the raw context.**

---

## 1. How Specs Define Tests

### 1.1 The `roadmap-from-plan` Skill

**Location:** `.claude/skills/roadmap-from-plan-workspace/skill-for-eval/SKILL.md`

This skill converts implementation plans into structured ROADMAP files. Each module entry includes a **Test plan** and **Exit criteria** section. The skill explicitly requires:

- Each test case describes WHAT is being verified (not just "tests pass")
- Exit criteria must be objectively verifiable (shape checks, loss thresholds, metric targets)
- Assumptions marked with `[ASSUMED]`
- Review tier assignment (1-3) determines scrutiny depth

**Key excerpt — formatting rules (lines 165-168):**
```
5. **Test plans**: Be specific. "Tests pass" is not a test plan. Each test describes WHAT is being verified.
6. **Exit criteria**: Objectively verifiable. Include shape checks, loss thresholds, metric targets.
7. **Assumptions**: Mark with `[ASSUMED]` — every assumption visible, not hidden.
```

**Test plan format template (lines 87-98):**
```
**Test plan:**
    ```
    1. [Specific test case — WHAT is being verified]
    2. [Another test case]
    ```

**Exit criteria:**
- [ ] [Specific, verifiable criterion]
- [ ] All tests pass
- [ ] py_compile passes on all new files
```

### 1.2 Example Roadmap Test Plans (from actual ROADMAPs)

**ROADMAP_POST_PROCESSING.md — Hysteresis Detection (Phase 15.1):**
```
Test plan:
1. Single sustained peak → 1 event
2. Two peaks separated by >gap_fill → 2 events
3. Two peaks separated by <=gap_fill → 1 merged event
4. All-noise input → 0 events
5. Short burst below min_duration → filtered out
6. Shoulder extension adds correct sustain-threshold windows
7. Peak probability and mean probability computed correctly
8. Config validation rejects invalid parameters
9. Time mapping from window indices to seconds
10. ADR-010 format conversion produces valid detection dicts

Exit criteria:
- [ ] All tests pass
- [ ] py_compile passes
- [ ] Running on 5 test WAV recordings produces sensible event counts
```

**ROADMAP_POST_PROCESSING.md — Temperature Scaling Calibration (Phase 15.3):**
```
Test plan:
1. fit() learns T ≈ 1.0 on already-calibrated data
2. calibrate() is identity when T = 1.0
3. T > 1 widens sigmoid (more uncertain), T < 1 sharpens
4. ECE decreases after calibration on synthetic miscalibrated data
5. Serialization round-trip preserves T value
6. SlidingInference uses calibration when model has calibration metadata
7. InferenceResult.logits has correct shape when return_logits=True

Exit criteria:
- [ ] Fitted T is in reasonable range (0.5-3.0)
- [ ] Calibrated probabilities have lower ECE than raw probabilities
- [ ] SlidingInference backward compatible
```

### 1.3 Other Skills/Agents That Touch Test Generation

| Component | Role | Location |
|-----------|------|----------|
| **test-writer** agent | Generates pytest tests for new/modified code | `.claude/agents/test-writer.md` |
| **master-reviewer** agent | Reviews tests against ROADMAP spec, flags coverage gaps | `.claude/agents/master-reviewer.md` |
| **detection-validator** agent | Validates detection algorithm changes, runs detection tests | `.claude/agents/detection-validator.md` |
| **dsp-reviewer** agent | Reviews DSP/math correctness (checks test math too) | `.claude/agents/dsp-reviewer.md` |
| **implement** skill | End-to-end module implementation (includes writing tests) | Defined in CLAUDE.md skills list |

### 1.4 CLAUDE.md Testing Conventions

**Test Protocol (Anti-Greenwashing) — the core testing rule:**

| Code State | Test Result | Action |
|------------|-------------|--------|
| Correct | Pass | Good |
| Buggy | Fail | Good (bug exposed) - fix code |
| Correct | Fail | Discuss - test expectations may be wrong |
| Buggy | Pass | **DANGEROUS** - tests not catching bug |
| Unknown | Fail | **STOP** - don't assume which is wrong, discuss |

**NEVER modify test expected values to make tests pass without discussion.**

**Test execution command:**
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

**From Common Mistakes to Avoid:**
- Don't claim completion without running py_compile and tests
- Don't modify test expectations to pass without discussion

**From Stop Conditions & Red Flags:**
- "Uncertain whether code or test expectation is wrong" → STOP

---

## 2. Current Test Inventory

### 2.1 Test File Tree and Sizes

**`tests/` directory — 39 test files, 14,547 lines total:**

| File | Lines | Domain |
|------|-------|--------|
| `test_dataset_splits.py` | 849 | Training data splits |
| `test_dataset_quality.py` | 801 | Data quality validation |
| `test_energy_detector.py` | 743 | Energy-based USV detection |
| `test_classification/test_repertoire_stats.py` | 710 | Repertoire statistics |
| `test_cnn_model.py` | 709 | CNN classifier |
| `test_spectrogram_extractor.py` | 694 | Spectrogram extraction |
| `test_app_qt_integration.py` | 611 | PyQt6 app integration |
| `test_linker.py` | 567 | Detection/event linking |
| `test_migration.py` | 563 | Data migration |
| `test_classification/test_deepsqueak_import.py` | 561 | DeepSqueak format import |
| `test_atomize.py` | 560 | Atomization pipeline |
| `test_training_cycle.py` | 551 | Active learning cycle |
| `test_classification/test_raven_export.py` | 545 | Raven export adapter |
| `test_atomizer.py` | 544 | Atomizer module |
| `test_app_save_workflows.py` | 537 | Workflow persistence |
| `test_event_triggered.py` | 470 | Event-triggered analysis (PETH) |
| `test_lmt.py` | 457 | LMT behavioral integration |
| `test_dataset_assembler.py` | 441 | Training data assembly |
| `test_notion_client.py` | 422 | Notion API integration |
| `test_tagger.py` | 379 | Tag/annotation system |
| `test_hysteresis.py` | 352 | Hysteresis detection |
| `test_event_scoring.py` | 268 | Event scoring |
| `test_mover.py` | 247 | File movement |
| `conftest.py` | 219 | Shared fixtures |
| `test_storage_zarr.py` | 215 | Zarr storage |
| `test_processor.py` | 207 | Signal processing |
| `test_metrics.py` | 159 | Evaluation metrics |
| `test_render_tiles.py` | 153 | PNG tile rendering |
| `test_stft_core.py` | 147 | STFT computation |
| `test_calibration.py` | 141 | Temperature calibration |
| `test_saved_detection_tracker.py` | 138 | Detection artifact tracking |
| `test_label_storage.py` | 109 | Label persistence |
| `test_saved_detection_ghosts.py` | 103 | Ghost detection tracking |
| `test_config.py` | 101 | Config validation |
| `test_streaming_equivalence.py` | 70 | Streaming vs batch |
| `test_param_lab_imports.py` | 64 | Param Lab imports |
| `test_param_lab_heuristic.py` | 48 | Param Lab heuristic |
| `test_param_lab_segment.py` | 38 | Param Lab segments |
| `test_classification/__init__.py` | 0 | Package init |

**`usv_language/tests/` directory — 21 test files, 6,074 lines total:**

| File | Lines | Domain |
|------|-------|--------|
| `test_analysis.py` | 961 | VQ-VAE analysis tools |
| `test_hidden_state_vqvae.py` | 828 | Hidden state VQ-VAE |
| `test_probing.py` | 511 | Probing framework |
| `test_null_models.py` | 500 | Null models |
| `test_statistical_tests.py` | 474 | Statistical tests |
| `test_information_theory.py` | 465 | Information theory metrics |
| `test_spectrogram_transformer.py` | 412 | Spectrogram transformer |
| `test_bout_extractor.py` | 341 | Bout extraction |
| `test_acoustic_properties.py` | 273 | Acoustic properties |
| `test_bout_dataset.py` | 251 | Bout dataset |
| `test_bout_spectrogram.py` | 144 | Bout spectrogram |
| `test_trainer.py` | 116 | Training loop |
| `test_bout_normalization.py` | 111 | Bout normalization |
| `test_dataset.py` | 104 | Dataset loading |
| `test_preprocessing.py` | 103 | Preprocessing |
| `conftest.py` | 101 | Shared fixtures |
| `test_vqvae.py` | 88 | VQ-VAE model |
| `test_quantizer.py` | 78 | Quantizer |
| `test_encoder_decoder.py` | 75 | Encoder/decoder |
| `test_transformer.py` | 70 | Transformer |
| `test_losses.py` | 68 | Loss functions |

**Total: ~60 test files, ~20,600 lines across both test suites.**

### 2.2 Conftest Files

**Three conftest files exist:**

#### Root `conftest.py` (repo-wide bootstrap)
```python
"""Repo-wide pytest bootstrap.

Windows in this environment can fail to initialize PyTorch DLLs if PyQt6
loads first in the same process. Preloading torch here makes test collection
order-independent without changing application code.
"""
from __future__ import annotations
import sys

if sys.platform == "win32":
    import torch  # noqa: F401
```

#### `tests/conftest.py` (USV detection test fixtures)
```python
"""Shared pytest fixtures for USV spectrogram tests."""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path
from typing import Tuple
import numpy as np
import pytest
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.config import SpectrogramConfig
from usv_spectrogram.spectrogram import compute_spectrogram_db
from usv_spectrogram.detection.config import DetectionConfig

@pytest.fixture
def sample_wav_path() -> Path:
    """250 kHz mono WAV with 60 kHz tone + noise, 0.1s duration."""
    # ... synthetic signal generation, yields path, cleans up ...

@pytest.fixture
def sample_config() -> SpectrogramConfig:
    """Return a default SpectrogramConfig for testing."""
    return SpectrogramConfig()

@pytest.fixture
def sample_spectrogram(sample_config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-computed spectrogram tuple from synthetic 70 kHz tone."""
    # ... returns (spec_db, freqs_hz, times_s) ...

@pytest.fixture
def detection_config() -> DetectionConfig:
    """Return a default DetectionConfig for testing."""
    return DetectionConfig()

@pytest.fixture
def create_tone_wav():
    """Factory fixture: create WAV files with configurable tones.
    Parameters: freq_hz, duration_ms, amplitude, sample_rate, noise_level, start_offset_ms.
    Tracks paths for batch cleanup."""
    # ... factory pattern with cleanup ...

@pytest.fixture
def create_multi_tone_wav():
    """Factory fixture: create WAV files with multiple tones at different times.
    Parameters: tones (list of dicts), sample_rate, total_duration_ms, noise_level."""
    # ... factory pattern with cleanup ...
```

#### `usv_language/tests/conftest.py` (ML/VQ-VAE test fixtures)
```python
"""Shared pytest fixtures for usv_language tests."""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import usv_language.src  # noqa: F401

@pytest.fixture
def synthetic_spectrogram() -> np.ndarray:
    """Synthetic spectrogram (n_freq=170, n_frames=1024), dB range [-100, 0]."""

@pytest.fixture
def synthetic_hdf5(synthetic_spectrogram, tmp_path):
    """Temporary HDF5 file with 3 synthetic spectrograms.
    Uses pytest.importorskip("h5py") for lazy import (fixed after BLOCKER)."""

@pytest.fixture
def synthetic_wav(tmp_path):
    """Temporary WAV file with 60 kHz tone at 300 kHz SR."""

@pytest.fixture
def wav_dir_with_files(tmp_path):
    """Directory with 3 synthetic WAV files at different frequencies."""
```

### 2.3 Documented Test Patterns (`docs/architecture/patterns.md`, Section 3)

**Pattern 3 — Test Fixture Pattern:**
- Tests use synthetic WAV data (never real recordings)
- `yield` for cleanup
- Factory fixtures for parameterized creation
- Track created temp files in a list for batch cleanup
- Use `tmp_path` (pytest builtin) for directory-based temp files

---

## 3. Recurring Test Issues (from Reviews)

### 3.1 Compiled Findings by Category

#### BLOCKERS (must-fix before proceeding)

| Module | Finding | Root Cause |
|--------|---------|------------|
| Information Theory | conftest.py imports h5py at module level — all usv_language tests fail | Import-time dependency on optional package |
| Hysteresis Detection | Handoff claims 14 tests; code has 18 — false completion claim | Test count mismatch in documentation |

#### Coverage Gaps (most frequent issue type — WARNING level)

| Module | Missing Test | Why It Matters |
|--------|-------------|----------------|
| Dataset Assembler | Jitter failure with long USVs (>= 40ms) | Silent `return []` path never exercised |
| Dataset Assembler | All-labels-deleted JSON case | Edge case with empty results |
| Raven Export | CLI `--dry-run` behavior | Recommended first step untested |
| Probing | `-1.0` sentinel filter in `_filter_labels` | Greenwashing risk |
| LMT Data Access | Real SQLite integration test stub | No integration test at all |
| LMT Data Access | Time_range upper-boundary exclusion | Boundary math untested |
| Training Cycle | ROADMAP test plan item 1 — integration test | Entirely absent |
| Hidden State VQ-VAE | `compare_layers.py` — zero automated coverage | Major utility untested |
| Null Models | AAFT autocorrelation preservation | B1-class bugs undetectable |
| Calibration | `return_logits=True` shape verification | Latent shape regression risk |
| Calibration | `fit()` shape validation | Mismatched shapes accepted silently |
| Event Scoring | Grid search on synthetic data | Would have caught bugs earlier |
| Event Scoring | One detection spanning two GT events | Spec item entirely missing |
| FastAPI Backend | Cache TTL expiry branch | Untested code path |

#### Test Quality Issues (WARNING/SUGGESTION level)

| Module | Issue | Detail |
|--------|-------|--------|
| DeepSqueak Import | Test used corrupt file instead of empty Excel | Zero-byte file ≠ header-only Excel |
| Analysis Tools | `test_concept_injection_shape` is shape-only | Doesn't verify injection changes output |
| Information Theory | `test_zipf_mle_vs_entropy_agree` depends on fixture coincidence | Only works because α=1.5 in fixture |
| Spectrogram Transformer | Test docstring says 8 tests, actually 11 | Documentation/code mismatch |
| Event-Triggered | `test_uniform_not_significant` fragile | n_permutations too low for robust test |
| Hidden State VQ-VAE | Overfit test uses 1:1 bottleneck ratio | Production uses 8:1 — gap undocumented |

### 3.2 Patterns in the Issues

1. **Coverage gaps from ROADMAP test plans** are the #1 recurring issue — test plans specify items that don't get implemented
2. **Edge cases** (empty input, boundary conditions, rare code paths) are systematically under-tested
3. **Test count/documentation mismatches** happen frequently
4. **Fixture quality** issues (wrong file format, fragile random data, import-time side effects) cause subtle failures
5. **Shape-only tests** for ML code miss behavioral correctness
6. **The master-reviewer consistently catches these** — the review process works, but the issues shouldn't reach review in the first place

---

## 4. Current Workflow — How Tests Get Written

### 4.1 Git History of Test File Creation

All test files were committed alongside their implementation modules, in bulk commits:

```
f36d59aa chore: bulk sync — agents skills, vault notes, bug hunt fixes, parts-finder expansion
32824e0f feat: DeepSqueak classification bridge, Raven export, and Parts Finder
a3c07581 feat: vacation workstreams — information theory, null models, probing, statistical tests
4579f151 feat: LMT data access layer
2fa9ad4c feat: implement Phase 9.1 Dataset Assembler
0b63ec66 Fix bugs, add 109 tests, sampling script, and close-event auto-move
3819d39f Complete Phase 1 & 2: Detection pipeline and spectrogram extraction
20261cab Major refactor: Claude Code migration, modular Streamlit, expanded tests
```

**Key observation:** Tests are always committed WITH the implementation, never separately. This means tests are written by the same agent session that writes the code.

### 4.2 Who Writes Tests

**The implementation agent writes tests** during the `/implement` workflow. The flow is:

1. `/implement` skill reads the ROADMAP `/implement` block (which includes test plan)
2. Implementation agent writes source code AND test files in the same session
3. Agent runs `pytest` to verify
4. Agent writes handoff doc claiming test counts
5. `master-reviewer` agent reviews, finds coverage gaps
6. Implementation agent fixes gaps in a second pass
7. Fixes are re-reviewed (for BLOCKERs)

**The `test-writer` agent exists but is positioned as a supplementary tool** — it's available for "after implementing new features" per CLAUDE.md, but the primary test writing happens inside `/implement`.

### 4.3 What Guidance the Test Writer Gets

The `test-writer` agent definition is lightweight (65 lines). It specifies:
- **Philosophy:** Test behavior not implementation, one assertion per test
- **Naming:** `test_<function>_<scenario>_<expected_outcome>()`
- **Structure:** Arrange-Act-Assert
- **Patterns:** Fixtures, parametrization, mocking, edge cases
- **Convention:** Tests in `tests/`, files named `test_<module>.py`

**What it does NOT receive:**
- No ROADMAP test plan items to check against
- No reference to the anti-greenwashing protocol
- No domain-specific fixture patterns (WAV generation, spectrogram creation)
- No awareness of the master-reviewer's common findings
- No guidance on ML-specific testing (shape checks, gradient flow, overfit tests)

### 4.4 The Master-Reviewer as Test Quality Gate

The `master-reviewer` agent (232 lines) is the most sophisticated test-related component. It:

1. Reads the ROADMAP test plan and checks every item is implemented
2. Runs `pytest -v --tb=short` and verifies claimed test counts
3. Checks conftest.py for problematic fixtures (monkeypatches, import-time mocks)
4. Flags false completion claims
5. Has explicit categories: "Test anti-greenwashing" and "Missing specified test cases"
6. Does a math-trace pass for Tier 2-3 modules

**The reviewer catches most issues but operates post-hoc.** Tests are already written by the time the reviewer sees them.

---

## 5. Raw Excerpts

### 5.1 `roadmap-from-plan` Skill — Full Contents

```markdown
---
name: roadmap-from-plan
description: "Convert implementation plans into structured ROADMAP files with self-contained
  /implement blocks. Use this skill whenever the user has a plan, design doc, brainstorm notes,
  or conversation output from web Claude (or any source) that needs to become an actionable
  implementation roadmap."
---

Convert the following implementation plan into structured ROADMAP format: $ARGUMENTS

## What This Skill Does

You are converting a high-level implementation plan into the structured ROADMAP format used by
this project. Each step becomes a module entry with a self-contained `/implement` block — meaning
a fresh Claude Code session can execute it without needing the original plan.

This is a two-part job:
1. **Format conversion** — plan steps become implementable ROADMAP modules
2. **Knowledge capture** — theoretical insights in the plan get routed to the knowledge graph

## Step 1: Resolve the Input
[file path, pasted text, or URL resolution]

## Step 2: Read Project State
1. Read ROADMAP.md — find last phase number
2. Scan for other ROADMAP_*.md files
3. Read DECISIONS.md — understand existing ADRs
4. Read docs/architecture/patterns.md — established patterns

## Step 3: Analyze the Plan (and Flag Gaps)
Parse each step for: what it builds, dependencies, data structures, algorithms/logic,
test requirements, exit criteria.

Gap detection — flag under-specified steps. Mark assumptions with [ASSUMED].
Theoretical content scan — note scientific rationale for KG extraction.

## Step 4: Generate ROADMAP Entries
[module entry format with /implement block, test plan, exit criteria]

## Step 5: Phase Gate
## Step 6: Present and Write
## Step 7: Extract Theoretical Knowledge to KG

## Formatting Rules
5. **Test plans**: Be specific. "Tests pass" is not a test plan.
   Each test describes WHAT is being verified.
6. **Exit criteria**: Objectively verifiable. Include shape checks, loss thresholds, metric targets.
```

### 5.2 `test-writer` Agent — Full Contents

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

## Pytest Patterns to Use
1. Fixtures — reusable test data, @pytest.fixture, scope appropriately
2. Parametrization — @pytest.mark.parametrize
3. Mocking — mock external dependencies, use pytest-mock or unittest.mock
4. Edge Cases — empty inputs, boundary values, invalid inputs

## Project Test Conventions
- Tests live in tests/ directory
- Test files named test_<module>.py
- Run with: .\.venv\Scripts\python.exe -m pytest tests/ -v

## Key Existing Tests
- tests/test_param_lab_heuristic.py
- tests/test_param_lab_segment.py
- tests/test_streaming_equivalence.py

## Output
1. Read the code to understand behavior
2. Identify key scenarios to test
3. Write focused tests
4. Run them to verify they pass
```

### 5.3 `master-reviewer` Agent — Test-Related Sections

```markdown
### 4. Run and verify the tests
- Run the handoff's exact test command first, then the full suite:
  .\.venv\Scripts\python.exe -m pytest tests/ -v --tb=short
- Check test infrastructure — read conftest.py for problematic fixtures
  (monkeypatches, import-time mocks, silent skips)
- Verify claimed test count — actual count from pytest output must match handoff
- Flag false completion — "all tests pass" but failures/skips = BLOCKER

### 5. Check for problems:

**ML RIGOR:**
- Test anti-greenwashing: were any test expected values modified to make tests pass?

**SPEC COMPLIANCE:**
- Are any specified test cases missing from the test plan?

**INTEGRATION CORRECTNESS:**
- Does it use synthetic WAV fixtures in tests, not real recordings? (Pattern 3)

**CODE QUALITY:**
- Are tests actually testing the right things (not just asserting no exception)?
```

### 5.4 `detection-validator` Agent — Test-Related Sections

```markdown
## Validation Steps
1. Run detection tests:
   .\.venv\Scripts\python.exe -m pytest tests/test_energy_detector.py -v
2. Check algorithm correctness
3. Verify edge cases
4. Check config validation

## Output Format
### Test Coverage: COMPLETE/GAPS
[Missing tests if any]
```

### 5.5 CLAUDE.md — Testing Sections

**Test Protocol (Anti-Greenwashing):**

| Code State | Test Result | Action |
|------------|-------------|--------|
| Correct | Pass | Good |
| Buggy | Fail | Good (bug exposed) - fix code |
| Correct | Fail | Discuss - test expectations may be wrong |
| Buggy | Pass | DANGEROUS - tests not catching bug |
| Unknown | Fail | STOP - don't assume which is wrong, discuss |

**NEVER modify test expected values to make tests pass without discussion.**

**Stop Conditions:**
- Same approach tried twice without new rationale
- Evidence contradicts hypothesis
- Uncertain whether code or test expectation is wrong

**Common Mistakes:**
- Don't claim completion without running py_compile and tests
- Don't modify test expectations to pass without discussion

### 5.6 `docs/architecture/patterns.md` — Test Fixture Pattern (Section 3)

```markdown
## 3. Test Fixture Pattern

Tests use synthetic WAV data (never real recordings), `yield` for cleanup,
and factory fixtures for parameterized creation.

Rules:
- Use `yield` for fixtures that create files (enables cleanup)
- Track created temp files in a list for batch cleanup
- Synthetic signals: noise + pure tone at known frequency
- Use `tmp_path` (pytest builtin) for directory-based temp files
- Never depend on real WAV recordings in tests
```

### 5.7 Review Findings — Full Test-Related Findings

(See Section 3 above for compiled findings. All 17 reviews were checked; findings
from 16 modules contained test-relevant items.)

---

## 6. Infrastructure and Configuration

### 6.1 Pytest Configuration

**No explicit pytest configuration file exists.** No `pyproject.toml`, `setup.cfg`, `pytest.ini`, or `tox.ini`. The project relies entirely on pytest defaults for test discovery and collection.

### 6.2 Test Dependencies

From `requirements.txt` (only test dependency):
```
pytest
```

Frozen version (from `requirements_frozen.txt`): `pytest==9.0.2`

**Notable absences:**
- No `pytest-cov` (no coverage measurement)
- No `pytest-xdist` (no parallel test execution)
- No `pytest-timeout` (no test timeout enforcement)
- No `pytest-mock` listed (tests use `unittest.mock` directly)

### 6.3 CI/CD

**No CI/CD configured.** No GitHub Actions, GitLab CI, or pre-commit hooks for testing. Tests are run manually via the CLI command.

### 6.4 Pre-Commit Hooks

**No test-related hooks.** Existing hooks handle workflow enforcement (`check_plan_mode`), git automation (`auto-commit`), session tracking (`session-capture`, `session-orient`), vault validation (`validate-note`), and agent requirement checking (`check_agents_tag`).

### 6.5 Test Style Inconsistency

Two test styles coexist:
- **pytest-style** (majority): standalone functions with `assert` statements
- **unittest.TestCase** (minority): class-based tests with `self.assertEqual` (e.g., `test_stft_core.py`, `test_config.py`)

### 6.6 Platform-Specific Bootstrap

The root `conftest.py` preloads `torch` on Windows to prevent PyTorch/PyQt6 DLL initialization order issues. This affects all test collection.

---

## 7. Knowledge Graph Testing Context

### 7.1 Vault Notes About Testing

No dedicated testing methodology notes exist in `notes/`. The grep for "test|fixture|eval|assert" returned only general agent-memory and domain notes (about memory architectures, vocal comparison, etc.) — none specifically about testing strategy or validation methodology.

### 7.2 Decision Notes (ADRs) About Testing

No ADRs specifically about testing conventions. Testing guidance lives entirely in:
- CLAUDE.md (anti-greenwashing protocol)
- `docs/architecture/patterns.md` (Pattern 3: Test Fixture Pattern)
- ROADMAP test plans (per-module)
- Agent definitions (master-reviewer, test-writer)

---

## 8. Additional Findings (Not in Original Handoff)

### 8.1 Two Separate Test Suites

The project has **two independent test suites** that are not mentioned together anywhere:
- `tests/` — USV detection, app, classification (14,547 lines, 39 files)
- `usv_language/tests/` — VQ-VAE, transformer, analysis tools (6,074 lines, 21 files)

These have separate conftest files with different fixture ecosystems. The `usv_language` conftest uses `pytest.importorskip("h5py")` (a fix from a BLOCKER found in review), while the main conftest does not use any lazy imports.

### 8.2 Fixture Duplication Between Suites

Both conftest files independently create synthetic WAV fixtures:
- `tests/conftest.py`: `sample_wav_path()` — 250 kHz SR, 60 kHz tone
- `usv_language/tests/conftest.py`: `synthetic_wav()` — 300 kHz SR, 60 kHz tone

Note the **different sample rates** (250 kHz vs 300 kHz). The main tests use 250 kHz while `usv_language` uses the project standard 300 kHz (ADR-001). This is a potential source of confusion.

### 8.3 Test-Writer Agent is Under-Specified

Compared to the master-reviewer (232 lines of detailed guidance), the test-writer agent (65 lines) lacks:
- Reference to ROADMAP test plan items
- Anti-greenwashing protocol awareness
- Domain-specific fixture patterns (WAV generation with correct sample rates)
- ML-specific test patterns (shape checks, gradient flow, overfit tests, data leakage checks)
- Awareness of the two separate test suites
- Guidance on which conftest fixtures to use/extend

### 8.4 No Coverage Measurement

There is no coverage tooling installed or configured. Coverage gaps are found only through manual review by the master-reviewer agent. This means:
- No visibility into which code paths are tested
- Coverage gaps accumulate until review
- No way to measure improvement over time

### 8.5 Test Count Growth Over Time

From IMPLEMENTATION_PROGRESS.md, test count has grown significantly:
- Phase 1-2: initial test suite (commit 3819d39f)
- Phase 9.1: 434 total tests
- usv_language tests: grew from 130 → 184 → 232 → 254 → 270 over vacation workstreams
- Phase 15.x: 47+ tests for post-processing modules alone

### 8.6 The Review-Fix Cycle is the Actual Test Quality Mechanism

The master-reviewer's review → fix → re-review cycle is where most test quality improvement happens. Of the 16 modules with test-related review findings:
- All had fixes applied
- Most added 2-5 additional tests
- Several had BLOCKER-level issues (false completion, broken fixtures)

This means **the master-reviewer is effectively the test quality gate**, not the test-writer agent.

### 8.7 Mixed Test Granularity

Some modules have very fine-grained tests (hysteresis: 21 tests for one function), while others have coarse tests (VQ-VAE: 88 lines covering multiple model operations). There's no standard for test density relative to code complexity.
