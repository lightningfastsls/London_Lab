# Handoff: Spawn test-architect agents for Phase 17 (SIS Benchmark ROADMAP)

**Date:** 2026-04-17
**Purpose:** Generate pre-implementation test files for all 9 modules in `ROADMAP_SIS_BENCHMARK.md` before any implementation begins. Follows the `/implement` Phase 0 requirement from `CLAUDE.md`.

---

## What you're doing

The ROADMAP at `/home/shachar/projects/mickey_london_lab/ROADMAP_SIS_BENCHMARK.md` defines 9 modules (17.1 through 17.9) for a USV labeling SIS benchmark on the 5970 dataset. Each module has a `Test plan:` block with specific test cases.

Your job: spawn `test-architect` subagents to convert each module's `Test plan` into runnable pytest files. All tests MUST fail initially — the modules don't exist yet. This is TDD — tests become the spec the subsequent `/implement` run must satisfy.

## Read this first

1. `/home/shachar/projects/mickey_london_lab/ROADMAP_SIS_BENCHMARK.md` — the full roadmap (9 modules + gate, ~700 lines)
2. `/home/shachar/projects/mickey_london_lab/docs/architecture/patterns.md` — established patterns (dataclass, CLI, imports). Test-architect needs these for import bootstrapping.
3. `/home/shachar/projects/mickey_london_lab/tests/conftest.py` — existing test fixtures (tmp_path, synthetic WAV creators). Reuse these.
4. `/home/shachar/projects/mickey_london_lab/CLAUDE.md` — project conventions (frozen dataclasses, sample rate always 300000, STFT n_fft=512 hop=128)

## Spawn strategy

### Parallelism groups

All 9 modules can have their tests generated in parallel — test-architect reads the module spec and writes test files without needing other modules to exist. **Spawn all 9 in a single message** with 9 Agent tool calls.

### Dependency note (informational only)

Test files themselves don't depend on each other. But at *runtime* the dependency chain is:

- 17.1, 17.2 — no deps
- 17.3 → 17.2
- 17.4 → 17.3
- 17.5 → 17.3
- 17.6 → 17.2
- 17.7 → 17.5 + 17.6
- 17.8 → 17.1
- 17.9 → 17.1, 17.4, 17.7, (17.8)

The test-architect for each module just needs to see the module's own spec. It can import from the other modules (which don't exist yet) — the tests will fail with ImportError, which is the correct TDD starting state.

## Exact spawn instructions

For each module, spawn a `test-architect` subagent with a prompt of this form:

```
Write pre-implementation test files for module [N.M] of ROADMAP_SIS_BENCHMARK.md.

READ FIRST:
- /home/shachar/projects/mickey_london_lab/ROADMAP_SIS_BENCHMARK.md — find module [N.M] and use its /implement block as the spec
- /home/shachar/projects/mickey_london_lab/docs/architecture/patterns.md — Patterns 1 (frozen dataclass), 4 (script CLI), 7 (STFT core), 8 (import bootstrap)
- /home/shachar/projects/mickey_london_lab/tests/conftest.py — existing fixtures to reuse (especially create_tone_wav)

WRITE: [exact test file path from the module spec]

RULES:
- Tests must FAIL INITIALLY (modules don't exist yet)
- Every test case in the module's "Test plan:" block must become a pytest function
- Use synthetic data (np arrays, synthetic WAVs), never real recordings
- Import from the module being tested even though it doesn't exist — ImportError is the correct initial failure mode
- Follow Pattern 8 (path bootstrap) at top of each test file
- Use tmp_path for directory-based temp files, yield-cleanup for single files
- Each test function has a docstring stating what it verifies

OUTPUT: Single test file. Do not create implementation stubs. Do not modify the ROADMAP.

When done, run pytest on the new file and report: (a) number of tests, (b) failure mode (should be ImportError or missing symbol, not assertion failure).
```

### Test file paths per module

| Module | Test file path (from ROADMAP) |
|--------|-------------------------------|
| 17.1 | `tests/test_sis_baselines.py` |
| 17.2 | `tests/test_spectrogram_filter.py` |
| 17.3 | `tests/test_ridge_tracker.py` |
| 17.4 | `tests/test_imsa_classifier.py` |
| 17.5 | `tests/test_omer_vectorize.py` |
| 17.6 | `tests/test_amvoc_autoencoder.py` |
| 17.7 | `tests/test_cluster_sweep.py` |
| 17.8 | `tests/test_sim_optimizer.py` |
| 17.9 | `tests/test_sis_benchmark.py` |

### One-shot spawn template

In a single message, use 9 `Agent` tool calls with `subagent_type: "test-architect"`. Prompt skeleton for each (replace `{N.M}` and `{TEST_FILE}`):

```
Write pre-implementation test file {TEST_FILE} for module {N.M} of ROADMAP_SIS_BENCHMARK.md.

Read /home/shachar/projects/mickey_london_lab/ROADMAP_SIS_BENCHMARK.md — find module {N.M} and use its /implement block + Test plan as the specification.

Reference /home/shachar/projects/mickey_london_lab/docs/architecture/patterns.md for Patterns 1 (frozen dataclass), 4 (script CLI), 7 (STFT core), 8 (import bootstrap).

Reference /home/shachar/projects/mickey_london_lab/tests/conftest.py for existing fixtures.

Constraints:
- Tests must fail initially (modules don't exist yet) — ImportError is correct.
- Every bullet in the module's "Test plan" block becomes a pytest function.
- Synthetic data only. No real WAVs.
- Follow Pattern 8 (path bootstrap with REPO_ROOT = Path(__file__).resolve().parents[1]).
- Use tmp_path for tempdirs, yield-cleanup for single files.
- Each test has a docstring explaining what it verifies.

Output the single test file only. Do not create implementation stubs. Do not modify the ROADMAP.

When done: run `.venv/bin/python -m pytest {TEST_FILE} -q` and report (a) test count, (b) failure mode (expected: ImportError or missing symbol).
```

## Post-spawn verification

After all 9 agents complete, do this:

1. **Verify all 9 test files exist:**
   ```bash
   ls tests/test_sis_baselines.py tests/test_spectrogram_filter.py tests/test_ridge_tracker.py tests/test_imsa_classifier.py tests/test_omer_vectorize.py tests/test_amvoc_autoencoder.py tests/test_cluster_sweep.py tests/test_sim_optimizer.py tests/test_sis_benchmark.py
   ```

2. **Collect-only run (tests should be discoverable even if they can't run):**
   ```bash
   .venv/bin/python -m pytest tests/test_sis_*.py tests/test_spectrogram_filter.py tests/test_ridge_tracker.py tests/test_imsa_classifier.py tests/test_omer_vectorize.py tests/test_amvoc_autoencoder.py tests/test_cluster_sweep.py tests/test_sim_optimizer.py --collect-only -q
   ```
   Expected: N tests collected, all ImportError at module load (module not yet implemented). This is the **correct** TDD starting state.

3. **Count tests per module** (sanity check — should roughly match the "Test plan:" bullet counts):
   - 17.1: ~8 tests
   - 17.2: ~9 tests
   - 17.3: ~9 tests
   - 17.4: ~11 tests
   - 17.5: ~10 tests
   - 17.6: ~10 tests
   - 17.7: ~8 tests
   - 17.8: ~9 tests
   - 17.9: ~7 tests
   - **Total: ~81 tests**

4. **Commit in 9 separate commits, one per module** (per project convention):
   ```bash
   git add tests/test_sis_baselines.py
   git commit -m "test(17.1): pre-implementation test spec from ROADMAP_SIS_BENCHMARK"
   # ... repeat for each module
   ```

   Use a single commit only if the user explicitly requests it. The per-module structure makes bisecting easier if an implementation later breaks expected behavior.

## Red flags — stop and ask user if you see these

- **test-architect writes implementation stubs** — NOT allowed. Tests only. If an agent produces implementation code, delete it.
- **test-architect modifies the ROADMAP** — NOT allowed. The ROADMAP is the spec.
- **Test expectations disagree with the ROADMAP's Exit criteria** — flag to user, do not silently adjust.
- **Tests pass without implementation** — impossible unless the agent wrote implementation stubs or mocked out the missing module. Investigate.
- **Agent writes tests that require real WAV files or real CSVs** — redirect to synthetic data. The fixtures in `conftest.py` already cover synthetic WAV generation.

## If an agent fails or produces low-quality tests

Do not re-spawn with the same prompt — diagnose first:

- **Test-architect says "module spec unclear":** read that module's `/implement` block yourself; amend the spawn prompt with explicit clarification; re-spawn only that module.
- **Test-architect wrote only 2-3 tests (under-specified):** the module's Test plan has ≥7 bullets for all 9 modules. Re-spawn with the Test plan text copy-pasted into the prompt for emphasis.
- **Test-architect added its own test cases beyond the spec:** acceptable — these are bonus coverage. Keep them.

## After this handoff is complete

Mark task #4 ("Spawn test-architect for 9 modules") as completed. The next step is `/implement 17.1` — the first module. Implementation will start with the free decision-gate baseline module.

## Context from the original session

- User feedback drove a major restructuring: initial plan was Oren-only (5 modules); revised plan covers 4 hypothesis classes (9 modules). See `inbox/sis-benchmark-design-2026-04-17.md` for design rationale.
- User explicitly confirmed: retrain AMVOC from scratch on our data (not use pretrained), split clustering from vectorization, run SIM on every starting labeling (option a).
- User flagged context size pressure at end of design session — this handoff exists so a fresh session can do the mechanical test-generation work without re-loading the design discussion.

## Expected runtime

9 agents × ~2 minutes per spec-read + test-write = ~5 min wall-clock if fully parallel, more if sequential. Each agent runs independently; any failures can be retried without affecting other agents.
