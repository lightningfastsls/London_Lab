---
name: implement
description: End-to-end module implementation workflow. Uses built-in plan mode for context-efficient planning, then builds code and tests following the ROADMAP spec.
---

Implement the feature described by the user: $ARGUMENTS

Follow this sequence strictly:

## Phase 1: PLAN (Enter Plan Mode)

**Call the `EnterPlanMode` tool NOW** to enter built-in plan mode. This creates a context boundary — all exploration reads from this phase can be compacted before implementation begins, preventing context rot during long implementation sessions.

While in plan mode (read-only tools only — no code writing allowed):

1. Read `ROADMAP.md` — find the module's `/implement` block and test plan
2. Read `notes/index.md` + relevant topic maps — understand architectural constraints and prior decisions (search `type: decision` notes for STFT params, split-by-recording, etc.)
3. Read `docs/architecture/patterns.md` — follow established patterns (frozen dataclasses, candidate flow, etc.)
4. Read `docs/modules/*.md` for any module this one depends on or interacts with
5. Read existing code to understand current state
6. Identify all files that need to be created or modified
7. Note edge cases, DSP parameter constraints, and integration points

**Write your plan to the plan file**, including:
- New files to create (with exact paths)
- Existing files to modify
- Data structures with field types
- Algorithm description (for DSP/ML modules)
- DSP parameters used and their ADR references
- Open questions or assumptions you're making

**Call `ExitPlanMode` to present the plan for user approval.** Do NOT proceed until the user approves.

## Phase 2: IMPLEMENT (only after plan is approved)

**FIRST: Create your task list immediately** using TaskCreate. Include ALL of these as separate tasks:
- One task per implementation step (config, core logic, scripts, tests)
- **MANDATORY final task:** "Write implementation handoff (`docs/reviews/<module>-handoff.md`)"
  (This task persists in the list as a visible reminder even after test output bloats the context)

Then implement in this order, marking tasks in_progress/completed as you go:
1. **Config** — Frozen dataclasses in the appropriate config module
2. **Core logic** — Implementation in `src/usv_spectrogram/` or `usv_language/`
3. **Scripts** — CLI entry points in `scripts/`
4. **Tests** — Write tests covering: happy path, edge cases, DSP parameter validation
5. **Run module tests** — `.\.venv\Scripts\python.exe -m pytest tests/test_<module>.py -v`
6. **Run full test suite** — `.\.venv\Scripts\python.exe -m pytest tests/ -v`
7. Fix any failures — iterate until green

**DSP checks during implementation:**
- All STFT parameters match ADR-002 (n_fft=512, hop=128, sr=300000)
- All frequency ranges match the module's requirements
- Never use librosa defaults — always specify sr explicitly

## Phase 3: DOCUMENT
After tests pass:
1. Create or update `docs/modules/<module_name>.md` with:
   - Purpose, public interface, usage examples
   - Key decisions and ADR references
   - Integration points (what this module calls, what calls this module)
2. If you established a new reusable pattern -> update `docs/architecture/patterns.md`
3. If you made a non-obvious decision -> create a decision note in `notes/` (type: decision) and run `/reflect` to connect it
4. Write the handoff: `docs/reviews/<module>-handoff.md` (the task should be sitting in your list as `pending` — write it and mark completed)

## Phase 3.5: STALENESS AUDIT
Before moving to review, check for stale references:
- Grep for the module name in `ops/goals.md` — update if mentioned
- Grep for the module name in `docs/modules/` — ensure module doc matches implementation
- Append a dated entry to `IMPLEMENTATION_PROGRESS.md` (never modify existing entries)

## Phase 4: REVIEW

After writing the handoff, spawn the `master-reviewer` agent (subagent_type: `master-reviewer`). Use the review tier from ROADMAP.md to set the prompt:

1. **Spawn master-reviewer** with this prompt structure:
   ```
   Review module [MODULE NAME] in the USV Detection project.
   This is a TIER [N] ([tier name]) review.
   Read the handoff first: docs/reviews/[module]-handoff.md
   ```
   See `docs/reviews/REVIEW-TEMPLATE.md` for the full tier-specific prompts (Tier 1/2/3).

2. **Write the review file** to `docs/reviews/<module>-review.md` based on the master-reviewer's findings. The agent returns text — you write the file.

3. **If CHANGES NEEDED:** Fix all blockers, add a "Fixes Applied" section to the review file, re-run tests.
   - **BLOCKERs require re-review** — the implementor cannot self-report that blockers are resolved. Spawn master-reviewer again as a spot-check (can be Tier 1 scope focused on the specific fixes).
   - **Tier 3 recommendation:** For Tier 3 (critical) modules, recommend that re-review happens in a **separate session** for fresh context. The implementor and reviewer sharing the same conversation risks anchoring bias.

## Phase 5: REPORT
Summarize to the user:
- What was created (files, classes, functions)
- Test results (pass counts)
- Review verdict and any issues found
- Any known limitations

