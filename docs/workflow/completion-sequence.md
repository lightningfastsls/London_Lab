# Completion Sequence

Every implementation task follows this 7-step sequence. No step may be skipped.
The sequence ensures that work is validated before being declared complete and that
the next session has everything it needs to continue.

---

## The 7 Steps

### Step 1: Create Tasks
Before writing any code, create task items that track the work.

**What to do:**
- Break the approved plan into discrete, testable tasks
- Include a final "Write handoff" task (Step 6)
- Set task dependencies where order matters

**Why:** Tasks make progress visible and prevent scope creep. The handoff task
ensures you don't forget to document what you did.

### Step 2: Write Code
Implement the approved changes, one task at a time.

**What to do:**
- Mark task as `in_progress` before starting
- Keep changes focused — one logical change per task
- Run `py_compile` after every file edit
- Mark task as `completed` when done

**Rules:**
- Never modify test expectations to make tests pass
- Never skip py_compile ("it's probably fine")
- If blocked, surface it immediately (Struggle Protocol)

### Step 3: Run Module Tests
After completing code changes, run the tests for the specific module you changed.

**What to do:**
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_<module>.py -v
```

**Pass criteria:**
- All tests pass (no failures, no errors)
- No new warnings that indicate problems

**If tests fail:**
- Do NOT modify test expectations
- Analyze whether the code or test is wrong (see Test Protocol in CLAUDE.md)
- Fix the code, then re-run

### Step 4: Run Full Test Suite
After module tests pass, run the complete test suite to catch regressions.

**What to do:**
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

**Pass criteria:**
- All tests pass
- No regressions (tests that passed before still pass)

**If unrelated tests fail:**
- Investigate whether your change caused the failure
- If yes: fix it
- If no: document it in the handoff as a pre-existing issue

### Step 5: Fix Failures
Iterate on Steps 2-4 until everything passes.

**What to do:**
- Fix code (not tests) to resolve failures
- Re-run py_compile after each fix
- Re-run module tests, then full suite

**Stop condition:**
- If the same fix has been attempted twice without progress, invoke the Struggle Protocol
- Don't spiral — surface the blocker

### Step 6: Write Handoff
Document what was done so the next session can pick up seamlessly.

**What to do:**
Write a handoff following the template below (full version with field descriptions in `docs/reviews/REVIEW-TEMPLATE.md`).
Save to `docs/reviews/<module>-handoff.md`.

```markdown
# Implementation Handoff: [Module Name]

**Module:** [name]
**Review Tier:** [1 | 2 | 3]
**Date:** [YYYY-MM-DD]

## What Changed
- [3-5 bullet summary]

## Files Changed
- `path/to/file.py` (NEW) — description
- `path/to/other.py` (MODIFIED) — what changed

## Key Decisions Made
- [Non-obvious choices the reviewer should scrutinize]

## What I'm Unsure About
- [Areas needing extra scrutiny — directs reviewer's attention]

## Test Results
[pytest output summary]

## ROADMAP Exit Criteria Status
- [x] Criterion 1
- [ ] Criterion 2 (reason)

## Docs Written/Updated
- [List of doc files created or updated]
```

Full template with field descriptions: `docs/reviews/REVIEW-TEMPLATE.md`

**Where to write:** Append a dated entry to `IMPLEMENTATION_PROGRESS.md` (never modify existing entries). Update `ops/goals.md` with current status.

> **Do NOT update `docs/human/` files directly** — use `/refresh-human-docs` to regenerate them from KG + ops state.

### Step 7: Report
Communicate completion to the user.

**What to include:**
- Summary of what was accomplished
- Test results (pass count, any caveats)
- Files created or modified
- Any deferred work or known issues
- Suggestion for next session if appropriate

**Format:**
```
## Done: [Task Name]

**Changes:** [1-3 bullet summary]
**Tests:** X passed, 0 failed
**Files:** [list of new/modified files]
**Deferred:** [anything not done]
**Next session:** [suggestion if applicable]
```

---

## Quick Reference

| Step | Action | Validation |
|------|--------|------------|
| 1 | Create tasks | Tasks visible in task list |
| 2 | Write code | py_compile passes |
| 3 | Module tests | pytest on changed module passes |
| 4 | Full suite | pytest on all tests passes |
| 5 | Fix failures | All tests green |
| 6 | Write handoff | Handoff document complete |
| 7 | Report | User informed of results |

---

## When to Invoke Specialized Agents

During the completion sequence, these agents should be invoked at specific points:

| Agent | When | Step |
|-------|------|------|
| `dsp-reviewer` | Any STFT/signal processing change | Between Steps 4 and 6 |
| `detection-validator` | Any detection logic change | Between Steps 4 and 6 |
| `streamlit-expert` | Any Streamlit UI work | During Step 2 |
| `test-writer` | New features needing tests | During Step 2 |
| `pr-reviewer` | Before declaring complete | Between Steps 6 and 7 |

---

## Common Pitfalls

1. **Skipping Step 3** ("I'll just run the full suite") — Module tests are faster and
   give clearer error messages. Always run them first.

2. **Skipping Step 6** ("The code is self-documenting") — It isn't. The next session
   has no context. Write the handoff.

3. **Modifying tests in Step 5** — This is the most dangerous pitfall. If tests fail,
   the code is probably wrong. Discuss before changing test expectations.

4. **Declaring done without Step 4** — Module tests passing doesn't mean you haven't
   broken something else. Always run the full suite.

5. **Forgetting py_compile** — Syntax errors caught late waste time. Run it after
   every edit, not just at the end.

---

## Relationship to Existing Workflow

This completion sequence **extends** the existing CLAUDE.md workflow. It does not replace it.

| Existing CLAUDE.md Rule | Status in This Sequence |
|------------------------|------------------------|
| Approval Request before code | **Still required.** Step 2 assumes approval was already given. |
| py_compile after every edit | **Still required.** Explicitly enforced in Step 2. |
| Append to IMPLEMENTATION_PROGRESS.md | **Still required.** Part of Step 6 (append-only, never modify existing entries). Update ops/goals.md too. |
| Run specialized agents (dsp-reviewer, detection-validator) | **Still required.** Run DURING implementation (Step 2) or between Steps 4 and 6. These complement the master review, not replace it. |
| Don't modify test expectations | **Reinforced.** Explicitly stated in Steps 3, 4, and 5. |
| State Machine transitions | **Still enforced.** The completion sequence operates within the EXECUTION → VALIDATION states. |
| Struggle Protocol | **Still active.** Referenced in Step 5 as the stop condition for fix loops. |
