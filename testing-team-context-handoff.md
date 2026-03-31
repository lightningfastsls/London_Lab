# Context Gathering Handoff: Testing Infrastructure Audit

## Background
We're designing a dedicated test-writing agent team for the USV project. Before we can design it, we need a clear picture of how tests are currently created, structured, and what patterns exist. **This is a discovery task — gather and report, don't design anything.**

## What to gather

### 1. How Modules Get Specified
We need to understand where test plans and exit criteria come from — the current source of truth for "what does correctness look like."

- Find and read the `roadmap-from-plan` skill (check `/mnt/skills/`, `CLAUDE.md`, or wherever custom skills live). Report its full contents — this is likely where module specs including test expectations are generated.
- Find 2-3 actual roadmap files that this skill has produced. Show their test plan / exit criteria sections so we can see what the implementer gets as testing guidance.
- Check if there are other skills or agents that touch test generation (grep for "test", "fixture", "eval" in skill definitions).
- Read `CLAUDE.md` and `AGENTS.md` — extract anything related to testing conventions, commands, expectations, or how agents are supposed to handle tests.

### 2. Current Test Structure
We need to see what exists today.

- Show the full test directory tree (`find tests/ -type f -name "*.py"` or equivalent)
- Show test file sizes (`wc -l` on test files)
- Read ALL `conftest.py` files (root and subdirectory level) — these define shared fixtures and are critical context
- Read `docs/architecture/patterns.md` if it exists — specifically anything about test patterns, synthetic data, fixtures

### 3. Test Quality: What's Working and What's Not
We need to see both good and bad examples.

- Read all files in `docs/reviews/` — extract every finding (BLOCKER, WARNING, SUGGESTION) that mentions testing, fixtures, coverage, or test quality
- Pick the test file that got the harshest review feedback and read it in full
- Pick a test file that passed review cleanly and read it in full
- If there's an `IMPLEMENTATION_PROGRESS.md`, scan it for any testing-related entries or fixes

### 4. How Tests Get Written Today
We need to understand the current workflow — who/what writes tests, when in the process, and with what guidance.

- Check git log for test files: are they committed alongside implementation, or separately? (`git log --oneline --diff-filter=A -- "tests/"` or similar)
- Is there any agent or skill that currently writes tests? (check all agent/skill definitions)
- Look at the master-reviewer agent — it reviews tests but does it ever generate them?

### 5. Knowledge Graph Testing Context
- Search `notes/` for anything about testing methodology, validation approaches, evaluation criteria
- Check if there are any decision notes (ADRs) about testing conventions

### 6. Anything Else Relevant
- If you find other infrastructure I didn't think to ask about (CI config, pre-commit hooks, test coverage tools, pytest plugins in `pyproject.toml`), include it
- If the repo structure has changed in ways that make any of the above questions wrong, say so and report what you actually find

## Output
Create `docs/testing-infrastructure-audit.md` with:

1. **How Specs Define Tests** — what does the implementer receive as test guidance? (include actual excerpts from roadmap-from-plan skill and example roadmap files)
2. **Current Test Inventory** — file tree, sizes, fixture patterns, conftest contents
3. **Recurring Test Issues** — compiled from review findings
4. **Current Workflow** — who writes tests, when, with what guidance
5. **Raw Excerpts** — full contents of key files (conftest files, the roadmap-from-plan skill, relevant CLAUDE.md sections, review findings about tests). Don't summarize these — include them verbatim so the next step has primary sources.

**Do NOT design the testing agent — just gather and organize the raw context.**
