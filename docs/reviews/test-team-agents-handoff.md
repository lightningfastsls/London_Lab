# Handoff: Test Architect & Test Hardener Agent Team

**Date:** 2026-03-28
**Source:** `testing-team-handoff.md` (user-authored spec)
**Scope:** Infrastructure only — no agents run on existing modules

---

## What Was Built

Two new specialized agents that separate test authoring from code authoring, plus workflow integration across commands and documentation.

### New Agents

| Agent | Path | Lines | Purpose |
|-------|------|-------|---------|
| `test-architect` | `.claude/agents/test-architect.md` | ~160 | Writes failing pytest tests from ROADMAP `/implement` specs BEFORE implementation |
| `test-hardener` | `.claude/agents/test-hardener.md` | ~85 | Writes adversarial edge-case tests AFTER implementation to find coverage gaps |

Both use `model: sonnet` and tools: Read, Grep, Glob, Write, Bash.

### Deprecated Agent

| Agent | Path | Change |
|-------|------|--------|
| `test-writer` | `.claude/agents/test-writer.md` | Body replaced with redirect to test-architect / test-hardener. Frontmatter preserved for backward compat. |

### Workflow Integration

| File | Change |
|------|--------|
| `.claude/commands/implement.md` | Added **step 3.5** (check for pre-existing test files from test-architect) and **phase 4.5** (spawn test-hardener after master-reviewer approval) |
| `.claude/commands/roadmap-from-plan.md` | Added **step 7.5** (reminder to run test-architect after ROADMAP generation) |
| `.claude/skills/roadmap-from-plan-workspace/skill-for-eval/SKILL.md` | Same step 7.5 added (skill version mirrors command version) |

### Documentation Updates

| File | Change |
|------|--------|
| `CLAUDE.md` | Agents table: replaced test-writer with test-architect + test-hardener. Added testing workflow sequence. Extended anti-greenwashing note for pre-existing test files. |
| `docs/workflow/completion-sequence.md` | Agent table: replaced test-writer row with test-architect (before step 2) and test-hardener (between steps 5-6) |
| `docs/claude-code-guide.md` | Directory tree + agent table updated with both new agents and deprecated marker |
| `docs/PROJECT_STATE_REPORT.md` | Agent table updated |

### Files NOT Changed (intentional)

Historical/audit docs left untouched — they describe what was true at the time:
- `IMPLEMENTATION_PROGRESS.md`, `docs/testing-infrastructure-audit.md`, `WORKFLOW-AUDIT.md`, `cloudy-claude-architecture-audit.md`, `usv-and-skills-audit.md`, `testing-team-handoff.md`, `docs/historical/plan.md`

---

## Integration Design Choice

**Reminder-based (modified option b from handoff):**
- `roadmap-from-plan` reminds the user to run test-architect (step 7.5)
- `/implement` detects and uses pre-existing test files (step 3.5)
- No ROADMAP format changes, no programmatic agent invocation
- Matches the project's instruction-based agent spawning convention

**Why not option (a) — automatic invocation:** Agent spawning in this project is prose-based (commands instruct sessions to spawn agents). There's no programmatic mechanism to chain agents.

**Why not option (c) — do nothing:** Too passive. The reminder in roadmap-from-plan and the awareness in /implement make the workflow discoverable without being forced.

---

## Updated Workflow Sequence

```
/roadmap-from-plan  →  ROADMAP with /implement blocks
                       Step 7.5: "Consider running test-architect"
        ↓ (optional but recommended)
test-architect       →  tests/test_<module>.py (all tests FAIL)
        ↓
/implement           →  Step 3.5: reads pre-existing tests as spec
                        Builds module, writes additional tests only
        ↓
master-reviewer      →  Reviews implementation (Phase 4)
        ↓ (after APPROVED)
test-hardener        →  Adversarial edge cases (Phase 4.5)
        ↓
Report
```

---

## Next Steps

The ROADMAP_POST_PROCESSING.md has modules in two states:

**Already implemented (test-hardener appropriate):**
- 15.1 Hysteresis Detection — `src/usv_spectrogram/postprocessing/hysteresis.py` + `tests/test_hysteresis.py`
- 15.2 Event Scoring — `src/usv_spectrogram/postprocessing/event_scoring.py` + `tests/test_event_scoring.py`
- 15.3 Temperature Calibration — `src/usv_spectrogram/postprocessing/calibration.py` + `tests/test_calibration.py`

**Not yet implemented (test-architect appropriate):**
- 15.4 Batch Detection Pipeline (BLOCKED on 15.1)
- 15.5 Confidence-Based Sorting (BLOCKED on 15.1)
- 15.6 ROC Curve Generation (READY)
- 15.7 Pipeline Integration (BLOCKED on 15.2, 15.5)

Recommended: fresh chat to run test-hardener on 15.1-15.3 (parallel), then test-architect on 15.4-15.7.

---

## What I'm Unsure About

- **Skill vs command duplication:** `roadmap-from-plan` exists in both `.claude/commands/` and `.claude/skills/`. I updated both, but the duplication itself may cause drift over time.
- **test-hardener's skip-marking:** Using `@pytest.mark.skip(reason="BUG FOUND: ...")` keeps the suite green but could accumulate. No mechanism currently tracks or surfaces these.
