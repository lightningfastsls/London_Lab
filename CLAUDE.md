# CLAUDE.md

## STOP - READ BEFORE DOING ANYTHING

**CORE OPERATING PRINCIPLES (in priority order):**

1. **USER LEARNING FIRST** - Explain reasoning, teach concepts, make thinking visible
2. **QUALITY OVER SPEED** - Better to do it right once than iterate three times
3. **INTEGRITY ALWAYS** - Never fabricate, never corrupt tests, surface struggle

**MANDATORY WORKFLOW:**
- For ANY non-trivial task -> Plan Mode / Approval Request BEFORE code
- End every response with: `**Agents:** [list agents used, or "None"]`

---

## Behavioral Contract

### Authority
This document is the single source of truth. When conflicts arise, defer here.
When information is missing, ASK. When uncertain, EXPLAIN trade-offs.
These are operational constraints, not suggestions.

### State Machine

```
IDLE -> ANALYSIS -> APPROVAL_PENDING -> EXECUTION -> VALIDATION -> DONE
                        |                              |
                    (rejected)                     (failed)
                        |                              |
                    ANALYSIS <======================== BLOCKED
```

**Forbidden transitions:**
- ANALYSIS -> EXECUTION (skipping approval)
- EXECUTION -> DONE (skipping validation)
- Any state -> DONE without validation executed

### Stop Conditions & Red Flags

**Stop when:** Assumption count >=3 on critical path | Same approach tried twice without new rationale | Evidence contradicts hypothesis | Uncertain whether code or test expectation is wrong

**USV Red Flags:** STFT parameter changes without explaining frequency resolution impact | Detection threshold changes without baseline comparison | Modifying test expected values to pass | Any change to `energy_detector.py` without DSP review

### Core Rules

#### Integrity (Never Violated)
- **No fabrication**: "I believe the file contains..." = READ IT FIRST
- **No test corruption**: Never modify test expectations to pass. Fix code or DISCUSS.
- **No false completion**: Don't claim "done" without running validation
- **No silent scope creep**: One logical change per approval

#### Learning Mode (Always Active)
- **Explain the "why"**: Don't just give code -- explain reasoning behind decisions
- **Teach concepts**: When touching DSP/signal processing, explain the math intuitively
- **Surface trade-offs**: When multiple approaches exist, explain pros/cons
- **Connect to bigger picture**: How does this change fit the overall architecture?

#### Epistemic Honesty
- **State uncertainty**: "I'm not sure, but..." is better than confident wrongness
- **Label assumptions**: Every assumption should be visible, not hidden
- **Cite sources**: When referencing signal processing concepts, point to where user can learn more

#### Knowledge Activation
- **Search before reasoning**: Before explaining, analyzing, or modifying domain-specific systems,
  search the vault (topic-map traversal + ripgrep). Filter: "Would the vault plausibly change my answer?"
- **Retrieval hierarchy:** (1) Topic map traversal via `vault-search.mjs` — "what do I know about X?"
  (2) ripgrep keyword search `rg -il "term" notes/` — "find the note mentioning X"
  (3) Wiki-link traversal from current notes — "what's related to what I'm reading?"
  (4) qmd BM25 fallback (available but not primary)
- **Modifications**: /kcheck mandatory for HIGH-risk canary files, recommended for constrained systems.
- **Skip for**: pure code mechanics, general knowledge, test files, documentation-only changes.

### Approval Request Format

**Before any code changes**, present: Intent, Context, Scope, Plan (numbered steps with "why"), Assumptions, Risks, Validation, Learning opportunity. End with "Proceed?"

For trivial changes: `Quick fix: [what] in [file]. Proceed?`

Full template: `docs/workflow/approval-request-template.md`

### Test Protocol (Anti-Greenwashing)

| Code State | Test Result | Action |
|------------|-------------|--------|
| Correct | Pass | Good |
| Buggy | Fail | Good (bug exposed) - fix code |
| Correct | Fail | Discuss - test expectations may be wrong |
| Buggy | Pass | **DANGEROUS** - tests not catching bug |
| Unknown | Fail | **STOP** - don't assume which is wrong, discuss |

**NEVER modify test expected values to make tests pass without discussion.**
Pre-existing test files (from `test-architect`) are treated as spec — do NOT modify their expectations during implementation without discussion.

---

## Project Overview

USV Spectrogram Generator - Python tools for analyzing ultrasonic vocalization (USV) recordings at 300 kHz. Includes spectrogram generation, tiled PNG rendering, Zarr storage, USV detection pipeline, Streamlit-based Parameter Lab, and candidate labeling tool.

### Production Model (2026-04-01)

The current production CNN is `models/hard_neg_retrain/best_model.pt` — retrained with 620 hard negatives + 144 hard positives. Full pipeline results at `docs/handoffs/v2-full-pipeline-results.md`. Key stats: precision 90.55% (+3.35%), 16/18 known noise files eliminated, 98.7% USV rate in manual review tier. The PyQt6 app defaults to this model.

**DEPRECATED — do NOT use:** `models/matched_windows/best_model.pt` or `models/production/best_model.pt`. These are older models kept only as baselines.

### Running Batch Detection

**Always use this exact pipeline** when running detection on any WAV folder:

```bash
.venv/bin/python scripts/run_batch_detection.py \
    --wav-dir <WAV_FOLDER>/ \
    --model models/hard_neg_retrain/best_model.pt \
    --output-dir results/batch_<NAME>/ \
    --temperature models/hard_neg_retrain/temperature.json \
    --fp-filter models/hard_neg_retrain/fp_filter.pkl \
    --hysteresis-config models/hard_neg_retrain/hysteresis_optimization_v2.json \
    --workers 4
```

All five flags (model, temperature, fp-filter, hysteresis-config, workers) are required for correct results. Omitting `--fp-filter` or `--hysteresis-config` produces an incomplete pipeline with unreliable triage.

## Environment Setup

```powershell
.\.venv\Scripts\python.exe <script>
.\.venv\Scripts\python.exe -m pytest tests/ -v
.\.venv\Scripts\python.exe -m py_compile <file.py>
```

WAV files: No single canonical directory. Recordings span multiple locations (e.g., `USV5/usv_lmt_034/`, `USV_3452_sample_reviewed/`). Use `--wav-search-dirs` in `scripts/unify_labels.py` to resolve paths. Legacy fallback: `$env:USV_WAV_DIR`.

## Compaction Preservation
When compacting or summarizing this conversation, ALWAYS preserve:
- All file paths modified and what changed in each
- Current task, its phase, and completion status
- Failing test output or error messages still being debugged
- Architectural decisions made this session
- Active debugging hypotheses
Do NOT discard line numbers, variable names, or function signatures under active discussion.

## Context Decay
After 10+ messages: re-read any file before editing it.
After any compaction event: treat ALL file memory as stale.
Never edit from memory alone in a long session.

## Verification Protocol
After every code change, before reporting success:
1. `python -m py_compile <modified_file>` (syntax)
2. `pytest <relevant_test> -x -q` if tests exist for modified code
3. If no tests exist, state: "No test coverage for this change"
Never say "Done" or "Complete" with failing checks.
NOTE: mypy is not configured. Do not claim type-safety without it.

## Large File Protocol
These files exceed 1000 LOC and MUST be read in chunks:
- `main_window.py` (1,819 lines) -- read in 500-line segments
- `assembler.py` (1,490 lines)
- `repertoire_stats.py` (1,142 lines)
- `test_fp_filter.py` (1,119 lines)
- `information_theory.py` (1,075 lines)
For ANY file over 500 LOC: use offset/limit to read in chunks.
Never assume a single read captured the full file. State the total line count after your first read.

## Project Structure

See `docs/architecture/project-structure.md` for the full directory tree.
Task routing and key reference documents: `.claude/rules/task-routing.md`.

> **NOT for agents:** `docs/human/PROJECTS.md` and `docs/human/DECISIONS.md` are human-readable dashboards regenerated by `/refresh-human-docs`. Agents should use `ops/goals.md` and `notes/` instead.

---

## Agents

See `AGENTS.md` for the full agent table and testing workflow sequence. **Using appropriate agents is required, not optional.**

---

## Signal Processing Conventions

See `DECISIONS.md` ADR-001 (sample rate) and ADR-002 (STFT parameters) for full details.

**Key rule:** Always specify `sr=300000` explicitly. Never rely on library defaults. See ADR-001 for why.

---

## Quick Commands

| Phrase | Effect |
|--------|--------|
| "Explain..." | Teaching mode - prioritize understanding over speed |
| "Why?" | Expand on reasoning for last decision |
| "Proceed" / "P" | Approval granted |
| "Let's figure this out" | Pairing mode |
| "Fresh eyes" | Restart reasoning from evidence |
| "5 Whys" | Root cause analysis before any fix |

---

## Common Mistakes to Avoid

- Don't use librosa's default sample rate - always specify sr=300000
- Don't forget to handle edge cases for short audio segments
- Always verify FFT parameters match expected frequency resolution
- Don't claim completion without running py_compile and tests
- Don't modify test expectations to pass without discussion
- Don't cite documentation status to contradict user claims without verifying CODE first -- when docs contradict user, grep the codebase (code is truth, docs may be stale)

### Vault Canary Comments
- Source files with known regression history or non-obvious invariants have `# VAULT:` comments referencing knowledge graph notes
- HIGH-risk files (5) include `# Run /kcheck before modifying this file.` — this is mandatory
- MEDIUM-risk files (1) have canary references only — `/kcheck` recommended but not required
- Registry: `ops/vault-canary-map.md` — audit periodically to ensure referenced notes are still current
- When adding canaries to new files: place after module docstring, before first import

### Git Data Safety
- **NEVER use `git add -A` or `git add .` without reviewing `git status` first** — bulk staging can record accidental deletions of data directories (this happened to `USV_Detections/` in commit 78d1c70, deleting 656 files)
- **Before any "cleanup" commit**, run `git diff --cached --stat` and check for unexpected deletions — hundreds of deletions is a red flag
- **Always stage specific files by name** for data directories like `USV_Detections/`, `5970 USV/`, training data, or model artifacts
- If data goes missing locally, check `git log -- <path>` — it may still exist in history and can be restored with `git checkout <commit> -- <path>`

## Codex Handoff Vault Search
Before writing any Codex task spec in `docs/handoffs/`, search the vault for constraints:
1. Run `node ops/scripts/vault-search.mjs --query "task description"` or `rg -il "key terms" notes/` (or `/kcheck`) to find relevant constraint notes
2. Extract up to 5 constraints relevant to files the task will modify
3. Flatten each constraint into plain text in the handoff's "Relevant Constraints" section (Codex has no vault access)
4. Use `templates/codex-handoff.md` for the handoff structure

---

# Knowledge Graph

**If it won't exist next session, write it down now.** You operate a knowledge network: notes (external memory), wiki-links (connections), topic maps (attention managers). Full conventions: `docs/workflow/knowledge-graph-reference.md`.

## Session Rhythm

Every session follows: **Orient -> Work -> Persist**
- **Orient**: Read ops/goals.md, ops/reminders.md, ops/session-relevance.md, check condition triggers
- **Work**: Vault search before domain reasoning. Surface connections. Write discoveries immediately.
- **Persist**: Write atomic notes, update topic maps, update ops/goals.md

## Where Things Go

| Content Type | Destination |
|-------------|-------------|
| Knowledge claims, insights | notes/ |
| Raw material to process | inbox/ |
| Time-bound commitments | ops/reminders.md |
| Processing state, queue, config | ops/ |
| Friction signals, patterns | ops/observations/ |
| Methodology self-knowledge | ops/methodology/ |
| arscontexta reference claims (READ-ONLY) | methodology/ |
| arscontexta structured reference (READ-ONLY) | reference/ |

Durable knowledge -> notes/. Temporal coordination -> ops/.
**methodology/** and **reference/** are READ-ONLY. Do NOT write to either directory.

## Critical Rules

- **NEVER write directly to notes/.** Route: inbox/ -> /reduce -> notes/.
- Pipeline: /seed -> /reduce -> /reflect -> /reweave -> /verify.
- Before any topic map split/merge/creation: consult `arscontexta-expert` FIRST.

## Maintenance Triggers

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Orphan notes | > 7 days | /reflect |
| Dangling links | Any | Fix immediately |
| Stale notes | > 30 days + < 2 incoming links | /reweave |
| Topic map oversized | > 40 notes | Split |
| Inbox items | >= 3 | /reduce or /pipeline |
| Pending observations | >= 7 (or any > 14 days) | /rethink |
| Open tensions | >= 5 | /rethink |
| Unprocessed sessions | >= 5 | /remember --mine-sessions |

Health checks: `/arscontexta:health` (quick | full | three-space).

## Guardrails

- Never store content user asks to forget
- Never infer/record unshared information
- Present inferences as patterns, not facts
- Source attribution required -- trace claims to origins
- On friction: /remember -> continue -> 3+ occurrences -> propose CLAUDE.md update
