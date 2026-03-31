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

## Environment Setup

```powershell
.\.venv\Scripts\python.exe <script>
.\.venv\Scripts\python.exe -m pytest tests/ -v
.\.venv\Scripts\python.exe -m py_compile <file.py>
```

WAV files: No single canonical directory. Recordings span multiple locations (e.g., `USV5/usv_lmt_034/`, `USV_3452_sample_reviewed/`). Use `--wav-search-dirs` in `scripts/unify_labels.py` to resolve paths. Legacy fallback: `$env:USV_WAV_DIR`.

## Project Structure

See `docs/architecture/project-structure.md` for the full directory tree.
Key entry points are listed in the Task Routing table below.

## Task Routing

| Task Type | Start With | Reference Doc |
|-----------|-----------|---------------|
| Spectrogram / STFT changes | `spectrogram.py`, `_stft_core.py`, `config.py` | `docs/reference/usv_signal_processing_reference.md` |
| Detection pipeline | `detection/energy_detector.py`, `detection/config.py` | `docs/modules/energy-detector.md` |
| CNN training / evaluation | `models/cnn_classifier.py`, `models/trainer.py` | `docs/modules/cnn-classifier.md` |
| Training data assembly | `dataset/assembler.py`, `scripts/assemble_training_data.py` | `docs/modules/dataset-assembler.md` |
| PyQt6 desktop app | `app/main_window.py`, `app/core/`, `app/widgets/` | `docs/plans/USV_DETECTION_APP_IMPLEMENTATION.md` |
| Labeling tool (Streamlit) | `labeling/labeling_app.py` | `docs/LABELING_TOOL_QUICKSTART.md` |
| Parameter Lab (Streamlit) | `param_lab/app.py`, `param_lab/ui/` | — |
| Clustering / repertoire | `clustering/`, `classification/repertoire_stats.py` | `docs/modules/repertoire-stats.md` |
| DeepSqueak / Raven bridge | `classification/raven_export.py`, `classification/deepsqueak_import.py` | `docs/modules/raven-export.md`, `docs/modules/deepsqueak-import.md` |
| VQ-VAE / Transformer | `usv_language/models/`, `usv_language/training/` | `docs/plans/vq_vae_transformer_plan.md` |
| LMT behavioral integration | `lmt/`, `scripts/run_event_triggered_analysis.py` | `docs/modules/event-triggered-analysis.md` |
| Script index (all ~76) | — | `docs/scripts-index.md` |

All `src/` paths above are relative to `src/usv_spectrogram/` unless they start with `usv_language/`.

---

## Key Reference Documents

| Document | When to Read |
|----------|--------------|
| `ops/goals.md` | **Start of every session** (session state, active threads) |
| `notes/index.md` + topic maps | **Before any architectural/design choice** (domain knowledge) |
| `ROADMAP*.md` / plan files | Before implementing — check relevant plan (no single master ROADMAP) |
| `docs/architecture/patterns.md` | Before implementing (follow established patterns) |
| `docs/workflow/completion-sequence.md` | When implementing 2+ file changes (includes handoff rules) |
| `docs/reviews/REVIEW-TEMPLATE.md` | When writing handoff or requesting review (includes tier system) |
| `docs/workflow/approval-request-template.md` | Full approval request + struggle protocol templates |
| `docs/workflow/knowledge-graph-reference.md` | Full verbose KG section details |
| `docs/plans/USV_TRAINING_PIPELINE_PLAN.md` | Building training data generation pipeline |
| `docs/plans/USV_DETECTION_APP_IMPLEMENTATION.md` | Building PyQt6 desktop app for detection |
| `docs/reference/usv_signal_processing_reference.md` | Any signal processing work |
| `IMPLEMENTATION_PROGRESS.md` | **Append after implementation** (session archive, never modify existing entries) |

**After implementing a module**, also update: module doc (`docs/modules/<module>.md`), `docs/architecture/patterns.md` (if new pattern), create decision note in `notes/` + run `/reflect` (if non-obvious architectural decision).

> **NOT for agents:** `docs/human/PROJECTS.md` and `docs/human/DECISIONS.md` are human-readable dashboards regenerated by `/refresh-human-docs`. Agents should use `ops/goals.md` and `notes/` instead.

---

## Project-Specific Agents

| Task | Agent | When to Use |
|------|-------|-------------|
| Review STFT/DSP/math changes | `dsp-reviewer` | ANY change to energy computation, FFT, dB scaling |
| Implement Streamlit UI | `streamlit-expert` | ANY Streamlit UI work |
| Write pre-implementation tests | `test-architect` | BEFORE implementing a new module (reads ROADMAP spec) |
| Harden tests post-implementation | `test-hardener` | AFTER implementation passes review (finds coverage gaps) |
| Validate detection changes | `detection-validator` | ANY change to detection logic |
| Final review before commit | `pr-reviewer` | Before telling user "done" |
| KG architecture decisions | `arscontexta-expert` | Topic map strategy, note schema, methodology questions |

**Using appropriate agents is required, not optional.**

**Testing workflow sequence:** For new modules, the full test lifecycle is:
1. `roadmap-from-plan` generates the ROADMAP with test plan
2. `test-architect` writes failing tests from the spec (optional but recommended)
3. `/implement` builds the module (uses pre-existing tests if available, step 3.5)
4. `master-reviewer` reviews implementation and tests
5. `test-hardener` finds remaining coverage gaps (after review approval, phase 4.5)

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

## Philosophy

**If it won't exist next session, write it down now.** You are the primary operator of this knowledge system -- the agent who builds, maintains, and traverses a knowledge network. Notes are your external memory. Wiki-links are your connections. Topic maps are your attention managers.

## Discovery-First Design

**Every note must be findable by a future agent who doesn't know it exists.** Before writing to notes/, verify:
1. **Title as claim** -- reads naturally when linked: `since [[title]]`
2. **Description quality** -- adds info beyond the title
3. **Topic map membership** -- linked from at least one topic map
4. **Composability** -- linkable without dragging irrelevant context

## Session Rhythm

Every session follows: **Orient -> Work -> Persist**

- **Orient**: Read ops/goals.md, ops/reminders.md, ops/session-relevance.md (auto-generated), check condition triggers
- **Work**: Do the task. **Vault search before domain reasoning (Core Rules → Knowledge Activation).** Surface connections. Write down discoveries immediately.
- **Persist**: Write new insights as atomic notes, update topic maps, update ops/goals.md, capture methodology learnings

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

**methodology/** (249 research claims, READ-ONLY) — search here when explaining WHY a vault convention exists, justifying a KG design decision, or grounding a skill in research. **reference/** (routing indexes, constraints, READ-ONLY) — used by /ask, /architect, /recommend, /health. Do NOT write to either directory.

## Atomic Notes

One claim per note. Title IS the claim (composability test: "This note argues that [title]").
Full conventions + examples: `docs/workflow/knowledge-graph-reference.md` § Atomic Notes.

## Wiki Links

`[[title]]` basic | `since [[claim]]` as prose | Target: 3+ outgoing links/note. Never rename manually (`ops/scripts/rename-note.sh`).
Full conventions: `docs/workflow/knowledge-graph-reference.md` § Wiki Links.

## Topic Maps (MOCs)

Three-tier: `index.md -> topic maps -> notes`. Context phrases required on all links. Split at ~50 notes.
**Before any topic map split/merge/creation:** consult `arscontexta-expert` FIRST.
Full conventions + context phrase examples: `docs/workflow/knowledge-graph-reference.md` § Topic Maps.

## Processing Pipeline

**NEVER write directly to notes/.** Route: inbox/ -> /reduce -> notes/. Phases: /seed -> /reduce -> /reflect -> /reweave -> /verify.
Full phase details + extraction categories: `docs/workflow/knowledge-graph-reference.md` § Processing Pipeline.

## Semantic Search (qmd)

Two discovery layers: **wiki links** (explicit, curated) + **qmd semantic search** (implicit, content-based).
- Known connection -> wiki link
- Discovery -> semantic search
- Verification -> both (find missed connections)

## Schema

Required: `description` (max 200 chars), `topics` (wiki links to topic maps). Domain: `type`, `confidence`, `conditions`, `meta_state`.
Full field specs + templates: `docs/workflow/knowledge-graph-reference.md` § Schema.

## Maintenance

Condition-based, not scheduled. Specific conditions trigger specific actions.

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Orphan notes | Any persistent (> 7 days) | Run /reflect on orphaned notes |
| Dangling links | Any | Fix broken references immediately |
| Stale notes | > 30 days old + < 2 incoming links | Run /reweave |
| Topic map oversized | > 40 notes | Split into sub-topic-maps |
| Inbox items | >= 3 | Run /reduce or /pipeline |
| Pending observations | >= 7 (or any observation > 14 days unreviewed) | Run /rethink |
| Open tensions | >= 5 | Run /rethink |
| Unprocessed sessions | >= 5 | Run /remember --mine-sessions |

Health checks: `/arscontexta:health` (quick | full | three-space).

## Operational Learning Loop

Observations (ops/observations/) — friction, surprises, process gaps. Tensions (ops/tensions/) — contradictions.
Triggers: 7+ observations (or any > 14 days old) -> /rethink. 5+ tensions -> /rethink.

> **Reference details** — Operational Space (ops/ directory tree), Templates (4 template files), Graph Analysis (graph operations + /graph command), Research Provenance (source chain), Helper Functions (vault scripts): see `docs/workflow/knowledge-graph-reference.md`.

## Self-Improvement

On friction: /remember to capture observation -> continue work -> 3+ occurrences -> propose CLAUDE.md update. User says "remember this" -> update immediately.

## Guardrails

- Never store content user asks to forget
- Never infer/record unshared information
- Present inferences as patterns, not facts
- Source attribution required -- trace claims to origins

## Common Pitfalls

- **Collector's Fallacy**: Process before capturing more. Inbox >= 3 -> /reduce first.
- **Orphan Drift**: /reflect after note creation batches. No orphan > 7 days.
- **Verbatim Risk**: Restate insights in your own framing. No paper-abstract copying.
- **Topic Map Sprawl**: Start with 4-5 broad maps. Split only at ~35 notes.
