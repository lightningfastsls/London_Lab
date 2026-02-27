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

---

## Project Overview

USV Spectrogram Generator - Python tools for analyzing ultrasonic vocalization (USV) recordings at 300 kHz. Includes spectrogram generation, tiled PNG rendering, Zarr storage, USV detection pipeline, Streamlit-based Parameter Lab, and candidate labeling tool.

## Environment Setup

```powershell
.\.venv\Scripts\python.exe <script>
.\.venv\Scripts\python.exe -m pytest tests/ -v
.\.venv\Scripts\python.exe -m py_compile <file.py>
```

WAV files: `$env:USV_WAV_DIR` or fallback `<repo>/5970 USV`

## Project Structure

```
src/usv_spectrogram/       # Core library
  config.py                # SpectrogramConfig dataclass
  io_wav.py                # WAV loading utilities
  spectrogram.py           # STFT computation
  detection/               # USV detection pipeline
    config.py              # DetectionConfig dataclass
    candidate.py           # Candidate dataclass
    energy_detector.py     # EnergyDetector class
  param_lab/               # Streamlit app modules
  labeling/                # USV labeling tool
    labeling_app.py        # Streamlit labeling UI

scripts/                   # Entry points
  usv_labeling_tool.py     # Labeling tool launcher
tests/                     # Test files
methodology/               # arscontexta reference graph (249 research claims, READ-ONLY)
reference/                 # arscontexta structured reference docs (routing indexes, constraints, templates)
```

---

## Key Reference Documents

| Document | When to Read |
|----------|--------------|
| `ops/goals.md` | **Start of every session** (session state, active threads) |
| `notes/index.md` + topic maps | **Before any architectural/design choice** (domain knowledge) |
| `ROADMAP.md` | **Before implementing any module** |
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
| Write tests for code | `test-writer` | After implementing new features |
| Validate detection changes | `detection-validator` | ANY change to detection logic |
| Final review before commit | `pr-reviewer` | Before telling user "done" |
| KG architecture decisions | `arscontexta-expert` | Topic map strategy, note schema, methodology questions |

**Using appropriate agents is required, not optional.**

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

### Git Data Safety
- **NEVER use `git add -A` or `git add .` without reviewing `git status` first** — bulk staging can record accidental deletions of data directories (this happened to `USV_Detections/` in commit 78d1c70, deleting 656 files)
- **Before any "cleanup" commit**, run `git diff --cached --stat` and check for unexpected deletions — hundreds of deletions is a red flag
- **Always stage specific files by name** for data directories like `USV_Detections/`, `5970 USV/`, training data, or model artifacts
- If data goes missing locally, check `git log -- <path>` — it may still exist in history and can be restored with `git checkout <commit> -- <path>`

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

- **Orient**: Read ops/goals.md, ops/reminders.md, check condition triggers
- **Work**: Do the task. Surface connections. Write down discoveries immediately.
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

**methodology/ — arscontexta Reference Graph (249 research claims)**
Read-only reference material backing the knowledge management system. Search here when:
- Explaining WHY a vault convention exists (e.g., why atomic notes, why MOCs, why descriptions)
- Justifying a knowledge graph design decision
- Answering questions about note-taking methodology, knowledge systems, or agent cognition
- A skill like /reduce, /reflect, or /reweave needs theoretical grounding

**reference/ — arscontexta Structured Reference Docs (routing indexes, constraints, templates)**
Companion to methodology/. Contains structured reference files that skills like /ask, /architect, /recommend, and /health use as routing indexes into the research graph. Key files: `claim-map.md` (topic→claim routing), `dimension-claim-map.md` (config dimensions→claims), `interaction-constraints.md` (dimension interaction rules), `failure-modes.md` (10 failure patterns), `three-spaces.md` (self/notes/ops boundaries). READ-ONLY.
Do NOT write to this directory. Do NOT mix these files into notes/ graph metrics.

## Atomic Notes

Every note makes exactly one claim. The title IS the claim, written as a complete proposition.

**Composability test:** "This note argues that [title]" -- if it doesn't work, fix the title.

**Good:** `energy detection at 10dB threshold misses low-amplitude calls below 40kHz`
**Bad:** `detection notes` (topic, not claim) | `STFT parameters` (category, not proposition)

## Wiki Links

- `[[note title]]` -- basic | `since [[claim]]` -- as prose (preferred) | `contradicts [[finding]]` -- typed
- Link density target: 3+ outgoing links per note
- Never rename manually -- use `./ops/scripts/rename-note.sh`
- Dangling links = demand signals. Orphan notes = need /reflect.

## Topic Maps (MOCs)

Three-tier navigation: `index.md -> domain topic maps -> individual notes`

**Context phrases required** on topic map links:
Bad: `- [[STFT window size affects frequency resolution]]`
Good: `- [[STFT window size affects frequency resolution]] -- determines temporal/spectral trade-off at 300kHz`

Split when a topic map exceeds ~35 notes.

## Processing Pipeline

**NEVER write directly to notes/.** Route through: inbox/ -> /reduce -> notes/. Direct writes skip quality gates.

Pipeline phases: /seed (research) -> /reduce (extract) -> /reflect (connect) -> /reweave (backward pass) -> /verify (quality check)

Processing depth configured in ops/config.yaml (deep | standard | quick).

## Semantic Search (qmd)

Two discovery layers: **wiki links** (explicit, curated) + **qmd semantic search** (implicit, content-based).
- Known connection -> wiki link
- Discovery -> semantic search
- Verification -> both (find missed connections)

## Schema

Required fields: `description` (max 200 chars, adds context beyond title), `topics` (wiki links to topic maps).

Domain fields: `type` (finding|decision|method|hypothesis|baseline|open-question|pattern), `confidence` (proven|likely|experimental|speculative), `conditions`, `meta_state` (current|outdated|superseded).

Templates in `templates/` are the single source of truth for schema.

## Maintenance

Condition-based, not scheduled. Specific conditions trigger specific actions.

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Orphan notes | Any persistent (> 7 days) | Run /reflect on orphaned notes |
| Dangling links | Any | Fix broken references immediately |
| Stale notes | > 30 days old + < 2 incoming links | Run /reweave |
| Topic map oversized | > 40 notes | Split into sub-topic-maps |
| Inbox items | >= 3 | Run /reduce or /pipeline |
| Pending observations | >= 10 | Run /rethink |
| Open tensions | >= 5 | Run /rethink |
| Unprocessed sessions | >= 5 | Run /remember --mine-sessions |

Health checks: `/arscontexta:health` (quick | full | three-space).

## Vault Self-Knowledge

Methodology knowledge lives in ops/methodology/. Browse it for system configuration rationale.

## Operational Learning Loop

- **Observations** (ops/observations/) -- friction, surprises, process gaps. Category: friction|surprise|process-gap|methodology.
- **Tensions** (ops/tensions/) -- contradictions between notes or implementation vs methodology. Status: pending|resolved|dissolved.
- Triggers: 10+ observations -> /rethink. 5+ tensions -> /rethink.

## Operational Space

```
ops/
+-- derivation.md, derivation-manifest.md, config.yaml
+-- goals.md, reminders.md, tasks.md
+-- methodology/, observations/, tensions/
+-- queue/, sessions/, health/, queries/
```

## Templates

- `templates/note.md` -- Research note
- `templates/topic-map.md` -- Topic map / MOC
- `templates/source-capture.md` -- Inbox source capture
- `templates/observation-note.md` -- Operational observation

## Graph Analysis

Vault = graph database: nodes (markdown), edges (wiki links), properties (YAML frontmatter). Key operations: triangle detection, orphan detection, bridge detection, link density (target: 3+). Use /graph for interactive analysis.

## Research Provenance

Preserve the chain: `source query -> inbox file (metadata preserved) -> /reduce -> notes/`. Every claim traceable to its origin.

## Helper Functions

- **Never rename manually** -- use `./ops/scripts/rename-note.sh "old" "new"`
- Scripts: `orphan-notes.sh`, `dangling-links.sh`, `backlinks.sh`, `link-density.sh`, `validate-schema.sh`

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
