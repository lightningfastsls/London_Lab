# CLAUDE.md Removed Content Archive

Created: 2026-02-20
Purpose: Safety net for the CLAUDE.md slim-down. Every section removed or condensed
is preserved here verbatim so it can be restored if a behavior regresses.

---

## REMOVED: Recently Created Skills (lines 829-848)

```markdown
## Recently Created Skills (Pending Activation)

Skills created during /setup are listed here until confirmed loaded. After restarting Claude Code, the SessionStart hook verifies each skill is discoverable and removes confirmed entries.

- /reduce -- Extract insights from source material (created 2026-02-18)
- /reflect -- Find connections between notes (created 2026-02-18)
- /reweave -- Update old notes with new context (created 2026-02-18)
- /verify -- Quality-check notes (created 2026-02-18)
- /validate -- Schema validation (created 2026-02-18)
- /seed -- Research and deposit sources (created 2026-02-18)
- /ralph -- Orchestrated processing (created 2026-02-18)
- /pipeline -- End-to-end pipeline (created 2026-02-18)
- /tasks -- Task queue management (created 2026-02-18)
- /stats -- Vault metrics (created 2026-02-18)
- /graph -- Graph analysis (created 2026-02-18)
- /next -- Next action recommendation (created 2026-02-18)
- /learn -- Research and grow graph (created 2026-02-18)
- /remember -- Capture friction (created 2026-02-18)
- /rethink -- Triage observations/tensions (created 2026-02-18)
- /refactor -- Restructure notes (created 2026-02-18)
```

**Reason removed:** All 16 skills confirmed working since 2026-02-18. Dead weight.

---

## REMOVED: Token Usage (lines 278-280)

```markdown
## Token Usage

See `docs/workflow/token-optimization.md`. Key rules: use haiku for exploration, sonnet for implementation, don't re-read files already in context.
```

**Reason removed:** Pointer-to-a-pointer. No behavioral enforcement.

---

## REMOVED: Collaboration Modes (lines 130-137)

```markdown
### Collaboration Modes

| Mode | Trigger | Behavior |
|------|---------|----------|
| **Autonomous** | Default | Full approval request → execute → validate → report |
| **Teaching** | "Explain..." or complex DSP | Prioritize explanation over code, use analogies |
| **UserDuck** | "Let me think aloud" | You explain your reasoning, I redirect/question |
| **Pairing** | "Let's figure this out" | Back-and-forth exploration, neither drives exclusively |
```

**Reason removed:** Redundant with Quick Commands table. Moved to docs/workflow/approval-request-template.md.

---

## REMOVED: Struggle Protocol (lines 114-128)

```markdown
### Struggle Protocol

When stuck, don't spiral. STOP and surface it:

\```
🚨 BLOCKED

**What I understand**: [specific]
**What I tried**: [list with outcomes]
**Where I'm stuck**: [specific blocker]
**What would help**: [specific request]
**Learning angle**: [Is there a concept here worth exploring together?]
\```

This is collaboration, not failure. Hiding struggle IS failure.
```

**Reason removed:** Core behavior (surface struggle) is in Core Rules. Format template moved to docs/workflow/approval-request-template.md.

---

## REMOVED: Session Workflow (lines 189-219)

```markdown
## Session Workflow

### 0. On Session Start
1. Read this contract
2. Read `IMPLEMENTATION_PROGRESS.md` for current state
3. Read `DECISIONS.md` for architectural constraints
4. If implementing a module: read `ROADMAP.md` for the spec
5. If working on detection app: read `docs/plans/USV_DETECTION_APP_IMPLEMENTATION.md`
6. If working on training pipeline: read `docs/plans/USV_TRAINING_PIPELINE_PLAN.md`
7. Build mental models (DoR, DoD, Stop Conditions, Red Flags)

### 1. Before Implementation (REQUIRED)
- Use Plan Mode / Approval Request for non-trivial tasks
- Explain the approach and why (learning mode)

### 2. During Implementation
- Keep diffs small and focused
- Run `py_compile` after every edit
- Use subagents for their specialties (see below)
- Explain what you're doing as you go

### 3. After Implementation
- Update `IMPLEMENTATION_PROGRESS.md`
- Run tests to verify no regressions
- Summarize what was learned/changed

### 4. Validation (Before "Done")
- Run `detection-validator` for detection algorithm changes
- Run `dsp-reviewer` for STFT/signal processing changes
- Run `pr-reviewer` for final quality check
```

**Reason removed:** Duplicated by Key Reference Documents table + Mandatory Workflow + Core Rules + Agent table.

---

## REMOVED: Documentation: Document As You Build (lines 250-262)

```markdown
## Documentation: Document As You Build

### Required Docs Per Module
1. **Module doc** (`docs/modules/<module>.md`) — purpose, public interface, usage, decisions
2. **Architecture patterns** (`docs/architecture/patterns.md`) — update if you establish a new pattern
3. **ADR** (`DECISIONS.md`) — add if you make a non-obvious architectural decision

### Before Building a New Module
1. Read `ROADMAP.md` for the module spec and dependencies
2. Read `DECISIONS.md` for architectural constraints
3. Read `docs/architecture/patterns.md` for established patterns
4. Read `docs/modules/*.md` for any module you'll interact with
```

**Reason removed:** Duplicated by Key Reference Documents table + completion sequence doc.

---

## REMOVED: Model Selection Guide (lines 265-275)

```markdown
## Model Selection Guide

| Task Type | Model | Rationale |
|-----------|-------|-----------|
| Planning & Architecture | `opus` | Complex reasoning, design decisions |
| Algorithm Implementation | `sonnet` | Good balance of capability and speed |
| Code Reviews | `sonnet` | Thorough analysis needed |
| Documentation Writing | `haiku` | Fast, straightforward task |
| Simple Edits/Fixes | `haiku` | Quick, low complexity |
| Codebase Exploration | `haiku` | Fast searches, no complex reasoning |
```

**Reason removed:** Can't switch own model. Subagent definitions in agent configs handle this.

---

## CONDENSED: Mental Models — Definition of Ready (lines 42-47)

```markdown
**Definition of Ready (before proposing changes):**
- Intent clear (feature / bugfix / refactor / exploration)
- Target files identified
- Success criteria observable
- Assumptions stated and counted (max 2 on critical path)
- Scope bounded (what's IN and what's OUT)
```

**Reason condensed:** Covered by Approval Request format fields.

---

## CONDENSED: Mental Models — Definition of Done (lines 49-54)

```markdown
**Definition of Done (before declaring complete):**
- Code complete per approval
- py_compile passes on touched files
- Tests pass (or tests written if new behavior)
- IMPLEMENTATION_PROGRESS.md updated
- User can verify the change works
```

**Reason condensed:** Covered by completion sequence doc.

---

## CONDENSED: Approval Request Format (lines 87-112)

Original had full code-block template with bracketed placeholders and comments.
Condensed to field list only. Full template moved to docs/workflow/approval-request-template.md.

---

## CONDENSED: Implementation Completion Sequence (lines 222-231)

```markdown
## Implementation Completion Sequence

**Non-negotiable** for any task creating/modifying 2+ files. Full details: `docs/workflow/completion-sequence.md`

**Key steps:** Create tasks (including handoff task) -> Write code -> Run module tests -> Run full suite -> Fix failures -> Write handoff (`docs/reviews/<module>-handoff.md`) -> Report

**Critical rule:** The handoff task must be created at the START as a TaskCreate item. It persists in the task list as a visible reminder even after 50+ tool calls of test debugging.

**When to skip:** Single-file changes, documentation-only, exploratory tasks.
```

**Reason condensed:** Replaced with pointer in Key Reference Documents table.

---

## CONDENSED: Module Reviews (lines 234-247)

```markdown
## Module Reviews

Reviews use a **tiered system** matched to module complexity. Full templates: `docs/reviews/REVIEW-TEMPLATE.md`

| Tier | For | Budget | Model |
|------|-----|--------|-------|
| 1 — Housekeeping | Config, cleanup, small fixes | 10 calls | Sonnet |
| 2 — Standard | New modules, scripts, pipelines | 30 calls | Sonnet |
| 3 — Critical | ML models, DSP changes, detection algorithm | 60 calls | Sonnet |

**Workflow:** Implementor writes handoff -> main session spawns master-reviewer -> main session writes review file -> implementor fixes issues.

**Rule:** Handoff is mandatory input for review. Never skip it.
```

**Reason condensed:** Replaced with pointer in Key Reference Documents table.

---

## KG REMOVED: Self-Evolution (lines 606-621)

```markdown
## Self-Evolution

This system evolves through use. Configuration is a starting point, not a destination.

### Expect These Changes
- **Schema expansion** -- New fields when a querying need emerges
- **Topic map splits** -- When a topic exceeds ~35 notes
- **Processing refinement** -- Pipeline patterns encoded as methodology updates
- **New note types** -- Tension notes, synthesis notes, methodology notes

### Signs of Friction (act on these)
- Notes accumulating without connections -> increase connection-finding frequency
- Can't find what you know exists -> add more topic map structure
- Schema fields nobody queries -> remove them
- Processing feels perfunctory -> simplify the cycle
```

**Reason removed:** Aspirational guidance with no behavioral enforcement.

---

## KG REMOVED: Task Management (lines 651-658)

```markdown
## Task Management

### Processing Queue (ops/queue/)
Pipeline tasks tracked in JSON. Each note gets one queue entry that progresses through phases (create -> reflect -> reweave -> verify). Fresh context per phase ensures quality.

### Maintenance Queue
Maintenance work lives alongside pipeline work in the same queue. /next evaluates conditions against vault state: fired conditions create maintenance queue entries, satisfied conditions auto-close them.
```

**Reason removed:** Queue mechanics are in /tasks and /ralph skills.

---

## KG REMOVED: Infrastructure Routing (lines 682-696)

```markdown
## Infrastructure Routing

When users ask about system structure, schema, or methodology:

| Pattern | Route To |
|---------|----------|
| "How should I organize/structure..." | /arscontexta:architect |
| "Research best practices for..." | /arscontexta:ask |
| "What does my system know about..." | Check ops/methodology/ directly |
| "I want to add a new area/domain..." | /arscontexta:add-domain |
| "What should I work on..." | /arscontexta:next |
| "Help / what can I do..." | /arscontexta:help |
| "Walk me through..." | /arscontexta:tutorial |
| "Research / learn about..." | /arscontexta:learn |
```

**Reason removed:** Skill routing is in skill descriptions already loaded into context.

---

## KG REMOVED: Self-Extension (lines 781-795)

```markdown
## Self-Extension

### Building New Skills
Create `.claude/skills/skill-name/SKILL.md` with YAML frontmatter, instructions, quality gates, and output format.

### Building Hooks
Create `.claude/hooks/` scripts triggered on SessionStart, PostToolUse (Write), or Stop events.

### Extending Schema
Add domain-specific YAML fields to templates. Base fields (description, type) are universal. Add fields that make YOUR notes queryable for YOUR use case.

### Growing Topic Maps
When a topic map exceeds ~35 notes, split it. Create sub-topic-maps that link back to the parent.
```

**Reason removed:** Only relevant when building extensions. Not default behavior.

---

## KG REMOVED: Derivation Rationale (lines 813-825)

```markdown
## Derivation Rationale

This knowledge system was derived on 2026-02-18 using the Research preset with these key choices:

- **Granularity: Atomic** -- One claim per note for maximum composability across detection, classification, DSP, and training domains
- **Organization: Flat** -- Topics cross-cut; folders would force artificial hierarchy
- **Linking: Explicit+Implicit** -- Wiki links + qmd semantic search for discovery
- **Processing: Heavy** -- Full pipeline with all quality gates from day one
- **Self-space: Disabled** -- Goals route to ops/goals.md; identity in this context file
- **Semantic search: qmd** -- Opted in for implicit connection discovery

Full derivation record: `ops/derivation.md`
Configuration: `ops/config.yaml`
```

**Reason removed:** Duplicates ops/derivation.md verbatim.

---

## KG CONDENSED: Various sections

The following KG sections were condensed (not removed). Full versions are in
docs/workflow/knowledge-graph-reference.md:

- Philosophy (12 → ~5 lines)
- Discovery-First Design (12 → ~5 lines)
- Session Rhythm (21 → ~8 lines)
- Where Things Go (12 → ~10 lines)
- Atomic Notes (22 → ~10 lines)
- Wiki Links (18 → ~8 lines)
- Topic Maps (30 → ~12 lines)
- Processing Pipeline (28 → ~10 lines)
- Semantic Search (25 → ~6 lines)
- Schema (19 → ~8 lines)
- Maintenance (32 → ~15 lines)
- Vault Self-Knowledge (9 → ~2 lines)
- Operational Learning Loop (13 → ~6 lines)
- Operational Space (18 → ~8 lines)
- Templates (11 → ~4 lines)
- Graph Analysis (16 → ~4 lines)
- Research Provenance (7 → ~2 lines)
- Helper Functions (15 → ~5 lines)
- Self-Improvement (7 → ~3 lines)
- Guardrails (8 → ~4 lines)
- Common Pitfalls (12 → ~6 lines)

---

## MEMORY.md Removed Lines

```markdown
  - Binary: `node D:/npm-global/node_modules/@tobilu/qmd/dist/qmd.js`
  - MCP config: `.mcp.json` uses `node` command directly + `XDG_CACHE_HOME=D:\qmd-cache` env var
  - Index: `D:\qmd-cache\qmd\index.sqlite` (consolidated on D:)
  - Models: `D:\qmd-cache\models\` (junction from `C:\Users\light\.cache\qmd`)
  - PowerShell: set `$env:XDG_CACHE_HOME = 'D:\qmd-cache'` and `$env:HOME = $env:USERPROFILE` (needed for git)
  - After adding notes: run `qmd update && qmd embed`
- **Hook invocation**: `.cmd` wrappers call `powershell.exe` via `cmd.exe /c .claude\\hooks\\<name>.cmd`. SessionStart uses direct `powershell.exe ... || true`. All `.ps1` scripts set `$ErrorActionPreference = 'SilentlyContinue'` and log errors to stderr. Stop hooks read stdin via `[Console]::In.ReadToEnd()` + `ConvertFrom-Json`.
```

**Reason removed:** Implementation details rarely needed session-to-session. Retrievable via `.mcp.json` and hook files directly.
