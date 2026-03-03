# Plugin Audit: arscontexta + USV Pipeline Integration

**Generated:** 2026-02-26
**Scope:** arscontexta plugin internals, all skills/agents/commands, typical session walkthrough, friction patterns

---

## Table of Contents

- [Part 1: arscontexta Plugin Internals](#part-1-arscontexta-plugin-internals)
  - [1. Complete File Inventory](#1-complete-file-inventory)
  - [2. Slash Command Reference](#2-slash-command-reference)
  - [3. Session Hooks](#3-session-hooks)
  - [4. Knowledge Graph Pipeline Flow](#4-knowledge-graph-pipeline-flow)
  - [5. Config Schema](#5-config-schema)
  - [6. Hardcoded Assumptions](#6-hardcoded-assumptions)
- [Part 2: All Skills and Agent Definitions](#part-2-all-skills-and-agent-definitions)
  - [7. Complete Skill Inventory](#7-complete-skill-inventory)
  - [8. Agent Definitions](#8-agent-definitions)
  - [9. Command Definitions](#9-command-definitions)
- [Part 3: Typical USV Pipeline Session](#part-3-typical-usv-pipeline-session)
  - [10. Common Task Types](#10-common-task-types)
  - [11. Typical Session Walkthrough](#11-typical-session-walkthrough)
  - [12. Recurring Friction Points](#12-recurring-friction-points)

---

# Part 1: arscontexta Plugin Internals

## 1. Complete File Inventory

The arscontexta plugin (`arscontexta@agenticnotetaking`) is **not a marketplace binary** — it's a locally-generated system. The plugin generates its artifacts directly into the project via a `/setup` command. Files are tagged `generated_from: "arscontexta-v1.6"` in frontmatter.

### Plugin Configuration & Guards

| File | Purpose | Size |
|------|---------|------|
| `.claude/settings.json` | Enables plugin (`arscontexta@agenticnotetaking: true`) | 6 lines |
| `.arscontexta` | Vault guard — hooks check for this before executing | 4 lines |
| `.mcp.json` | Registers qmd MCP server for semantic search | 19 lines |

### Skill Definitions (21 files)

| File | Lines | Model | Context |
|------|-------|-------|---------|
| `.claude/skills/architect/SKILL.md` | ~568 | opus | fork |
| `.claude/skills/ask/SKILL.md` | ~626 | opus | fork |
| `.claude/skills/graph/SKILL.md` | short | sonnet | fork |
| `.claude/skills/health/SKILL.md` | ~200+ | opus | fork |
| `.claude/skills/learn/SKILL.md` | short | (default) | fork |
| `.claude/skills/next/SKILL.md` | short | sonnet | fork |
| `.claude/skills/note-history/SKILL.md` | ~80 | haiku | fork |
| `.claude/skills/pipeline/SKILL.md` | short | sonnet | fork |
| `.claude/skills/ralph/SKILL.md` | short | (default) | fork |
| `.claude/skills/recommend/SKILL.md` | ~200+ | opus | fork |
| `.claude/skills/reduce/SKILL.md` | ~350+ | (default) | fork |
| `.claude/skills/refactor/SKILL.md` | short | (default) | fork |
| `.claude/skills/reflect/SKILL.md` | ~300+ | (default) | fork |
| `.claude/skills/remember/SKILL.md` | short | sonnet | fork |
| `.claude/skills/rethink/SKILL.md` | short | (default) | fork |
| `.claude/skills/reweave/SKILL.md` | short | (default) | fork |
| `.claude/skills/seed/SKILL.md` | short | sonnet | fork |
| `.claude/skills/stats/SKILL.md` | short | sonnet | fork |
| `.claude/skills/tasks/SKILL.md` | short | sonnet | fork |
| `.claude/skills/validate/SKILL.md` | short | sonnet | fork |
| `.claude/skills/verify/SKILL.md` | short | (default) | fork |

**Note on "short" skills:** Several skill files contain only frontmatter + a `[Full ... content]` placeholder, suggesting their full prompts are loaded from the plugin at runtime or were truncated during generation.

### Hook Scripts (12 files: 6 PowerShell + 6 CMD wrappers)

| File | Trigger | Blocking? |
|------|---------|-----------|
| `.claude/hooks/session-orient.ps1` | SessionStart | No (advisory) |
| `.claude/hooks/session-orient.cmd` | (CMD wrapper) | — |
| `.claude/hooks/session-capture.ps1` | Stop | No (advisory) |
| `.claude/hooks/session-capture.cmd` | (CMD wrapper) | — |
| `.claude/hooks/validate-note.ps1` | PostToolUse (Write) | No (advisory) |
| `.claude/hooks/validate-note.cmd` | (CMD wrapper) | — |
| `.claude/hooks/auto-commit.ps1` | PostToolUse (Write, async) | No (advisory) |
| `.claude/hooks/auto-commit.cmd` | (CMD wrapper) | — |
| `.claude/hooks/check_agents_tag.ps1` | Stop | No (advisory) |
| `.claude/hooks/check_agents_tag.cmd` | (CMD wrapper) | — |
| `.claude/hooks/check_plan_mode.ps1` | PreToolUse (Edit\|Write) | No (advisory) |
| `.claude/hooks/check_plan_mode.cmd` | (CMD wrapper) | — |

### Read-Only Reference Material

| Directory | Files | Purpose |
|-----------|-------|---------|
| `methodology/` | 249 `.md` | Research claims (cognitive science, knowledge systems) |
| `methodology/index.md` | 1 | Topic map for methodology claims |
| `reference/` | 20 `.md` + 1 `.yaml` | Structured routing indexes for skills |
| `reference/templates/` | 10 `.md` | Note type templates (base, research, learning, MOC, etc.) |
| `reference/test-fixtures/` | 5 `.md` | Test scenarios for validation |
| `reference/kernel.yaml` | 1 | 15 universal primitives (invariant core) |

### Operational State (arscontexta-managed)

| File/Dir | Purpose |
|----------|---------|
| `ops/config.yaml` | 8-dimension system configuration |
| `ops/derivation.md` | Why each config choice was made |
| `ops/derivation-manifest.md` | Machine-readable manifest for skill runtime |
| `ops/goals.md` | Active goals (read by session-orient hook) |
| `ops/reminders.md` | Time-bound commitments (parsed by hook) |
| `ops/tasks.md` | Task stack |
| `ops/goals-archive.md` | Auto-archived completed goals |
| `ops/last-session.md` | Session bridging context (written by session-capture) |
| `ops/queue/queue.json` | Processing queue state + maintenance thresholds |
| `ops/sessions/*.json` | Session metadata (one per conversation) |
| `ops/health/*.md` | Health check reports |
| `ops/methodology/` | System self-knowledge (2 files) |
| `ops/observations/` | Friction signals (5 files, 3 pending) |
| `ops/tensions/` | Contradictions (6 files, 4 pending) |

### External Dependencies

| Package | Location | Version | Purpose |
|---------|----------|---------|---------|
| `@tobilu/qmd` | `AppData/Roaming/npm/node_modules/` | 1.0.7 | Hybrid search engine (BM25 + vector + LLM rerank) |

### Total File Count

| Category | Count |
|----------|-------|
| Skills | 21 |
| Hooks (PS1 + CMD) | 12 |
| Methodology claims | 249 |
| Reference docs + templates + fixtures | 36 |
| Operational state | ~25 |
| Config/guards | 3 |
| **Total arscontexta-related files** | **~346** |

---

## 2. Slash Command Reference

Every arscontexta skill is user-invocable via `/skill-name`. Here is the detailed reference for each, based on reading their actual SKILL.md implementations.

### Pipeline Commands (Knowledge Processing)

#### `/seed [file]`
- **What it does:** Adds a source file to the processing queue. Checks for duplicates, creates an archive folder structure, moves the source from inbox, creates an "extract" task in the queue, and updates `ops/queue/queue.json`.
- **Reads:** `ops/queue/queue.json` (duplicate check), inbox files
- **Writes:** `inbox/<file>` (moves source), `ops/queue/queue.json` (adds task), archive folder
- **Arguments:** File path to source material
- **Dependencies:** None — entry point of pipeline
- **Model:** sonnet

#### `/reduce [file]`
- **What it does:** The extraction engine. Reads source material and produces atomic notes. The most complex skill — 350+ lines of instructions including "The Comprehensive Extraction Principle" which mandates <10% skip rate on domain-relevant sources. Zero extraction from a domain source is classified as a BUG.
- **Reads:** `ops/derivation-manifest.md` (vocabulary), `ops/config.yaml` (depth, selectivity), `ops/queue/queue.json` (handoff mode), source file, existing notes (dedup check via qmd)
- **Writes:** `notes/*.md` (new atomic notes), enrichment tasks to queue
- **Arguments:** File path or `--handoff` (for queue-driven processing)
- **Dependencies:** Source should be `/seed`'d first (establishes provenance)
- **Processing depth behavior:**
  - deep: Maximum extraction, multiple passes, enrichment tasks for near-duplicates
  - standard: Balanced — extract all core claims, single pass
  - quick: Fast extraction, obvious claims only

#### `/reflect [note]`
- **What it does:** The forward-connection phase. Finds connections between a note and the rest of the vault using dual discovery: (1) browse relevant topic maps for related notes, (2) run semantic search via qmd. Adds inline wiki-links where genuine connections pass the "articulation test" — you must be able to say WHY they connect. Updates topic maps.
- **Reads:** Target note, topic maps, `ops/derivation-manifest.md`, `ops/config.yaml`
- **Writes:** Target note (adds links), topic maps (adds entry with context phrase)
- **Arguments:** `[[note name]]`, "recent"/"new", or empty (prompts for target). `--handoff` for queue mode.
- **Dependencies:** Notes must exist (run after `/reduce`)
- **Uses MCP:** `mcp__qmd__search`, `mcp__qmd__vector_search`, `mcp__qmd__deep_search`

#### `/reweave [note]`
- **What it does:** The backward-connection phase that `/reflect` doesn't do. Revisits EXISTING notes that predate newer content, adds connections from old → new, sharpens claims, considers splits. Asks "what would be different if written today?"
- **Reads:** Target note (or scans for stale candidates), newer notes, topic maps
- **Writes:** Old notes (adds new links), may split oversized notes
- **Arguments:** Note name, or empty (finds stale candidates automatically)
- **Dependencies:** Should run after `/reflect` (needs established connections to propagate backward)
- **Uses MCP:** All qmd tools

#### `/pipeline [file]`
- **What it does:** End-to-end orchestrator. Runs the full sequence: seed → reduce → reflect/reweave/verify for all extracted claims → archive. Spawns subagents for each phase to prevent context contamination.
- **Reads:** Everything the sub-commands read
- **Writes:** Everything the sub-commands write
- **Arguments:** Source file path
- **Dependencies:** None — runs the full chain
- **Model:** sonnet | **Spawns:** Task subagents

#### `/ralph [N]`
- **What it does:** Queue processor. Takes N tasks from `ops/queue/queue.json` and processes them, spawning isolated subagents per task to prevent context contamination between phases.
- **Reads:** `ops/queue/queue.json`, task files
- **Writes:** Delegates to subagent skills
- **Arguments:** `N [--parallel] [--batch id] [--type extract] [--dry-run]`
- **Dependencies:** Queue must have tasks (populated by `/seed` or `/reduce`)
- **Spawns:** Task subagents with fresh context per phase

### Quality & Maintenance Commands

#### `/verify [note]`
- **What it does:** Combined quality gate — three checks: (1) Recite test — cold-read the description, predict what the note contains, compare to actual content (tests description quality). (2) Validate — schema compliance check. (3) Review — health checks (link density, orphan status).
- **Reads:** Target note, templates (for schema), topic maps (for orphan check)
- **Writes:** May fix minor issues (description, missing fields)
- **Arguments:** Note name, or "all" for batch
- **Uses MCP:** `mcp__qmd__vector_search`

#### `/validate [note]`
- **What it does:** Schema validation only (subset of `/verify`). Checks frontmatter against domain templates — required fields, enum values, description quality, link health.
- **Reads:** Target note, `templates/*.md` (for schema definition)
- **Writes:** Nothing (read-only check)
- **Arguments:** Note name, or "all"
- **Model:** sonnet

#### `/health [mode]`
- **What it does:** Comprehensive vault diagnostics across 8 categories: schema compliance, orphan detection, link health, description quality, three-space boundaries, processing throughput, stale notes, MOC coherence.
- **Reads:** All notes, topic maps, ops files, templates, `reference/three-spaces.md`
- **Writes:** `ops/health/YYYY-MM-DD-report.md`
- **Arguments:** `quick` (categories 1-3), `full` (all 8), `three-space` (category 5 only)
- **Model:** opus

#### `/stats [--share]`
- **What it does:** Vault metrics snapshot — note count, link count, link density, topic map coverage, inbox status, processing throughput, growth trends.
- **Reads:** All notes (counts), topic maps, ops state
- **Writes:** Nothing (display only)
- **Model:** sonnet

### Discovery & Analysis Commands

#### `/graph [operation]`
- **What it does:** Interactive graph analysis. Routes natural language questions to graph operations — triangle detection (synthesis opportunities), bridge detection, cluster analysis, hub identification, sibling detection, forward/backward traversal.
- **Reads:** All notes (parses wiki links), topic maps
- **Writes:** Nothing (analysis only)
- **Arguments:** `health`, `triangles`, `bridges`, `clusters`, `hubs`, `siblings`, `forward [note]`, `backward [note]`, `query [question]`
- **Model:** sonnet

#### `/ask [question]`
- **What it does:** Queries the bundled research knowledge base (249 methodology claims + 20 reference docs). Routes through 3 tiers: (1) WHY — research claims in `methodology/`, (2) HOW — guidance docs in `reference/`, (3) WHAT IT LOOKS LIKE — domain examples. Returns research-backed answers with citations.
- **Reads:** `methodology/*.md`, `reference/*.md`, `ops/derivation.md` (for system-specific context)
- **Writes:** Nothing
- **Model:** opus | **Uses MCP:** All qmd tools

#### `/learn [topic]`
- **What it does:** Research a topic using web search (or Exa deep researcher if configured). Files results with full provenance into inbox, chains to processing pipeline.
- **Reads:** Web search results
- **Writes:** `inbox/*.md` (source captures with provenance metadata)
- **Arguments:** Topic to research
- **Uses:** `WebSearch`, `mcp__exa__*` (if configured)

#### `/note-history [note]`
- **What it does:** Git-based note evolution tracking. Reconstructs how a note changed over time — not raw diffs but interpreted semantic shifts. Can restore previous versions.
- **Reads:** Git history for target note
- **Writes:** Optionally restores previous version (with confirmation)
- **Arguments:** `[note] [--restore N] [--full]`
- **Model:** haiku (cheapest — just git operations)

### System Evolution Commands

#### `/architect [area]`
- **What it does:** Research-backed system evolution advisor. Analyzes health reports, friction patterns (ops/observations), and derivation history. Consults research graph for supporting claims. Generates proposals with research justification. Never auto-implements — proposals require approval.
- **Reads:** `ops/derivation.md`, `ops/health/*.md`, `ops/observations/*.md`, `ops/config.yaml`, `methodology/*.md`
- **Writes:** Nothing until approved; then may update config/ops files
- **Model:** opus

#### `/recommend [use case]`
- **What it does:** Architecture advice for new knowledge systems. Starts from tradition presets (research, therapy, engineering, etc.), customizes across 8 dimensions, validates coherence, outputs full config.
- **Reads:** `reference/tradition-presets.md`, `reference/methodology.md`, `reference/components.md`, `reference/dimension-claim-map.md`, `reference/interaction-constraints.md`
- **Writes:** Nothing (advisory output)
- **Model:** opus

#### `/rethink`
- **What it does:** The scientific method applied to the knowledge system itself. Triages pending observations and tensions, detects cross-cutting patterns, generates evolution proposals.
- **Reads:** `ops/observations/*.md`, `ops/tensions/*.md`, `ops/config.yaml`, `ops/derivation.md`
- **Writes:** May resolve observations/tensions (updates frontmatter status), may propose config changes
- **Can ask questions:** Uses `AskUserQuestion` tool

#### `/refactor [dimension]`
- **What it does:** Plans vault restructuring when configuration changes. Compares current `ops/config.yaml` against `ops/derivation.md`, identifies what shifted, shows restructuring plan, executes on approval.
- **Reads:** `ops/config.yaml`, `ops/derivation.md`, `ops/derivation-manifest.md`
- **Writes:** Vault files (restructures on approval)
- **Arguments:** Specific dimension or `--dry-run`

### Operational Commands

#### `/remember [description]`
- **What it does:** Captures friction as methodology observations. Three modes: (1) explicit — user describes the friction, (2) contextual — reviews recent corrections in conversation, (3) session mining — scans past session transcripts for patterns.
- **Reads:** Conversation context, `ops/observations/*.md` (dedup check)
- **Writes:** `ops/observations/<description>.md`
- **Model:** sonnet

#### `/tasks [action]`
- **What it does:** View and manage the task stack and processing queue.
- **Reads:** `ops/tasks.md`, `ops/queue/queue.json`
- **Writes:** `ops/tasks.md` (add/done/drop/reorder)
- **Arguments:** `add [description]`, `done [number]`, `drop [number]`, `reorder`, `status`
- **Model:** sonnet

#### `/next`
- **What it does:** Surfaces the most valuable next action by combining: task stack priority, queue state, inbox pressure, health status, and goals.
- **Reads:** `ops/tasks.md`, `ops/queue/queue.json`, `ops/goals.md`, inbox count, health state
- **Writes:** Nothing (recommendation only)
- **Model:** sonnet

---

## 3. Session Hooks (Detailed)

### Hook 1: `session-orient.ps1` (SessionStart)

**Trigger:** Fires automatically when a Claude Code session starts.
**Guard:** Checks for `.arscontexta` file — exits silently if not in an arscontexta vault.

**What it reads and surfaces:**

1. **Active Goals** — Parses `ops/goals.md`, extracts the `## Active Threads` section, displays all non-empty lines.

2. **Reminders with Overdue Detection** — Parses `ops/reminders.md` for unchecked items matching `- [ ] YYYY-MM-DD: Description`. Calculates days until due:
   - `OVERDUE (N days)` — past due
   - `DUE TODAY` — due today
   - `DUE in N days` — due within 3 days
   - Distant items: counted but not shown individually

3. **Last Session Summary** — Reads `ops/last-session.md` (written by session-capture hook), shows first 5 non-blank content lines. Skips YAML frontmatter.

4. **Pending Tasks** — Parses `ops/tasks.md` for `## Pending` and `## In Progress` sections.

5. **Vault Counts** — Counts notes, inbox items, and parses frontmatter `status: pending` in observations and tensions directories.

6. **Condition Triggers** — Checks counts against thresholds (from `ops/queue/queue.json` or defaults):
   - Inbox >= 3: `TRIGGER: Inbox has N items`
   - Observations >= 10: `TRIGGER: N pending observations`
   - Tensions >= 5: `TRIGGER: N pending tensions`

7. **Lifecycle Archival** (silent housekeeping):
   - Auto-archives completed goals older than 15 items to `ops/goals-archive.md`
   - Auto-archives completed tasks older than 7 days to `ops/tasks-archive.md`
   - Removes completed reminders older than 14 days

**Output format:**
```
=== Session Orient ===

Current Goals:
- [goal lines from ops/goals.md]

Reminders:
  OVERDUE (2 days): Fix dangling links
  DUE in 1 days: Weekly maintenance

Last session:
  Session abc123 (2026-02-25T14:30:00)
  Files changed: 3 src, 2 notes (5 total)

Vault: 171 notes | 3 inbox | 2 pending observations | 4 pending tensions
TRIGGER: Inbox has 3 items (threshold: 3)
=== End Orient ===
```

### Hook 2: `session-capture.ps1` (Stop)

**Trigger:** Fires when a Claude Code session ends (Stop event).
**Input:** JSON via stdin with `transcript_path`.

**What it does:**

1. **Saves session metadata** — Writes `ops/sessions/<conversation_id>.json` with:
   - `session_id`, `timestamp`, `transcript_path`, `status: "completed"`

2. **Writes bridging context** — Creates `ops/last-session.md` for the next session's orient hook:
   - Session ID and timestamp
   - Git status summary (N src, N tests, N notes, N ops changed)
   - Recent commits from last 2 hours
   - Whether `ops/goals.md` was updated

3. **State Update Rule enforcement** — If substantial changes happened (`src/` or `notes/`) but `ops/goals.md` wasn't updated, writes a warning to stderr:
   > "Session had substantial changes to src/ or notes/ but ops/goals.md was not updated. Did you complete a milestone?"

### Hook 3: `validate-note.ps1` (PostToolUse — Write)

**Trigger:** After any Write tool use targeting a `.md` file in `notes/`.
**Blocking:** No (all checks advisory — exits 0 always).

**What it checks:**
1. YAML frontmatter exists (starts with `---`)
2. `description` field present
3. `topics` field present
4. `type` field present (soft warning)
5. `type` value is valid enum: `finding|decision|method|hypothesis|baseline|open-question|pattern`
6. `description` length <= 200 characters

**Output:** `WARN:` messages for each violation.

### Hook 4: `auto-commit.ps1` (PostToolUse — Write, async)

**Trigger:** After any Write tool use targeting files in `notes/`, `ops/`, `inbox/`, or `self/`.
**Runs asynchronously** — doesn't block the session.

**What it does:**
1. `git add <written_file>`
2. If there are staged changes: `git commit -m "vault: update <filename>" --no-verify`

**Note:** Uses `--no-verify` because vault auto-commits are single-file markdown additions where pre-commit hooks (linting, formatting) don't apply.

### Hook 5: `check_agents_tag.ps1` (Stop)

**Trigger:** Session end.

**What it does:** Reads the session transcript, checks if it contains `**Agents:**`. If not, outputs:
> "Remember to end your response with: **Agents:** [list or None]"

**Purpose:** Enforces the CLAUDE.md behavioral contract requiring agent attribution.

### Hook 6: `check_plan_mode.ps1` (PreToolUse — Edit|Write)

**Trigger:** Before any Edit or Write tool use on `.py` files (excluding `plans/`).
**Blocking:** No (advisory only — always exits 0).

**What it does:** Outputs:
> "Editing code file — did you use EnterPlanMode for non-trivial tasks?"

**Design note:** The hook is deliberately non-blocking. The comment in the source explains: "Making this blocking would prevent trivial .py fixes without plan mode."

---

## 4. Knowledge Graph Pipeline Flow

### Complete Pipeline: Raw Input → Finished Note

```
                    ┌──────────────────────────────────────────┐
                    │           SOURCES (entry points)          │
                    │  /learn [topic]  — web research           │
                    │  /seed [file]    — manual file queue      │
                    │  User drops file in inbox/                │
                    └──────────┬───────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   1. INBOX            │
                    │   inbox/*.md          │
                    │   (provenance preserved│
                    │    in YAML frontmatter)│
                    └──────────┬───────────┘
                               │  /reduce (or /pipeline)
                               ▼
                    ┌──────────────────────┐
                    │   2. EXTRACTION       │
                    │   /reduce             │
                    │   - Reads source      │
                    │   - Classifies claims │
                    │   - Dedup via qmd     │
                    │   - Creates atomic    │
                    │     notes in notes/   │
                    │   - Validates schema  │
                    │     (via hook)        │
                    │   - Auto-commits      │
                    │     (via hook)        │
                    └──────────┬───────────┘
                               │  /reflect (or /pipeline)
                               ▼
                    ┌──────────────────────┐
                    │   3. FORWARD CONNECT  │
                    │   /reflect            │
                    │   - Dual discovery:   │
                    │     topic maps + qmd  │
                    │   - Articulation test │
                    │   - Add wiki-links    │
                    │   - Update topic maps │
                    └──────────┬───────────┘
                               │  /reweave (or /pipeline)
                               ▼
                    ┌──────────────────────┐
                    │   4. BACKWARD CONNECT │
                    │   /reweave            │
                    │   - Find stale notes  │
                    │   - Add new→old links │
                    │   - Sharpen claims    │
                    │   - Consider splits   │
                    └──────────┬───────────┘
                               │  /verify
                               ▼
                    ┌──────────────────────┐
                    │   5. QUALITY GATE     │
                    │   /verify             │
                    │   - Recite test       │
                    │   - Schema validation │
                    │   - Health check      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   DONE               │
                    │   Note is:           │
                    │   - Atomic (1 claim) │
                    │   - Connected (3+ links) │
                    │   - On topic map     │
                    │   - Schema compliant │
                    │   - Description tested│
                    └──────────────────────┘
```

### Orchestration Options

| Command | What Runs | Context Management |
|---------|-----------|-------------------|
| `/pipeline [file]` | Full chain: seed → reduce → reflect → reweave → verify | Spawns subagents per phase (fresh context) |
| `/ralph N` | N tasks from queue, any phase | Spawns subagents per task (isolation) |
| Manual sequence | User runs each command individually | Runs in current context (risks contamination) |

### Queue System

The queue (`ops/queue/queue.json`) tracks tasks by type:
- `extract` — needs `/reduce`
- `reflect` — needs `/reflect`
- `reweave` — needs `/reweave`
- `verify` — needs `/verify`

`/ralph` pulls tasks in FIFO order within each type, spawning isolated subagents to prevent one phase's context from contaminating the next.

---

## 5. Config Schema (ops/config.yaml)

### Dimension Configuration

| Dimension | Current Value | Options | What It Controls |
|-----------|--------------|---------|-----------------|
| `granularity` | `atomic` | atomic \| compound | One claim per note vs. multi-claim notes |
| `organization` | `flat` | flat \| hierarchical | No folders vs. folder taxonomy |
| `linking` | `explicit+implicit` | explicit \| explicit+implicit | Wiki links only vs. wiki links + semantic search |
| `processing` | `heavy` | light \| standard \| heavy | Quality gate intensity (how many pipeline steps) |
| `navigation` | `3-tier` | flat \| 2-tier \| 3-tier | Hub depth (index → domain maps → notes) |
| `maintenance` | `condition-based` | manual \| scheduled \| condition-based | When maintenance triggers fire |
| `schema` | `moderate` | minimal \| moderate \| heavy | Metadata field requirements |
| `automation` | `full` | manual \| assisted \| full | How much skills do autonomously |

### Feature Flags

| Feature | Current | What It Enables |
|---------|---------|----------------|
| `semantic-search` | `true` | qmd MCP server, vector_search in skills |
| `processing-pipeline` | `true` | /seed → /reduce → /reflect chain |
| `sleep-processing` | `false` | Background processing between sessions |
| `voice-capture` | `false` | Voice memo transcription pipeline |

### Processing Configuration

| Field | Current | What It Controls |
|-------|---------|-----------------|
| `processing.depth` | `standard` | How thorough extraction/connection is |
| `processing.chaining` | `suggested` | After /reduce, suggest /reflect (vs. auto-chain or manual) |
| `processing.extraction.selectivity` | `moderate` | How aggressively to extract (strict = fewer notes, permissive = more) |
| `processing.extraction.categories` | 7 categories | What types of claims to extract |
| `processing.verification.description_test` | `true` | Run cold-read test on descriptions |
| `processing.verification.schema_check` | `true` | Validate YAML frontmatter |
| `processing.verification.link_check` | `true` | Verify wiki link targets exist |
| `processing.reweave.scope` | `related` | How far backward to scan |
| `processing.reweave.frequency` | `after_create` | When to trigger reweave |

### Extraction Categories

| Category | What to Find |
|----------|-------------|
| `findings` | Empirical results, measurements, experimental outcomes |
| `decisions` | Architectural choices, parameter selections, design trade-offs |
| `methods` | Techniques, algorithms, procedures |
| `hypotheses` | Untested predictions, proposed mechanisms |
| `baselines` | Reference metrics, established benchmarks |
| `open-questions` | Unanswered questions, knowledge gaps |
| `patterns` | Recurring structures, reusable abstractions |

### Other Settings

| Field | Current | Purpose |
|-------|---------|---------|
| `processing_tier` | `auto` | Auto-detect processing depth from source type |
| `provenance` | `full` | Track source → note chain completely |
| `personality.enabled` | `false` | No agent personality (neutral-helpful) |
| `research.primary` | `web-search` | Default research tool |
| `research.fallback` | `web-search` | Fallback when primary fails |
| `research.default_depth` | `moderate` | How deep to research by default |

---

## 6. Hardcoded Assumptions and Paths

### In Hook Scripts

| Assumption | Location | Details |
|------------|----------|---------|
| Vault root is 2 directories up from hooks | All hooks | `Resolve-Path (Join-Path $PSScriptRoot "..\.."))` |
| `.arscontexta` file existence = valid vault | All hooks | Guard condition before execution |
| Notes are in `notes/` subfolder | `validate-note.ps1` | Path regex: `[/\\]notes[/\\]` |
| Auto-commit targets: `notes\|ops\|inbox\|self` | `auto-commit.ps1` | Path regex for which files to auto-commit |
| Code files are `.py` files | `check_plan_mode.ps1` | Only reminds for `.py` extensions |
| Reminders follow `- [ ] YYYY-MM-DD: Description` | `session-orient.ps1` | Regex-parsed date format |
| Completed items have `(done YYYY-MM-DD)` suffix | `session-orient.ps1` | For archival age calculation |
| Max 15 completed goals before archival | `session-orient.ps1` | Hardcoded `$maxCompleted = 15` |
| Completed tasks archived after 7 days | `session-orient.ps1` | Hardcoded age threshold |
| Completed reminders removed after 14 days | `session-orient.ps1` | Hardcoded age threshold |
| Git is available in PATH | `session-capture.ps1` | Uses `git -C $VaultRoot status` |
| "Substantial work" = changes in `src/` or `notes/` | `session-capture.ps1` | State Update Rule check |

### In Skill Definitions

| Assumption | Location | Details |
|------------|----------|---------|
| `ops/derivation-manifest.md` exists | All skills (Step 0) | Falls back to universal defaults if missing |
| `ops/config.yaml` exists | All skills (Step 0) | Falls back to defaults if missing |
| Templates in `templates/` | `/validate`, `/health` | Schema source of truth |
| `ops/queue/queue.json` exists | `/ralph`, `/seed`, `/reduce` | Queue state file |
| Wiki link format: `[[title]]` | `/reflect`, `/reweave`, `/graph` | Regex-based link parsing |
| Note files are `.md` in `notes/` | All skills | Hardcoded path |
| Topic maps identifiable by content | `/reflect`, `/health` | Heuristic detection (has "## Core Ideas" or similar) |

### In MCP Configuration

| Assumption | Location | Details |
|------------|----------|---------|
| qmd binary at `C:/Users/shach/AppData/Roaming/npm/node_modules/@tobilu/qmd/dist/qmd.js` | `.mcp.json` | Absolute path to user's npm global install |
| qmd cache at `C:\Users\shach\.qmd-cache` | `.mcp.json` | `XDG_CACHE_HOME` env var |

---

# Part 2: All Skills and Agent Definitions

## 7. Complete Skill Inventory

### arscontexta Skills (21 files in `.claude/skills/`)

| # | Skill | Model | Status | Purpose |
|---|-------|-------|--------|---------|
| 1 | `/architect` | opus | Active | Research-backed system evolution proposals |
| 2 | `/ask` | opus | Active | Query research knowledge base (3-tier) |
| 3 | `/graph` | sonnet | Active | Interactive graph analysis (triangles, bridges, etc.) |
| 4 | `/health` | opus | Active | 8-category vault diagnostics |
| 5 | `/learn` | default | Active | Web research → inbox with provenance |
| 6 | `/next` | sonnet | Active | Prioritized next-action recommendation |
| 7 | `/note-history` | haiku | Active | Git-based note evolution + restore |
| 8 | `/pipeline` | sonnet | Active | Full end-to-end processing chain |
| 9 | `/ralph` | default | Active | Queue processor with isolated subagents |
| 10 | `/reduce` | default | Active | Extraction engine (source → atomic notes) |
| 11 | `/refactor` | default | Active | Config-driven vault restructuring |
| 12 | `/reflect` | default | Active | Forward connections + topic map updates |
| 13 | `/remember` | sonnet | Active | Friction capture → observations |
| 14 | `/rethink` | default | Active | Evidence-driven system evolution |
| 15 | `/reweave` | default | Active | Backward connections (old ← new) |
| 16 | `/seed` | sonnet | Active | Add source to processing queue |
| 17 | `/stats` | sonnet | Active | Vault metrics snapshot |
| 18 | `/tasks` | sonnet | Active | Task stack management |
| 19 | `/validate` | sonnet | Active | Schema validation (subset of /verify) |
| 20 | `/verify` | default | Active | Combined quality gate (recite + validate + review) |
| 21 | `/recommend` | opus | Active | Architecture advice for new KG systems |

### claude-md-management Skills (2 commands)

| # | Skill | Status | Purpose |
|---|-------|--------|---------|
| 22 | `/revise-claude-md` | Active | Update CLAUDE.md with session learnings |
| 23 | `/claude-md-improver` | Active | Audit and improve CLAUDE.md quality |

### .codex Skills (4 files — NOT Claude Code)

| # | Skill | File | Status | Purpose |
|---|-------|------|--------|---------|
| 24 | `code-simplifier` | `.codex/skills/code-simplifier/SKILL.md` | **Dead weight** | Codex CLI refactoring skill |
| 25 | `spec-refiner` | `.codex/skills/spec-refiner/SKILL.md` | **Dead weight** | Codex CLI spec writing |
| 26 | `implementor-stage-gate` | `.codex/skills/implementor-stage-gate/SKILL.md` | **Dead weight** | Codex CLI staged implementation |
| 27 | `verify-app` | `.codex/skills/verify-app/SKILL.md` | **Dead weight** | Codex CLI verification |

**These `.codex/` skills reference `AGENTS.md` and a `tasks/<date>_<slug>/` folder structure that no longer exists. They are artifacts from an earlier experiment with OpenAI Codex CLI and can be safely deleted.**

---

## 8. Agent Definitions (`.claude/agents/`)

### `dsp-reviewer.md`
- **Model:** opus (most capable — DSP correctness is critical)
- **Tools:** Read, Grep, Glob (read-only — reviewer can't modify code)
- **Purpose:** Reviews signal processing code for mathematical correctness. Specializes in STFT, FFT bins, dB scaling, frequency band masking, Nyquist, zero-padding.
- **Key behavior:** Cross-checks code against vault notes about established DSP parameters (586 Hz frequency bins, 1.7 ms temporal resolution). Cites vault notes in findings.
- **When used:** ANY change to energy computation, FFT, dB scaling (per CLAUDE.md mandate)

### `detection-validator.md`
- **Model:** sonnet
- **Tools:** Read, Grep, Glob, Bash (can run tests)
- **Purpose:** Validates changes to USV detection algorithms. Checks energy detection pipeline (thresholds, duration filters, bandwidth, merging) and legacy Parameter Lab heuristic.
- **Key behavior:** Runs `pytest tests/test_energy_detector.py`, checks algorithm correctness, edge cases, config validation completeness. Cross-checks vault baselines (89.7% precision, 93.8% recall).
- **Outputs:** Structured validation report with Algorithm Correctness / Edge Cases / Config Validation / Test Coverage / Issues / Recommendations.

### `streamlit-expert.md`
- **Model:** sonnet
- **Tools:** Read, Grep, Glob, Edit, Write (can modify code)
- **Purpose:** Implements and reviews Streamlit UI. Expert in session state, caching (`@st.cache_data`, `@st.cache_resource`), layout, widget callbacks.
- **Key files:** `src/usv_spectrogram/param_lab/app.py` (650+ lines), `scripts/usv_parameter_lab.py`

### `test-writer.md`
- **Model:** sonnet
- **Tools:** Read, Grep, Glob, Edit, Write, Bash (full implementation capability)
- **Purpose:** Generates pytest tests. Follows AAA pattern (Arrange/Act/Assert), `test_<function>_<scenario>_<expected_outcome>` naming, fixtures, parametrization.
- **Key convention:** Tests in `tests/test_<module>.py`, run with `.venv\Scripts\python.exe -m pytest tests/ -v`

### `pr-reviewer.md`
- **Model:** opus
- **Tools:** Read, Grep, Glob, Bash (read + run tests)
- **Purpose:** Final quality gate before commit/PR. 5-section checklist: code quality, style, testing, security, documentation. Cross-checks vault for contradictions.
- **Outputs:** Summary + Issues Found (Critical/Warning/Suggestion) + Verdict (APPROVE / REQUEST_CHANGES / NEEDS_DISCUSSION)

### `master-reviewer.md`
- **Model:** sonnet
- **Tools:** Read, Grep, Glob, Bash (read + run tests)
- **Purpose:** Senior technical reviewer. Reviews implementations against ROADMAP spec, DECISIONS.md constraints, and established patterns. Fresh context — hasn't seen implementation happen.
- **Review order:** (1) DSP correctness, (2) ML rigor, (3) Spec compliance, (4) Integration correctness, (5) Code quality, (6) Documentation
- **Key feature:** "Fix Documentation Requirement" — if verdict is CHANGES NEEDED, review MUST include fix documentation steps. Prevents the "permanently stale at CHANGES NEEDED" failure mode.
- **Outputs:** Writes review content between `---BEGIN REVIEW FILE---` / `---END REVIEW FILE---` markers. Main session extracts and writes the file (reviewer lacks Write tool).
- **KG integration:** Reads topic maps and greps notes for relevant vault claims before reviewing.

---

## 9. Command Definitions (`.claude/commands/`)

### `/implement [module]`
- **File:** `.claude/commands/implement.md`
- **Type:** Full workflow orchestrator (most complex command)
- **Phases:**
  1. **PLAN** — Enters plan mode (read-only), reads ROADMAP, DECISIONS, patterns, existing code. Writes plan for approval.
  2. **IMPLEMENT** — Creates task list, implements config → core → scripts → tests. Runs module tests then full suite.
  3. **DOCUMENT** — Creates/updates `docs/modules/<module>.md`, patterns.md, DECISIONS.md, handoff.
  4. **REVIEW** — Spawns master-reviewer subagent, writes review file, fixes blockers.
  5. **REPORT** — Summarizes to user.
- **DSP guardrail:** Checks all STFT params match ADR-002 during implementation.

### `/roadmap-from-plan [file]`
- **File:** `.claude/commands/roadmap-from-plan.md`
- **Purpose:** Converts web Claude implementation plans into structured ROADMAP format with `/implement` blocks.
- **Key feature:** Step 6 — "Extract Theoretical Knowledge to KG" — scans source for theoretical content, offers to run `/reduce`. Added after the `web-claude-plans-are-dual-purpose-documents` observation.
- **Output:** Standalone `ROADMAP_<PLAN_NAME>.md` file (NOT appended to main ROADMAP).

### `/review-all [module]`
- **File:** `.claude/commands/review-all.md`
- **Purpose:** Comprehensive review — spawns master-reviewer (tier-appropriate) + dsp-reviewer (if DSP) + detection-validator (if detection). Writes unified review file.
- **Tier system:**
  - Tier 1 (Housekeeping): 10 tool calls
  - Tier 2 (Standard): 30 tool calls
  - Tier 3 (Critical): 60 tool calls

### `/run-app`
- **File:** `.claude/commands/run-app.md`
- **Command:** `.\.venv\Scripts\streamlit.exe run scripts/usv_parameter_lab.py`

### `/verify`
- **File:** `.claude/commands/verify.md`
- **Purpose:** Full verification: py_compile → pytest → flake8 → output check.
- **Note:** Collides with arscontexta `/verify` (which does recite + validate + review on notes). Context determines which runs.

### `/verify-quick`
- **File:** `.claude/commands/verify-quick.md`
- **Purpose:** Quick check: py_compile on modified files only, pytest if relevant tests exist.

### `/simplify [area]`
- **File:** `.claude/commands/simplify.md`
- **Purpose:** Code quality review — flatten conditionals, consolidate logic, improve naming, add type hints, add docstrings. Must keep functionality identical.

### `/commit-push-pr`
- **File:** `.claude/commands/commit-push-pr.md`
- **Purpose:** Git workflow: status → diff → commit → push → `gh pr create`.

### `/web-handoff [topic]`
- **File:** `.claude/commands/web-handoff.md`
- **Purpose:** Generates context summary for continuing in claude.ai. Includes project context, session summary, current state, and topic for deeper discussion. Excludes internal workflow details.

---

# Part 3: Typical USV Pipeline Session

## 10. Common Task Types

Based on the last 30+ commits and PROJECTS.md:

### Task Distribution

| Task Type | Frequency | Example Commits |
|-----------|-----------|-----------------|
| **Feature implementation** | ~45% | `feat: Phase 11.1 bout preprocessing on real data`, `feat: LMT data access layer` |
| **Documentation & tracking** | ~35% | `docs: update project tracking`, `docs: knowledge graph expansion — 54 new notes` |
| **Knowledge graph maintenance** | ~15% | `vault: capture VQ-VAE findings`, `vault: topic map updates` |
| **Fixes & chores** | ~5% | `fix: un-ignore important .txt files`, `chore: hook improvements` |

### Top 5 Task Patterns

1. **"Implement Phase N.M"** — Most common. User says `/implement [Module Name]`. Reads ROADMAP spec, enters plan mode, implements, tests, documents, reviews.

2. **"Convert this plan to ROADMAP"** — User brings a web Claude plan. `/roadmap-from-plan` converts it. Then `/reduce` extracts theoretical knowledge.

3. **"Run maintenance"** — `/health`, `/reflect`, `/reweave`, `/stats`. Fixes links, connects orphans, updates stale notes.

4. **"Research [topic]"** — `/learn [topic]` → `/seed` → `/reduce` → `/reflect`. Web research flows into the knowledge graph.

5. **"Fix [specific issue]"** — User identifies a bug or improvement. Quick approval → fix → `/verify-quick` → done.

---

## 11. Typical Implementation Session Walkthrough

Here is what happens step-by-step when you say "Implement the Information Theory Metrics module":

### Phase 0: Session Start (Automatic — ~5 seconds)

```
1. Claude Code starts
2. CLAUDE.md loaded into context (365 lines — always present)
3. .claudeignore applied (excludes .wav, .png, .venv, etc.)
4. session-orient.ps1 fires:
   - Reads ops/goals.md → shows "Active Threads"
   - Reads ops/reminders.md → shows "DUE in 1 days: Weekly maintenance"
   - Reads ops/last-session.md → shows what happened last time
   - Counts: 171 notes, 3 inbox, 2 observations, 4 tensions
   - Checks thresholds → "TRIGGER: Inbox has 3 items"
5. MCP server qmd available (search tools registered)
6. Session ready — orient output visible to me
```

**Token cost of session start:** ~400 tokens for CLAUDE.md + ~100 tokens for hook output = ~500 tokens before user's first message.

### Phase 1: User Request → Skill Dispatch

```
User: "/implement Information Theory Metrics"

7. Claude Code recognizes /implement as a skill
8. Loads .claude/commands/implement.md (88 lines)
9. $ARGUMENTS = "Information Theory Metrics"
10. Skill says "Call EnterPlanMode NOW"
```

### Phase 2: Plan Mode (Read-Only Exploration)

```
11. EnterPlanMode activated — Write/Edit tools disabled
12. Read ROADMAP.md — find the /implement block for this module
13. Read DECISIONS.md — understand ADR constraints (sr=300000, etc.)
14. Read docs/architecture/patterns.md — frozen dataclasses, candidate flow
15. Read docs/modules/*.md for dependent modules (e.g., spectrogram.py)
16. Read existing source code to understand integration points
17. Write plan to plan file (all files to create, data structures, algorithms)
18. ExitPlanMode — present plan to user
```

**Token cost of planning:** ~2000-4000 tokens for reading 4-6 files + plan output. All exploration reads are compacted when plan mode exits (context efficiency feature).

### Phase 3: User Approval

```
User: "P" (shorthand for "Proceed" per Quick Commands)

19. Approval granted — ANALYSIS → EXECUTION transition (state machine)
```

### Phase 4: Implementation

```
20. TaskCreate: Create task list (config, core logic, scripts, tests, handoff)
21. Mark "Implement config" as in_progress

22. Write src/usv_spectrogram/analysis/information_theory.py
    → check_plan_mode.ps1 fires: "[HOOK] Editing code file — did you use EnterPlanMode?"
    → (Yes, we did — hook is advisory)

23. Run py_compile on new file
24. Write tests/test_information_theory.py
25. Run module tests: pytest tests/test_information_theory.py -v
26. Fix any failures → re-run
27. Run full suite: pytest tests/ -v
28. All green → mark implementation tasks completed
```

**Agents spawned during implementation:** Potentially `test-writer` (sonnet) if tests are complex, `streamlit-expert` if UI changes involved.

### Phase 5: Documentation

```
29. Create docs/modules/information-theory.md
30. Update docs/architecture/patterns.md (if new pattern)
31. Update DECISIONS.md (if new ADR needed)
32. Write docs/reviews/information-theory-handoff.md
    → validate-note.ps1 does NOT fire (not in notes/)
    → auto-commit.ps1 does NOT fire (not in notes/ops/inbox/)
```

### Phase 6: Review

```
33. Spawn master-reviewer subagent (sonnet, ~30 tool calls for Tier 2):
    Prompt: "Review module Information Theory Metrics. Tier 2 review.
            Read handoff: docs/reviews/information-theory-handoff.md"

    Master-reviewer:
    a. Reads handoff
    b. Reads ROADMAP spec for this module
    c. Reads DECISIONS.md for ADR constraints
    d. Reads patterns.md
    e. Reads notes/index.md → relevant topic maps
    f. Greps notes/ for "information theory", "entropy", "mutual information"
    g. Reads source files
    h. Reads test files
    i. Runs pytest tests/ -v
    j. Reports findings: BLOCKER / WARNING / SUGGESTION
    k. Outputs review between ---BEGIN/END REVIEW FILE--- markers

34. Main session writes docs/reviews/information-theory-review.md
35. If CHANGES NEEDED: fix blockers, add "Fixes Applied" section, re-run tests
```

**Additional agents (conditional):**
- If DSP changes: also spawn `dsp-reviewer` (opus)
- If detection changes: also spawn `detection-validator` (sonnet)

### Phase 7: Report & Persist

```
36. Report to user: files created, test counts, review verdict
37. Update IMPLEMENTATION_PROGRESS.md (dated entry)

38. [Knowledge graph integration — if implementation produced insights]:
    → New finding? Write to inbox/ → /reduce later
    → Or user manually notes connections

39. End response with: **Agents:** master-reviewer, [others if used]
```

### Phase 8: Session End (Automatic)

```
40. User ends session (or /clear)
41. session-capture.ps1 fires:
    - Writes ops/sessions/<id>.json
    - Writes ops/last-session.md with:
      "Files changed: 4 src, 2 tests, 1 docs (7 total)"
      "Recent commits: feat: add information theory metrics"
    - Checks State Update Rule: src/ changed but goals.md not updated?
      → Warning if applicable
42. check_agents_tag.ps1 fires:
    - Checks transcript for "**Agents:**" — warns if missing
```

### Total Agent Spawns in a Typical Implementation

| Agent | Model | When | Purpose |
|-------|-------|------|---------|
| master-reviewer | sonnet | After handoff | Tier 2 code review |
| dsp-reviewer | opus | If DSP changes | Mathematical correctness |
| detection-validator | sonnet | If detection changes | Algorithm validation |
| test-writer | sonnet | If complex tests | Test generation |
| streamlit-expert | sonnet | If UI changes | Streamlit best practices |

---

## 12. Recurring Friction Points

Based on `ops/observations/`, `ops/tensions/`, health reports, and the `docs/workflow/claude-md-removed-content.md` archive:

### Friction Point 1: Token Budget Pressure

**Evidence:** `docs/workflow/claude-md-removed-content.md` documents 12+ sections removed from CLAUDE.md to keep it under ~1000 lines. The Knowledge Graph section alone is ~170 lines that loads every turn, even during pure coding sessions.

**Impact:** Every implementation session starts with ~500 tokens of overhead (CLAUDE.md + hook output). By Phase 6 (review), the context may be 50%+ full from reading ROADMAP, DECISIONS, source files, test output. The master-reviewer runs in a subagent specifically to get fresh context.

**Current mitigation:** Plan mode compacts exploration reads. Subagents for review. Skills use `context: fork` to prevent contamination. Still, the 365-line CLAUDE.md is a permanent tax.

### Friction Point 2: Master-Reviewer Bias as Subagent

**Evidence:** `ops/observations/master-reviewer-bias-when-run-as-implementor-subagent.md` (status: pending)

**What happens:** The `/implement` command spawns master-reviewer as a Task subagent within the implementor's chat session. The reviewer found real bugs (Phase 8.4: causal attention cross-contamination). BUT the implementor controls when to spawn the reviewer, interprets the findings, and marks own fixes without re-review.

**Risk:** Interpretation bias — the implementor may interpret "WARNING" findings favorably toward their own code.

**Proposed fix:** For Tier 3 reviews (critical: VQ-VAE, transformer, detection), require a separate chat session for master-reviewer. For Tier 2, current approach is acceptable.

### Friction Point 3: Documentation-Code Drift

**Evidence:** `ops/observations/stale-docs-caused-agent-to-distrust-user-about-pipeline.md` (status: pending)

**What happened:** PROJECTS.md said the PyQt6 app was "Not yet started" when it was fully built. Agent cited stale docs to contradict user's correct claims about the pipeline.

**Impact:** When docs are wrong, agents trust docs over users. This is backwards — docs should be verified against code, not the other way around.

**Current mitigation:** Updated PROJECTS.md. No systematic fix for ongoing drift.

### Friction Point 4: `/verify` Command Name Collision

**Two different `/verify` commands exist:**
1. `.claude/commands/verify.md` — py_compile + pytest + flake8 (code verification)
2. `.claude/skills/verify/SKILL.md` — recite + validate + review (note quality verification)

**Impact:** Context-dependent dispatch. When working on code, you get code verification. When working on notes, you get note verification. But the ambiguity can confuse the dispatch system.

### Friction Point 5: Dual-Purpose Repository

**This repo is simultaneously:**
- A Python codebase (USV detection pipeline, ~350 tests)
- A knowledge vault (171 notes, 249 methodology claims)

**Friction:** Every coding session loads KG instructions. Every KG session has coding constraints in context. The session-orient hook always fires, even when the user wants a quick code fix.

**Evidence:** The hook output at session start shows inbox triggers and maintenance reminders even when the user's intent is purely "fix this bug."

### Friction Point 6: Stale Worktree

**`.claude/worktrees/goofy-elbakyan/`** contains a full repository snapshot including:
- Model checkpoints (`.pt` files)
- Training data (`.csv`, `.npy` files)
- Labeling archives
- A complete copy of all `.claude/` configs

**Impact:** Glob searches return duplicate results from the worktree. Disk usage is inflated. The worktree appears to be from a previous implementation session that wasn't cleaned up.

### Friction Point 7: Web Claude Plan Knowledge Loss

**Evidence:** `ops/observations/web-claude-plans-are-dual-purpose-documents.md` (status: implemented)

**What happened:** A vacation master plan contained 28 extractable domain claims. The `/roadmap-from-plan` skill only captured task specs, losing theoretical knowledge.

**Fix applied:** Added Step 6 to `/roadmap-from-plan` — now scans for theoretical content and offers to run `/reduce`. This observation led to a concrete skill improvement.

### Friction Point 8: git add -A Data Loss Incident

**Evidence:** `ops/observations/bulk-git-commit-deleted-656-detection-files.md` (status: resolved)

**What happened:** Commit 78d1c70 accidentally deleted 656 files from `USV_Detections/` during a "clean git status" commit that used bulk staging.

**Fix applied:** Git Data Safety section added to CLAUDE.md. Now: never `git add -A` without reviewing status, check `git diff --cached --stat` for unexpected deletions, stage data directories by specific file name. Data restored from git history.

### Pattern Summary

| Pattern | Occurrences | Root Cause |
|---------|-------------|------------|
| Token/context pressure | Ongoing | 365-line CLAUDE.md + dual-purpose repo |
| Agent trust hierarchy confusion | 2 incidents | Stale docs + reviewer as subagent |
| Data safety | 1 major incident | Bulk git operations without review |
| Skill/command collisions | 1 known | `/verify` name shared between code and note domains |
| Knowledge loss at boundaries | 1 incident, fixed | Plan conversion didn't extract theory |
| Worktree housekeeping | Ongoing | No automated cleanup |

---

*This audit is read-only. No files were modified except creation of PLUGIN-AUDIT.md.*
