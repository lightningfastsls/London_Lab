# Workflow Audit: Claude Code Configuration

**Generated:** 2026-02-26
**Scope:** All Claude Code instruction files, configs, plugins, MCP servers, and supporting infrastructure

---

## Executive Summary

This project has a **sophisticated, multi-layered Claude Code configuration** combining:
- A 365-line root `CLAUDE.md` with behavioral contract, knowledge graph rules, and project conventions
- 2 custom agents (`.claude/agents/`)
- 6 custom slash commands (`.claude/commands/`)
- 4 Codex skills (`.codex/skills/`) — from a separate AI tool, not Claude Code
- 2 plugins: `arscontexta` (knowledge graph) and `claude-md-management`
- 1 MCP server: `qmd` (semantic search over markdown vault)
- A session-start hook that auto-orients the agent
- An extensive docs/workflow/ system with templates and processes
- A full knowledge graph infrastructure (171 notes, 249 methodology claims, 20 reference docs)

The system is well-structured but has **notable complexity** — there are effectively 4 instruction layers that all contribute to how I behave in any given session.

---

## 1. CLAUDE.md Files

### Root: `CLAUDE.md` (365 lines)
**Location:** `./CLAUDE.md`
**Loaded:** Automatically at session start (always in context)

**Contents breakdown:**
| Section | Lines | Purpose |
|---------|-------|---------|
| Core Operating Principles | 1-14 | Priority hierarchy: Learning > Quality > Integrity |
| Behavioral Contract | 16-82 | State machine, stop conditions, rules, test protocol |
| Project Overview | 86-120 | USV Spectrogram Generator description, env setup, structure |
| Key Reference Documents | 124-140 | Conditional reading table — what to read and when |
| Project-Specific Agents | 142-155 | 5 required agents for different task types |
| Signal Processing Conventions | 158-162 | sr=300000 rule, ADR references |
| Quick Commands | 166-176 | Natural language shortcuts ("Proceed", "Fresh eyes", etc.) |
| Common Mistakes | 178-192 | Domain pitfalls + git data safety (from real incident) |
| Knowledge Graph | 195-365 | Full KG operating manual (philosophy, design, pipeline, schema, maintenance) |

**Key behavioral constraints defined here:**
- Mandatory approval before any code changes (state machine)
- Never modify test expectations without discussion
- Always use specialized agents (dsp-reviewer, test-writer, etc.)
- End every response with `**Agents:** [list]`
- Knowledge graph session rhythm: Orient -> Work -> Persist
- Never write directly to `notes/` — route through pipeline

### Worktree Copy: `.claude/worktrees/goofy-elbakyan/CLAUDE.md`
**Status:** Stale duplicate from a previous worktree session. Identical to root.

---

## 2. `.claude/` Directory Structure

```
.claude/
+-- settings.json              # Plugin enablement
+-- agents/
|   +-- streamlit-expert.md    # Streamlit UI specialist (model: sonnet)
|   +-- test-writer.md         # pytest test generator (model: sonnet)
+-- commands/
|   +-- commit-push-pr.md      # /commit-push-pr — git workflow
|   +-- run-app.md             # /run-app — launch Streamlit app
|   +-- verify-quick.md        # /verify-quick — py_compile + quick tests
|   +-- simplify.md            # /simplify — code simplification
|   +-- verify.md              # /verify — full verification
|   +-- web-handoff.md         # /web-handoff — context summary for claude.ai
+-- worktrees/
    +-- goofy-elbakyan/        # STALE WORKTREE (see Section 8)
```

### `.claude/settings.json`
```json
{
  "enabledPlugins": {
    "claude-md-management@claude-plugins-official": true,
    "arscontexta@agenticnotetaking": true
  }
}
```

**What this does:** Enables two plugins that add slash commands and session hooks.

---

## 3. Custom Agents (`.claude/agents/`)

These are subagent definitions that Claude Code spawns for specialized tasks via the `Task` tool.

### `streamlit-expert.md`
- **Model:** sonnet (cheaper/faster)
- **Tools:** Read, Grep, Glob, Edit, Write
- **Purpose:** Implements/reviews Streamlit UI following caching, session state, and layout best practices
- **Key files it knows about:** `src/usv_spectrogram/param_lab/app.py`, `scripts/usv_parameter_lab.py`

### `test-writer.md`
- **Model:** sonnet
- **Tools:** Read, Grep, Glob, Edit, Write, Bash
- **Purpose:** Generates pytest tests following AAA pattern, fixtures, parametrization
- **Naming convention:** `test_<function>_<scenario>_<expected_outcome>()`

**Note:** Three additional agents are referenced in CLAUDE.md but defined at the system level, not in `.claude/agents/`:
- `dsp-reviewer` — DSP/signal processing review
- `detection-validator` — detection logic validation
- `pr-reviewer` — final pre-commit review
- `master-reviewer` — checks against ROADMAP spec

---

## 4. Custom Slash Commands (`.claude/commands/`)

These are user-invocable commands (type `/command-name` in Claude Code).

| Command | File | What It Does |
|---------|------|-------------|
| `/run-app` | `run-app.md` | Launches Streamlit Parameter Lab |
| `/verify-quick` | `verify-quick.md` | py_compile on changed files + pytest |
| `/verify` | `verify.md` | Full verification (syntax + tests + lint + output) |
| `/simplify` | `simplify.md` | Code quality review (accepts `$ARGUMENTS` for focus area) |
| `/commit-push-pr` | `commit-push-pr.md` | Git add, commit, push, gh pr create |
| `/web-handoff` | `web-handoff.md` | Generates context summary for continuing in claude.ai |

---

## 5. Plugins

### `arscontexta@agenticnotetaking`
**What it provides:**
- 27 slash commands for knowledge graph operations (see Section 6 for full list)
- Session-start hook that runs "Orient" — reads goals, reminders, checks condition triggers
- The hook output appears as `<user-prompt-submit-hook>` at the start of each session
- All knowledge graph pipeline operations: `/seed`, `/reduce`, `/reflect`, `/reweave`, `/verify`
- Health diagnostics: `/health`
- Graph analysis: `/graph`
- Research capability: `/learn`
- Task/queue management: `/tasks`, `/ralph`, `/next`

**This is the most impactful plugin** — it transforms Claude Code from a coding assistant into a knowledge management system operator.

### `claude-md-management@claude-plugins-official`
**What it provides:**
- `/revise-claude-md` — Update CLAUDE.md with session learnings
- `/claude-md-improver` — Audit and improve CLAUDE.md quality

---

## 6. All Slash Commands (Combined)

### From `.claude/commands/` (project-specific):
| Command | Source |
|---------|--------|
| `/run-app` | commands/run-app.md |
| `/verify-quick` | commands/verify-quick.md |
| `/verify` | commands/verify.md |
| `/simplify` | commands/simplify.md |
| `/commit-push-pr` | commands/commit-push-pr.md |
| `/web-handoff` | commands/web-handoff.md |

### From `arscontexta` plugin (knowledge graph):
| Command | Purpose |
|---------|---------|
| `/architect` | Research-backed KG evolution advice |
| `/ask` | Query research knowledge base |
| `/graph` | Interactive graph analysis |
| `/health` | Vault health diagnostics |
| `/learn` | Research a topic + grow KG |
| `/next` | Surface most valuable next action |
| `/note-history` | Git-based note evolution tracking |
| `/pipeline` | End-to-end source processing |
| `/ralph` | Queue processing with fresh context |
| `/recommend` | Architecture advice |
| `/reduce` | Extract knowledge from source material |
| `/refactor` | Plan vault restructuring |
| `/reflect` | Find connections, update MOCs |
| `/remember` | Capture friction as methodology notes |
| `/rethink` | Challenge assumptions against evidence |
| `/reweave` | Update old notes with new connections |
| `/seed` | Add source to processing queue |
| `/stats` | Vault statistics snapshot |
| `/tasks` | View/manage task stack and queue |
| `/validate` | Schema validation for notes |
| `/verify` | Combined quality verification |

### From `claude-md-management` plugin:
| Command | Purpose |
|---------|---------|
| `/revise-claude-md` | Update CLAUDE.md with learnings |
| `/claude-md-improver` | Audit CLAUDE.md quality |

### From project CLAUDE.md (implementation workflow):
| Command | Purpose |
|---------|---------|
| `/roadmap-from-plan` | Convert web Claude plans to ROADMAP |
| `/implement` | End-to-end module implementation |
| `/commit-push-pr` | Commit, push, create PR |
| `/verify-quick` | Quick verification |
| `/simplify` | Simplify code |
| `/verify` | Full verification |
| `/web-handoff` | Web Claude handoff |

**Note:** `/verify` exists in both `.claude/commands/` AND the arscontexta plugin. The plugin version does combined recite+validate+review; the commands/ version does py_compile+pytest+flake8. The context of invocation determines which runs.

---

## 7. MCP Server: `qmd`

**What it is:** A local search engine over the markdown vault.
**Collection:** `mickey_london_lab` (indexed at session start)
**Current state:** 0 documents indexed (may need re-indexing)

**Available tools:**
| Tool | Speed | Method |
|------|-------|--------|
| `search` | ~30ms | BM25 keyword matching |
| `vector_search` | ~2s | Semantic/meaning-based |
| `deep_search` | ~10s | Auto-expanded query + reranked |
| `get` | instant | Single document retrieval |
| `multi_get` | instant | Batch glob retrieval |
| `status` | instant | Index health check |

**Note:** The vector embeddings appear unavailable — only BM25 search works. This limits semantic discovery.

---

## 8. `.claudeignore`

**Location:** `./.claudeignore`
**Purpose:** Prevents Claude Code from reading/indexing large or irrelevant files.

**What's excluded:**
- Binary data: `*.wav`, `*.png`, `*.jpg`, `*.jpeg`, `*.zarr/`
- Data directories: `5970 USV/`, `spectrograms*/`
- Generated outputs: `candidates*.csv`, `*.pyc`, `__pycache__/`
- Specific large docs: `usv_signal_processing_reference.md`, `USV_DETECTION_IMPLEMENTATION_PLAN.md`
- Old task folders: `tasks/`
- Virtual environment: `.venv/`, `venv/`

---

## 9. `.codex/` Skills (NOT Claude Code)

**Important:** These are **OpenAI Codex** skill definitions, not Claude Code configurations. They do NOT affect Claude Code behavior.

```
.codex/skills/
+-- code-simplifier/SKILL.md     # Refactor after verification
+-- spec-refiner/SKILL.md        # Create task briefs + handoff notes
+-- implementor-stage-gate/SKILL.md  # Stage-gated implementation
+-- verify-app/SKILL.md          # Verification transcript writer
```

These reference an `AGENTS.md` file and a `tasks/` folder workflow that predates the current Claude Code setup. They appear to be from an earlier experiment with Codex CLI.

---

## 10. Stale Worktree

**Location:** `.claude/worktrees/goofy-elbakyan/`
**Branch:** Unknown (has a `.git` file pointing to worktree)
**Status:** Contains a full repository snapshot including model checkpoints, training data, analysis results

**Contents include:** `.pt` model files, `.csv` datasets, `.npy` arrays, analysis markdown files, labeling archives. This is a large directory that could be cleaned up.

---

## 11. Context Loading Map

### Loaded AUTOMATICALLY at every session start:

| What | How | Size |
|------|-----|------|
| `CLAUDE.md` (root) | Claude Code reads project CLAUDE.md | 365 lines |
| `.claudeignore` | Claude Code reads for file exclusion rules | 28 lines |
| `.claude/settings.json` | Plugin enablement | 6 lines |
| Session Orient hook | arscontexta plugin fires on session start | Variable |
| MCP server status (qmd) | Registered at startup | Metadata only |
| Available slash commands | Listed in system prompt | Command names only |
| Agent definitions | Available but not loaded until invoked | Metadata only |

### Loaded CONDITIONALLY (when relevant task arises):

| What | Trigger | Referenced By |
|------|---------|---------------|
| `IMPLEMENTATION_PROGRESS.md` | Start of every session; after implementation | CLAUDE.md |
| `ROADMAP.md` | Before implementing any module | CLAUDE.md |
| `DECISIONS.md` | Before architectural/design choices | CLAUDE.md |
| `docs/architecture/patterns.md` | Before implementing | CLAUDE.md |
| `docs/workflow/completion-sequence.md` | Multi-file changes | CLAUDE.md |
| `docs/workflow/approval-request-template.md` | Before code changes | CLAUDE.md |
| `docs/workflow/knowledge-graph-reference.md` | KG operations | CLAUDE.md |
| `docs/plans/*.md` | Specific implementation tasks | CLAUDE.md |
| `docs/reference/usv_signal_processing_reference.md` | Signal processing work | CLAUDE.md |
| `ops/goals.md` | Session orient | arscontexta hook |
| `ops/reminders.md` | Session orient | arscontexta hook |
| `ops/config.yaml` | KG operations, pipeline | arscontexta skills |

### NEVER loaded automatically (must be explicitly requested):

| What | Why |
|------|-----|
| `methodology/*.md` (249 files) | Read-only research claims, too large for auto-load |
| `reference/*.md` (20 files) | Read-only structured reference, loaded by specific skills |
| `notes/*.md` (171 files) | Knowledge graph notes, accessed via search/links |
| `docs/reviews/*-handoff.md` | Historical handoffs, only for review |
| `docs/historical/*.md` | Archived documents |
| `.codex/skills/*.md` | Not used by Claude Code |

---

## 12. Supporting Infrastructure

### Workflow Documents (`docs/workflow/`)
| File | Purpose | Used By |
|------|---------|---------|
| `approval-request-template.md` | Approval + Struggle Protocol templates | CLAUDE.md behavioral contract |
| `completion-sequence.md` | 7-step implementation sequence | Every implementation task |
| `knowledge-graph-reference.md` | Verbose KG operating manual | Extends CLAUDE.md KG section |
| `claude-md-removed-content.md` | Content removed from CLAUDE.md for token savings | Historical reference |
| `token-optimization.md` | Token budget management | Meta-documentation |
| `session-1-2-fixes.md` | Early session fix log | Historical |

### Templates (`templates/`)
| Template | Purpose |
|----------|---------|
| `note.md` | Research note schema |
| `topic-map.md` | Topic map / MOC schema |
| `source-capture.md` | Inbox source capture |
| `observation-note.md` | Operational observation |

### Operational State (`ops/`)
| File/Dir | Purpose |
|----------|---------|
| `config.yaml` | KG system configuration (dimensions, processing depth, features) |
| `derivation.md` | WHY each config choice was made |
| `derivation-manifest.md` | Tracks config history |
| `goals.md` | Current goals (read at session start) |
| `reminders.md` | Time-bound commitments |
| `tasks.md` | Task stack |
| `methodology/` | System self-knowledge (2 files) |
| `observations/` | Friction signals (5 files) |
| `tensions/` | Contradictions to resolve (5 files) |
| `health/` | Health check reports (2 files) |
| `sessions/` | Session logs |
| `queue/` | Processing queue state |

### Root-Level Tracking Documents
| File | Purpose | Auto-loaded? |
|------|---------|-------------|
| `CLAUDE.md` | Master instruction file | Yes |
| `ROADMAP.md` | Module implementation specs | No (on-demand) |
| `DECISIONS.md` | Architecture Decision Records | No (on-demand) |
| `IMPLEMENTATION_PROGRESS.md` | Progress tracker | No (should be per-session) |
| `PROJECTS.md` | Multi-project tracking | No |

---

## 13. Relationship Diagram

```
SESSION START
    |
    v
[CLAUDE.md]  <-- Always loaded, 365 lines of instructions
    |
    +-- defines --> Behavioral Contract (state machine, stop conditions)
    +-- defines --> Agent requirements (dsp-reviewer, test-writer, etc.)
    +-- defines --> Knowledge Graph operating manual
    +-- references --> docs/workflow/*.md (loaded on-demand)
    +-- references --> ROADMAP.md, DECISIONS.md (loaded on-demand)
    |
    v
[arscontexta plugin]  <-- Session orient hook fires
    |
    +-- reads --> ops/goals.md, ops/reminders.md
    +-- checks --> condition triggers (inbox >= 3, etc.)
    +-- provides --> 27 slash commands for KG operations
    +-- uses --> methodology/ (249 claims, read-only)
    +-- uses --> reference/ (20 docs, read-only)
    |
    v
[qmd MCP server]  <-- Semantic search available
    |
    +-- indexes --> notes/ (171 files)
    +-- provides --> search, vector_search, deep_search
    |
    v
[.claude/agents/]  <-- Spawned on-demand
    +-- streamlit-expert (sonnet) -- Streamlit UI tasks
    +-- test-writer (sonnet) -- pytest generation
    |
[.claude/commands/]  <-- Invoked by user
    +-- /run-app, /verify, /verify-quick, /simplify, /commit-push-pr, /web-handoff
```

---

## 14. Gaps and Redundancies

### Redundancies

1. **`/verify` command collision** — Both `.claude/commands/verify.md` (py_compile+pytest) and the arscontexta plugin provide `/verify` (recite+validate+review). These serve different purposes but share the same name. The plugin version targets note quality; the command version targets code quality.

2. **Worktree duplication** — `.claude/worktrees/goofy-elbakyan/` contains a full stale copy of the repository including all `.claude/` configs, model checkpoints, and data files. This should be cleaned up — it adds dead weight to glob searches and disk usage.

3. **`.codex/skills/` are dead weight** — These Codex CLI skill definitions are not used by Claude Code. They reference a `tasks/` folder and `AGENTS.md` file from a previous workflow. Consider removing or archiving.

4. **KG instructions appear twice** — The Knowledge Graph section in CLAUDE.md (~170 lines) is a condensed version of `docs/workflow/knowledge-graph-reference.md`. Both are maintained. The CLAUDE.md version loads every session; the verbose version is loaded on-demand.

### Gaps

5. **No hooks directory** — Despite CLAUDE.md referencing hook-based enforcement and the arscontexta plugin using a session-start hook, there's no `.claude/hooks/` directory for custom shell hooks. All hooks are plugin-managed. If you want project-specific hooks (e.g., auto-running py_compile on file save), you'd need to create this.

6. **No user-level `settings.json`** — There's no `~/.claude/settings.json` for user-wide preferences. All configuration is project-scoped. This is fine if you only use Claude Code in this project, but means preferences don't carry to other repos.

7. **`IMPLEMENTATION_PROGRESS.md` not auto-loaded** — CLAUDE.md says to read this "at start of every session" but it's not part of the auto-loading chain. The session orient hook reads `ops/goals.md` and `ops/reminders.md` but not `IMPLEMENTATION_PROGRESS.md`. You have to either remember to ask, or add it to the hook.

8. **qmd shows 0 documents indexed** — The MCP search engine reports 0 docs in the `mickey_london_lab` collection. Either the vault hasn't been indexed in this session or the indexer isn't running. This means semantic search over notes is non-functional until re-indexed.

9. **Missing agents for 3/5 required roles** — CLAUDE.md requires 5 specialized agents, but only 2 have `.claude/agents/` definitions (streamlit-expert, test-writer). The other 3 (dsp-reviewer, detection-validator, pr-reviewer) are defined at the system level as built-in agent types. This works but means their prompts aren't customizable from this project.

10. **No `.cursorrules` or `.clinerules`** — No configuration for other AI coding tools (Cursor, Cline, etc.). This is fine if you only use Claude Code, but worth noting if you use multiple tools.

### Observations

11. **Token budget pressure** — The 365-line CLAUDE.md loads every turn. The KG section alone is ~170 lines. The `docs/workflow/claude-md-removed-content.md` file suggests you've already done token optimization passes. Consider whether the full KG manual needs to be in CLAUDE.md or could be deferred to a plugin-managed context.

12. **Dual-purpose project** — This repo serves both as a **codebase** (USV detection pipeline) and a **knowledge vault** (notes/, methodology/, ops/). The CLAUDE.md reflects this duality, but it means every coding session pays the token cost of KG instructions even when not doing KG work.

13. **Strong incident-driven learning** — Several CLAUDE.md rules trace to specific incidents (git data safety from commit 78d1c70, test anti-greenwashing from actual test corruption). This is excellent practice — rules grounded in real failures are more effective.

---

## 15. File Inventory

### Total files by category:

| Category | Count | Location |
|----------|-------|----------|
| CLAUDE.md files | 2 | Root + stale worktree copy |
| Agent definitions | 2 | `.claude/agents/` |
| Slash command definitions | 6 | `.claude/commands/` |
| Plugin configs | 1 | `.claude/settings.json` |
| Codex skills (unused) | 4 | `.codex/skills/` |
| Workflow docs | 6 | `docs/workflow/` |
| Plan docs | 8 | `docs/plans/` |
| Review/handoff docs | 22 | `docs/reviews/` |
| Module docs | 9 | `docs/modules/` |
| Templates | 4 | `templates/` |
| Operational state | ~20 | `ops/` |
| Knowledge graph notes | 171 | `notes/` |
| Methodology claims | 249 | `methodology/` (READ-ONLY) |
| Reference docs | 20 | `reference/` (READ-ONLY) |

---

*This audit is read-only. No files were modified.*
