# Claude Code Setup in This Repo — A Tour for New Users

This guide walks through every Claude Code extension point used in this repo, from basic (`CLAUDE.md`) to advanced (hooks, agents, knowledge management). Use it as a reference for what's possible and how the pieces connect.

---

## Table of Contents

1. [The Two Instruction Files](#1-the-two-instruction-files-claudemd-vs-agentsmd)
2. [The `.claude/` Directory](#2-the-claude-directory-the-extension-hub)
3. [Permissions](#3-permissions-settingslocaljson)
4. [Custom Agents](#4-custom-agents-claudeagents)
5. [Slash Commands](#5-slash-commands-claudecommands)
6. [Skills](#6-skills-claudeskills--from-arscontexta-plugin)
7. [Hooks](#7-hooks-claudehooks--event-driven-automation)
8. [Plugins](#8-plugins-settingsjson)
9. [Output Style](#9-output-style)
10. [Auto-Memory](#10-auto-memory)
11. [The Knowledge Graph](#11-the-knowledge-graph-notes-ops-inbox)
12. [How It All Fits Together](#12-how-it-all-fits-together)
13. [Getting Started — What to Steal First](#13-getting-started--what-to-steal-first)

---

## 1. The Two Instruction Files: `CLAUDE.md` vs `AGENTS.md`

| File | Who reads it | Purpose |
|------|-------------|---------|
| **`CLAUDE.md`** (root) | The main Claude Code session | Project instructions: behavioral rules, project structure, environment setup, coding conventions, knowledge graph instructions |
| **`AGENTS.md`** (root) | Subagents (spawned workers) | Same content — ensures subagents also follow the rules |

Claude Code automatically reads `CLAUDE.md` at the start of every session. It's the single most important file for controlling Claude's behavior.

### Key Sections in Our CLAUDE.md

**Behavioral Contract** — A state machine that prevents Claude from writing code before getting approval:

```
IDLE -> ANALYSIS -> APPROVAL_PENDING -> EXECUTION -> VALIDATION -> DONE
```

Forbidden transitions (e.g., `ANALYSIS -> EXECUTION`) are enforced by a hook (see section 7).

**Stop Conditions** — When Claude should pause and ask rather than guess:
- 3+ assumptions on the critical path
- Same approach tried twice without new rationale
- Evidence contradicts hypothesis

**Test Protocol (Anti-Greenwashing)** — A truth table preventing Claude from silently changing test expectations to make tests pass:

| Code State | Test Result | Action |
|------------|-------------|--------|
| Correct | Pass | Good |
| Buggy | Fail | Good — fix code |
| Correct | Fail | Discuss — test expectations may be wrong |
| Buggy | Pass | **DANGEROUS** — tests not catching bug |
| Unknown | Fail | **STOP** — don't assume which is wrong |

**Quick Commands** — Natural language shortcuts:

| Phrase | Effect |
|--------|--------|
| "Proceed" / "P" | Approval granted, start coding |
| "Fresh eyes" | Restart reasoning from evidence |
| "5 Whys" | Root cause analysis before any fix |
| "Explain..." | Teaching mode — prioritize understanding |

**Git Data Safety** — Rules born from real incidents:
- Never `git add -A` without reviewing status first (once accidentally deleted 656 files)
- Always stage specific files by name

---

## 2. The `.claude/` Directory (The Extension Hub)

This is where all Claude Code extensions live:

```
.claude/
├── settings.json          # Plugins enabled (checked into git)
├── settings.local.json    # Permissions, hooks, output style (local only, NOT in git)
├── agents/                # 6 custom subagents
│   ├── master-reviewer.md
│   ├── dsp-reviewer.md
│   ├── detection-validator.md
│   ├── pr-reviewer.md
│   ├── test-architect.md
│   ├── test-hardener.md
│   ├── test-writer.md       # Deprecated — redirects to test-architect/test-hardener
│   ├── streamlit-expert.md
│   └── arscontexta-expert.md
├── commands/              # 7 slash commands (legacy format)
│   ├── implement.md
│   ├── commit-push-pr.md
│   ├── simplify.md
│   ├── verify-quick.md
│   ├── verify-code.md
│   ├── run-app.md
│   ├── roadmap-from-plan.md
│   ├── web-handoff.md
│   └── review-all.md
├── skills/                # 22 skills (new format, from arscontexta plugin)
│   ├── reduce/SKILL.md
│   ├── reflect/SKILL.md
│   ├── ... (20 more)
└── hooks/                 # 7 event hooks
    ├── session-orient.ps1 + .cmd
    ├── session-capture.ps1 + .cmd
    ├── check_agents_tag.ps1 + .cmd
    ├── check_plan_mode.ps1 + .cmd
    ├── validate-note.ps1 + .cmd
    └── auto-commit.ps1 + .cmd
```

**Important**: `settings.json` is shared (git-tracked), while `settings.local.json` is machine-specific (gitignored). Permissions and hooks go in the local file.

---

## 3. Permissions (`settings.local.json`)

An allowlist-based system controlling what Claude can run without asking:

```json
{
  "permissions": {
    "allow": [
      "Read(*)",
      "Edit",
      "Write",
      "WebFetch",
      "WebSearch",
      "Bash(git *)",
      "Bash(python*)",
      "Bash(pytest*)",
      "Bash(pip *)",
      "Bash(gh *)",
      "Bash(ls*)",
      "Bash(mkdir *)",
      "Bash(echo*)"
    ],
    "deny": [
      "Read(.env)",
      "Read(.env.*)",
      "Read(**/secrets/**)",
      "Write(.env)",
      "Write(.env.*)"
    ]
  }
}
```

- **Allow**: File operations, git, python, pytest, pip, GitHub CLI, basic shell commands
- **Deny**: Reading/writing `.env` files and secrets — prevents accidental credential exposure
- Anything not on the allowlist triggers a confirmation prompt

---

## 4. Custom Agents (`.claude/agents/`)

Subagents are specialized Claude instances spawned for specific tasks. Each is a markdown file with YAML frontmatter specifying name, tools, and optionally a model override.

| Agent | Model | Tools | Purpose |
|-------|-------|-------|---------|
| `master-reviewer` | sonnet | Read, Grep, Glob, Bash | Post-implementation code review against ROADMAP spec |
| `dsp-reviewer` | **opus** | Read, Grep, Glob | Reviews signal processing math (higher model for harder math) |
| `detection-validator` | — | Read, Grep, Glob, Bash | Validates USV detection algorithm changes |
| `pr-reviewer` | — | Read, Grep, Glob, Bash | Final quality check before commits |
| `test-architect` | sonnet | Read, Grep, Glob, Write, Bash | Writes failing tests from ROADMAP specs BEFORE implementation |
| `test-hardener` | sonnet | Read, Grep, Glob, Write, Bash | Adversarial coverage hardening AFTER implementation |
| `test-writer` | — | *(Deprecated)* | Redirects to test-architect / test-hardener |
| `streamlit-expert` | — | Read, Grep, Glob, Edit, Write | Streamlit UI implementation |
| `arscontexta-expert` | — | Read, Grep, Glob, + MCP tools | Knowledge graph architecture decisions |

### Example: Agent File Structure

```markdown
---
name: dsp-reviewer
description: Reviews DSP and signal processing code for mathematical correctness
model: opus
tools:
  - Read
  - Grep
  - Glob
---

# DSP/Signal Processing Reviewer

You are a specialist in digital signal processing...

## Your Expertise
- STFT computation and windowing functions
- FFT bin calculations and frequency resolution
...
```

### Key Pattern: Fresh-Context Review

The `master-reviewer` agent reads a "handoff document" written by the implementer, then independently reviews the code. Because it's a separate agent, it has **fresh context** — it hasn't seen the implementation happen, so it's less likely to nod along with mistakes.

---

## 5. Slash Commands (`.claude/commands/`)

Commands are markdown files that expand into prompts when you type `/command-name`. This is the simpler, legacy format.

| Command | What it does |
|---------|-------------|
| `/implement` | Full workflow: Plan Mode → code → test → document → review |
| `/commit-push-pr` | Commit + push + create GitHub PR |
| `/simplify` | Review code for reuse, quality, efficiency |
| `/verify-quick` | Quick verification pass |
| `/verify-code` | Full implementation verification |
| `/run-app` | Launch the Streamlit app |
| `/roadmap-from-plan` | Convert a plan into ROADMAP format with `/implement` blocks |
| `/web-handoff` | Generate context handoff for web Claude |
| `/review-all` | Review all changed files |

### Example: `/implement` Workflow

The `/implement` command orchestrates a 5-phase workflow:

1. **PLAN** — Enters Plan Mode (read-only), reads ROADMAP + architecture docs, presents plan for approval
2. **IMPLEMENT** — Creates task list, writes code + tests in order (config → core → scripts → tests)
3. **DOCUMENT** — Creates module docs, updates architecture patterns, writes handoff
4. **REVIEW** — Spawns the `master-reviewer` agent to independently review
5. **REPORT** — Summarizes what was built, test results, review verdict

---

## 6. Skills (`.claude/skills/` — from arscontexta plugin)

Skills are the newer, richer format with more metadata in frontmatter (allowed tools, context forking, version tracking). There are **22 skills** for knowledge management.

### The Core Pipeline

```
/seed  →  /reduce  →  /reflect  →  /reweave  →  /verify
  │          │           │             │            │
  │          │           │             │            └─ Quality gate
  │          │           │             └─ Update OLD notes with new connections
  │          │           └─ Find connections, update topic maps
  │          └─ Extract knowledge claims from source
  └─ Queue source material for processing
```

### All Skills

| Skill | Purpose |
|-------|---------|
| `/seed` | Queue a source file for processing |
| `/reduce` | Extract structured knowledge claims from source material |
| `/reflect` | Find connections between notes, update topic maps (MOCs) |
| `/reweave` | Backward pass — update old notes with connections to newer notes |
| `/verify` | Combined quality gate (schema + description + health) |
| `/validate` | Schema validation for notes |
| `/pipeline` | End-to-end: seed → reduce → reflect → reweave → verify |
| `/ralph` | Queue processor — spawns isolated subagents per task |
| `/tasks` | View/manage the task stack and processing queue |
| `/stats` | Vault statistics and knowledge graph metrics |
| `/graph` | Interactive knowledge graph analysis |
| `/next` | Surface the most valuable next action |
| `/learn` | Research a topic using web search, file results |
| `/remember` | Capture friction as methodology notes |
| `/rethink` | Challenge system assumptions against evidence |
| `/refactor` | Plan vault restructuring from config changes |
| `/health` | Run vault health diagnostics (8 categories) |
| `/architect` | Research-backed evolution advice for the knowledge system |
| `/ask` | Query the research knowledge graph for methodology guidance |
| `/recommend` | Get architecture advice grounded in research |
| `/note-history` | Show how a note evolved over time (git history) |
| `/refresh-human-docs` | Regenerate human-readable project dashboards |

### Skill File Structure

```markdown
---
name: reduce
description: Extract structured knowledge from source material...
version: "1.0"
generated_from: "arscontexta-v1.6"
user-invocable: true
allowed-tools: Read, Write, Grep, Glob, mcp__qmd__vector_search
context: fork
---

## Runtime Configuration (Step 0)
Read these files to configure domain-specific behavior:
1. `ops/derivation-manifest.md` — vocabulary mapping
2. `ops/config.yaml` — processing depth
...
```

Key differences from commands:
- `allowed-tools` restricts what the skill can use
- `context: fork` means it runs in an isolated context
- `generated_from` tracks which plugin version created it

---

## 7. Hooks (`.claude/hooks/` — Event-Driven Automation)

Hooks run shell commands in response to Claude Code lifecycle events. They're configured in `settings.local.json`.

| Event | Hook | What it does |
|-------|------|-------------|
| **SessionStart** | `session-orient.ps1` | Shows goals, overdue reminders, last session summary, vault stats, trigger warnings |
| **Stop** | `check_agents_tag.cmd` | Enforces `**Agents:** [list]` footer on every Claude response |
| **Stop** | `session-capture.cmd` | Saves session state to `ops/last-session.md` for next-session continuity |
| **PreToolUse** (Edit/Write) | `check_plan_mode.cmd` | **Blocks code writing** if Claude hasn't gotten approval — enforces the state machine |
| **PostToolUse** (Write) | `validate-note.cmd` | Validates note schema when writing to `notes/` |
| **PostToolUse** (Write) | `auto-commit.cmd` | Auto-commits note changes asynchronously |

### Hook Configuration in `settings.local.json`

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [{
          "type": "command",
          "command": "powershell.exe -NoProfile -File .claude/hooks/session-orient.ps1 || true"
        }]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [{
          "type": "command",
          "command": "cmd.exe /c .claude\\hooks\\check_plan_mode.cmd"
        }]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Write",
        "hooks": [
          { "type": "command", "command": "cmd.exe /c .claude\\hooks\\validate-note.cmd" },
          { "type": "command", "command": "cmd.exe /c .claude\\hooks\\auto-commit.cmd", "async": true }
        ]
      }
    ]
  }
}
```

**Key concepts:**
- `matcher` filters which tools trigger the hook (e.g., only `Edit|Write`)
- `async: true` runs the hook without blocking Claude's response
- The `SessionStart` hook output appears as a `<system-reminder>` at the top of the conversation

### Windows Adaptation

Claude Code hooks run via bash, but this is a Windows machine. The solution:
`.cmd` wrapper → calls `powershell.exe` → runs the actual `.ps1` script.

---

## 8. Plugins (`settings.json`)

```json
{
  "enabledPlugins": {
    "claude-md-management@claude-plugins-official": true,
    "arscontexta@agenticnotetaking": true
  }
}
```

| Plugin | What it provides |
|--------|-----------------|
| **claude-md-management** | Skills to audit and improve CLAUDE.md files |
| **arscontexta** | The entire knowledge management system — 22 skills, config generation, vault scaffolding, 249 research claims backing the methodology |

Plugins are like package managers for Claude Code capabilities. They add skills, agent definitions, and sometimes hooks.

---

## 9. Output Style

```json
"outputStyle": "Explanatory"
```

Set in `settings.local.json`. This tells Claude to include educational `★ Insight` blocks with explanations alongside its work. Other options exist (concise, verbose, etc.), but Explanatory is good for learning.

---

## 10. Auto-Memory

Claude Code has a persistent memory directory at:
```
~/.claude/projects/<project-path-hash>/memory/MEMORY.md
```

This file survives across sessions and is loaded into every conversation's context. It stores:

- **Project status** — What's done, what's in progress, where to resume
- **Environment quirks** — Windows workarounds, GPU configuration, tool versions
- **Critical rules** — Operational patterns learned from past mistakes
- **Integration details** — API keys location, database IDs, external service config

The first ~200 lines of `MEMORY.md` are injected into every session. Keep it concise and high-signal.

There's also a repo-tracked mirror at `docs/SESSION_MEMORY.md` for version control.

---

## 11. The Knowledge Graph (`notes/`, `ops/`, `inbox/`)

This is the **arscontexta** system — a Zettelkasten-style knowledge base operated by Claude itself.

### Directory Structure

```
notes/              # Atomic knowledge claims (one claim per note, wiki-linked)
  index.md          # Entry point → topic maps → individual notes
  detection.md      # Topic map: USV detection pipeline
  classification.md # Topic map: CNN training, labeling
  ...
inbox/              # Raw material waiting to be processed
ops/                # Operational state
  goals.md          # Active threads, completed milestones
  tasks.md          # Task stack
  reminders.md      # Time-bound commitments
  config.yaml       # Processing depth, pipeline settings
  queue/            # Processing queue
  sessions/         # Session logs
  observations/     # Friction signals, surprises
  tensions/         # Contradictions to resolve
  methodology/      # System self-knowledge
templates/          # Note/topic-map/observation templates
methodology/        # 249 research claims backing the system (READ-ONLY)
reference/          # Structured reference docs (READ-ONLY)
```

### How Notes Work

Every note makes exactly one claim. The title IS the claim:

- **Good**: `energy detection at 10dB threshold misses low-amplitude calls below 40kHz`
- **Bad**: `detection notes` (too vague) or `STFT parameters` (category, not claim)

Notes link to each other with `[[wiki links]]` and are organized by topic maps (Maps of Content).

### Navigation Hierarchy

```
index.md
  └── Topic maps (detection, classification, signal-processing, ...)
        └── Individual atomic notes
```

---

## 12. How It All Fits Together

```
User types a message
  │
  ├─ SessionStart hook fires
  │    └─ session-orient.ps1 reads goals, reminders, vault state
  │       └─ Output appears as context at top of conversation
  │
  ├─ CLAUDE.md loads (behavioral contract, project context)
  │
  ├─ Claude analyzes the request
  │    └─ Reads relevant files, plans approach
  │
  ├─ PreToolUse hook fires on Edit/Write
  │    └─ check_plan_mode.cmd BLOCKS if no approval yet
  │
  ├─ User says "Proceed"
  │    └─ Claude writes code
  │
  ├─ PostToolUse hooks fire on Write
  │    ├─ validate-note.cmd checks schema (if writing to notes/)
  │    └─ auto-commit.cmd commits note changes (async)
  │
  ├─ /implement spawns master-reviewer agent
  │    └─ Independent review with fresh context
  │
  ├─ Stop hooks fire on every Claude response
  │    ├─ check_agents_tag.cmd enforces **Agents:** footer
  │    └─ session-capture.cmd saves state for next session
  │
  └─ Auto-memory updated with session outcomes
```

---

## 13. Getting Started — What to Steal First

If you're just starting with Claude Code, here's a progression from simple to advanced:

### Level 1: CLAUDE.md (Start Here)
Create a `CLAUDE.md` in your project root with:
- Project overview (what it is, what language/framework)
- Environment setup commands (how to run, test, build)
- Coding conventions (style, patterns to follow)
- Common mistakes to avoid

This alone makes Claude dramatically more useful.

### Level 2: Permissions
Add a `settings.local.json` with an allowlist so Claude doesn't ask permission for every git command or file read. Deny access to secrets.

### Level 3: Custom Agents
Create a reviewer agent in `.claude/agents/`. A fresh-context reviewer catches things the main session misses because it hasn't been staring at the same code for 30 minutes.

### Level 4: Slash Commands
Create `/commit`, `/test`, or `/deploy` commands for workflows you repeat often. Each is just a markdown file describing the steps.

### Level 5: Hooks
Add a `SessionStart` hook that shows project status. Add a `PreToolUse` hook to enforce approval before code changes. These are the guardrails that make Claude reliable.

### Level 6: Knowledge Management (arscontexta)
This is the deep end — a full knowledge graph operated by Claude across sessions. Start with the arscontexta plugin (`/arscontexta:setup`) if you want Claude to build and maintain a research knowledge base for your project.

---

## Quick Reference: File Locations

| What | Where |
|------|-------|
| Main instructions | `CLAUDE.md` (root) |
| Subagent instructions | `AGENTS.md` (root) |
| Shared settings (plugins) | `.claude/settings.json` |
| Local settings (permissions, hooks) | `.claude/settings.local.json` |
| Custom agents | `.claude/agents/*.md` |
| Slash commands | `.claude/commands/*.md` |
| Skills (plugin-generated) | `.claude/skills/*/SKILL.md` |
| Hooks (event scripts) | `.claude/hooks/*.ps1` + `.cmd` wrappers |
| Auto-memory | `~/.claude/projects/<hash>/memory/MEMORY.md` |
| Knowledge graph | `notes/`, `inbox/`, `ops/`, `templates/` |
| Research backing | `methodology/`, `reference/` (read-only) |
