# arscontexta Audit Report

## Date: 2026-03-20
## Repo: /mnt/d/mickey_london_lab (USV Spectrogram Generator)

---

## 1. Architecture Overview

arscontexta is **not** a simple notes folder or a standalone MCP server. It is a **knowledge management kernel** that integrates with Claude Code through four mechanisms:

1. **Hook-driven lifecycle automation** — PowerShell scripts registered in `.claude/settings.local.json` fire on SessionStart, PreToolUse, PostToolUse, and Stop events. These handle session orientation, plan-mode enforcement, note validation, auto-commit, and session capture.

2. **MCP semantic search** — A `qmd` MCP server (Node.js, SQLite-backed) provides keyword (BM25) and vector (embedding) search across all vault content. Auto-approved for frictionless use.

3. **Skill graph** — 27 user-invocable skills (`.claude/skills/*/SKILL.md`) implement the processing pipeline, maintenance, and knowledge activation. Skills read runtime config from `ops/derivation-manifest.md` to adapt vocabulary and behavior.

4. **Agent roster** — 7 specialized agents (`.claude/agents/*.md`) provide domain-specific review (DSP, detection, Streamlit, testing, architecture, PR quality, vault methodology).

The `.arscontexta` file is a guard marker that prevents a globally-installed plugin from interfering with this local vault instance. The system was generated from `arscontexta-v1.6` with a research snapshot from 2026-02-18.

### Configuration Hierarchy

```
Level 1: CLAUDE.md          — Behavioral contract, red flags, mandatory workflows
Level 2: ops/config.yaml    — System behavior tuning (dimensions, features, processing depth)
Level 3: ops/derivation-manifest.md — Runtime skill config (vocabulary, extraction categories)
Level 4: .claude/settings.local.json — Session permissions + hooks
Level 5: reference/kernel.yaml — Universal primitives (15 core concepts)
```

---

## 2. Vault Contents Summary

| Space | Files | Size | Purpose |
|-------|-------|------|---------|
| `notes/` | 523 | 1.6 MB | Primary knowledge — atomic claims with frontmatter |
| `methodology/` | 249 | 3.1 MB | READ-ONLY research claims (TFT cognitive science) |
| `reference/` | 22 | 565 KB | READ-ONLY routing indexes + constraints |
| `ops/` | ~40+ | ~207 KB | Operational state (goals, queue, health, observations, tensions) |
| `inbox/` | 0 | 0 | Processing queue (currently empty, 34 archived) |
| **TOTAL** | **~834** | **~5.5 MB** | |

- **Total notes:** 523 (all markdown with YAML frontmatter)
- **Topic maps (MOCs):** 23
- **Content language:** English (100%)
- **Estimated tokens:** ~1.4M tokens raw (but never loaded all at once — see Section 3)

### Skills (27 total)

| Category | Skills |
|----------|--------|
| **Processing pipeline** | `/seed`, `/reduce`, `/reflect`, `/reweave`, `/verify`, `/pipeline`, `/ralph` |
| **Knowledge activation** | `/learn`, `/ask`, `/kcheck`, `/recommend` |
| **Maintenance** | `/health`, `/validate`, `/refactor`, `/refresh-human-docs` |
| **Analysis** | `/graph`, `/stats`, `/next`, `/tasks`, `/note-history` |
| **Evolution** | `/rethink`, `/remember`, `/architect` |
| **Tooling** | `/skill-creator`, `/sync` |
| **Eval workspaces** | `reduce-workspace`, `roadmap-from-plan-workspace` (with iteration/benchmark infrastructure) |

### Agents (7 total)

| Agent | Domain |
|-------|--------|
| `arscontexta-expert` | KG architecture, topic maps, methodology reasoning |
| `master-reviewer` | Implementation review vs ROADMAP spec |
| `pr-reviewer` | Final quality gate before commit |
| `dsp-reviewer` | Signal processing / math correctness |
| `detection-validator` | USV detection algorithm changes |
| `streamlit-expert` | Streamlit UI implementation |
| `test-writer` | pytest generation for new code |

### Registry/Manifest Files

- `notes/index.md` — Knowledge system entry point, lists all 23 topic maps
- `ops/goals.md` — Active threads, completed phases (20 phases documented)
- `ops/vault-canary-map.md` — HIGH/MEDIUM risk file registry with vault note references
- `ops/config.yaml` — Live system configuration (8 dimensions)
- `ops/derivation-manifest.md` — Runtime config for skills
- `ops/queue/queue.json` — Processing pipeline state (schema v3)
- `reference/kernel.yaml` — 15 universal primitives defining the system

---

## 3. Context Loading Mechanism

Context loading is **hook-driven and selective**, not bulk-loaded.

### Session Start (`session-orient.ps1`)

On every new session, the hook:
1. Reads `ops/goals.md` — surfaces active threads
2. Reads `ops/reminders.md` — shows overdue/due-soon items
3. Reads `ops/last-session.md` — previous session summary
4. Reads `ops/tasks.md` — pending/in-progress items
5. **Knowledge Activation**: For each active thread (up to 5):
   - Runs `qmd search` (keyword, BM25) with thread keywords
   - Runs `qmd vsearch` (vector, embedding) with thread description
   - Merges, deduplicates, caps at 4 notes per thread
   - Writes results to `ops/session-relevance.md`
6. Checks maintenance thresholds (inbox count, observation count, tension count)
7. Archives old completed goals/tasks/reminders

### What Gets Loaded Per Session

| Content | Loaded | Mechanism |
|---------|--------|-----------|
| CLAUDE.md (~15 KB) | Always | Claude Code native |
| Session orient brief (~2-5 KB) | Always | Hook output |
| Session relevance notes (~1-3 KB) | Always | Hook generates, Claude reads |
| Skill text (~2-10 KB each) | On invocation | User triggers `/skill` |
| Agent persona (~1-3 KB each) | On invocation | Claude spawns agent |
| Individual notes (~1-3 KB each) | On demand | qmd search or direct read |

**Token footprint per session start:** ~20-30 KB (~5-8K tokens) — very light.

**Key design principle:** The vault is ~5.5 MB (~1.4M tokens) but **never loaded in bulk**. The session-orient hook surfaces only the ~4-8 most relevant notes per session. Additional notes are pulled on demand via qmd semantic search during work.

### Session End (`session-capture.ps1`)

On stop:
1. Saves `ops/sessions/{sessionId}.json` with metadata
2. Writes `ops/last-session.md` (bridging context for next session)
3. Warns if goals.md wasn't updated after substantial changes

### Other Hooks

- **`check_plan_mode.ps1`** (PreToolUse on Edit/Write): Advisory reminder to use plan mode for `.py` edits
- **`validate-note.ps1`** (PostToolUse on Write): Checks `notes/` files for required YAML frontmatter
- **`auto-commit.ps1`** (PostToolUse on Write, async): Auto-commits vault changes with `vault: update <basename>`
- **`check_agents_tag.ps1`** (Stop): Reminds if response missing `**Agents:**` tag

---

## 4. Skill & Agent Activation

### Skills

- **Discovery:** Claude Code finds skills via `.claude/skills/*/SKILL.md` file structure
- **Trigger:** User invokes with `/skillname [args]` or natural language matching skill description
- **Loading:** `context: fork` — skill runs in isolated context, returns results to main session
- **Runtime config:** Each skill reads `ops/derivation-manifest.md` for vocabulary mapping
- **Full text required:** Yes — the SKILL.md is the complete instruction set

### Agents

- **Discovery:** Claude Code finds agents via `.claude/agents/*.md`
- **Trigger:** Spawned by main session when task matches agent domain (per CLAUDE.md routing table)
- **Loading:** Agent persona loaded when spawned; has limited tool access per its definition
- **Full text required:** Yes — agent .md is the complete persona + instructions

---

## 5. Content Ingestion

### `/learn` Command — Research Entry Point

Three depth modes:
- **Quick:** 2 search queries, 2-3 sources, specific factual answer
- **Moderate:** 3-5 queries, 5-8 sources, comprehensive comparison
- **Deep:** 8-12 queries, 15-30+ sources, full survey

Process:
1. Check existing vault knowledge via qmd
2. Execute web searches, chase primary sources
3. Cross-verify claims (minimum 2 independent sources)
4. Write structured inbox file: `inbox/{topic-slug}-research-{date}.md`
5. Chain to `/seed` based on `ops/config.yaml` chaining mode (manual/suggested/automatic)

**Critical invariant:** NEVER writes directly to `notes/`. All content routes: `inbox/ -> /reduce -> notes/`.

### Processing Pipeline

```
/learn -> inbox/ -> /seed -> /reduce -> notes/ -> /reflect -> /reweave -> /verify -> archive
```

Each phase is condition-triggered, not scheduled. State tracked in `ops/queue/queue.json`.

### Manual Ingestion

Users can also drop files directly into `inbox/` and run `/seed` + `/reduce`.

---

## 6. Relationship with CLAUDE.md / AGENTS.md

### CLAUDE.md

CLAUDE.md is **the behavioral contract** — loaded by Claude Code at session start (always in context). It:
- Defines the state machine (IDLE -> ANALYSIS -> APPROVAL -> EXECUTION -> VALIDATION -> DONE)
- Encodes the Knowledge Graph section (~40% of CLAUDE.md) with vault conventions
- References arscontexta indirectly via the Knowledge Activation rule ("search before reasoning")
- Points to vault infrastructure (notes/, ops/, methodology/, reference/)
- Defines task routing table, agent routing table, and key reference documents
- Establishes stop conditions, red flags, and test protocol

### AGENTS.md

AGENTS.md defines the agent roster with routing rules. It complements `.claude/agents/*.md` files by providing the high-level "when to use which agent" guidance.

### How They Interact

```
CLAUDE.md (always loaded)
    |-- References vault structure (notes/, ops/, methodology/)
    |-- Defines Knowledge Activation rule -> triggers qmd search
    |-- Defines agent routing -> spawns .claude/agents/*.md
    |-- Defines skill triggers -> activates .claude/skills/*/SKILL.md
    '-- Points to ops/goals.md, ops/vault-canary-map.md, etc.

Session-orient hook
    |-- Reads ops/ state files
    |-- Runs qmd search for relevant notes
    '-- Generates session-relevance.md brief

Skills + Agents
    |-- Read ops/derivation-manifest.md for runtime config
    |-- Operate on notes/, ops/, inbox/
    '-- Use qmd MCP tools for semantic search
```

**Would replacing arscontexta affect CLAUDE.md/AGENTS.md?** Yes, significantly. The Knowledge Graph section of CLAUDE.md (~40% of content) directly encodes arscontexta conventions. The hook system, skill graph, agent roster, and processing pipeline are all arscontexta infrastructure. Replacing it would require rewriting CLAUDE.md's KG section, all 27 skills, all 7 agents, and all 6 hooks.

---

## 7. Identified Pain Points

### A. Dangling Links (39 active)
- **Category A (16):** Archived inbox sources moved but wiki links still reference inbox path
- **Category B (13):** Cross-space references (methodology/, ops/tensions/) not in default resolver path
- **Category C (10):** Genuinely missing targets (Windows filename truncation, stale refs)
- **Status:** Known since 2026-03-02, unfixed

### B. Description Quality Drift
- `/reduce` produced weak descriptions (title restatements) during bulk ingestion
- **Resolution:** Self-check gate added 2026-03-07, but similar gaps exist in other skills
- **Root cause:** Instruction-only quality gates fail under throughput pressure

### C. Maintenance Overhead Creep
- Weekly routine takes 20-25 min vs. 15 min target
- Caused by bulk ingestion backlog + one-time compliance cleanup
- Phase 5.2 scored 3/5 on this criterion

### D. Reference Document Non-Compliance
- Only 4 of 14 non-exempt reference files meet PRD template (28.6%)
- Missing: Purpose, Derivation Questions, Curated Claims sections
- **Impact:** Engine cannot reliably parse non-compliant docs for derived constraints

### E. Windows Filename Truncation
- Colons in note titles create zero-byte files (Windows path limitation)
- Results in unresolvable dangling links
- 2 known instances

### F. Inbox Ghost Items
- `/reduce` has no archive step — processed files linger in inbox
- Currently resolved manually but will recur
- **Status:** Known, fix proposed but not implemented

### G. Observation-to-Action Pipeline Gap
- Threshold for /rethink trigger is 10 observations
- Patterns can linger below threshold for weeks before action
- Phase 5.2 recommended lowering to 7, not yet implemented

### H. Schema Compliance Degradation
- Dropped from 100% to 91.8% during bulk ingestion (39 notes missing `topics:` field)
- Wiki-links/note average dropped from 8.6 to 6-7

---

## 8. What Must Be Preserved

Any replacement or augmentation **must** provide equivalent capability for:

1. **Atomic note structure with YAML frontmatter** — The entire skill graph depends on parsing `description`, `type`, `topics`, `confidence` fields
2. **Wiki-link graph** — `[[title]]` links are the primary connection mechanism; 3000+ links across 523 notes
3. **Topic map hierarchy** — Three-tier navigation (index -> topic maps -> notes) is the attention management layer
4. **Processing pipeline** — The seven-phase workflow (seed->reduce->reflect->reweave->verify) with queue state tracking
5. **Session orient/capture lifecycle** — Hook-driven context priming at start, state persistence at end
6. **Semantic search (dual mode)** — Both keyword (BM25) and vector search, auto-approved, used by hooks and skills
7. **Condition-triggered maintenance** — Threshold-based actions (orphans, dangling links, stale notes, inbox pressure)
8. **Three-space separation** — notes/ (durable knowledge), ops/ (temporal coordination), methodology/ (read-only research)
9. **Vault canary system** — HIGH-risk file -> mandatory /kcheck before modification
10. **Runtime config adaptation** — Skills reading `ops/derivation-manifest.md` to adapt vocabulary
11. **Git integration** — Auto-commit on vault writes, session capture, data safety protocol
12. **Research grounding** — 249 methodology claims providing cognitive science backing for design decisions

---

## 9. What Could Be Improved

### Quick Wins
1. **Archive step in /reduce** — Prevent inbox ghost items (blocks future backlog)
2. **Windows filename sanitization** — Strip colons/invalid chars from note titles
3. **Reference document compliance** — Fill missing PRD template sections in 9 non-compliant files

### Design Improvements
4. **Cross-space link resolution** — Resolver should check methodology/, ops/methodology/, ops/tensions/ paths
5. **Attention-quality metrics** — MOCs theoretically manage attention but no metric tracks effectiveness
6. **Maintenance SLOs** — Weekly cost has no formal target; thresholds are hardcoded without calibration feedback
7. **Quality gate enforcement** — Description checks, schema checks are advisory (hooks don't block writes)

### Potential OpenViking-Relevant Improvements
8. **Retrieval precision** — qmd keyword search suffers from query term dilution on long descriptions (documented in methodology). A better embedding/retrieval layer could help.
9. **Automatic connection discovery** — `/reflect` requires manual invocation. An always-on similarity detection layer could surface connections proactively.
10. **Multi-modal content** — Vault is text-only. USV spectrograms, detection PNGs exist in the repo but aren't indexed in the knowledge graph.
11. **Tiered summarization** — No L0/L1/L2 abstraction levels. Every note is full-text. A summary layer could reduce retrieval payload.
12. **Cross-session memory extraction** — Session transcripts exist in ops/sessions/ but aren't automatically mined for knowledge (requires manual `/remember --mine-sessions`)

---

## Appendix: Representative Content Samples

### Note Example (Decision type)
```yaml
---
description: "Nyquist requires 2x highest frequency; 300 kHz captures full 20-150 kHz range"
type: decision
confidence: proven
topics: [[signal-processing]]
---
# 300 kHz sample rate is the canonical standard for mouse USV recording

[~1.9 KB of explanation with 5 wiki-links]
```

### Topic Map Example (detection.md)
- Synthesis paragraph explaining the detection domain
- 29 core claims organized by theme (energy detection, candidate generation, filtering)
- 3 sub-maps cross-linked
- Open questions section
- Related areas section

### Methodology Example (READ-ONLY)
```yaml
---
kind: research
confidence: likely
topics: [[agent-cognition]]
---
# LLM attention degrades as context fills

"Smart zone" is first ~40% of context window. Beyond that, attention diffuses.
[~50 lines with 14+ inbound references from notes/]
```

### Processing Queue Example (queue.json entry)
```json
{
  "id": "seed-004",
  "type": "extract",
  "phase": "complete",
  "completed_phases": ["reduce", "reflect", "reweave"],
  "notes": ["note-1.md", "note-2.md", "..."],
  "stats": { "connections": 28, "backward_links": 22, "dangling": 0 }
}
```

---

## Appendix: System Maturity Metrics

| Metric | Baseline (2026-02-20) | Current (2026-03-09) | Status |
|--------|----------------------|----------------------|--------|
| Notes | 117 | ~525 | Growing (4.5x) |
| Topic maps | 6 | ~23 | Expanded |
| Orphan notes | 0 | 0 | Maintained |
| Schema compliance | 100% | 91.8% | Degraded (39 missing `topics:`) |
| Wiki-links/note (avg) | 8.6 | 6-7 | Diluted by bulk ingestion |
| Dangling links | 0 | 39 | FAIL (archival + cross-space refs) |
| Weekly maintenance | N/A | 20-25 min | Above target (<=15 min) |
| Reference compliance | N/A | 28.6% (4 of 14) | FAIL |
