# KG System Portability Blueprint

> How to bootstrap the arscontexta knowledge graph system into a new repo that has no KG.

**Philosophy:** This is a *blueprint*, not a template. It teaches construction rather than providing a pre-built copy. The domain-agnostic infrastructure transfers verbatim; the domain-specific parts are re-derived for your new project. See `methodology/derivation generates knowledge systems from composable research claims not template customization.md` for why this matters.

**Source repo:** `mickey_london_lab` (or whichever repo currently hosts the system)
**Target:** Any git repo where you want the KG system running

---

## Inventory: What the System Consists Of

### Layer 1: Research Substrate (READ-ONLY) — copy verbatim
| Source | Files | Size | Domain-specific? |
|--------|-------|------|-----------------|
| `methodology/` | 249 | ~250 research claims | No — these are universal KG theory |
| `reference/` | 24 | routing indexes, constraints, test fixtures | No |

These are the "fuel" for the derivation engine. Every skill that makes architectural decisions (`/architect`, `/recommend`, `/ask`) reads from these.

### Layer 2: Skills — copy verbatim, strip 3 files
| Source | Count | Domain-specific? |
|--------|-------|-----------------|
| `.claude/skills/` | 27 skill directories | **95% agnostic.** 3 files have minor examples to replace |

**Files needing edits (examples only, not logic):**
- `.claude/skills/learn/SKILL.md` — 4 illustrative examples reference USV/DeepSqueak
- `.claude/skills/reduce/SKILL.md` — 1 worked extraction example references DeepSqueak
- `ops/scripts/vault-search.mjs` — `COMPOUND_TERMS` array has ~12 USV-specific terms to strip

### Layer 3: Agent Definitions — selective copy
| Agent | Transfer? | Why |
|-------|-----------|-----|
| `arscontexta-expert.md` | Yes | Core KG architecture agent |
| `master-reviewer.md` | Yes | Reviews implementations against ROADMAP |
| `pr-reviewer.md` | Yes | Final quality review |
| `test-architect.md` | Yes | Pre-implementation test design |
| `test-hardener.md` | Yes | Post-implementation coverage gaps |
| `test-writer.md` | Yes | General test writing |
| `dsp-reviewer.md` | **No** | USV signal processing specific |
| `detection-validator.md` | **No** | USV detection specific |
| `streamlit-expert.md` | **Maybe** | Only if new project uses Streamlit |

### Layer 4: Hooks — copy verbatim, adapt platform
| Hook | Purpose | Domain-specific? |
|------|---------|-----------------|
| `session-orient.*` | Load goals + reminders at session start | No |
| `session-capture.*` | Log session state at end | No |
| `validate-note.*` | Check YAML frontmatter on Write to `notes/` | No |
| `auto-commit.*` | Auto-commit vault changes | No |
| `check-plan-mode.*` | Block writes during plan mode | No |
| `check-agents-tag.*` | Ensure Agents: tag on responses | No |

All hooks exist in `.sh`, `.ps1`, and `.cmd` variants. Copy only the variants matching your target platform.

### Layer 5: Templates — copy verbatim
| Template | Purpose |
|----------|---------|
| `templates/note.md` | Atomic note schema + structure |
| `templates/topic-map.md` | MOC structure |
| `templates/source-capture.md` | Inbox source format |
| `templates/observation-note.md` | Friction/surprise capture |
| `templates/codex-handoff.md` | Codex task specs (optional — only if using Codex) |

### Layer 6: Ops Scripts — copy verbatim, strip compound terms
| Script | Purpose | Domain-specific? |
|--------|---------|-----------------|
| `ops/scripts/vault-search.mjs` | 3-layer vault search engine | Strip ~12 USV terms from `COMPOUND_TERMS` |
| `ops/scripts/topic-map-index.mjs` | Topic map regeneration | No |
| `ops/scripts/audit-topic-map-coverage.sh` | Coverage audit | No |
| `ops/scripts/test-vault-search.sh` | Search smoke tests | No |

### Layer 7: CLAUDE.md — transfer ~60%, re-derive ~40%

**Sections that transfer verbatim (infrastructure):**
- `## STOP - READ BEFORE DOING ANYTHING` (core principles)
- `## Behavioral Contract` (state machine, stop conditions framework, core rules)
- Test Protocol table
- Quick Commands
- `# Knowledge Graph` (entire section — Philosophy through Guardrails + Common Pitfalls)

**Sections to re-derive for new domain:**
- `## Project Overview` — describe your project
- `## Environment Setup` — your venv, build tools, test commands
- `## Project Structure` / `## Task Routing` — your codebase map
- `## Key Reference Documents` — your doc inventory
- `## Project-Specific Agents` — your domain expert agents
- Domain-specific conventions (replaces Signal Processing Conventions)
- `## Common Mistakes to Avoid` — your project's landmines
- Stop Conditions Red Flags — your domain's red flags
- Git Data Safety — your sensitive data directories
- Vault Canary Comments — skip initially, add when HIGH-risk files emerge

### Layer 8: Operational Scaffolding — copy structure, re-derive content

| File/Dir | Action |
|----------|--------|
| `ops/config.yaml` | Re-derive: choose your 8 dimension values |
| `ops/derivation.md` | Re-derive: document your configuration choices + justifications |
| `ops/derivation-manifest.md` | Re-derive: machine-readable manifest |
| `ops/goals.md` | Copy structure, empty content |
| `ops/reminders.md` | Copy structure, empty content |
| `ops/queue/queue.json` | Create empty: `{"schema_version": 3, "tasks": []}` |
| `ops/methodology/` | Start empty — accumulates through `/remember` |
| `ops/observations/` | Start empty — accumulates through friction |
| `ops/tensions/` | Start empty — accumulates through contradictions |
| `ops/sessions/` | Start empty — populated by session-capture hook |
| `ops/health/` | Start empty — populated by `/health` |
| `ops/cache/` | Start empty — runtime cache |
| `ops/last-session.md` | Create empty |
| `ops/vault-canary-map.md` | Create empty |

### Layer 9: Notes Vault — start fresh

| File/Dir | Action |
|----------|--------|
| `notes/index.md` | Create: hub with 3-5 starter topic maps for your domain |
| `notes/` | Start empty — grows through /seed -> /reduce -> /reflect |
| `inbox/` | Start empty |
| `archive/` | Start empty |
| `archive/inbox/` | Start empty (processed sources go here) |

### Layer 10: Settings — copy structure, adapt

| File | Action |
|------|--------|
| `.claude/settings.json` | Copy plugin refs + hook structure; adapt hook commands to platform |
| `.claude/settings.local.json` | Re-derive permissions for your project |

---

## Bootstrap Sequence

### Phase 0: Prepare (in source repo)

No preparation needed in the source repo. Everything is copied from its current location.

> **Future automation note:** A `scripts/export-kg-system.sh` could automate Phases 1-4 (the mechanical copy steps). Phases 5-8 require human decisions.

### Phase 1: Scaffold Directory Structure

In your target repo:

```bash
mkdir -p notes inbox archive/inbox templates methodology reference
mkdir -p ops/{queue,scripts,sessions,health,cache,methodology,observations,tensions}
mkdir -p .claude/{skills,agents,hooks}
mkdir -p docs/human
```

### Phase 2: Copy Verbatim Layers

From source repo to target (adjust paths as needed):

```bash
SOURCE=/path/to/mickey_london_lab
TARGET=/path/to/your-new-repo

# Layer 1: Research substrate
cp -r "$SOURCE/methodology/"*.md "$TARGET/methodology/"
cp -r "$SOURCE/reference/"*.md "$TARGET/reference/"

# Layer 2: Skills (all 27 directories)
cp -r "$SOURCE/.claude/skills/"* "$TARGET/.claude/skills/"

# Layer 3: Domain-agnostic agents
for agent in arscontexta-expert master-reviewer pr-reviewer test-architect test-hardener test-writer; do
  cp "$SOURCE/.claude/agents/$agent.md" "$TARGET/.claude/agents/"
done

# Layer 4: Hooks (copy all variants, prune later)
cp "$SOURCE/.claude/hooks/"* "$TARGET/.claude/hooks/"

# Layer 5: Templates
cp "$SOURCE/templates/"*.md "$TARGET/templates/"

# Layer 6: Ops scripts
cp -r "$SOURCE/ops/scripts/"* "$TARGET/ops/scripts/"
```

### Phase 3: Strip Domain Examples

Three files need minor edits:

**3a. `.claude/skills/learn/SKILL.md`**
Find the argument hints / research depth examples referencing USV, DeepSqueak, bioacoustics, FFT. Replace with examples from your domain. The surrounding skill logic is untouched.

**3b. `.claude/skills/reduce/SKILL.md`**
Find the worked extraction example referencing "DeepSqueak v3 switched from Faster R-CNN to YOLO v2". Replace with a domain-appropriate example showing claim extraction.

**3c. `ops/scripts/vault-search.mjs`**
In the `COMPOUND_TERMS` array (~line 43-48), remove USV-specific terms:
```
Remove: vq-vae, deepsqueak, stft, usv, lmt, 300-khz, raven, deep-squeak,
        bootssnap, entropy-rate, zipf, codebook, spectrogram, bout
Keep:   rlhf, ppo, grpo, dpo, icl, lora, peft, cnn, mcp, k-means,
        topic-map, wiki-link
```
Add any compound terms from your domain that the tokenizer should treat as single tokens.

### Phase 4: Adapt Hooks for Target Platform

**If target is Linux/Mac (most common):**
- Keep `.sh` variants
- Delete `.ps1` and `.cmd` variants
- Update `.claude/settings.local.json` hook commands to use `.sh` directly

**If target is Windows:**
- Keep `.ps1` variants
- Update hook commands to use `powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File .claude/hooks/<name>.ps1`

### Phase 5: Re-derive Configuration

This is where derivation happens — you make conscious choices about how your knowledge system should work.

**5a. `ops/config.yaml`**

Start from the source config and adjust. The 8 dimensions:

| Dimension | Question to Answer | Source Default |
|-----------|--------------------|---------------|
| granularity | One claim per note (atomic) or multi-paragraph entries? | atomic |
| organization | Flat with links, or hierarchical folders? | flat |
| linking | Wiki-links only, semantic search only, or both? | explicit+implicit |
| processing | Full pipeline (heavy) or lightweight capture? | heavy |
| navigation | Hub + topic maps + notes (3-tier) or simpler? | 3-tier |
| maintenance | Condition-triggered or scheduled? | condition-based |
| schema | Strict YAML frontmatter or freeform? | moderate |
| automation | Full hooks+skills or manual? | full |

> **For most research/engineering projects:** the source defaults are well-validated. Change only what your domain demands.

**5b. `ops/derivation.md`**

Document your choices. Copy the structure from the source file:
- Configuration Dimensions table (with your positions + justifications)
- Vocabulary Mapping (if you rename any terms)
- Active Feature Blocks checklist
- Failure Mode Risks (assess for your domain)

**5c. `ops/derivation-manifest.md`**

Machine-readable version. Copy structure, update dimension values and vocabulary.

### Phase 6: Compose CLAUDE.md

Start from the source CLAUDE.md and work section by section:

1. **Copy verbatim:** Everything from `## STOP - READ BEFORE DOING ANYTHING` through Core Rules, Test Protocol, Quick Commands
2. **Copy verbatim:** The entire `# Knowledge Graph` section (Philosophy through Common Pitfalls)
3. **Re-write:** `## Project Overview` — 2-3 sentences about your project
4. **Re-write:** `## Environment Setup` — your build/test/run commands
5. **Re-write:** `## Task Routing` — map task types to your codebase entry points
6. **Re-write:** `## Key Reference Documents` — your doc inventory
7. **Re-write:** `## Project-Specific Agents` — list only agents you copied + any new ones
8. **Re-write:** Domain conventions section (replaces Signal Processing Conventions)
9. **Re-write:** `## Common Mistakes to Avoid` — start with 2-3 known landmines
10. **Adapt:** Stop Conditions Red Flags — your domain's "stop and think" triggers
11. **Adapt:** Git Data Safety — your sensitive directories
12. **Skip for now:** Vault Canary Comments (add when you identify HIGH-risk files)
13. **Skip for now:** Codex Handoff section (add if you integrate with Codex)

### Phase 7: Initialize Notes Vault

**7a. Create `notes/index.md`:**

```markdown
---
description: Root navigation hub for [your domain] knowledge system
type: moc
topics: "[[index]]"
---

# index

Welcome to your [domain] knowledge system.

## Topic Maps
- [[topic-1]] -- brief description
- [[topic-2]] -- brief description
- [[topic-3]] -- brief description

## Getting Started
1. Read ops/goals.md to orient on current threads
2. Capture your first note in notes/
3. Connect it to a topic map above
```

**7b. Create 3-5 starter topic map files** in `notes/`, using `templates/topic-map.md` as the template. Pick the top-level themes of your project.

> **Gall's Law reminder:** Start with 3-5 broad maps. Do NOT try to pre-design the full taxonomy. Let it emerge through friction. Split maps when they exceed ~40 notes.

**7c. Create empty operational files:**

```bash
# ops/goals.md
cat > ops/goals.md << 'EOF'
---
description: Current active threads and what the agent is working on
type: moc
---

# goals

## Active Threads

## Waiting

## Recently Completed

## Completed
EOF

# ops/reminders.md
cat > ops/reminders.md << 'EOF'
# Reminders

<!-- Checked at session start. Due items surface in orientation. -->
<!-- Format: - [ ] YYYY-MM-DD: Description -->
<!-- Completed: - [x] YYYY-MM-DD: Description (done YYYY-MM-DD) -->
EOF

# ops/queue/queue.json
echo '{"schema_version": 3, "tasks": []}' > ops/queue/queue.json

# ops/last-session.md
touch ops/last-session.md

# ops/vault-canary-map.md
echo '# Vault Canary Map' > ops/vault-canary-map.md
```

### Phase 8: Configure Settings

**8a. `.claude/settings.json`:**

```json
{
  "enabledPlugins": {
    "arscontexta@agenticnotetaking": true
  }
}
```

Add `claude-md-management@claude-plugins-official` if you want the CLAUDE.md management plugin.

**8b. `.claude/settings.local.json`:**

Copy the hook definitions from the source, adapting commands for your platform. Example for Linux:

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
      "Bash(ls*)",
      "Bash(mkdir *)"
    ],
    "deny": [
      "Read(.env)",
      "Read(.env.*)"
    ]
  },
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "bash .claude/hooks/session-orient.sh || true"
          }
        ]
      }
    ],
    "Stop": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "bash .claude/hooks/check_agents_tag.sh || true"
          }
        ]
      },
      {
        "hooks": [
          {
            "type": "command",
            "command": "bash .claude/hooks/session-capture.sh || true"
          }
        ]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "bash .claude/hooks/check_plan_mode.sh || true"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Write",
        "hooks": [
          {
            "type": "command",
            "command": "bash .claude/hooks/validate-note.sh"
          },
          {
            "type": "command",
            "command": "bash .claude/hooks/auto-commit.sh || true",
            "async": true
          }
        ]
      }
    ]
  }
}
```

Adapt the `permissions.allow` list for your project's tools (Python, npm, cargo, etc.).

### Phase 9: Verify

1. **Session start:** Open a Claude Code session. The session-orient hook should fire and show your (empty) goals.
2. **Health check:** Run `/health quick` — should pass schema + orphan checks trivially.
3. **Stats:** Run `/stats` — should report 0 notes, 0 inbox, 249 methodology claims.
4. **Pipeline smoke test:**
   - Drop a source file into `inbox/`
   - Run `/seed` to queue it
   - Run `/reduce` to extract notes
   - Run `/reflect` to find connections
   - Verify notes appear in `notes/` with correct frontmatter and topic map links
5. **Search:** Run vault-search on a term — should find methodology notes.

---

## What NOT to Copy

| Source Content | Why Not |
|---------------|---------|
| `notes/*.md` (all ~520 notes) | Domain-specific USV research knowledge |
| Domain-specific agents (`dsp-reviewer`, `detection-validator`) | Specialized for USV signal processing |
| `ops/goals.md` content | Operational state of the USV project |
| `ops/methodology/*.md` | Accumulated self-knowledge specific to this vault's usage patterns |
| `ops/observations/`, `ops/tensions/` | Operational learning specific to this vault |
| `ops/sessions/` | Session history |
| `IMPLEMENTATION_PROGRESS.md` | USV project progress |
| `docs/` (most of it) | USV-specific architecture, plans, handoffs, reviews |
| `src/`, `scripts/`, `tests/` | USV codebase |
| Any `ROADMAP*.md` files | Project-specific implementation plans |

---

## Post-Bootstrap: The Evolution Path

After bootstrapping, the system follows the **seed-evolve-reseed** lifecycle:

1. **Seed (you just did this):** Minimal viable configuration. 3-5 topic maps, empty vault, full pipeline.
2. **Evolve (next weeks/months):** Use the system. Friction reveals what to add:
   - New topic maps when a theme accumulates 5+ notes
   - New domain-specific agents when review patterns recur
   - Schema field additions when 20%+ of notes manually include a value
   - Canary comments when a file accumulates regression history
3. **Reseed (when drift accumulates):** Re-derive using `ops/derivation.md` enriched by operational observations. The `/architect` skill helps with this.

**Condition-based maintenance kicks in automatically** via the thresholds in CLAUDE.md's Knowledge Graph section (orphans > 7 days, inbox >= 3, stale notes > 30 days, etc.).

---

## Future: Automation Script

Phases 1-4 (mechanical copying) could be automated with a script:

```bash
#!/bin/bash
# scripts/export-kg-system.sh — future automation
# Usage: ./scripts/export-kg-system.sh /path/to/target-repo

# Phase 1: scaffold
# Phase 2: copy verbatim layers
# Phase 3: strip domain examples (sed replacements)
# Phase 4: platform detection + hook adaptation
```

This is left as future work. The manual process should be executed at least once to build understanding of what each component does — per Gall's Law and the bootstrapping principle.
