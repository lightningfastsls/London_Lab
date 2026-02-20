# Knowledge Graph — Full Reference

This is the verbose reference for sections condensed in CLAUDE.md.
Consult this when you need the full detail behind a condensed section.

---

## Philosophy (Full)

**If it won't exist next session, write it down now.**

You are the primary operator of this knowledge system. Not an assistant helping organize notes, but the agent who builds, maintains, and traverses a knowledge network. The human provides direction and judgment. You provide structure, connection, and memory.

Notes are your external memory. Wiki-links are your connections. Topic maps are your attention managers. Without this system, every session starts cold. With it, you start knowing what you're working on.

This knowledge graph coexists with the USV research codebase. Domain knowledge about detection, classification, signal processing, training pipelines, and experimental methods lives in `notes/`. Operational state lives in `ops/`. The codebase and knowledge graph reinforce each other.

---

## Discovery-First Design (Full)

**Every note you create must be findable by a future agent who doesn't know it exists.**

Before writing anything to notes/, ask:

1. **Title as claim** -- Does the title work as prose when linked? `since [[title]]` reads naturally?
2. **Description quality** -- Does the description add information beyond the title? Would an agent searching for this concept find it?
3. **Topic map membership** -- Is this note linked from at least one topic map?
4. **Composability** -- Can this note be linked from other notes without dragging irrelevant context?

If any answer is "no," fix it before saving. Discovery-first is not a polish step -- it's a creation constraint.

---

## Session Rhythm (Full)

Every session follows: **Orient -> Work -> Persist**

### Orient
Read orientation state at session start. Check condition-based triggers for maintenance items.
- `ops/goals.md` -- current threads, what's active
- `ops/reminders.md` -- time-bound commitments (surface overdue items)
- Workboard reconciliation -- surfaces condition-based maintenance triggers automatically
- `CLAUDE.md` -- methodology and identity

### Work
Do the actual task. Surface connections as you go. If you discover something worth keeping, write it down immediately -- it won't exist next session otherwise.

### Persist
Before session ends:
- Write any new insights as atomic notes
- Update relevant topic maps
- Update ops/goals.md with current threads
- Capture anything learned about methodology
- Session capture: stop hooks save transcript to ops/sessions/ and auto-create mining tasks

---

## Atomic Notes (Full)

Every note makes exactly one claim. The title IS the claim, written as a complete proposition.

**The composability test:** Can you complete "This note argues that [title]"? If not, the title needs work.

**Good titles:**
- `energy detection at 10dB threshold misses low-amplitude calls below 40kHz`
- `recording-level splits prevent data leakage in USV classification`
- `constrained jittering preserves temporal structure better than random cropping`

**Bad titles:**
- `detection notes` (topic, not claim)
- `STFT parameters` (category, not proposition)
- `meeting 2026-02-18` (event, not insight)

**One claim per note** means:
- Each note can be linked independently without dragging irrelevant context
- Notes compose into arguments by linking claims together
- Contradictions are visible (two notes can disagree)
- Evolution is trackable (a claim can be marked outdated when evidence changes)

---

## Wiki Links (Full)

Wiki links (`[[note title]]`) create edges in your knowledge graph. They are the primary connection mechanism.

### Link Patterns
- `[[note title]]` -- basic link
- `since [[claim about STFT resolution]]` -- link as prose (preferred)
- `contradicts [[earlier finding]]` -- explicit relationship

### Link Philosophy
- Link when there is a genuine intellectual relationship, not just topic similarity
- Every link should be followable -- a reader clicking through should find relevant content
- Prefer typed relationships: "supports", "contradicts", "extends", "applies to"
- Link density target: 3+ outgoing links per note

### Wiki Link Rules
- Never rename a note manually -- use the rename script to update all references
- Dangling links (pointing to nonexistent notes) are demand signals -- create the missing note or fix the link
- Orphan notes (no incoming links) need connections -- run /reflect to find them

---

## Topic Maps (Full)

Topic maps are attention management hubs. They organize notes into navigable clusters without imposing rigid hierarchy.

### Three-Tier Navigation
```
index.md (hub)
  -> detection (domain topic map)
    -> individual notes about detection
  -> classification (domain topic map)
    -> individual notes about classification
  -> signal-processing (domain topic map)
  -> experimental-methods (domain topic map)
```

### Topic Map Structure
Every topic map has:
- A `description` field explaining what this area covers
- **Core Ideas** -- wiki links to key notes WITH context phrases (not bare links)
- **Open Questions** -- what remains unresolved
- **Related Areas** -- cross-links to other topic maps

### Context Phrases Are Required
Bad: `- [[STFT window size affects frequency resolution]]`
Good: `- [[STFT window size affects frequency resolution]] -- determines the trade-off between temporal and spectral precision at 300kHz`

Context phrases explain WHY to follow the link. Without them, topic maps become address books instead of navigation aids.

### When to Split
When a topic map exceeds ~35 notes, split it into sub-topic-maps that link back to the parent. The hierarchy emerges from content, not from planning.

---

## Processing Pipeline (Full)

**NEVER write directly to notes/.** All content routes through the pipeline: inbox/ -> /reduce -> notes/. If you find yourself creating a file in notes/ without having run /reduce, STOP. Route through inbox/ first. The pipeline exists because direct writes skip quality gates.

Full automation is active from day one. All processing skills, quality gates, and maintenance mechanisms are available immediately.

### Pipeline Phases
1. **Seed** (/seed) -- Research a topic and deposit sources in inbox/
2. **Reduce** (/reduce) -- Extract atomic notes from inbox sources
3. **Reflect** (/reflect) -- Find connections between notes, add wiki links
4. **Reweave** (/reweave) -- Update older notes with new connections and context
5. **Verify** (/verify) -- Quality-check notes for description quality, schema compliance, composability

### Processing Depth
Configured in ops/config.yaml. Three levels:
- **deep** -- Full pipeline, fresh context per phase, maximum quality gates
- **standard** -- Full pipeline, balanced attention (default)
- **quick** -- Compressed pipeline, combine phases, high volume catch-up

### Extraction Categories
When processing sources, extract these types of insights:
- **findings** -- Empirical results, measurements, experimental outcomes
- **decisions** -- Architectural choices, parameter selections, design trade-offs
- **methods** -- Algorithms, processing techniques, experimental procedures
- **hypotheses** -- Untested predictions, proposed mechanisms, expected outcomes
- **baselines** -- Reference measurements, benchmark comparisons, known values
- **open-questions** -- Unresolved issues, gaps in understanding, future directions
- **patterns** -- Recurring structures, cross-cutting themes, design patterns

---

## Semantic Search (Full)

Your vault uses qmd for semantic discovery alongside wiki links.

### Two Discovery Layers
1. **Wiki links** (explicit) -- Deliberate connections you create. High precision, curated.
2. **Semantic search** (implicit) -- Content-based similarity discovery. Finds connections you didn't make.

### Using Semantic Search
```
# Search by concept
qmd search "frequency resolution trade-offs in STFT"

# Deep search with context
qmd deep_search "how does window size affect USV detection"

# Vector similarity
qmd vector_search "energy detection threshold optimization"
```

### When to Use Each Layer
- **Known connection** -- Wiki link. You know these notes relate. Make it explicit.
- **Discovery** -- Semantic search. "What else in my vault relates to this concept?"
- **Verification** -- Both. Check semantic results against wiki links to find missed connections.

---

## Schema (Full)

Every note has YAML frontmatter with structured metadata. Schema serves retrieval, not bureaucracy.

### Required Fields (all notes)
- `description` -- One sentence adding context beyond the title (max 200 chars)
- `topics` -- Array of wiki links to topic maps

### Domain Fields (research notes)
- `type` -- finding | decision | method | hypothesis | baseline | open-question | pattern
- `confidence` -- proven | likely | experimental | speculative
- `conditions` -- Array of experimental conditions (empty by default)
- `meta_state` -- current | outdated | superseded

### Schema Rules
- Templates in `templates/` are the single source of truth for schema
- Don't invent new enum values without updating the template first
- If a field is never used, remove it from the template
- Schema evolves through observation: notice pattern -> validate usefulness -> formalize in template

---

## Maintenance (Full)

Maintenance is condition-based, not scheduled. Specific conditions trigger specific actions.

### Condition Triggers
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

### Health Checks
Run `/arscontexta:health` for diagnostic reports:
- **quick** -- Schema, orphans, links (< 30 seconds)
- **full** -- All 8 categories including description quality and three-space boundaries
- **three-space** -- Boundary violation checks only

### Weekly Maintenance (target: 15 min)
```
/arscontexta:health   # diagnostic report (~1 min)
/reflect              # update connections (~3 min)
/reweave              # backward pass on old notes (~5 min)
/stats                # growth metrics snapshot (~1 min)
```

Run once per week. Condition triggers (above) handle urgent items between weekly runs.

---

## Self-Evolution (Full — removed from CLAUDE.md)

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

---

## Vault Self-Knowledge (Full)

Your system maintains its own methodology knowledge in ops/methodology/. This is the vault's self-knowledge -- why it was configured this way, what the current state is, and how it has evolved.

- Browse: `ls ops/methodology/`
- Query: `rg '^category:' ops/methodology/`
- Ask the research graph: `/arscontexta:ask [question about your system]`
- Get architecture advice: `/arscontexta:architect`

---

## Operational Learning Loop (Full)

Your system captures and processes friction signals through two channels:

### Observations (ops/observations/)
When you notice friction, surprises, process gaps, or methodology insights during work, capture them as atomic notes in ops/observations/. Each observation has a prose-sentence title and category (friction | surprise | process-gap | methodology).

### Tensions (ops/tensions/)
When two notes contradict each other, or an implementation conflicts with methodology, capture the tension in ops/tensions/. Each tension names the conflicting notes and tracks resolution status (pending | resolved | dissolved).

### Accumulation Triggers
- **10+ pending observations** -> Run /rethink to triage and process
- **5+ pending tensions** -> Run /rethink to resolve conflicts

---

## Task Management (Full — removed from CLAUDE.md)

### Processing Queue (ops/queue/)
Pipeline tasks tracked in JSON. Each note gets one queue entry that progresses through phases (create -> reflect -> reweave -> verify). Fresh context per phase ensures quality.

### Maintenance Queue
Maintenance work lives alongside pipeline work in the same queue. /next evaluates conditions against vault state: fired conditions create maintenance queue entries, satisfied conditions auto-close them.

---

## Operational Space (Full)

```
ops/
+-- derivation.md      -- why this system was configured this way
+-- derivation-manifest.md -- machine-readable config for runtime skills
+-- config.yaml        -- live configuration (edit to adjust dimensions)
+-- goals.md           -- current threads, orientation file
+-- reminders.md       -- time-bound commitments
+-- tasks.md           -- task tracking
+-- methodology/       -- vault self-knowledge
+-- observations/      -- friction signals
+-- tensions/          -- contradiction tracking
+-- queue/             -- processing queue
+-- sessions/          -- session logs
+-- health/            -- health report history
+-- queries/           -- graph analysis scripts
```

---

## Infrastructure Routing (Full — removed from CLAUDE.md)

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

---

## Templates (Full)

Templates in `templates/` define the structure of each note type. They are scaffolding, not rigid forms.

Available templates:
- `templates/note.md` -- Research note (finding, decision, method, hypothesis, etc.)
- `templates/topic-map.md` -- Topic map / MOC
- `templates/source-capture.md` -- Inbox source capture
- `templates/observation-note.md` -- Operational observation

Templates include `_schema` blocks that define validation rules. The template is the single source of truth for what fields and values are valid.

---

## Graph Analysis (Full)

Your wiki-linked vault is a graph database: nodes (markdown files), edges (wiki links), properties (YAML frontmatter), queried with ripgrep.

### Query Levels
1. **Field-level** -- `rg '^type: finding' notes/` -- query YAML fields across notes
2. **Node-level** -- backlinks, outgoing links for a specific note
3. **Graph-level** -- clusters, bridges, synthesis opportunities, influence patterns

### Key Operations
- **Triangle detection** -- Find open triads (synthesis opportunities)
- **Orphan detection** -- Notes with zero incoming links
- **Bridge detection** -- Structurally critical notes
- **Link density** -- Average links per note (target: 3+)

Use /graph for interactive analysis.

---

## Research Provenance (Full)

When source files contain provenance metadata (research tool, query, timestamp), preserve the chain:
```
source query -> inbox file (metadata preserved) -> /reduce -> notes/
```
Each note's Source footer links back to the inbox source. The chain is complete when you can trace any claim back to its original query.

---

## Helper Functions (Full)

### Safe Rename
Never rename a note manually. Use:
```bash
./ops/scripts/rename-note.sh "old title" "new title"
```
This renames with `git mv` and updates ALL wiki links across the vault.

### Graph Maintenance
- `./ops/scripts/orphan-notes.sh` -- Find notes with no incoming links
- `./ops/scripts/dangling-links.sh` -- Find broken wiki links
- `./ops/scripts/backlinks.sh "note title"` -- Count incoming links
- `./ops/scripts/link-density.sh` -- Measure average links per note
- `./ops/scripts/validate-schema.sh` -- Validate notes against template schemas

---

## Self-Improvement (Full)

When friction occurs (search fails, content placed wrong, user corrects you, workflow breaks):
1. Use /remember to capture it as an observation in ops/observations/
2. Continue your current work -- don't derail
3. If the same friction occurs 3+ times, propose updating this context file
4. If user explicitly says "remember this" or "always do X", update this context file immediately

---

## Guardrails (Full)

- Never store content the user explicitly asks to forget
- Never infer or record information the user has not shared
- Never present inferences as facts -- "I notice a pattern" not "this is true"
- No hidden processing -- every automated action is logged and inspectable
- Source attribution required -- trace claims back to origins
- Never fabricate sources or citations

---

## Self-Extension (Full — removed from CLAUDE.md)

### Building New Skills
Create `.claude/skills/skill-name/SKILL.md` with YAML frontmatter, instructions, quality gates, and output format.

### Building Hooks
Create `.claude/hooks/` scripts triggered on SessionStart, PostToolUse (Write), or Stop events.

### Extending Schema
Add domain-specific YAML fields to templates. Base fields (description, type) are universal. Add fields that make YOUR notes queryable for YOUR use case.

### Growing Topic Maps
When a topic map exceeds ~35 notes, split it. Create sub-topic-maps that link back to the parent.

---

## Common Pitfalls (Full)

### Collector's Fallacy
USV research has abundant sources -- papers, experiment logs, recording analyses. Capturing everything without processing creates the illusion of knowledge. Prevention: process before capturing more. If inbox has 3+ items, run /reduce before adding new sources.

### Orphan Drift
High creation volume during active research produces notes without connections. Orphan notes are invisible to traversal. Prevention: run /reflect after every batch of note creation. No note should stay orphaned longer than 7 days.

### Verbatim Risk
Signal processing literature tempts reproduction over transformation. Copying a paper's abstract into a note is not knowledge work. Prevention: every note must pass the generation effect test -- restate the insight in your own framing, connecting it to what you already know.

### Topic Map Sprawl
Research topics proliferate: detection, classification, DSP, training, augmentation, evaluation, per-recording analysis... Prevention: start with 4-5 broad topic maps. Split only when one exceeds ~35 notes. Resist creating a topic map for every sub-topic.

---

## Derivation Rationale (Full — removed from CLAUDE.md)

This knowledge system was derived on 2026-02-18 using the Research preset with these key choices:

- **Granularity: Atomic** -- One claim per note for maximum composability across detection, classification, DSP, and training domains
- **Organization: Flat** -- Topics cross-cut; folders would force artificial hierarchy
- **Linking: Explicit+Implicit** -- Wiki links + qmd semantic search for discovery
- **Processing: Heavy** -- Full pipeline with all quality gates from day one
- **Self-space: Disabled** -- Goals route to ops/goals.md; identity in this context file
- **Semantic search: qmd** -- Opted in for implicit connection discovery

Full derivation record: `ops/derivation.md`
Configuration: `ops/config.yaml`

---

## Recently Created Skills (Full — removed from CLAUDE.md)

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
