# Three-Space Architecture Reference

## Purpose

This document defines the three-space architecture (Self, Notes, Ops) that every generated system must implement. It serves as the derivation engine's primary structural reference for workspace partitioning, ensuring that each space's durability profile, growth pattern, and query characteristics remain distinct. The six failure modes of conflation establish why space boundaries are architectural constraints, not organizational preferences.

---

## Derivation Questions

1. **How should workspace directories be partitioned?** Every generated system divides into self/, notes/, and ops/ based on durability, growth pattern, and load pattern — not by content topic.
2. **When should self-space be enabled vs disabled?** Self-space is configurable (off for research vaults, on for personal assistant vaults) depending on whether persistent agent identity adds value beyond what the context file provides.
3. **Where does a specific piece of content belong?** The memory type routing decision tree classifies content by asking: is it about the agent? Is it durable? Is it domain knowledge? Is it operational coordination?
4. **What breaks when spaces are conflated?** Six documented failure modes predict the specific decay pattern for each possible conflation, enabling the derivation engine to generate targeted warnings.
5. **How does content move between spaces?** Promotion is one-directional: ops/ -> notes/ or self/. Durable knowledge never becomes temporal scaffolding.

---

## Curated Claims

### Space Definitions

#### Self space holds the agent's persistent mind

**Summary:** Self space stores agent identity, methodology, goals, and accumulated operational wisdom. It is permanent, slow-growing (tens of files), and fully loaded at every session start. Without it, every session starts from zero.

**Derivation Implication:** When self-space is enabled, the derivation engine must scaffold identity.md, methodology.md, and goals.md. When disabled, these functions route to ops/goals.md (orientation), ops/methodology/ (self-knowledge), and the context file (identity).

**Source:** arscontexta three-space architecture — self-space primitive.

#### Notes space is the user's knowledge graph

**Summary:** Notes space holds durable, composable, worth-finding-again knowledge. It grows steadily (10-50 claims/week for research, 2-5 memories/week for companion), uses progressive disclosure (MOC navigation, semantic search, link traversal), and is the reason the system exists.

**Derivation Implication:** The derivation engine must generate structural constants (flat folder, prose-sentence titles, MOC navigation, wiki links, topics footer) while adapting vocabulary and schema fields to the domain.

**Source:** arscontexta three-space architecture — notes-space primitive.

#### Ops space provides operational coordination

**Summary:** Ops space holds temporal content that flows through, gets processed, and either graduates to notes/self or gets archived. It fluctuates in size, is loaded only in targeted fashion, and keeps the knowledge graph clean by separating scaffolding from durable knowledge.

**Derivation Implication:** The derivation engine must generate ops/ with derivation.md, derivation-manifest.md, reminders.md, sessions/, health/, observations/, queue/, and user-overrides.md. Content promotion rules must be included in the context file.

**Source:** arscontexta three-space architecture — ops-space primitive.

### Conflation Failure Modes

#### Conflating ops into notes pollutes search with processing debris

**Summary:** When processing queue state, session logs, and health reports end up in notes/, search returns temporal content alongside real knowledge. Note counts inflate, MOCs accumulate operational entries, and the graph becomes noisy.

**Derivation Implication:** The derivation engine must generate clear routing rules that prevent operational state from entering notes/. Generated context files must include the "What Does NOT Belong in Notes" list.

**Source:** arscontexta conflation failure analysis — ops-into-notes.

#### Conflating self into notes creates schema confusion and search pollution

**Summary:** When agent identity and methodology end up in the user's knowledge graph, schema fields diverge, search mixes agent self-knowledge with domain content, and progressive disclosure loads irrelevant content.

**Derivation Implication:** When self-space is enabled, the derivation engine must enforce that agent self-knowledge routes exclusively to self/, not notes/. The routing decision tree must be included.

**Source:** arscontexta conflation failure analysis — self-into-notes.

#### Conflating notes into ops traps insights in temporal storage

**Summary:** When genuine insights stay in session logs or observation files without promotion, they are lost when ops/ is archived. Knowledge cannot compound because session-trapped insights cannot be linked. The vault appears thinner than the work invested.

**Derivation Implication:** The derivation engine must generate the content promotion rule (ops/ -> notes/) and include promotion triggers in the context file. The /reflect phase must be part of the processing pipeline.

**Source:** arscontexta conflation failure analysis — notes-into-ops.

#### Conflating self into ops scatters identity across session logs

**Summary:** When agent identity is distributed across 50 session logs instead of curated self/ files, orientation fails because the agent cannot load all logs. Identity drifts without an authoritative source.

**Derivation Implication:** When self-space is enabled, session logs (ops/) must not serve as the primary location for identity evolution. The derivation engine must route identity learnings to self/identity.md or self/methodology.md.

**Source:** arscontexta conflation failure analysis — self-into-ops.

#### Conflating ops into self bloats identity with temporal state

**Summary:** When queue status, health metrics, and processing state enter self/, it becomes too large to load at session start. Temporal content creates noise in identity orientation.

**Derivation Implication:** The derivation engine must enforce that self/ contains only durable self-knowledge. Current processing state always routes to ops/ regardless of self-space configuration.

**Source:** arscontexta conflation failure analysis — ops-into-self.

#### Conflating notes into self stores domain knowledge as identity

**Summary:** When domain knowledge is stored in self/ because it "felt personally relevant," self/ bloats beyond loadable size. Search in notes/ misses content hidden in self/. The distinction between agent self-knowledge and domain knowledge collapses.

**Derivation Implication:** The derivation engine must include the design rule: "Only what the agent needs about itself." Domain knowledge always routes to notes/, even when the agent finds it interesting.

**Source:** arscontexta conflation failure analysis — notes-into-self.

---

### Supplementary Reference

The following tables and specifications preserve detailed implementation guidance that supports the curated claims above.

#### Self Space — Core Files (when enabled)

| File | Contents | Update Trigger |
|------|----------|----------------|
| `identity.md` | Who the agent is — personality, voice, approach, values | Rarely (personality doesn't change often) |
| `methodology.md` | How the agent works — quality standards, processing principles, operational patterns | When operational learnings accumulate (evolves as agent learns) |
| `goals.md` | Current threads — what's active, deferred, completed | Every session (the orientation file) |

#### Self Space — Optional Extensions

| File/Directory | Included When | Purpose |
|---------------|---------------|---------|
| `relationships.md` | Domain involves multiple people | Key people, preferences, interaction patterns |
| `memory/` | Agent needs atomic self-knowledge beyond core files | Prose-titled atomic notes mirroring the notes/ pattern |
| `journal/` | Agent captures raw session observations | Processing input for self-knowledge — analogous to inbox |
| `sessions/` | Session logs need graduated storage | Session-specific logs that might graduate to memory/ or methodology.md |

#### Self Space — Disabled Fallback Routing

| Function | Fallback Location | Notes |
|----------|-------------------|-------|
| Goals / orientation | `ops/goals.md` | Current threads, active work — the session orientation file |
| Methodology / self-knowledge | `ops/methodology/` | Vault configuration rationale, pipeline config, evolution history |
| Identity | Context file | Agent personality baked into the context file directly |

The key insight is that self/ serves two distinct purposes: (1) agent identity/personality and (2) operational orientation. Research vaults typically do not need a persistent agent personality — the context file handles identity. Operational orientation (goals, methodology) routes to ops/ where it belongs alongside other operational state.

#### Toggle Mechanism

Self space is toggled via `/architect`:

```
/architect enable self    # Creates self/ with identity.md, methodology.md, goals.md
/architect disable self   # Migrates goals to ops/goals.md, archives self/
```

The toggle preserves content — disabling self/ moves goals to ops/ rather than deleting them. Enabling self/ creates the directory and scaffolds the core files.

#### Self Space Design Rule

**Only what the agent needs about itself.** Self/ is not a second knowledge graph — it holds agent identity, operational learning, and current orientation. Domain knowledge lives in notes/. Processing scaffolding lives in ops/. Self/ answers: "Who am I? How do I work? What am I working on?"

#### Session Rhythm Integration

Self space integrates with the session rhythm primitive, but is not required by it:

```
Orient -> read orientation state (self/ if enabled, ops/goals.md if not)
Work   -> do the actual task, surface connections
Persist -> update orientation state (self/ or ops/goals.md)
```

The session rhythm primitive depends on markdown-yaml, not on self-space. When self/ is disabled, the orient/persist cycle still works — it just reads from and writes to ops/ instead. The context file always provides methodology and identity; self/ adds a richer, evolving layer on top.

#### Notes Space — Structural Constants

| Constant | Implementation | Why It's Universal |
|----------|---------------|-------------------|
| Flat folder | No subfolders for organization | Prevents folder reorganization from breaking links |
| Prose-sentence titles | Each note makes one claim, titled as a sentence | Enables wiki-link-as-prose pattern |
| MOC navigation | Hub -> domain -> topic -> notes | Manages attention at scale |
| Wiki links | `[[note title]]` creates graph edges | Spreading activation without infrastructure |
| Topics footer | Every note declares MOC membership | Bidirectional navigation |

#### Notes Space — Domain Adaptation

| Aspect | Universal Pattern | Domain Adaptation |
|--------|-------------------|-------------------|
| Folder name | `notes/` | Vocabulary transform: `reflections/`, `concepts/`, `decisions/`, `memories/` |
| Note title style | Prose sentence | Domain phrasing: "client showed progress on..." vs "the evidence suggests..." |
| Schema fields | `description`, `topics` | Domain fields: `person`, `session_date`, `confidence`, `alternatives` |
| MOC vocabulary | Hub, domain, topic | Domain groupings: "themes", "project areas", "study guides" |

#### Notes Space Design Rule

**Durable, composable, worth finding again.** If it won't be queried or linked, it doesn't belong here. Session-specific observations start in ops/ and get promoted when they earn permanence. Raw capture starts in inbox/ and gets processed into notes/ through the processing pipeline.

#### What Does NOT Belong in Notes

- Processing queue state -> ops/queue/
- Session logs -> ops/sessions/
- Agent self-knowledge -> self/
- Health reports -> ops/health/
- Temporary scaffolding -> ops/

#### Ops Space — Contents

| Directory | Contents | Lifecycle |
|-----------|----------|-----------|
| `derivation.md` | The original derivation rationale — dimension positions, tradition mapping, vocabulary choices, rationale for each decision | Semi-permanent — updated only during reseed |
| `derivation-manifest.md` | Version tracking — arscontexta version, research snapshot date, feature blocks enabled, coherence validation results | Semi-permanent — updated during reseed |
| `reminders.md` | User-delegated time-bound actions — flat markdown, checked at orient, items removed on completion | Active rotation — items added and removed regularly |
| `sessions/` | Session logs — what happened today, handoff notes for next session | Rolling archive — logs older than 30 days can be archived without knowledge loss |
| `health/` | Schema validation results, orphan lists, link health metrics — point-in-time snapshots | Superseding — yesterday's report is superseded by today's |
| `observations/` | Operational learnings captured during work — pre-promotion holding area | Graduating — observations get promoted to notes/ or self/ when they earn permanence |
| `queue/` | Processing queue state — what needs extraction, connection, verification | Flowing — items move through and complete |
| `user-overrides.md` | User customizations that reseed must preserve as immutable | Semi-permanent — grows as user modifies generated content |

#### Reminders Specification

`ops/reminders.md` is a flat markdown file for user-delegated time-bound actions:

```markdown
# Reminders

- [ ] 2026-02-15: Follow up with Sarah about the new job
- [ ] 2026-03-01: Follow up with Sarah about job offer
- [x] 2026-02-10: Send reading list to Alex (done 2026-02-10)
```

**Behavior:**
- Checked at orient (session start) — due items surface in the morning briefing
- Completed items are marked with `[x]` and date, then archived when the list grows long
- No complex scheduling — if the user needs recurring reminders, that's a different tool

#### Content Promotion Rule

**Content moves from temporal to durable, never the reverse.** Promotion is one-directional:

```
ops/observations/ -> notes/ (when observation proves durable)
ops/observations/ -> self/methodology.md (when observation is about agent operation)
ops/sessions/ -> self/memory/ (when session insight is personally significant)
```

Content never moves FROM notes/ or self/ INTO ops/. Durable knowledge doesn't become temporal scaffolding.

#### The Promotion Pattern

1. Content enters ops/ at low ceremony (friction logs, session notes, queue entries)
2. When it demonstrates persistence — same observation recurs, insight proves useful across sessions, pattern is confirmed — it gets promoted
3. Promotion means creating a proper note in notes/ or adding to self/, not moving the ops entry
4. The ops entry can then be archived, its value extracted

#### Filesystem Layout — Claude Code Platform

```
project-root/
├── CLAUDE.md                    # context file (methodology + operational instructions)
├── .claude/
│   ├── hooks/                   # event-driven automation
│   ├── skills/                  # methodology-as-code
│   └── settings.json            # platform configuration
├── self/
│   ├── identity.md
│   ├── methodology.md
│   ├── goals.md
│   ├── relationships.md         # optional
│   ├── memory/                  # optional
│   └── journal/                 # optional
├── notes/                       # or domain-specific name (reflections/, concepts/, etc.)
│   ├── index.md                 # hub MOC
│   ├── [domain-mocs].md         # domain/topic MOCs
│   └── [prose-titled-notes].md  # atomic notes
├── inbox/                       # or domain-specific name (journal/, encounters/, etc.)
├── archive/                     # processed sources
├── templates/                   # note templates
└── ops/
    ├── derivation.md
    ├── derivation-manifest.md
    ├── reminders.md
    ├── user-overrides.md
    ├── sessions/
    ├── health/
    ├── observations/
    └── queue/
```

#### Memory Type Routing Decision Tree

```
Is this about the agent itself?
├── YES: Is it durable self-knowledge?
│   ├── YES -> self/ (identity, methodology, goals, memory)
│   └── NO -> ops/ (session log, current processing state)
│
└── NO: Is this domain knowledge?
    ├── YES: Is it durable, composable, worth finding again?
    │   ├── YES -> notes/ (atomic note with proper schema)
    │   └── NO -> ops/ (observation, friction log, session note)
    │       └── May be promoted to notes/ later if it persists
    │
    └── NO: Is this operational coordination?
        └── YES -> ops/ (queue state, health report, session handoff)
```

#### Quick Routing Rules

| Content Type | Destination | Why |
|-------------|-------------|-----|
| "I work best when..." | self/methodology.md | Agent operational learning |
| "The user prefers..." | self/relationships.md | Agent knowledge about user |
| "Today I processed..." | ops/sessions/ | Temporal processing state |
| "Spaced repetition helps memory" | notes/ | Domain knowledge |
| "The reduce skill over-extracts" | ops/observations/ | Operational friction (may promote) |
| "Queue has 12 items" | ops/queue/ | Temporal coordination state |
| "Schema validation passed" | ops/health/ | Point-in-time diagnostic |
| "My goal this quarter is..." | self/goals.md | Agent orientation |
| "Remember to follow up by Friday" | ops/reminders.md | Time-bound action |

---

## Exclusion Notes

### Four-space models (adding archive/ as a top-level space)
**Reason:** Archive is a lifecycle state of content, not a distinct space with its own durability profile. Processed sources in archive/ are inert — they don't grow, get queried differently, or have unique load patterns. Elevating archive/ to a space would add complexity without preventing any failure mode.

### Per-domain space partitioning (separate spaces per knowledge domain)
**Reason:** Domains are organized within notes/ via MOCs, not via separate top-level spaces. Separate domain spaces would break cross-domain linking and fragment the knowledge graph.

---

## Cross-Reference

- **Failure modes that afflict each space:** See `failure-modes.md` for the full failure mode taxonomy. Conflation failures (this document) are structural; failure-modes.md covers operational decay (collector's fallacy, orphan drift, schema erosion).
- **How personality affects each space:** See `personality-layer.md` for how warmth/formality dimensions change the voice of self/identity.md, skill instructions, and health reports.
- **What goes in each space per domain:** See `use-case-presets.md` for domain-specific routing decisions (therapy reflections vs research claims vs PM decisions).
- **Kernel primitives that depend on three-space separation:** `self-space` (configurable), `session-rhythm`, `discovery-first`, `task-stack`, `methodology-folder`, and `session-capture` all assume clean space boundaries. See `kernel.yaml`.

---

## Version
- **Last curated:** 2026-03-20
- **Source claims evaluated:** 11
- **Claims included:** 9 (3 space definitions, 6 conflation failure modes)
- **Claims excluded:** 2
