# Methodology Tradition Presets

## Purpose

This document defines six methodology traditions (Zettelkasten, PARA, Evergreen, Cornell, GTD, Memory Palace) as named points in an 8-dimensional configuration space. The derivation engine uses these as coherent reference anchors — each tradition represents a region where dimension interactions have been resolved through long practice. They are starting points for derivation, not templates to copy.

## Derivation Questions

- What coherent starting configurations exist for a given use case?
- What dimension values does tradition X recommend, and why?
- Which traditions can be mixed, and what interaction constraints apply?
- What processing philosophy does each tradition assume?
- When does a tradition's coherence break down?
- What is a tradition's "best for" profile?

## Curated Claims

### Configuration Space Framing

Traditions are not competing paradigms but different trade-off choices along shared dimensions. Zettelkasten maximizes connection density at the cost of heavy processing. PARA maximizes project organization at the cost of cross-domain synthesis. Each optimizes for a different purpose.

Derived systems can mix choices across traditions when the mixing is principled — each dimension choice justified independently by research claims, with interaction constraints checked for coherence.

---

### Tradition Configurations

#### Zettelkasten optimizes for deep synthesis through maximum-commitment atomic configuration

**Summary:** Zettelkasten places every dimension at its intensive pole — atomic granularity, flat organization, explicit bidirectional linking, heavy processing. It is the highest-commitment configuration, internally consistent because heavy processing sustains atomic granularity's demands. Breaks when processing discipline lapses (notes created but never connected).

**Derivation Implication:** When a user's signals indicate deep synthesis, cross-domain connection density, or long-term knowledge accumulation, Zettelkasten is the primary tradition anchor. Its atomic granularity and heavy processing requirements should be adopted together or not at all.

**Source:** Luhmann's Zettelkasten practice; arscontexta methodology research on dimension interactions.

**Configuration:**

| Dimension | Value | Rationale |
|-----------|-------|-----------|
| Granularity | atomic | One claim per note maximizes composability and reuse |
| Organization | flat | Associative ontology adapts while hierarchy brittles |
| Linking | explicit, bidirectional | Dense typed connections are the primary value mechanism |
| Processing | heavy (formulation + linking) | Careful formulation at creation time ensures quality |
| Navigation | 3-4 tier (emergent hubs) | Deep hierarchy manages attention across thousands of notes |
| Maintenance | continuous | Living documents, not finished artifacts |
| Schema | moderate | Enough for querying, not so much as to impede capture |
| Automation | convention | Traditionally manual; agent operation enables automation |

**Process step:** Formulate — transform source material into your own words as a falsifiable claim.

**Best for:** Research, academic work, long-term intellectual projects, concept-heavy domains.

---

#### PARA optimizes for project execution through minimum-commitment coarse configuration

**Summary:** PARA places every dimension at its extensive pole — coarse granularity, hierarchical organization, minimal linking, light processing. It is the lowest-commitment configuration, internally consistent because coarse notes with minimal linking don't demand deep processing. Breaks when cross-domain synthesis is needed (folder silos prevent connection finding).

**Derivation Implication:** When a user's signals indicate project execution, quick filing, or task-heavy workflows, PARA is the primary anchor. Its coarse granularity and light processing work together — adopting PARA's linking without its granularity creates orphan risk.

**Source:** Tiago Forte's PARA method; arscontexta methodology research on dimension interactions.

**Configuration:**

| Dimension | Value | Rationale |
|-----------|-------|-----------|
| Granularity | coarse | Topic-level or project-level documents |
| Organization | hierarchical (4 folders) | Clear filing taxonomy reduces decision fatigue |
| Linking | minimal (folder membership) | Folder placement IS the primary organizational mechanism |
| Processing | light (progressive summarization) | Highlight → bold → summarize, minimal transformation |
| Navigation | 2 tier (folder browsing) | Four top-level folders, project subfolders |
| Maintenance | condition-based review | Triage when project status changes accumulate |
| Schema | minimal | Actionability field (project vs area vs resource) |
| Automation | manual | Designed for human operation with minimal tooling |

**Process step:** Summarize — progressively distill highlights without transformation.

**Best for:** Project management, task-heavy workflows, getting things done, storage-oriented systems.

---

#### Evergreen Notes optimize for evolving understanding through continuous rewriting

**Summary:** Similar to Zettelkasten in structure (atomic, flat, explicit linking) but diverges on processing philosophy: continuous rewriting vs. careful initial formulation. The temporal dimension of processing differs even though intensity is comparable. Breaks when revision never happens (aspirational evergreen that's actually write-once).

**Derivation Implication:** When a user values evolving understanding over initial precision, Evergreen is preferred over Zettelkasten. The distinction matters for processing pipeline design — Evergreen systems need revision triggers, not just creation-time quality gates.

**Source:** Andy Matuschak's Evergreen Notes practice; arscontexta methodology research on processing philosophy.

**Configuration:**

| Dimension | Value | Rationale |
|-----------|-------|-----------|
| Granularity | atomic | Concept-oriented, one idea per note |
| Organization | flat | No folder hierarchy; links define structure |
| Linking | explicit, contextual | Links with surrounding prose context explain WHY |
| Processing | heavy (continuous rewriting) | Notes evolve over time through repeated revisiting |
| Navigation | 3 tier (link browsing) | Follow connections, not categories |
| Maintenance | continuous | "Evergreen" means perpetual growth and revision |
| Schema | moderate | Enough metadata to support discovery |
| Automation | convention | Designed for personal practice; automation optional |

**Process step:** Rewrite — revisit and improve notes as understanding deepens, rather than formulating once.

**Best for:** Personal knowledge development, conceptual exploration, writing projects, intellectual growth.

---

#### Cornell Note-Taking optimizes for retention through structured multi-phase processing

**Summary:** The most structured processing pipeline among traditions. Each of the 5 Rs (Record, Reduce, Recite, Reflect, Review) has a distinct purpose. Uses medium granularity (per-session documents) with temporal organization and implicit linking through cue columns. Breaks when processing phases are skipped (especially the Recite self-testing step that provides the core retention benefit).

**Derivation Implication:** When a user's signals indicate learning, study, or retention as primary goals, Cornell's processing phases should be adopted. Its cue-generation step can be mixed with other traditions' granularity choices, but the multi-phase pipeline is the core value.

**Source:** Walter Pauk's Cornell Note-Taking System; arscontexta methodology research on processing pipelines.

**Configuration:**

| Dimension | Value | Rationale |
|-----------|-------|-----------|
| Granularity | medium (per-session) | One document per lecture/session, structured internally |
| Organization | temporal | Chronological primary axis |
| Linking | implicit (cue columns) | Cue column creates retrieval prompts, not explicit graph edges |
| Processing | heavy (5 Rs structured) | Record → Reduce → Recite → Reflect → Review |
| Navigation | 2-3 tier (index-based) | Master index points to session documents |
| Maintenance | spaced review | Revisiting at increasing intervals based on note maturity |
| Schema | moderate | Cue/notes/summary structure per document |
| Automation | convention | Paper-native; agent adaptation requires skill encoding |

**Process step:** Generate cues — create self-testing prompts that force active recall.

**Best for:** Learning, study, courses, structured knowledge acquisition.

---

#### GTD optimizes for stress-free execution where schema IS the processing

**Summary:** Dense schema with light processing seems contradictory, but the schema IS the processing — classifying an item as @computer/2-minutes/low-energy is the routing that GTD calls processing. Task-sized granularity with hierarchical context organization and minimal linking. Breaks when system review lapses (trust in the system collapses).

**Derivation Implication:** When a user's signals indicate task management, execution, or operational workflows, GTD's dense-schema-as-processing pattern applies. Its automation-friendly design makes it the most natural tradition for agent-operated systems, but its task-sized granularity limits knowledge synthesis.

**Source:** David Allen's Getting Things Done; arscontexta methodology research on schema-processing interaction.

**Configuration:**

| Dimension | Value | Rationale |
|-----------|-------|-----------|
| Granularity | task-sized | Next actions, not knowledge claims |
| Organization | hierarchical (contexts) | @context, @project, @waiting-for organize for execution |
| Linking | minimal | Context membership, not knowledge connections |
| Processing | light (capture + route) | Quick triage: actionable? → project/action/reference/trash |
| Navigation | 2 tier (context lists) | Browse by context for situation-appropriate actions |
| Maintenance | condition-based review | Per-session: today's actions. Threshold-triggered: full system review when items accumulate |
| Schema | dense (action metadata) | Due dates, contexts, energy levels, priorities |
| Automation | automation-friendly | Repeating tasks, condition-based triggers, reminders |

**Process step:** Route — classify each item by actionability and context, not by content.

**Best for:** Task management, productivity, execution-heavy workflows, operational systems.

---

#### Memory Palace optimizes for spatial-mnemonic retrieval through physical metaphor hierarchy

**Summary:** The most unusual configuration — spatial rather than textual organization. Uses moderate granularity with hierarchical spatial organization (palace, room, locus) and explicit spatial-sequence linking. Breaks when the spatial metaphor is forced onto non-sequential information. Agent adaptation is speculative because the method relies on visual imagination that agents don't have.

**Derivation Implication:** Memory Palace has limited direct applicability for agent-operated systems (no spatial cognition), but its spatial-hierarchical navigation pattern transfers. When a user mentions sequential recall or spatial organization preferences, this tradition's navigation structure can be borrowed without adopting the full configuration.

**Source:** Classical Method of Loci; arscontexta methodology research on non-textual organization.

**Configuration:**

| Dimension | Value | Rationale |
|-----------|-------|-----------|
| Granularity | moderate | Items organized in spatial loci, grouped by room/path |
| Organization | hierarchical (spatial) | Palace → room → locus creates physical metaphor hierarchy |
| Linking | explicit (spatial sequence) | Loci connect by physical adjacency in the imagined space |
| Processing | moderate (visualization) | Creating vivid, unusual images for each locus |
| Navigation | 3 tier (palace → room → locus) | Spatial navigation provides strong retrieval cues |
| Maintenance | condition-based (palace review) | Walk the palace when associations weaken or new content needs encoding |
| Schema | minimal | Location and image descriptors |
| Automation | manual | Intrinsically human-cognitive; agent adaptation speculative |

**Process step:** Visualize — encode information as vivid spatial-mnemonic images.

**Best for:** Sequential recall, presentation preparation, memorization tasks. Limited applicability for agent-operated systems (no spatial cognition), but the spatial-hierarchical navigation pattern transfers.

---

### Use-Case Derivation Summary

These are derived from tradition presets, adapted for specific use cases. Each cell shows the recommended starting position.

| Dimension | Research | Life Mgmt | Learning | Relationships | Therapy | Creative | Companion |
|-----------|----------|-----------|----------|---------------|---------|----------|-----------|
| Granularity | atomic | moderate | atomic | moderate | moderate | moderate | coarse |
| Organization | flat | flat | flat | flat | flat | flat | flat |
| Linking | explicit+implicit | explicit | explicit+implicit | explicit | explicit | explicit+implicit | explicit |
| Processing | heavy | light | heavy | light | moderate | moderate | light |
| Nav depth | 3-tier | 2-tier | 3-tier | 2-tier | 2-tier | 2-tier | 2-tier |
| Maintenance | condition-based (tight) | condition-based (lax) | condition-based (tight) | condition-based (lax) | condition-based (tight) | condition-based (lax) | condition-based (lax) |
| Schema | moderate | minimal | moderate | moderate | moderate | minimal | minimal |
| Automation | convention | convention | convention | convention | convention | convention | convention |

**Derivation from traditions:**
- Research → primarily Zettelkasten, with Cornell processing phases (the 6 Rs)
- Life Management → PARA-influenced, with flat organization (Zettelkasten adaptation)
- Learning → Cornell core, with Zettelkasten atomicity for long-term retention
- Relationships → custom — moderate schema for preference tracking, light processing
- Therapy → custom — moderate processing for pattern detection, tight condition-based review for continuity
- Creative → Evergreen-influenced — living ideas that evolve through revisiting
- Companion → lightest configuration — memory building without processing overhead

**Why flat organization for all:** The research strongly favors flat-associative for agent-operated systems. Hierarchy can be added via MOCs without folder nesting. This is the one dimension where the recommendation holds across all use cases.

**Why convention automation for all:** Following Gall's Law (complex systems evolve from simple working systems), all systems start at convention level. Users add automation at friction points rather than deploying full automation from day one.

**Automation upgrades per platform:** The presets above are the floor, not the ceiling. During init, tier-1 platforms (Claude Code) should prompt the user about adding hooks-blueprint and validation-hooks. A Research use case on Claude Code can reasonably start at "convention trending toward automation" — the preset is the starting point, and the init wizard should suggest upgrades when the platform supports them.

---

### Mixing Traditions

When a user selects "Custom / Mixed" in the init wizard, present traditions as reference points:

1. **Show the tradition presets table** — each row is a coherent starting point
2. **Let user pick per-dimension** — selecting values from different traditions
3. **Check interaction constraints** — flag combinations where traditions conflict
4. **Document the mix** — include which tradition each choice came from in the derivation rationale

**Valid mixing examples:**
- Zettelkasten atomicity + Cornell processing phases + PARA project folders = research system with structured extraction and active project tracking
- Evergreen continuous rewriting + GTD condition-based review = creative system with disciplined maintenance

**Invalid mixing (interaction constraints fire):**
- Zettelkasten atomicity + PARA minimal linking → atomic notes need explicit connections
- GTD dense schema + Zettelkasten manual operation → dense schemas need validation support
- Memory Palace spatial hierarchy + Evergreen flat organization → contradictory organizational principles

---

## Exclusion Notes

- **Bullet Journal (Ryder Carroll):** Evaluated but excluded — primarily analog/paper-based with rapid logging conventions that don't map cleanly to agent-operated dimension space. Could be revisited if physical-digital bridge patterns emerge.
- **Commonplace Book tradition:** Excluded — coarse granularity with no processing pipeline makes it a degenerate case (capture-only). Its value is historical context, not derivation guidance.
- **Building a Second Brain (Tiago Forte):** Not treated as a separate tradition — it largely overlaps with PARA + progressive summarization. PARA configuration already captures its dimension values.
- **Roam/logseq-native outlining:** Excluded as a tradition — outliner structure is an implementation detail, not a dimension-level configuration choice. Outlining affects how notes are edited, not how the system is configured.

---

## Version

- **Date:** 2026-03-20
- **Source claim count:** 6 traditions evaluated + 4 exclusion candidates = 10
- **Included count:** 6 (Zettelkasten, PARA, Evergreen, Cornell, GTD, Memory Palace)
- **Excluded count:** 4 (Bullet Journal, Commonplace Book, BASB, Roam-native outlining)
