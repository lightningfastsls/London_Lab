# Use-Case Presets Reference

## Purpose

This document defines three use-case presets (Research, Personal Assistant, Experimental) as pre-validated coherent starting points in the configuration space. The derivation engine uses these as anchors — matching user signals to the closest preset, then adjusting individual dimensions based on conversation specifics. Each preset includes dimension values, block configuration, vocabulary mappings, failure mode risks, and a preset selection algorithm.

## Derivation Questions

- Which preset best matches a given user's signals and stated goals?
- What dimension values, block configurations, and vocabulary does each preset recommend?
- How should the derivation engine adjust dimensions when user signals diverge from the preset?
- What failure modes are most likely for each use case?
- When should the engine route to the Experimental preset instead of forcing a fit?
- How are tradition anchors translated into use-case-specific configurations?

## Curated Claims

### Use-Case Preset Configurations

#### Research preset optimizes for deep synthesis with full automation from day one

**Summary:** The Research preset combines Zettelkasten's atomic granularity and explicit linking with Cornell's multi-phase processing pipeline. Full automation from day one — all pipeline skills at full depth, all blocks active. This is the reference implementation and the most fully specified preset.

**Derivation Implication:** When user signals include academic language, source tracking, cross-domain synthesis, or high-volume knowledge processing, Research is the primary anchor. Its high processing and automation values mean users must accept significant upfront system complexity.

**Source:** Derived from Zettelkasten + Cornell traditions; arscontexta use-case analysis.

**Display name:** Knowledge Research

**Closest tradition:** Zettelkasten + Cornell processing phases

| Dimension | Value | Rationale |
|-----------|-------|-----------|
| Atomicity | 0.8 | One claim per note maximizes composability and cross-source comparison |
| Organization | 0.3 | Flat-associative with semantic search |
| Linking | 0.7 | Explicit typed connections + semantic discovery |
| Processing | 0.8 | Full pipeline: extract, connect, verify, reweave |
| Session | 0.7 | Fresh per task |
| Maintenance | 0.6 | Event-driven |
| Search | 0.8 | Semantic primary |
| Automation | 0.8 | Full automation from day one |

**Block configuration:**

| Category | Blocks |
|----------|--------|
| Always | atomic-notes, wiki-links, mocs, processing-pipeline, schema, maintenance, self-evolution, session-rhythm, templates, ethical-guardrails, helper-functions, graph-analysis |
| Conditional | semantic-search (if qmd opted in), multi-domain (if needed) |
| Optional | personality |
| Disabled | self-space (goals route to ops/) |

**Key settings:**
- `self_space: false`
- `qmd: true` (opted in during onboarding)
- `personality: "neutral-analytical"`
- `processing_depth: "full quality gates from day one"`

**Extraction categories:** claims, evidence, methodology-comparisons, contradictions, open-questions, design-patterns, design-dimensions

**Key vocabulary:**

| Level | Universal | Research |
|-------|-----------|----------|
| Note type | note | claim |
| Processing verb | reduce/extract | reduce |
| Connection verb | reflect/connect | reflect |
| Navigation unit | MOC | topic map |
| Collection name | notes/ | notes/ |
| Capture zone | inbox/ | inbox/ |

**Starter MOCs:** `{DOMAIN:domain}-overview`, `methods`, `open-questions`

**Domain-specific failure mode risks:**
- Collector's Fallacy (HIGH) — source material is abundant
- Orphan Drift (HIGH) — high creation volume without mandatory connection
- Verbatim Risk (HIGH) — source material tempts reproduction over transformation
- MOC Sprawl (HIGH) — topics proliferate in research domains
- Productivity Porn (HIGH) — meta-system building displaces actual research

**Example user statements:**
- "I read 5-10 papers a week on [topic] and need to track claims across disciplines"
- "I'm building a literature review and need to see connections between sources"
- "I want an academic knowledge base for my dissertation research"

---

#### Personal Assistant preset optimizes for life reflection with warm-supportive personality

**Summary:** A custom configuration combining moderate processing for pattern detection with warm personality voice for personal content. All skills available and active from day one, adapted for personal use. Full pipeline with personal extraction categories. Includes self-space for relationship tracking and growth awareness.

**Derivation Implication:** When user signals include personal growth language, relationship tracking, emotional awareness, or "remember what I care about" framing, Personal Assistant is the primary anchor. Its self-space and warm personality are the key differentiators from Research.

**Source:** Custom configuration; arscontexta use-case analysis on personal knowledge systems.

**Display name:** Personal Assistant

**Closest tradition:** Custom — moderate processing for pattern detection, warm voice for personal content

| Dimension | Value | Rationale |
|-----------|-------|-----------|
| Atomicity | 0.5 | Mixed — per-session reflections, not atomic decomposition |
| Organization | 0.5 | Light hierarchy for life areas |
| Linking | 0.3 | Associative connections between reflections |
| Processing | 0.7 | Full pipeline with personal extraction categories |
| Session | 0.3 | Continuous context across sessions |
| Maintenance | 0.4 | Condition-based check-ins |
| Search | 0.3 | Keyword primary, semantic optional |
| Automation | 0.6 | Semi-automated with personal touch |

**Block configuration:**

| Category | Blocks |
|----------|--------|
| Always | atomic-notes, wiki-links, mocs, processing-pipeline, schema, maintenance, self-evolution, session-rhythm, templates, ethical-guardrails, helper-functions, graph-analysis, personality, self-space |
| Conditional | semantic-search (if qmd opted in), multi-domain (if needed) |
| Optional | (none) |
| Disabled | (none) |

**Key settings:**
- `self_space: true`
- `qmd: "user_choice"` (choice during onboarding)
- `personality: "warm-supportive"`
- `processing_depth: "full quality gates"`

**Extraction categories:** reflections, relationship-dynamics, goals, habits, gratitude, lessons

**Key vocabulary:**

| Level | Universal | Personal Assistant |
|-------|-----------|-------------------|
| Note type | note | reflection |
| Processing verb | reduce/extract | surface |
| Connection verb | reflect/connect | find patterns |
| Navigation unit | MOC | life area |
| Collection name | notes/ | reflections/ |
| Capture zone | inbox/ | journal/ |

**Starter MOCs:** `life-areas`, `people`, `goals`

**Domain-specific failure mode risks:**
- Journal without reflection (HIGH) — capture without pattern detection
- Cognitive Outsourcing (medium) — pattern detection without human validation
- Emotional avoidance (medium) — system enables intellectual distancing from feelings
- Capture without connection (medium) — reflections stay isolated

**Example user statements:**
- "I want to track my growth and notice patterns in my life"
- "I need something that remembers what I care about across sessions"
- "Help me be more thoughtful about my relationships and goals"

---

#### Experimental preset enables user co-design through conversation-driven derivation

**Summary:** Rather than starting from a pre-configured anchor, the Experimental preset walks the user through each design decision with relevant thinking notes surfaced as context. All dimension values are null until user-chosen during onboarding. In-depth onboarding with thinking notes surfaced for every design decision — the user co-designs the system.

**Derivation Implication:** When no preset scores above 2.0 in affinity scoring, or the user explicitly asks to customize, route here. The engine must surface trade-off explanations for every dimension and check interaction constraints after each choice. Configuration paralysis is the primary risk.

**Source:** arscontexta derivation engine design; interaction constraint research.

**Display name:** Experimental / Build Your Own

**Closest tradition:** None — derived from conversation

| Dimension | Value | Rationale |
|-----------|-------|-----------|
| Atomicity | null | User-chosen during onboarding |
| Organization | null | User-chosen during onboarding |
| Linking | null | User-chosen during onboarding |
| Processing | null | User-chosen during onboarding |
| Session | null | User-chosen during onboarding |
| Maintenance | null | User-chosen during onboarding |
| Search | null | User-chosen during onboarding |
| Automation | null | User-chosen during onboarding |

**Block configuration:**

| Category | Blocks |
|----------|--------|
| Always | wiki-links, processing-pipeline, schema, maintenance, self-evolution, session-rhythm, templates, ethical-guardrails, helper-functions, graph-analysis |
| Conditional | atomic-notes, mocs, semantic-search, personality, self-space, multi-domain |
| Optional | (none) |
| Disabled | (none) |

**Key settings:**
- `self_space: "user_choice"`
- `qmd: "user_choice"`
- `personality: "derived_from_conversation"`
- `processing_depth: "full quality gates by default, user can opt down"`

**Extraction categories:** User-defined during init, guided by examples from research and personal-assistant presets

**Starter MOCs:** User-chosen during onboarding

**Domain-specific failure mode risks:**
- Configuration paralysis (HIGH) — too many choices without guidance
- Incoherent configuration (medium) — user choices may violate interaction constraints
- Under-specified system (medium) — skipping decisions leads to gaps

**Example user statements:**
- "I want to understand the trade-offs before choosing"
- "My use case doesn't fit the other presets"
- "I want to build something custom for [unusual domain]"

---

### Tradition Reference Summary

Traditions are named coherence points in the 8-dimensional space. They are reference anchors, not templates.

| Dimension | Zettelkasten | PARA | Cornell | Evergreen | GTD |
|-----------|-------------|------|---------|-----------|-----|
| Granularity | Atomic | Coarse | Medium (per-session) | Atomic | Task-sized |
| Organization | Flat | Hierarchical (4 folders) | Temporal | Flat | Hierarchical (contexts) |
| Linking | Explicit, bidirectional | Minimal (folder membership) | Implicit (cue columns) | Explicit, contextual | Minimal |
| Processing | Heavy (formulation + linking) | Light (progressive summarization) | Heavy (5 Rs structured) | Heavy (continuous rewriting) | Light (capture + route) |
| Navigation | 3-4 tier (emergent hubs) | 2 tier (folder browsing) | 2-3 tier (index-based) | 3 tier (link browsing) | 2 tier (context lists) |
| Maintenance | Continuous | Condition-based review | Spaced review | Continuous | Condition-based review |
| Schema | Moderate | Minimal | Moderate (cue/notes/summary) | Moderate | Dense (action metadata) |
| Automation | Convention | Manual | Convention | Convention | Automation-friendly |

**How traditions relate to presets:** Use-case presets are derived from traditions, adapted for specific domains. Research draws primarily from Zettelkasten + Cornell. Personal Assistant is a custom configuration combining moderate processing with warm personality. Experimental derives everything from conversation, using traditions as reference points during design decisions.

---

### Preset Selection Algorithm

The derivation engine maps conversation signals to the closest preset, then adjusts individual dimensions.

### Step 1: Signal Collection

Listen for signals in the user's description and follow-up answers. Each signal maps to one or more dimensions with a confidence level:

| Signal Type | Examples | Maps To |
|-------------|---------|---------|
| Volume indicators | "5-10 papers/week", "2-3 books/month" | Volume projection, navigation, maintenance |
| Processing verbs | "track claims", "remember reactions", "document decisions" | Granularity, processing intensity |
| Connection words | "across disciplines", "between projects", "patterns" | Linking philosophy, semantic search need |
| Frequency indicators | "a few times a week", "whenever I have a session", "after each project" | Maintenance trigger signals |
| Domain markers | "research papers", "personal growth", "relationships" | Closest preset match |
| Emotional register | "feel seen", "like a friend", "professional" | Personality dimensions |
| Design curiosity | "understand trade-offs", "custom", "unusual domain" | Experimental preset signal |

### Step 2: Preset Affinity Scoring

Score the user's signals against each preset. Each matching signal adds affinity:

```
For each preset:
  affinity = sum of (signal_weight x match_strength)
  where match_strength is:
    1.0 = signal directly matches preset characteristic
    0.5 = signal partially matches
    0.0 = no match
```

**Example:** "I read papers and want to track claims across disciplines"
- Research: "papers" (1.0) + "claims" (1.0) + "across disciplines" (1.0) = 3.0
- Personal Assistant: low scores
- Experimental: low scores (no design-curiosity signals)

Select the preset with highest affinity as the starting point. If no preset scores above 2.0 and the user shows design curiosity, route to Experimental.

### Step 3: Dimension Adjustment

Starting from the selected preset, adjust individual dimensions based on specific signals that diverge from the preset:

```
For each dimension:
  if user_signal contradicts preset_position:
    adjust dimension to signal-indicated position
    check interaction constraints
    cascade adjustments to dependent dimensions
```

**Example:** Research preset selected, but user says "I don't need deep analysis, just capture my reactions"
- Processing: heavy -> light (user signal overrides preset)
- Cascade check: light processing + atomic granularity -> WARN (atomic needs processing to recreate context)
- Resolution: suggest moderate granularity to match light processing, or confirm user wants atomic + light with the trade-off acknowledged

### Step 4: Coherence Validation

After all adjustments, check the final configuration against interaction constraints (see `interaction-constraints.md`). Flag hard violations (BLOCK), warn on soft violations (WARN with compensating mechanisms).

### Step 5: Vocabulary Derivation

Using the adjusted preset as the base vocabulary, override with any domain-native terms the user provided during conversation:

```
For each universal term:
  if user provided a domain-native equivalent:
    use user's term
  else:
    use preset's default term

  Verify: would this term feel natural to the user?
```

---

### Novel Domain Handling

When no preset cleanly matches the user's description, the Experimental preset provides the framework for co-design. Unlike the Research and Personal Assistant presets which start with pre-configured dimensions, Experimental walks the user through each design decision with relevant thinking notes surfaced as context.

### When to Route to Experimental

- No preset scores above 2.0 in affinity scoring
- User explicitly asks to customize or "build my own"
- Domain is unusual (competitive gaming, wine tasting, legal case tracking)
- User shows curiosity about trade-offs and design decisions

### Conversation-Driven Derivation

For each dimension:

1. Surface the relevant thinking note(s) that explain the trade-off
2. Present the dimension as a spectrum with examples from Research and Personal Assistant
3. Let the user choose their position
4. Check interaction constraints after each choice
5. Cascade adjustments to dependent dimensions

### Novel Schema Fields

Novel domains often need schema fields that no preset provides. Derive them from the domain's characteristics:

| Domain Characteristic | Suggests Schema Field |
|----------------------|----------------------|
| Temporal dynamics (things become outdated) | `meta_state: current | outdated | speculative` |
| Confidence tracking | `confidence: proven | likely | experimental` |
| Sequential progression | `prerequisites: ["[[concept]]"]` |
| Entity tracking | `person: Name`, `entity: Name` |
| Decision history | `superseded_by: "[[newer-decision]]"` |

### Worked Example: Wine Tasting Notes

**User:** "I'm getting into wine and want to track what I taste — flavors, regions, pairings, and which wines remind me of others."

**Affinity scores:**
- Personal Assistant: 1.5 ("remember things", personal interest)
- Research: 1.0 ("track", systematic)
- Experimental: route here (no preset above 2.0)

**Conversation-driven configuration:**

| Dimension | Position | Rationale |
|-----------|----------|-----------|
| Atomicity | Moderate | Per-wine notes, not atomic flavor claims |
| Organization | Flat | Wines cross regions and varietals |
| Linking | Explicit | Direct connections: "this Barolo reminded me of that Nebbiolo" |
| Processing | Light | Capture tasting notes, connect to similar wines |
| Session | Continuous | Low volume, ongoing |
| Maintenance | Condition-based | Low volume — review when 5+ unconnected tasting notes accumulate |
| Search | Moderate | Structured fields for filtering + semantic for "wines like this" |
| Automation | Convention | Light overhead |

**Novel vocabulary:** "tasting note" (not "reflection" or "claim"), "wine library" (not "life area"), "cellar" (not "archive")

**Novel schema field:** `pairing: ["food pairing notes"]` — domain-specific field not in any preset.

---

## Exclusion Notes

- **Team/Collaborative preset:** Evaluated but excluded — multi-user knowledge systems introduce access control, merge conflicts, and shared-vs-personal note boundaries that require a separate architectural layer. May be added when collaboration primitives are designed.
- **Developer/Engineering preset:** Evaluated but excluded — code-centric knowledge management (ADRs, runbooks, incident postmortems) overlaps heavily with Research preset. Domain vocabulary differs but dimension values are nearly identical. Can be served by Research with vocabulary overrides.
- **Student preset:** Considered as distinct from Learning use case — excluded because the Learning tradition reference in tradition-presets.md already covers this. A Student preset would be Cornell + Zettelkasten atomicity, which is exactly the Learning row in the use-case derivation summary.
- **Journaling-only preset:** Excluded — journaling without pattern detection or connection is a degenerate case (capture-only). Personal Assistant with reduced processing covers this better than a dedicated preset.

---

## Version

- **Date:** 2026-03-20
- **Source claim count:** 3 presets + selection algorithm + novel domain handling + 4 exclusion candidates = 9
- **Included count:** 3 presets (Research, Personal Assistant, Experimental) + 2 supporting algorithms
- **Excluded count:** 4 (Team/Collaborative, Developer/Engineering, Student, Journaling-only)
