# Dimension Interaction Constraints

## Purpose

Defines the coherence rules that constrain the valid configuration space across the eight derivation dimensions. The derivation engine uses this document to detect incoherent configurations (BLOCK), flag friction-prone combinations (WARN), document cascade effects between dimensions, and suggest compensating mechanisms when soft constraints are overridden. The valid configuration space is much smaller than the combinatorial product — eight dimensions with three positions each produces 6,561 theoretical combinations, and most are incoherent.

---

## Derivation Questions

1. **Is this configuration coherent?** Given a set of dimension values, check whether any hard or soft constraints are violated.
2. **What cascades does a dimension change trigger?** When a user changes one dimension, which other dimensions experience pressure to shift?
3. **Can a soft constraint violation be compensated?** When a WARN fires, is there a compensating mechanism that reduces the friction to acceptable levels?
4. **Which constraints are invariant vs. configurable?** Kernel primitive constraints identify which system components cannot be disabled regardless of user preference.
5. **What should the derivation rationale include?** Which constraints were active, how they were resolved, and what overrides were accepted.

---

## Curated Claims

### Primary Cascades

#### Atomic granularity creates the widest cascade

**Summary:** Atomic granularity forces explicit linking (notes lack internal context), forces deep navigation (thousands of small nodes need MOC hierarchy), forces heavy processing (extraction, reflection, reweaving to maintain links), and pressures toward semantic search (keyword search fails across vocabularies).

**Derivation Implication:** When the engine sets granularity to atomic, it must cascade: increase linking to explicit, navigation depth to 3+ tiers, and processing to medium or heavy. Failure to cascade produces navigational vertigo and disconnected notes.

**Source:** Granularity cascade analysis from dimension interaction modeling.

#### Coarse granularity permits relaxed downstream dimensions

**Summary:** Coarse granularity permits lightweight linking (internal proximity provides context), permits shallow navigation (fewer nodes to organize), and permits light processing (each note is more self-contained).

**Derivation Implication:** Coarse granularity loosens constraints on other dimensions, enabling lighter-weight system configurations.

**Source:** Inverse of granularity cascade analysis.

#### Full automation enables dense downstream configurations

**Summary:** Full automation (hooks + skills + pipelines) enables dense schemas (validation catches errors), enables heavy processing (pipelines handle volume), and enables condition-based maintenance (automated trigger evaluation).

**Derivation Implication:** When automation is set to full, the engine can safely recommend denser schemas and heavier processing without unsustainable manual burden.

**Source:** Automation cascade analysis from dimension interaction modeling.

#### Manual operation pressures toward minimal configurations

**Summary:** Manual operation (convention only) pressures toward minimal schemas (less to remember/validate by hand), pressures toward light processing (each step costs attention), and pressures toward lax maintenance conditions (manual burden).

**Derivation Implication:** When automation is set to manual/convention, the engine should recommend lighter schemas, processing, and maintenance to avoid unsustainable burden.

**Source:** Inverse of automation cascade analysis.

#### High volume requires deep navigation, semantic search, and automated maintenance

**Summary:** Vaults exceeding ~200 notes require deep navigation (shallow structures become unnavigable), semantic search (grep alone misses vocabulary divergence), and automated maintenance (manual review at scale is impractical).

**Derivation Implication:** Volume acts as an environmental variable that tightens constraints on navigation, search, and maintenance dimensions. The engine must check volume estimates against these thresholds.

**Source:** Volume cascade analysis from dimension interaction modeling.

#### Low volume permits lightweight system configurations

**Summary:** Vaults under ~50 notes permit shallow navigation (agent can hold full structure in context), keyword search (grep works when vocabulary is consistent), and manual maintenance (small enough to review fully).

**Derivation Implication:** Small vaults can safely use minimal configurations without the overhead of deep navigation or automation.

**Source:** Inverse of volume cascade analysis.

---

### Hard Constraints (BLOCK)

These produce systems that will fail. The derivation engine must block these configurations.

#### Atomic granularity with 2-tier navigation at volume above 100 is incoherent

**Summary:** `atomic + navigation_depth == "2-tier" + volume > 100` — Atomic notes at this volume need at least 3-tier navigation. 2-tier creates navigational vertigo.

**Derivation Implication:** BLOCK. The engine must not generate a system with this combination. Recommend increasing navigation depth to 3-tier minimum.

**Source:** Granularity cascade + [[navigational vertigo emerges in pure association systems]].

#### Full automation requires platform hook support

**Summary:** `automation == "full" + platform_lacks_hooks` — Full automation requires hooks and skills. Convention-only platforms cannot support it.

**Derivation Implication:** BLOCK. The engine must detect platform capabilities and cap automation level accordingly.

**Source:** [[four abstraction layers separate platform-agnostic from platform-dependent]].

#### Heavy processing without pipeline skills is unsustainable

**Summary:** `processing == "heavy" + automation == "manual" + no_pipeline_skills` — Heavy processing without pipeline skills creates unsustainable manual burden.

**Derivation Implication:** BLOCK. Either add automation or reduce processing intensity.

**Source:** Automation cascade analysis + [[throughput matters more than accumulation]].

---

### Soft Constraints (WARN)

These produce friction but can work with compensating mechanisms.

#### Atomic granularity with light processing leaves notes disconnected

**Summary:** `atomic + processing == "light"` — Atomic notes need processing to recreate decomposed context. Light processing may leave notes disconnected.

**Derivation Implication:** WARN. Recommend medium processing. Compensating mechanism: semantic search can partially bridge missing explicit links.

**Source:** Granularity cascade + [[enforcing atomicity can create paralysis when ideas resist decomposition]].

#### Dense schema without automated validation creates maintenance burden

**Summary:** `schema == "dense" + automation == "convention"` — Dense schemas without automated validation create maintenance burden.

**Derivation Implication:** WARN. Recommend adding validation hooks or reducing schema density. Compensating mechanism: good templates reduce manual validation burden at capture (moderate effectiveness).

**Source:** Automation cascade + [[schema evolution follows observe-then-formalize not design-then-enforce]].

#### Implicit linking without semantic search tool falls back to explicit-only

**Summary:** `linking == "explicit+implicit" + no_semantic_search` — Implicit linking (semantic search) is enabled but no search tool is configured.

**Derivation Implication:** WARN. The system will work with explicit links only. Not a failure, but the implicit layer provides no value.

**Source:** [[controlled disorder engineers serendipity through semantic linking]].

#### Large vaults without maintenance conditions risk drift

**Summary:** `volume > 200 + maintenance_conditions_disabled` — Large vaults need condition-based maintenance to prevent link rot and orphan accumulation.

**Derivation Implication:** WARN. Recommend enabling at least orphan detection and dangling link checks.

**Source:** Volume cascade + [[random note resurfacing prevents write-only memory]].

#### Heavy processing with lax maintenance thresholds creates backlog

**Summary:** `processing == "heavy" + maintenance_conditions_too_lax"` — Heavy processing generates maintenance targets faster than lax thresholds can catch.

**Derivation Implication:** WARN. Consider lowering condition thresholds (e.g., orphan count, stale node percentage).

**Source:** [[gardening cycle implements tend prune fertilize operations]].

#### Coarse granularity with heavy processing has diminishing returns

**Summary:** `coarse + processing == "heavy"` — Coarse notes are self-contained enough that heavy processing provides diminishing returns.

**Derivation Implication:** WARN. Recommend medium processing.

**Source:** Granularity cascade inverse + [[processing effort should follow retrieval demand]].

#### Flat organization with 2-tier navigation gets crowded above 50 notes

**Summary:** `flat + navigation_depth == "2-tier" + volume > 50` — Flat organization with only 2 tiers gets crowded as notes accumulate.

**Derivation Implication:** WARN. Recommend adding topic-level MOCs (3-tier).

**Source:** [[MOCs are attention management devices not just organizational tools]].

---

### Kernel Primitive Constraints

These constraints apply to the 15 kernel primitives and their INVARIANT/CONFIGURABLE status.

#### Session capture is INVARIANT and cannot be disabled

**Summary:** Primitive 15: `session_capture == false` — Every vault saves session transcripts. The operational learning loop depends on this evidence.

**Derivation Implication:** BLOCK. The engine must always enable session capture regardless of user preferences.

**Source:** Kernel primitive invariant analysis.

#### Methodology folder is INVARIANT and cannot be disabled

**Summary:** Primitive 14: `methodology_folder == false` — Meta-skills (/ask, /architect, /rethink) require ops/methodology/ to reason about system state.

**Derivation Implication:** BLOCK. The engine must always include the methodology folder.

**Source:** Kernel primitive invariant analysis.

#### Schema enforcement is INVARIANT and cannot be disabled

**Summary:** Primitive 7: `schema_enforcement == false` — Without validation, metadata drift corrupts retrieval within weeks.

**Derivation Implication:** BLOCK. The engine must always enforce schema validation.

**Source:** Kernel primitive invariant analysis + [[metadata reduces entropy enabling precision over recall]].

#### Wiki links are INVARIANT and cannot be disabled

**Summary:** Primitive 3: `wiki_links == false` — Wiki links are the universal reference form and the foundation of the graph database.

**Derivation Implication:** BLOCK. The engine must always enable wiki links.

**Source:** Kernel primitive invariant analysis + [[propositional link semantics transform wiki links from associative to reasoned]].

#### Task stack is INVARIANT and cannot be disabled

**Summary:** Primitive 13: `task_stack == false` — Without task tracking, the agent has no lifecycle visibility and cannot answer "what should I work on?"

**Derivation Implication:** BLOCK. The engine must always include the task stack.

**Source:** Kernel primitive invariant analysis.

#### Self space defaults depend on preset type

**Summary:** Self space is OFF by default for Research presets (knowledge graph is the focus) and ON by default for Personal Assistant presets (agent identity is central). Toggle only with explicit justification.

**Derivation Implication:** WARN if default is overridden. Include justification in derivation rationale.

**Source:** Kernel primitive configurable analysis.

#### Semantic search opt-in affects implicit linking effectiveness

**Summary:** Primitive: `semantic_search == false + linking == "explicit+implicit"` — Without qmd, implicit linking falls back to keyword overlap and MOC traversal.

**Derivation Implication:** WARN. System functions but implicit layer is degraded.

**Source:** [[spreading activation models how agents should traverse]].

#### Disabling all maintenance thresholds suppresses maintenance surfacing

**Summary:** `condition_thresholds_all_zero` — The vault will not surface maintenance tasks.

**Derivation Implication:** WARN. Recommend enabling at least orphan detection and dangling link checks.

**Source:** [[backward maintenance asks what would be different if written today]].

---

### Compensating Mechanisms

Some dimension mismatches can be compensated rather than blocked. Retained as a supplementary table for quick lookup during derivation.

| Mismatch | Compensating Mechanism | Effectiveness |
|----------|----------------------|---------------|
| Atomic + medium processing | Semantic search compensates for missing explicit links | Moderate — finds connections but doesn't create them |
| Dense schema + convention | Good templates reduce manual validation burden | Moderate — helps at capture, not at maintenance |
| High volume + shallow nav | Strong semantic search enables discovery without deep hierarchy | Moderate — works for retrieval, not for orientation |
| Manual + moderate processing | Batch processing sessions compensate for missing automation | Low — depends on user discipline |

**Key insight:** Granularity cascades are hard — you cannot atomic-granularity your way out of missing navigation depth. Automation cascades are soft — you can manually maintain a dense schema if you're disciplined enough. It's friction, not failure.

---

### Cross-Dimension Interaction Matrix

Each cell describes the pressure that the row dimension's pole creates on the column dimension. Retained as supplementary reference for the derivation engine's constraint checker.

| Row → Col | Granularity | Organization | Linking | Processing | Nav Depth | Maintenance | Schema | Automation |
|-----------|-------------|-------------|---------|------------|-----------|-------------|--------|------------|
| **Atomic granularity** | — | pressures flat | forces explicit | forces heavy | forces deep | pressures active conditions | neutral | pressures automation |
| **Coarse granularity** | — | permits hierarchical | permits light | permits light | permits shallow | permits lax conditions | neutral | permits manual |
| **Flat organization** | neutral | — | requires explicit links | neutral | requires MOC overlay | neutral | neutral | neutral |
| **Hierarchical org** | neutral | — | folder membership as linking | neutral | folder browsing sufficient | neutral | neutral | neutral |
| **Explicit+implicit linking** | neutral | neutral | — | neutral | neutral | neutral | neutral | requires semantic search tool |
| **Heavy processing** | neutral | neutral | enables dense links | — | produces material for deep nav | generates maintenance targets | enables field discovery | benefits from automation |
| **Light processing** | neutral | neutral | produces few links | — | needs little nav | generates few targets | minimal fields emerge | works without automation |
| **Full automation** | neutral | neutral | neutral | enables heavy | neutral | enables active conditions | enables dense | — |
| **Manual operation** | neutral | neutral | neutral | pressures light | neutral | pressures infrequent | pressures minimal | — |

---

## Derivation Application

When generating a system:

1. **Start from use-case preset** (or tradition preset) — these are pre-validated coherence points
2. **Allow user customization** — but check each change against interaction constraints
3. **Cascade recommendations** — if user changes granularity from moderate to atomic, recommend increasing processing and navigation depth
4. **Document justification** — include interaction constraint reasoning in the derivation rationale section of the generated context file
5. **Flag unresolved tensions** — if user overrides a warning, note it in the generated system for future reseeding

The derivation rationale should include which constraints were active and how they were resolved. This enables principled reseeding when friction patterns emerge later.

---

## Exclusion Notes

### Pairwise interaction exhaustive enumeration
**Reason:** The full 8x8 matrix of all possible pairwise dimension interactions was evaluated, but many cells are "neutral" (no meaningful pressure). Only cells with documented pressure effects are included as claims. The neutral interactions are visible in the cross-dimension interaction matrix but do not warrant individual claim entries.

### Platform-specific constraint implementations
**Reason:** Several constraints could be expressed as platform-specific rules (e.g., "Obsidian requires X plugin for semantic search"). These were excluded because the constraint system operates at the abstract dimension level. Platform mapping belongs in the tradition-presets or platform-capability layer.

### Quantitative threshold calibration
**Reason:** Exact numeric thresholds (e.g., "volume > 200" vs "volume > 150") are based on operational experience rather than research claims. They are included in constraint rules but not treated as curated claims with research provenance.

---

## Version
- **Last curated:** 2026-03-20
- **Source claims evaluated:** 34
- **Claims included:** 27
- **Claims excluded:** 7
