# Dimension-Claim Map

## Purpose

Maps specific research claims from the methodology corpus to the eight configuration dimensions used by the derivation engine. Every dimension choice in a generated system must trace back to at least one claim in this map, ensuring that configuration decisions are evidence-grounded rather than arbitrary. This document is the primary lookup table for claim-to-dimension attribution during derivation.

---

## Derivation Questions

1. **Which research claims support this dimension's default position?** When the engine sets a default value for a dimension, this map provides the evidential basis.
2. **What claims justify softening or shifting a dimension away from its default?** Each dimension has boundary conditions where claims support alternative positions.
3. **Which dimensions share claims, and what does that imply about coupling?** Cross-referenced claims reveal interaction pressure between dimensions.
4. **Is a user's custom dimension choice supported by any research evidence?** When a user overrides a default, the engine checks whether any claims support that override.

---

## Curated Claims

### Dimension 1: Granularity (atomic ↔ coarse)

**Default position:** Atomic (composability maximizes reuse). Soften for reference-heavy domains.

#### Enforcing atomicity can create paralysis when ideas resist decomposition

**Summary:** The atomic pole has real cost — some ideas need compound expression.

**Derivation Implication:** When to soften atomicity.

**Source:** [[enforcing atomicity can create paralysis when ideas resist decomposition]]

#### Three capture schools converge through agent-mediated synthesis

**Summary:** Agent processing dissolves capture granularity tradeoffs.

**Derivation Implication:** Default recommendation.

**Source:** [[three capture schools converge through agent-mediated synthesis]]

#### Summary coherence tests composability before filing

**Summary:** If can't summarize in 1-3 sentences, it bundles claims.

**Derivation Implication:** Split signal.

**Source:** [[summary coherence tests composability before filing]]

---

### Dimension 2: Organization (flat ↔ hierarchical)

**Default position:** Flat with MOC overlay. Add folders only when file counts exceed tool limits.

#### Associative ontologies beat hierarchical taxonomies

**Summary:** Heterarchy adapts while hierarchy brittles.

**Derivation Implication:** Default recommendation.

**Source:** [[associative ontologies beat hierarchical taxonomies]]

#### Topological organization beats temporal for knowledge work

**Summary:** Concept-based beats date-based.

**Derivation Implication:** Organization axis.

**Source:** [[topological organization beats temporal for knowledge work]]

#### Navigational vertigo emerges in pure association systems

**Summary:** Without MOCs, unlinked neighbors become unreachable.

**Derivation Implication:** When hierarchy is needed.

**Source:** [[navigational vertigo emerges in pure association systems]]

#### Faceted classification treats notes as multi-dimensional objects

**Summary:** Ranganathan's PMEST: facets compose multiplicatively.

**Derivation Implication:** Alternative to folder hierarchy.

**Source:** [[faceted classification treats notes as multi-dimensional objects]]

---

### Dimension 3: Linking Philosophy (explicit-only ↔ explicit+implicit)

**Default position:** Explicit+implicit (wiki links primary, semantic search supplemental).

#### Propositional link semantics transform wiki links from associative to reasoned

**Summary:** Moving from "related" to "this causes/enables/contradicts that."

**Derivation Implication:** Link quality standard.

**Source:** [[propositional link semantics transform wiki links from associative to reasoned]]

#### Inline links carry richer relationship data than metadata fields

**Summary:** Prose context encodes WHY notes connect.

**Derivation Implication:** Link format.

**Source:** [[inline links carry richer relationship data than metadata fields]]

#### Concept-orientation beats source-orientation

**Summary:** Organizing by concept enables cross-domain edges.

**Derivation Implication:** Link target design.

**Source:** [[concept-orientation beats source-orientation]]

#### Controlled disorder engineers serendipity through semantic linking

**Summary:** Luhmann: perfect order yields zero surprise.

**Derivation Implication:** When to add implicit.

**Source:** [[controlled disorder engineers serendipity through semantic linking]]

#### Spreading activation models how agents should traverse

**Summary:** Graph traversal as primary discovery mechanism.

**Derivation Implication:** Traversal pattern.

**Source:** [[spreading activation models how agents should traverse]]

---

### Dimension 4: Processing Intensity (heavy ↔ light)

**Default position:** Medium (4-phase skeleton). Heavy for research/synthesis. Light for capture-heavy domains.

#### Fresh context per task preserves quality better than chaining phases

**Summary:** Session isolation keeps phases in smart zone.

**Derivation Implication:** Phase architecture.

**Source:** [[fresh context per task preserves quality better than chaining phases]]

#### Throughput matters more than accumulation

**Summary:** Processing velocity, not note count.

**Derivation Implication:** Health metric.

**Source:** [[throughput matters more than accumulation]]

#### Processing effort should follow retrieval demand

**Summary:** JIT processing over front-loading.

**Derivation Implication:** When to invest.

**Source:** [[processing effort should follow retrieval demand]]

#### Every knowledge domain shares a four-phase processing skeleton

**Summary:** Capture, process, connect, verify — only process step varies.

**Derivation Implication:** Pipeline structure.

**Source:** [[every knowledge domain shares a four-phase processing skeleton]]

#### Structure without processing provides no value

**Summary:** Structural motions without generation produce nothing.

**Derivation Implication:** Minimum processing.

**Source:** [[structure without processing provides no value]]

#### The generation effect requires active transformation not just storage

**Summary:** Moving files is not processing.

**Derivation Implication:** Quality gate.

**Source:** [[the generation effect requires active transformation not just storage]]

---

### Dimension 5: Navigation Depth (2-tier ↔ 4-tier)

**Default position:** 3-tier (hub → domain → topic). Add 4th tier at >100 notes per topic.

#### MOCs are attention management devices not just organizational tools

**Summary:** Reduce context-switching cost by 23 minutes.

**Derivation Implication:** Why depth matters.

**Source:** [[MOCs are attention management devices not just organizational tools]]

#### Progressive disclosure means reading right not reading less

**Summary:** Each layer reveals more but costs more tokens.

**Derivation Implication:** Layer design.

**Source:** [[progressive disclosure means reading right not reading less]]

#### Basic level categorization determines optimal MOC granularity

**Summary:** Cognitive sweet spot for categorization depth.

**Derivation Implication:** Tier count.

**Source:** [[basic level categorization determines optimal MOC granularity]]

#### Community detection algorithms can inform when MOCs should split or merge

**Summary:** Algorithmic signals for structural maintenance.

**Derivation Implication:** Maintenance trigger.

**Source:** [[community detection algorithms can inform when MOCs should split or merge]]

---

### Dimension 6: Maintenance Sensitivity (tight thresholds ↔ lax thresholds)

**Default position:** Condition-based for all domains. High-volume active domains use tight thresholds (low orphan/inbox tolerance). Stable reference domains use lax thresholds. Threshold sensitivity scales with the domain's rate of change.

#### Backward maintenance asks what would be different if written today

**Summary:** Living documents, not finished artifacts.

**Derivation Implication:** Maintenance philosophy.

**Source:** [[backward maintenance asks what would be different if written today]]

#### Incremental formalization happens through repeated touching

**Summary:** Many small touches over time.

**Derivation Implication:** Threshold pattern.

**Source:** [[incremental formalization happens through repeated touching]]

#### Gardening cycle implements tend prune fertilize operations

**Summary:** Separated maintenance phases.

**Derivation Implication:** Phase structure.

**Source:** [[gardening cycle implements tend prune fertilize operations]]

#### Random note resurfacing prevents write-only memory

**Summary:** Counteracts structural attention bias.

**Derivation Implication:** Anti-stagnation.

**Source:** [[random note resurfacing prevents write-only memory]]

#### Spaced repetition scheduling could optimize vault maintenance

**Summary:** Front-loaded review intervals.

**Derivation Implication:** Scheduling pattern.

**Source:** [[spaced repetition scheduling could optimize vault maintenance]]

#### Derived systems follow a seed-evolve-reseed lifecycle

**Summary:** Minimum viable → friction-driven → principled restructuring.

**Derivation Implication:** Evolution pattern.

**Source:** [[derived systems follow a seed-evolve-reseed lifecycle]]

---

### Dimension 7: Schema Density (minimal ↔ dense)

**Default position:** Minimal (description + topics). Add fields when querying patterns emerge.

#### Metadata reduces entropy enabling precision over recall

**Summary:** Pre-computed representations shrink search space.

**Derivation Implication:** Why density helps.

**Source:** [[metadata reduces entropy enabling precision over recall]]

#### Schema evolution follows observe-then-formalize not design-then-enforce

**Summary:** Start minimal, grow based on evidence.

**Derivation Implication:** Evolution pattern.

**Source:** [[schema evolution follows observe-then-formalize not design-then-enforce]]

#### Schema fields should use domain-native vocabulary not abstract terminology

**Summary:** Every abstractly-named field forces translation at capture.

**Derivation Implication:** Naming constraint.

**Source:** [[schema fields should use domain-native vocabulary not abstract terminology]]

#### Type field enables structured queries without folder hierarchies

**Summary:** Content-kind metadata provides filtering axis.

**Derivation Implication:** Minimum useful field.

**Source:** [[type field enables structured queries without folder hierarchies]]

#### Descriptions are retrieval filters not summaries

**Summary:** Lossy compression optimized for decision-making.

**Derivation Implication:** Description design.

**Source:** [[descriptions are retrieval filters not summaries]]

---

### Dimension 8: Automation Level (full ↔ manual)

**Default position:** Convention (context file instructions). Add automation at friction points.

#### Hook enforcement guarantees quality while instruction enforcement merely suggests it

**Summary:** Convention-to-automation is the sharpest capability gap.

**Derivation Implication:** When to automate.

**Source:** [[hook enforcement guarantees quality while instruction enforcement merely suggests it]]

#### Skills encode methodology so manual execution bypasses quality gates

**Summary:** Skills contain selectivity gates that instructions don't.

**Derivation Implication:** Automation value.

**Source:** [[skills encode methodology so manual execution bypasses quality gates]]

#### Four abstraction layers separate platform-agnostic from platform-dependent

**Summary:** Foundation, convention, automation, orchestration.

**Derivation Implication:** Layer mapping.

**Source:** [[four abstraction layers separate platform-agnostic from platform-dependent]]

#### The determinism boundary separates hook methodology from skill methodology

**Summary:** Deterministic ops → hooks. Judgment ops → skills.

**Derivation Implication:** Automation design.

**Source:** [[the determinism boundary separates hook methodology from skill methodology]]

#### Methodology development should follow documentation to skill to hook

**Summary:** Hardening trajectory: understand → encode → enforce.

**Derivation Implication:** Maturity path.

**Source:** [[methodology development should follow documentation to skill to hook]]

#### Complex systems evolve from simple working systems

**Summary:** Gall's Law: start simple, automate at friction points.

**Derivation Implication:** Evolution constraint.

**Source:** [[complex systems evolve from simple working systems]]

---

### Cross-Dimension Interactions

These claims describe pressures between dimensions. Retained as a supplementary table for quick cross-reference during derivation.

| Interaction | Claim | Effect |
|-------------|-------|--------|
| Granularity → Linking | configuration dimensions interact | Atomic granularity forces explicit linking |
| Granularity → Navigation | configuration dimensions interact | Atomic + flat requires deep MOC hierarchy |
| Granularity → Processing | configuration dimensions interact | Atomic notes need heavy processing to recreate lost context |
| Automation → Schema | configuration dimensions interact | Full automation enables dense schemas (validation catches errors) |
| Automation → Processing | configuration dimensions interact | Manual operation pressures toward light processing |
| Volume → Navigation | small-world topology requires hubs | Large vaults need deeper navigation |
| Volume → Maintenance | random note resurfacing prevents write-only memory | Large vaults need more frequent maintenance |

---

## Methodology Tradition Presets

See `tradition-presets.md` for the full tradition configurations, use-case presets, and mixing rules. That file is the single source of truth for preset definitions.

---

## Exclusion Notes

### Volume as a formal dimension
**Reason:** Volume (note count) appears in cross-dimension interactions but is not one of the 8 configuration dimensions. It acts as an environmental variable that modifies constraint thresholds rather than a settable dimension. Claims about volume are captured in cross-dimension interactions rather than given their own dimension section.

### Platform capability claims
**Reason:** Several methodology claims address platform-specific features (e.g., Obsidian dataview, specific search tools). These were excluded from dimension mapping because the derivation engine operates platform-agnostically. Platform claims inform the automation dimension indirectly through the "four abstraction layers" claim.

---

## Version
- **Last curated:** 2026-03-20
- **Source claims evaluated:** 43
- **Claims included:** 38
- **Claims excluded:** 5
