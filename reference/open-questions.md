# Open Questions and Deferred Items

## Purpose

This document serves as the project's deliberate-gap tracker — cataloging what we know we don't know, what we consciously deferred, and what open research questions shape future versions. The derivation engine consults this when users ask about capabilities that don't yet exist, and the architect command references it when proposing evolution paths. Unlike claim-based reference documents, this tracks uncertainty and future direction rather than established knowledge.

## Derivation Questions

- What capabilities does the derivation engine NOT yet support, and what is the workaround?
- Which deferred features are concrete enough for v1.1 vs requiring v2 architectural changes?
- What open research questions would, if answered, unlock new derivation dimensions or heuristics?
- How should the architect command handle user requests that touch deferred capabilities?
- What is the relative priority of open questions based on impact and feasibility?
- Which deferred items have dependencies on each other (e.g., observation export before contribution workflow)?

## Curated Claims

### Deferred Items (v1.1)

These items are specified but not implemented in v1.0. They are concrete enough to ship in the next minor release.

These items are specified but not implemented in v1.0. They are concrete enough to ship in the next minor release.

#### 1. Scan Mode for Health Command

**Summary:** Read-only health check mode that separates diagnosis from prescription.

**Derivation Implication:** Enables CI/monitoring integration; unblocks automated health tracking without side effects.

**Source:** Internal observation — users want metrics without commitment to action.

**What:** `/health --scan` performs a lightweight, non-destructive health check that outputs metrics without modifying any files. Currently, `/health` both diagnoses and proposes fixes. Scan mode separates diagnosis from prescription.

**Why deferred:** Health command works without scan mode — the combined mode handles most use cases. Scan mode becomes valuable when users want to monitor without committing to action (CI pipelines, automated monitoring).

**When to ship:** When the first user asks for health metrics in their CI pipeline, or when automated monitoring reveals demand for read-only health checks.

#### 2. Migration Assistant

**Summary:** Guided import flow for existing note collections into generated Ars Contexta systems.

**Derivation Implication:** Would extend the derivation engine to analyze existing structure, not just conversation signals. Unlocks adoption from non-greenfield users.

**Source:** Anticipated user need — existing vault owners wanting to adopt methodology.

**What:** A guided flow for importing existing note collections into a generated Ars Contexta system. Detects common formats (Obsidian, Notion export, plain markdown), maps existing structure to the three-space architecture, and proposes schema additions for existing frontmatter.

**Why deferred:** Migration is engineering-heavy and domain-specific (Obsidian exports differ from Notion exports). New systems (greenfield) are the primary v1 target. Migration matters for adoption but doesn't affect the derivation engine's correctness.

**When to ship:** When user feedback shows significant demand from existing vault owners who want to adopt Ars Contexta methodology on top of their current content.

#### 3. Multi-Platform Support

**Summary:** Agent platform portability beyond Claude Code with shared notes/self and platform-specific ops/automation.

**Derivation Implication:** Would require the derivation engine to generate platform-variant implementations, expanding the generation target from one platform to many.

**Source:** Internal architectural anticipation — multi-platform is a predictable adoption path.

**What:** Supporting additional agent platforms beyond Claude Code, with shared notes/ and self/ while maintaining platform-specific ops/ and automation.

**Why deferred:** Requires solving platform-specific session isolation, context file format adaptation, and shared state reconciliation. Currently focused on Claude Code as the sole supported platform.

**When to ship:** When a second platform stabilizes enough for reliable operation, and when user demand validates multi-platform as a real use case.

#### 4. arscontexta.dev Website

**Summary:** Web frontend for human-browsable research graph plus MCP endpoint for agent queries.

**Derivation Implication:** Distribution channel — does not affect derivation logic but extends reach to non-agent audiences.

**Source:** Internal product strategy.

**What:** A web frontend rendering the research graph through a sliding-pane reader for human browsing, plus an MCP endpoint for agent architect queries. Built on the same knowledge graph that the derivation engine reasons from.

**Why deferred:** The website is a distribution channel, not a core capability. The plugin and MCP server deliver the derivation engine's value directly. The website extends reach to human audiences who want to browse the research without installing anything.

**When to ship:** After v1.0 plugin and MCP server are stable, when human-readable research presentation becomes the growth bottleneck.

#### 5. MCP Hosted Server

**Summary:** Cloud-hosted MCP server removing the need for local `npx arscontexta-mcp` installation.

**Derivation Implication:** Lowers onboarding friction; does not change derivation logic but affects deployment architecture.

**Source:** Internal product strategy — trade-off between reach and operational complexity.

**What:** A hosted version of the Ars Contexta MCP server that users connect to without running locally. Provides `arscontexta_query`, `arscontexta_recommend`, and `arscontexta_dimensions` tools through a cloud endpoint.

**Why deferred:** The local MCP server (`npx arscontexta-mcp`) works for v1. Hosting adds infrastructure, authentication, rate limiting, and operational costs. The trade-off is reach (easier onboarding) vs complexity (ops burden).

**When to ship:** When the local MCP server sees enough adoption that hosting becomes a friction reducer rather than a premature investment.

#### 6. Advanced Derivation Heuristics

**Summary:** Richer signal analysis including conversation flow, confidence calibration, and automated heuristic testing.

**Derivation Implication:** Directly improves derivation accuracy — would replace keyword-to-dimension mapping with sequence-aware, confidence-weighted analysis.

**Source:** Internal observation — basic heuristics work but lack nuance for edge cases.

**What:** Richer conversation analysis beyond keyword-to-dimension mapping. Includes: conversation flow analysis (not just individual signals but signal sequences), confidence calibration across dimensions (some signals are stronger than others), and automated heuristic testing against the conversation pattern corpus.

**Why deferred:** The basic heuristics (keyword signals -> dimension positions -> cascade checking) work for the five tested patterns. Advanced heuristics add complexity before we have enough deployment data to validate the improvement.

**When to ship:** After collecting derivation logs from 50+ real conversations, when patterns in heuristic failures reveal specific improvement targets.

#### 7. Sleep-Time Compute

**Summary:** Background processing during inactive periods with morning briefing output.

**Derivation Implication:** Would add a `sleep-pipeline` feature block activated when processing >= moderate, generating scheduling config and sleep skill.

**Source:** Internal observation — temporal separation of capture and processing is a strong research-backed pattern.

**What:** Background inference during inactive periods — processing inbox items, finding connections between existing notes, running backward maintenance, detecting synthesis opportunities. Produces a morning briefing in ops/.

**Why deferred:** Requires platform-specific scheduling (hook-suggested on Claude Code). The core derivation engine and processing pipeline work without background compute. Sleep-time compute is an acceleration mechanism, not a prerequisite.

**When to ship:** After the processing pipeline skills are proven reliable, when the incremental value of background processing justifies the scheduling complexity.

#### 8. Observation Export

**Summary:** Anonymized observation bundling for recursive improvement loop feedback.

**Derivation Implication:** Prerequisite for the v2 Organization Contribution Workflow; enables aggregate learning from deployed systems.

**Source:** Internal need — recursive improvement requires deployment data.

**What:** `/health --export-observations` packages ops/observations/ into an anonymized JSON bundle with system context (dimension positions, note count, feature blocks) but no user content. Always manual, always opt-in, always reviewable before sharing.

**Why deferred:** The recursive improvement loop benefits from observation export, but v1 can improve through direct user feedback and our own vault's operational learning. The formal export pipeline adds engineering complexity (format specification, privacy verification, submission channel) that isn't necessary for initial validation.

**When to ship:** When the deployed user base is large enough that aggregate observation data would meaningfully improve the research graph beyond what our own vault provides.

---

### Deferred Items (v2)

These items require architectural decisions or ecosystem maturity that v1 doesn't address.

#### 1. Organization Contribution Workflow

**Summary:** Full feedback loop from deployed system observations through quality gates to research graph integration.

**Derivation Implication:** Would close the recursive improvement loop — deployed systems improve the research that generates future systems.

**Source:** Internal architectural vision — ecosystem feature requiring governance decisions.

**What:** A full feedback loop where deployed systems submit anonymized observations, which pass through quality gates (staging, deduplication, human review, claim creation with attribution) before integration into the research graph. The export side is specified (observation export above); this is the import and curation side.

**Why deferred to v2:** Requires: (a) enough deployed systems generating observations to justify a curation pipeline, (b) governance decisions about how community contributions are attributed and validated, (c) infrastructure for staging, review, and integration. This is an ecosystem feature, not a product feature.

#### 2. Programmatic Skills API Deployment

**Summary:** Programmatic skill deployment via `/v1/skills` API for organization-level management.

**Derivation Implication:** Would enable self-evolving systems and admin-deployed methodology; changes distribution from manual to programmatic.

**Source:** Claude.ai Skills API capability — distribution engineering problem.

**What:** Using the Claude.ai Skills API (`/v1/skills`) to programmatically deploy, update, and manage skills on Claude accounts — including organization-level deployment to Team/Enterprise Claude plans. Enables: admin-deployed methodology for entire teams, self-evolving systems that create their own skills via API, automated skill updates during reseed.

**Why deferred to v2:** Per-account authentication, skill versioning (mid-session updates), per-role customization (junior vs senior), and compliance auditing are distribution engineering problems that don't block the core product. Plugin and marketplace distribution serve v1 needs.

---

### Open Research Questions

These questions don't have clear answers yet. They inform the research graph's direction and may lead to new claims, new dimensions, or new architectural decisions.

#### Constraint-Surprise Metric

**Summary:** Principled stopping criterion for init wizard conversations based on information gain per turn.

**Derivation Implication:** Would replace the arbitrary "3-5 questions" heuristic with a scoring function that stops when marginal constraint drops below threshold.

**Source:** Internal observation + HyperMapper (arXiv:1810.05236) active learning analogy.

**Question:** Can we measure diminishing information gain per conversation turn to determine when the derivation conversation should stop?

**Context:** The init wizard currently asks 3-5 follow-up questions. But some conversations reveal everything in 2 turns (unambiguous signals) while others need 7+ (contradictory signals, novel domains). A principled stopping criterion would replace the arbitrary "3-5 questions" heuristic.

**Approach:** Measure information gain per turn — how much each user response reduces uncertainty across the 8 dimensions. When gain drops below a threshold, the conversation is complete. This is analogous to active learning in machine learning — sample where uncertainty is highest.

**Related work:** HyperMapper (arXiv:1810.05236) uses active learning and multi-objective Bayesian optimization for design space exploration. The init conversation is functionally design space exploration — each user response samples a region of the 8-dimensional configuration space.

**What a solution looks like:** A scoring function that takes (signals_extracted, dimensions_resolved, confidence_levels) and returns (conversation_sufficient: boolean, next_best_question: string | null). When marginal constraint from the next question drops below threshold, stop asking.

#### DialogComplete Model

**Summary:** RL-based classifier for conversation readiness — alternative to constraint-surprise metric.

**Derivation Implication:** Would automate "ready to derive" vs "needs more information" classification; requires constraint-surprise metric as foundation.

**Source:** Internal speculation — depends on corpus of labeled derivation conversations.

**Question:** Can an RL-based classifier determine when a derivation conversation has enough information to produce a confident configuration?

**Context:** Alternative to the constraint-surprise metric. Instead of measuring information gain, train a model to classify conversation state as "ready to derive" vs "needs more information" based on which dimensions have confident positions.

**Research direction:** Would require a corpus of successful derivation conversations labeled with "minimum sufficient" turn count. The reward signal would be: did the derived system need reseeding within 30 days? Currently speculative — the constraint-surprise metric would need to exist first as foundation.

#### Graph Density Scoring

**Summary:** Quality-weighted connectivity metric that goes beyond naive link counting.

**Derivation Implication:** Would directly improve /health command recommendations by identifying weakest connection points and most-needed new links.

**Source:** Internal observation — current orphan/density metrics are crude.

**Question:** How do we measure whether a knowledge graph has "enough" connections without just counting links?

**Context:** Health checks count orphans (notes with no incoming links) and link density (links per note). But these are crude — a note with 20 links to vaguely related content is worse than a note with 3 precisely contextualized connections.

**Approach:** A quality-weighted density metric that considers:
- Link context quality (does the inline prose explain WHY?)
- Relationship type diversity (all "extends" is monotone; mix of extends/contradicts/enables is richer)
- Cluster connectivity (are topic clusters well-connected internally AND bridged externally?)
- Small-world properties (average path length, clustering coefficient)

**What a solution looks like:** A composite metric that can answer: "Is this graph navigable? Where are the weakest connection points? Which notes would benefit most from new links?" This would feed directly into the /health command's connection recommendations.

#### Optimal MOC Split/Merge Thresholds

**Summary:** Domain-sensitive threshold functions replacing the heuristic 50+ split / <10 merge rules.

**Derivation Implication:** Would make generated maintenance thresholds domain-aware — research MOCs may need splitting earlier than companion MOCs.

**Source:** Internal practice observation — current thresholds are experiential, not principled.

**Question:** When exactly should a MOC split into sub-MOCs, and when should small MOCs merge?

**Context:** Heuristic thresholds — split at 50+ links, merge at <10 links. These come from practice in the research vault, not from principled analysis.

**Approach:** Thresholds should derive from cognitive science (working memory limits, context window constraints) and graph theory (community detection algorithms identifying natural cluster boundaries). The optimal split point likely depends on domain — a research MOC with 40 densely-linked claims may need splitting, while a companion MOC with 40 loosely-linked memories may not.

**What a solution looks like:** Domain-sensitive threshold functions: `should_split(moc, domain_config) -> (bool, suggested_subclusters)` using community detection on the link subgraph within the MOC.

#### Context-Bench Integration

**Summary:** Standardized benchmark suite for evaluating generated knowledge systems.

**Derivation Implication:** Would enable objective measurement of derivation quality — retrieval, processing, maintenance, and composition metrics.

**Source:** Internal need — current validation is manual (4 worked tests).

**Question:** How do we standardize evaluation of generated knowledge systems?

**Context:** Evaluation is manual — derivation-validation.md runs 4 worked tests (self-derivation, cross-domain, novel domain, multi-domain composition). These verify structural coherence but not operational effectiveness.

**Approach:** A benchmark suite that tests generated systems on:
- Retrieval quality: can the agent find relevant notes given a query?
- Processing fidelity: does the processing pipeline produce quality output?
- Maintenance sustainability: does the system degrade gracefully over time?
- Cross-domain composition: do multi-domain systems maintain coherence?

**What a solution looks like:** A test harness that generates systems for each use-case preset, populates them with synthetic content, runs processing and maintenance cycles, and measures quality metrics over simulated time.

#### Cross-Platform Hook Equivalence

**Summary:** Behavioral equivalence testing for platform-variant hook implementations.

**Derivation Implication:** Would validate that generated systems produce identical behavior across platforms — essential for multi-platform support.

**Source:** Internal architectural concern — variants are hand-specified, not tested.

**Question:** Can we ensure generated systems work equally well across Claude Code and future platforms?

**Context:** The kernel specifies platform-variant implementations for hooks and tree injection. But the variants are hand-specified, not systematically tested for behavioral equivalence.

**Approach:** Platform adapters that translate abstract hook specifications into platform-native implementations, with behavioral equivalence tests verifying that each platform's hooks produce equivalent outcomes.

#### Multi-Agent Collaboration in Generated Systems

**Summary:** Generated multi-agent systems with shared knowledge and per-agent identity.

**Derivation Implication:** Would require the derivation engine to generate coordination primitives (shared notes/, per-agent self/, conflict resolution).

**Source:** Internal observation — research vault's Cornelius/Heinrich pattern is hand-crafted, not generatable.

**Question:** Can a generated system support multiple agents with different roles?

**Context:** The research vault supports two operators (Cornelius/Heinrich) with shared methodology and separate infrastructure. But this is hand-crafted, not generated. Multi-agent templates would need: shared notes/ and ops/, per-agent self/, task distribution across agents, and conflict resolution when agents modify the same notes.

**Research direction:** Federated wiki patterns, CRDTs for markdown, and stigmergic coordination (agents leaving traces for others) are potential approaches. No generated system has tested multi-agent operation yet.

#### Vocabulary Transformation Quality Metrics

**Summary:** Empirical comparison of deep vs shallow vocabulary transformation on agent compliance.

**Derivation Implication:** If shallow transformation is equivalent, the derivation engine can be significantly simpler — fewer generation targets per system.

**Source:** Internal efficiency question — full transformation is expensive.

**Question:** Does deep vocabulary transformation actually improve agent compliance compared to shallow transformation?

**Context:** Full transformation is expensive — every skill instruction, every context file section, every template field. If shallow transformation (key terms only) produces equivalent agent behavior, the derivation engine can be simpler.

**Approach:** Generate two systems for the same domain — one with full transformation, one with shallow. Compare: agent compliance with methodology, user satisfaction with system voice, and long-term engagement.

#### Personality Drift Detection

**Summary:** Runtime verification that agent behavior matches encoded personality profile.

**Derivation Implication:** Would add a maintenance capability to generated systems — periodic personality audits as a health check sub-command.

**Source:** Internal observation — no mechanism currently verifies runtime personality compliance.

**Question:** How do we detect when the agent's actual behavior has drifted from its encoded personality?

**Context:** Personality is encoded in ops/derivation.md and expressed through generated files. But there's no mechanism to verify that the agent's runtime behavior matches the encoded profile. A warm agent might gradually become clinical if the user's content becomes more analytical.

**Approach:** Periodic personality audits that sample recent agent outputs (health reports, skill completions, session logs), score them against the personality encoding, and flag drift when measured personality diverges from encoded personality.

**What a solution looks like:** A /architect sub-command that reads recent ops/sessions/ entries, scores linguistic markers (warmth, formality, emotional acknowledgment), and compares against the personality encoding.

#### Selective Forgetting Algorithms

**Summary:** Principled forgetting policies based on access patterns, temporal validity, and confidence decay.

**Derivation Implication:** Would add a `forgetting-policy` domain-specific configuration to generated systems — graduated obscurity instead of binary archive/active.

**Source:** Internal observation — "archive, don't delete" may create retrieval noise at scale.

**Question:** Should knowledge systems deliberately forget?

**Context:** The current philosophy is "archive, don't delete." Content moves to archive/ when no longer active, but nothing is destroyed. This preserves history but may create retrieval noise over time.

**Approach:** Principled forgetting based on:
- Access patterns: notes never accessed or linked in 12+ months
- Temporal validity: content that is explicitly dated and has expired
- Supersession chains: when a decision is superseded 3 times, the original may be archivable
- Confidence decay: speculative notes that were never confirmed

**What a solution looks like:** A forgetting policy per domain that moves notes from notes/ to a deep-archive/ (still recoverable, but excluded from search indices and MOCs). Not deletion — graduated obscurity.

#### Governance Layer for Enterprise Domains

**Summary:** Enterprise-specific governance feature block with data classification, access control, and compliance mapping.

**Derivation Implication:** Would add a new feature block activated by organizational domain signals — blocking for enterprise adoption.

**Source:** Internal anticipation — ethical guardrails exist but lack enterprise-grade governance.

**Question:** What additional constraints apply when generated systems handle sensitive content in organizational contexts?

**Context:** Ethical guardrails exist for therapy (no diagnosis) and general content (privacy, transparency, autonomy). Enterprise use cases add: data classification, access control, audit trails, compliance requirements.

**Approach:** A governance feature block activated by organizational domain signals, adding content classification fields, access control metadata, audit trail requirements, retention policies, and compliance mapping.

#### Multi-Domain Composition Rules Validation

**Summary:** Systematic testing of domain-pair compositions to build a compatibility matrix.

**Derivation Implication:** Would validate or refine the five composition rules — identifying which pairs compose cleanly, need adaptation, or conflict.

**Source:** Internal observation — composition rules are stated but not exhaustively tested.

**Question:** What are the limits of composing multiple domains in a single system?

**Context:** derivation-validation.md Test 4 demonstrates Research + Relationships composition. The five composition rules are stated but not exhaustively tested.

**Approach:** Systematic testing of composition pairs (Research + Therapy, PM + Creative, Learning + Companion, three-domain compositions) to build a composition compatibility matrix showing which domain pairs compose cleanly, which require adaptation, and which conflict.

#### Background Compute Integration

**Summary:** Generated sleep-pipeline feature block with scheduling config and morning briefing template.

**Derivation Implication:** Would activate when processing >= moderate, adding temporal separation of capture and processing to generated systems.

**Source:** Internal observation — strongest research-backed pattern not yet generated.

**Question:** How should generated systems leverage background processing (sleep/nightly pipelines)?

**Context:** The components blueprint includes a "Sleep Skill" for nightly processing. But integration with platform scheduling is not generated — it's left to the user. Temporal separation of capture and processing is one of the strongest patterns from the research.

**Approach:** A `sleep-pipeline` feature block that generates scheduling configuration, sleep skill, and morning briefing template, activated when processing >= moderate.

#### Optimal Context File Size

**Summary:** Empirical determination of when generated CLAUDE.md becomes counterproductive due to context window crowding.

**Derivation Implication:** Would set size constraints on the generation engine — feature block inclusion may need to consider cumulative context file size.

**Source:** Internal observation — vault CLAUDE.md is ~30KB, generated files are 5-15KB, no data on optimal range.

**Question:** Is there a point where the generated CLAUDE.md becomes counterproductive?

**Context:** Our vault's CLAUDE.md is ~30KB. Generated context files are 5-15KB depending on feature blocks. At some point, methodology instructions may crowd out task-relevant context.

**Approach:** Test generated systems at various context file sizes on standard vault operations. Measure: task completion quality, hallucination rate, methodology compliance, context window utilization.

---

## Priority Assessment

| Question | Impact | Feasibility | Priority |
|----------|--------|-------------|----------|
| Constraint-surprise metric | High — affects all derivations | Medium — needs formalization | High |
| Graph density scoring | High — affects all health checks | Medium — needs quality metrics | High |
| Context-Bench integration | High — enables objective measurement | Medium — needs test harness | High |
| MOC split/merge thresholds | Medium — affects maintenance | Medium — community detection is well-studied | Medium |
| Cross-platform hook equivalence | Medium — affects portability | High — engineering problem | Medium |
| Personality drift detection | Medium — affects user trust | Medium — needs linguistic analysis | Medium |
| Background compute | Medium — enables sleep pipeline | High — well-understood scheduling | Medium |
| Governance layer | High for enterprise — blocking for adoption | Medium — domain expertise needed | Medium |
| Multi-domain composition rules | Medium — affects advanced users | Medium — systematic testing needed | Medium |
| Optimal context file size | Medium — affects all generated systems | High — empirical testing | Medium |
| Vocabulary quality metrics | Low — current approach works | Medium — needs user testing | Low |
| Selective forgetting | Medium — affects long-term health | Low — needs longitudinal data | Low |
| Multi-agent collaboration | High — next capability frontier | Low — needs coordination primitives | Low (future) |
| DialogComplete model | High — but speculative | Low — needs prerequisite metrics | Low (future) |

---

## Integration with the Derivation Engine

The derivation engine references this document in three situations:

1. **User asks about a deferred feature:** The engine explains the feature, why it's deferred, and the current workaround (if any). It does not pretend the feature exists.

2. **Architect command encounters an open question:** The command acknowledges the uncertainty and recommends the conservative approach. "We don't yet know the optimal context file size, so the recommendation is to start with the standard template and adjust based on observed friction."

3. **Reseed considers open research:** When new research claims address an open question, the reseed audit mode flags it: "New evidence about optimal context file size — consider reviewing your context file length."

---

## Cross-Reference

- **Research claims that ground these questions:** See `claim-map.md` for TFT research claims referenced throughout.
- **Interaction constraints that complicate answers:** See `interaction-constraints.md` — many open questions involve dimension interactions.
- **Failure modes these questions would prevent:** See `failure-modes.md` for the failure modes that better metrics would detect earlier.
- **Current validation tests:** See `derivation-validation.md` for the 4 worked tests that validate the derivation engine today.
- **Personality layer questions:** See `personality-layer.md` for how personality drift detection relates to the encoding format.
- **Three-space architecture questions:** See `three-spaces.md` for the memory type routing that selective forgetting would extend.

---

## Exclusion Notes

- **Resolved: Plugin marketplace submission process.** Originally tracked as a deferred item but resolved during v1.0 development — standard marketplace submission, no special handling needed. Removed from deferred list.
- **Resolved: Context file format specification.** Was an open question about whether to use YAML frontmatter or markdown-only context files. Resolved in favor of markdown-only (CLAUDE.md). No longer tracked.
- **Dropped: Real-time collaboration.** Evaluated as a potential v2 item but dropped — real-time multi-user editing of knowledge graphs is a fundamentally different product category. CRDTs and operational transforms are out of scope for a methodology-focused system.
- **Subsumed: Hook testing framework.** Originally a separate open question but subsumed by Cross-Platform Hook Equivalence — the testing framework is part of the equivalence solution, not a standalone item.

---

## Version

- **Date:** 2026-03-20
- **Source claim count:** 24 (8 v1.1 deferred items + 2 v2 deferred items + 14 open research questions)
- **Included count:** 24 items tracked
- **Excluded count:** 4 (2 resolved, 1 dropped, 1 subsumed)
