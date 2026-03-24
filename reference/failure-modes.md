# Knowledge System Failure Modes

## Purpose

This document catalogs the ten predictable ways knowledge systems decay, serving as the derivation engine's reference for generating domain-appropriate warnings in context files. Each failure mode has documented causes, vulnerable domains, and prevention patterns. The init wizard uses this to select the top 3-4 failure modes for a given domain and include them as "Common Pitfalls" in the generated context file.

---

## Derivation Questions

1. **Which failure modes should a generated context file warn about?** Cross-reference the user's domain against the domain vulnerability matrix; include all HIGH-risk modes and mention medium-risk modes briefly.
2. **What prevention patterns should be generated for each failure mode?** Each claim includes a derivation implication specifying the structural countermeasure (pipeline phase, WIP limit, health check, etc.).
3. **How should warnings be phrased for the domain?** Warnings should use domain-native vocabulary (the personality layer controls voice; this document controls content selection).
4. **What early warning signs should health checks detect?** Each failure mode includes quantitative warning signs that map to health check thresholds.
5. **Which failure modes are universal vs domain-specific?** Collector's Fallacy and Productivity Porn are universal HIGH-risk; others vary by domain per the vulnerability matrix.

---

## Curated Claims

### Accumulation Failures

#### The collector's fallacy causes inbox to grow without synthesis

**Summary:** Accumulating without processing. Inbox grows indefinitely because capture is easy and feels productive while processing is hard and feels slow. The gap widens until the inbox becomes psychologically overwhelming.

**Derivation Implication:** Generated systems must include a processing pipeline with explicit reduce phase, WIP limits on inbox, and condition-based inbox review triggers. Warning signs: inbox > 20 items, new captures outnumber processed items 3:1.

**Source:** arscontexta failure mode analysis — collector's fallacy. Vulnerable domains: all (universal), especially Research and Learning.

#### Verbatim risk produces storage without the generation effect

**Summary:** Copying content instead of transforming it. Copy-paste is faster than reformulation, producing notes that are summaries rather than claims. The composability test ("This note argues that [title]") fails for verbatim content.

**Derivation Implication:** Generated systems must include the generation effect gate, composability test, source attribution that separates original from transformation, and review checkpoints for paraphrase vs genuine reformulation. Warning signs: notes that read like summaries, titles that are topic labels not claims.

**Source:** arscontexta failure mode analysis — verbatim risk. Vulnerable domains: Research, Learning.

### Graph Fragmentation Failures

#### Orphan drift fragments the knowledge graph into disconnected islands

**Summary:** Notes created but never connected. The reflect phase gets skipped "just this once" which becomes always. The system fragments because creation feels complete without connection.

**Derivation Implication:** Generated systems must include mandatory reflect phase after note creation, MOC maintenance in the creation workflow, periodic orphan detection via health checks, and topics footer as minimum connection requirement. Warning signs: >10% of notes with no incoming wiki-links, MOCs not updated in 30+ days.

**Source:** arscontexta failure mode analysis — orphan drift. Vulnerable domains: Research, Learning.

#### Link rot creates dead ends in the knowledge graph

**Summary:** Wiki-links pointing to notes that were renamed or deleted. Manual renames don't update references; archiving notes breaks incoming links.

**Derivation Implication:** Generated systems must include rename scripts that update all references, periodic link health checks, archive-instead-of-delete policy, and dangling link detection. Warning signs: >5% of wiki-links resolve to nothing, renames done without reference updates.

**Source:** arscontexta failure mode analysis — link rot. Vulnerable domains: all systems with heavy linking, worse at scale.

#### MOC sprawl creates navigation overhead exceeding navigation value

**Summary:** Too many MOCs created, none properly maintained. Creating a MOC feels like organizing, but an unmaintained MOC provides false confidence about topic coverage and is worse than no MOC.

**Derivation Implication:** Generated systems must include MOC creation threshold (20+ related notes, not speculative), merge-back rule for small MOCs (<10 notes), MOC health checks (size, freshness, coverage), and max 4 tiers of hierarchy. Warning signs: MOCs with <5 links, MOCs not updated in 60+ days, >50% of MOCs with no recent additions.

**Source:** arscontexta failure mode analysis — MOC sprawl. Vulnerable domains: Research, Creative.

### Schema and Structural Failures

#### Schema erosion makes YAML fields inconsistent and queries unreliable

**Summary:** YAML fields inconsistently used, enum values drift from template, validation ignored. Schemas add friction at capture time, and without enforcement the path of least resistance is skipping fields.

**Derivation Implication:** Generated systems must use templates as single source of truth, include validation automation (hooks or batch scripts), maintain minimal viable schema (only require fields that are actively queried), and support schema evolution. Warning signs: >20% of notes missing required fields, enum values appearing that aren't in the template.

**Source:** arscontexta failure mode analysis — schema erosion. Vulnerable domains: systems with moderate+ schema density.

#### Temporal staleness presents outdated content as current

**Summary:** Content becomes outdated but isn't flagged. Knowledge systems don't inherently track temporal validity — a note written 6 months ago appears identical to one written yesterday.

**Derivation Implication:** Generated systems in time-sensitive domains must include meta_state or staleness fields in schema, maintenance condition thresholds matched to domain's rate of change, periodic staleness sweeps, and date-aware health checks. Warning signs: notes referencing outdated information, no maintenance in high-change-rate domains.

**Source:** arscontexta failure mode analysis — temporal staleness. Vulnerable domains: Gaming strategy, PM, any time-sensitive domain.

### Human-System Boundary Failures

#### Cognitive outsourcing erodes human judgment capacity

**Summary:** Delegating judgment entirely to the system until the human can no longer evaluate quality. The system gets good enough that the human stops checking, and quality drifts without the human noticing.

**Derivation Implication:** Generated systems must include periodic human review checkpoints, never auto-implement system changes (propose, don't execute), maintain human judgment checkpoints in the pipeline, and flag confidence levels. Warning signs: human hasn't reviewed notes in 30+ days, system changes implemented without review, no recent human corrections.

**Source:** arscontexta failure mode analysis — cognitive outsourcing. Vulnerable domains: Research (unchecked claims compound), Therapy (pattern detection without human validation).

#### Over-automation corrupts quality by encoding judgment in hooks

**Summary:** Hooks encoding judgment rather than verification, corrupting quality silently. The boundary between "check if description exists" (verification) and "write a good description" (judgment) blurs when automation feels like progress.

**Derivation Implication:** Generated systems must enforce the determinism boundary (hooks for verification, skills for judgment), require automation to fail loudly not fix silently, include hook behavior review when false positives are suspected, and use graduated enforcement (warn before block). Warning signs: hooks auto-fixing content, perfect quality metrics with shallow notes, no validation failures.

**Source:** arscontexta failure mode analysis — over-automation. Vulnerable domains: any system at automation level.

#### Productivity porn displaces knowledge work with meta-work

**Summary:** Building the system instead of using it. System design is interesting and feels productive while actual knowledge work (processing, connecting, synthesizing) is harder and less visible.

**Derivation Implication:** Generated systems must follow Gall's Law (complex systems evolve from simple ones that work), time-box system improvement to <20% of total work time, and track content creation vs system modification ratio. Warning signs: more time on CLAUDE.md than on notes, system redesigns before 100 notes, template modifications outnumber note creations.

**Source:** arscontexta failure mode analysis — productivity porn. Vulnerable domains: all (universal), especially system builders and PM.

---

### Supplementary Reference

#### Domain Vulnerability Matrix

Which failure modes are highest risk per use case. Select the top 3-4 for inclusion in the generated context file's "Common Pitfalls" section.

| Failure Mode | Research | Learning | Therapy | Relationships | Creative | PM | Companion |
|-------------|----------|----------|---------|---------------|----------|-----|-----------|
| Collector's Fallacy | HIGH | HIGH | medium | low | medium | medium | low |
| Orphan Drift | HIGH | HIGH | medium | low | medium | low | low |
| Link Rot | medium | medium | low | low | medium | low | low |
| Schema Erosion | medium | medium | medium | medium | low | medium | low |
| MOC Sprawl | HIGH | medium | low | low | medium | low | low |
| Verbatim Risk | HIGH | HIGH | low | low | low | low | low |
| Cognitive Outsourcing | HIGH | medium | HIGH | low | medium | low | medium |
| Over-Automation | medium | medium | medium | low | low | medium | low |
| Productivity Porn | HIGH | medium | low | low | medium | HIGH | low |
| Temporal Staleness | low | medium | low | low | low | HIGH | low |

**Reading the matrix:** Include all HIGH-risk modes in the generated context file. Mention medium-risk modes briefly. Omit low-risk modes.

#### Integration with Init

When generating a context file in Step 5b, include a "Common Pitfalls" section:

1. Look up the user's use case in the domain vulnerability matrix
2. Select all HIGH-risk failure modes (typically 3-4)
3. Write each warning in domain-native vocabulary
4. Include the prevention pattern, not just the warning
5. Place after Self-Extension Blueprints, before System Evolution

Example for Research:
```markdown
## Common Pitfalls

### The Collector's Fallacy
Saving sources feels productive but isn't. If your inbox grows faster than you process it, stop capturing and start extracting. WIP limit: process what you have before adding more.

### Orphan Notes
A note without connections is a note that will never be found again. Every note needs at least one MOC link (Topics footer) and ideally inline connections to related notes. Run health checks to catch orphans.

### Verbatim Copying
Summarizing a source is not the same as extracting a claim. Each note must transform the material — your words, your framing, your argument. If the title doesn't pass "This note argues that [title]", it's not a claim yet.

### Productivity Porn
It's tempting to keep perfecting the system instead of using it. The vault serves the research, not the other way around. If you're spending more time on methodology than on claims, recalibrate.
```

---

## Exclusion Notes

### Information overload (too many notes to navigate)
**Reason:** This is a symptom of other failure modes (MOC sprawl, orphan drift, collector's fallacy) rather than a distinct failure mode. Addressing the root causes via their respective prevention patterns resolves the navigation problem.

### Platform lock-in (dependency on specific tools)
**Reason:** arscontexta's markdown-yaml primitive already addresses this at the kernel level. Platform independence is a design constraint, not a failure mode of knowledge systems per se.

---

## Version
- **Last curated:** 2026-03-20
- **Source claims evaluated:** 12
- **Claims included:** 10 (2 accumulation, 3 graph fragmentation, 2 schema/structural, 3 human-system boundary)
- **Claims excluded:** 2
