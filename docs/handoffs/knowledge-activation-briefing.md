# Knowledge Activation Gap: Briefing for Collaborative Problem-Solving

## Context

This briefing comes from a USV (ultrasonic vocalization) research project that uses Claude Code as the primary development agent. The project maintains a **knowledge graph vault** of 505 atomic markdown notes, 17 topic maps, and semantic search via qmd (a local markdown search engine with keyword + vector search). The vault is managed by an arscontexta-based methodology with structured skills (/reduce, /reflect, /reweave, etc.).

The system also uses **Codex** (OpenAI's async coding agent) for implementation work, with a handoff protocol between the two agents.

## The Problem

**The vault accumulates knowledge that the agent rarely activates.**

The agent (Claude Code) sees these things automatically:
- `CLAUDE.md` (~5.4k tokens) — behavioral contract, task routing, conventions
- `MEMORY.md` (~3.1k tokens) — cross-session state, environment notes
- `ops/goals.md` — active threads, read at session start via a PowerShell hook

The 505 notes in `notes/` are **only surfaced if the agent actively decides to search** using qmd. If the agent doesn't think to look, the knowledge doesn't exist for that session. This creates a paradox: significant effort goes into maintaining high-quality notes, but the retrieval step is voluntary and often skipped.

### Why This Matters

The vault has notes like:
- "saved-previous ghost detections form three aligned detection state tiers in the app" — an architectural pattern an agent MUST know before modifying the detection app
- "DeepSqueak import previously required exact subdirectory name matches while Raven export already supported prefix matches" — a design constraint that caused a real bug

If an agent modifies these systems without encountering these notes, it risks re-introducing fixed bugs or violating architectural invariants. The notes exist precisely to prevent this — but they only work if they're found.

## Research Diagnosis (from arscontexta methodology)

The arscontexta research corpus (249 claims backing the knowledge system design) addresses this gap explicitly:

### 1. Metacognitive Confidence Diverges from Retrieval Capability
> "A vault with well-organized folders, good descriptions, and dense links may feel navigable while actual retrieval fails. Structural quality signals can produce false confidence that masks actual retrieval failures."

The agent *feels* like it has access to 505 notes because it knows the system exists. But "knowing the system exists" ≠ actually retrieving relevant notes. The divergence is silent and self-reinforcing.

### 2. The System Was Designed for Pull, Not Push
The vault architecture (descriptions as filters, flat structure, sentence titles, MOC navigation) was optimized for findability *given a search*. The research assumed the search would happen. It didn't address what **triggers** the search.

### 3. Spreading Activation Needs a Trigger Node
> "Graph traversal IS spreading activation. When you follow wiki links to load context, you're replicating what the brain does when priming related concepts."

But spreading activation requires a starting point. In the current system, the agent must voluntarily enter the graph. Without that initial activation, the entire graph sits dormant.

### 4. The Programmable Notes Vision
> "Notes that act based on their properties [...] shifts the architecture from pull (agent searches for relevant content) to push (content declares its relevance)."

This is the theoretical answer: notes that surface themselves. But it's flagged as speculative research, not validated architecture.

### 5. Scale Threshold
At 505 notes, we're exactly at the regime where "dense semantic connections are required" for retrieval to work. The infrastructure is appropriate (17 topic maps, qmd, dense links) — but the infrastructure is only activated when the agent decides to use it.

## Current Platform Constraints

| Capability | Status |
|---|---|
| Claude Code hooks (PreToolUse, PostToolUse, SessionStart, Stop) | Can block/allow actions, run scripts. **Cannot inject content into conversation context.** |
| qmd search | Keyword (~30ms), vector (~2s), deep (~10s). Available as MCP tools. |
| Context window | 200k tokens. Typical session uses ~44% at midpoint. ~33k reserved for autocompact. |
| CLAUDE.md + MEMORY.md | Always loaded. Total ~8.5k tokens. |
| Session orient hook | Runs PowerShell at session start. Currently reads goals.md and reminders.md, outputs text to stdout (which appears in the first message). |
| Skills | Can include arbitrary retrieval steps in their procedures. Currently /reduce and /reflect consult methodology claims, but /implement and review workflows do not search the vault. |

**Key constraint:** Hooks output to stdout, which appears as context in the agent's first response. This IS a form of injection — limited to session start, but it works. Mid-session injection is not possible through hooks.

## Proposed Three-Layer Activation Architecture

### Layer 1: Goal-Aware Orient (Immediate, Highest ROI)

**What:** Enhance the session-start hook to run qmd searches against active goal threads and write results to `ops/session-relevance.md`. Add "Read ops/session-relevance.md" to the orient procedure in CLAUDE.md.

**How it works:**
1. Orient hook reads `ops/goals.md`, extracts active thread titles/descriptions
2. For each active thread, runs `qmd vector_search` against notes/
3. Writes a compact relevance brief: note titles + descriptions only (progressive disclosure)
4. Agent reads the brief during orientation, decides which notes to load fully

**Token cost:** ~3,000-4,000 tokens for 20 relevant descriptions. Well within budget.

**Why this is the highest-ROI change:** It fires every session with zero agent discretion required. The agent doesn't need to "think to search" — the search happens before the agent starts working.

**Limitation:** Only fires at session start. Mid-session task pivots still require voluntary search.

**Implementation questions:**
- Should the hook run qmd via CLI (`qmd search "query"`) or via the MCP server?
- How to extract good search queries from goals.md automatically?
- Should the brief include topic map pointers in addition to note descriptions?
- How to handle Codex sessions (different agent, different orient hook)?

### Layer 2: Skill-Level Pre-Retrieval Gates (Medium Effort)

**What:** Modify key skills (/implement, /reduce, /reflect, /reweave) to include a mandatory qmd search step before their core logic.

**How it works:**
```
BEFORE starting work:
1. Extract the core topic/claim from the task
2. Run qmd deep_search with that topic
3. Read descriptions of top 5-10 results
4. Load full content of any directly relevant notes
5. THEN proceed with the skill's core logic
```

**Why this works:** The skill invocation IS the trigger node for spreading activation. The task description provides the activation seed. The qmd search performs spreading activation through the semantic layer. This bypasses metacognitive confidence ("I think I know what's relevant") with actual retrieval.

**Token cost:** ~1,500-2,000 tokens per gate (10 descriptions). Manageable.

**Implementation questions:**
- Which skills are highest priority for gates? (/implement and /reduce seem most impactful)
- Should the gate be a shared utility function or duplicated per skill?
- How to extract good search queries from skill arguments?
- Should the gate be skippable if the agent already loaded relevant notes?

### Layer 3: Property-Triggered Surfacing (Future, Higher Complexity)

**What:** Notes surface themselves based on metadata properties when context matches.

**Candidates:**
- Notes with `type: decision` surface when the agent is about to modify files in the same domain
- Notes with `type: baseline` surface when evaluating performance
- Notes with `meta_state: outdated` surface during /reduce of new sources on the same topic

**Implementation:** A periodic scan script that queries notes by properties + task context, writing results to a staging file.

**Why defer this:** Per the research, "complex systems evolve from simple working systems." Validate Layers 1-2 first. Property triggers add rule complexity and false-positive risk.

## The Deeper Questions (For Discussion)

### 1. Is usage tracking worth implementing?

**Pro:** Tracking which notes get retrieved vs dormant reveals retrieval failures (poorly described notes, missing links, dead topic map areas).

**Con:** The research warns against using usage as a pruning signal:
> "Notes that SHOULD receive attention but don't generate demand signals represent the blind spot. Random selection surfaces notes independent of demand signals."

Usage tracking measures the symptom (low retrieval) without treating the cause (no activation mechanism). Implement activation first, then track to evaluate.

### 2. Is 505 notes too many?

The research says no — link density matters more than note count. But it also says the vault is only earning ~40-60% of its potential value because activation is the bottleneck, not storage. The right response is better activation, not fewer notes.

### 3. Should notes have "activation energy" levels?

Some notes are critical invariants (sample rate = 300 kHz, always specify sr explicitly). Others are historical findings that matter in specific contexts. Could notes declare their own activation priority?

### 4. What about the Codex side?

Codex uses `AGENTS.md` and `docs/codex_index.md` for orientation. It has no access to qmd or the vault's semantic search. Should Codex-facing orientation include relevant vault knowledge, or is that Claude Code's responsibility during handoff review?

### 5. The context budget trade-off

Every activation mechanism trades context space for knowledge coverage. The research warns:
> "LLM attention degrades as context fills. The first 40% of context is the 'smart zone.'"

The goal is not "inject everything relevant" but "inject just enough to trigger the agent's own spreading activation through the graph." Seeds, not dumps.

## What I Need Help With

1. **Architecture design** for Layer 1 (goal-aware orient) — specifically, how to extract good search queries from goals.md and how to format the relevance brief for maximum utility at minimum token cost.

2. **Evaluating the three-layer approach** — is this the right decomposition? Are there simpler alternatives we're missing?

3. **The hook injection constraint** — creative workarounds for surfacing knowledge mid-session, not just at session start. Could a "knowledge check" be a lightweight skill the agent is instructed to run before certain categories of work?

4. **Cross-agent knowledge flow** — how should Codex benefit from vault knowledge when it can't search qmd directly?

## File References

If you need to see any of these:
- Vault structure: `notes/index.md` (master index) → topic maps → individual notes
- Current orient hook: `.claude/hooks/session-orient.ps1`
- Goals file: `ops/goals.md`
- Methodology claims referenced: `methodology/metacognitive-confidence-can-diverge-from-retrieval-capability.md`, `methodology/retrieval-utility-should-drive-design-over-capture-completeness.md`, `methodology/spreading-activation-models-how-agents-should-traverse.md`, `methodology/programmable-notes-could-enable-property-triggered-workflows.md`, `methodology/flat-files-break-at-retrieval-scale.md`
- Research reference routing: `reference/claim-map.md`
