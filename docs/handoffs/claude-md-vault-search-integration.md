# CLAUDE.md Refactoring Task: Integrate Vault Search Rule

## Context

This project uses a knowledge graph vault (523 notes in `notes/`, searchable via `qmd` semantic search and wiki-link navigation). The vault contains design decisions, domain constraints, research findings, and architecture rationale accumulated over months of work on a USV (ultrasonic vocalization) analysis pipeline.

The agent (Claude Code) operates under instructions defined in `CLAUDE.md` (attached separately). This file is loaded into the agent's context at the start of every conversation and governs all behavior.

## The Problem

We discovered that when the agent performs domain-related tasks (explaining code architecture, analyzing design decisions, answering questions about why things were built a certain way), it does NOT search the vault first. It relies entirely on its parametric knowledge + reading source code, missing design rationale, constraints, and prior findings that live in vault notes.

Example: the user asked the agent to explain the VQ-VAE architecture. The agent read the source files and gave a thorough technical explanation, but never checked whether the vault contained notes about WHY specific design choices were made (e.g., why K=64 codebook entries, why L2 normalization, the ADR-007 two-phase architecture decision). The vault has exactly these notes, but they were never consulted.

Currently, CLAUDE.md only mandates vault search for **file modifications** of high-risk files (the "Mid-Session Knowledge Checks" section). There is no rule covering explanations, analysis, or general domain reasoning.

## Research Backing

We consulted the project's methodology research graph (249 research claims). Key findings:

1. **Activation timing matters as much as retrieval quality** (vault note + methodology claim) — The decision of *when* to retrieve is the primary bottleneck, not how good the retrieval is. The vault has excellent search infrastructure; the gap is entirely in activation triggers.

2. **FLARE pattern** (vault note) — Using the agent's *intended action* as the retrieval query catches constraints from related systems that file-specific searches miss. The existing `/kcheck` skill already implements this pattern but its documented scope is artificially narrow.

3. **Three-layer timing model** (vault note) — The system has three activation layers: session-level (orient hook, broad context), planning-level (/kcheck, task-focused), and file-level (canary comments, specific). The gap is at the planning layer — it only fires for file modifications, not for domain reasoning tasks.

4. **External memory shapes cognition** (methodology claim) — Without a retrieval trigger, vault knowledge never enters the agent's context window. The agent reasons from parametric memory alone, which may be incomplete or miss project-specific constraints.

5. **Metacognitive confidence diverges from retrieval capability** (methodology claim) — The agent may feel confident answering from parametric knowledge while missing vault-specific findings. It doesn't know what it doesn't know.

### Failure Modes to Guard Against

- **Context pollution**: Loading irrelevant search results wastes high-quality attention slots
- **False confidence**: Finding *some* related notes creates illusion that all constraints are surfaced
- **Ritual without reasoning**: If vault search becomes a mindless checkbox, it loses value — must NOT be automated as a hook
- **Over-searching**: Pure code mechanics tasks (syntax, debugging) don't benefit from vault search

The recommended filter is: **"Would checking the vault plausibly change my answer?"** If yes, search. If not, skip.

## Current CLAUDE.md State

- **439 lines** total — already long
- The file has a clear structure: Core Principles (top) -> Behavioral Contract -> Project Overview -> Task Routing -> Knowledge Graph -> Maintenance
- The "Mid-Session Knowledge Checks" section (around line 200) currently reads:

```markdown
## Mid-Session Knowledge Checks
Before modifying files in high-risk directories (detection app, export adapters,
labeling pipeline), run `/kcheck "<brief description of planned changes>"`.
This is mandatory for HIGH-risk canary files and recommended for any non-trivial
modification to existing systems.

Skip /kcheck for: new standalone files, test files, documentation-only changes.
```

## Constraints on the Solution

1. **Primacy effect**: Instructions earlier in the file have stronger behavioral influence on the LLM. The vault search rule should be near the top, not buried at line 400.

2. **Net-zero line budget**: The file is already 439 lines. The solution should not add significant length. Prefer replacing/merging over adding new sections.

3. **No hook automation**: The expert specifically recommended against making this a hook. It should remain a judgment-based behavioral expectation.

4. **Compatibility with existing /kcheck**: The `/kcheck` skill already supports arbitrary task descriptions (not just file modifications). The skill doesn't need to change — only the documented trigger conditions in CLAUDE.md.

5. **The Behavioral Contract section (lines 17-82)** is the highest-impact location. It contains Core Rules (Integrity, Learning Mode, Epistemic Honesty) that the agent follows most reliably because of their position and grouping.

## Our Current Proposal (For Your Review)

We proposed adding a 4th core rule block inside the Behavioral Contract's "Core Rules" section, after "Epistemic Honesty" (line 62):

```markdown
#### Knowledge Activation (Before Domain Tasks)
- **Search before reasoning**: Before explaining, analyzing, or modifying domain-specific code,
  search the vault (qmd + topic maps). "Would the vault plausibly change my answer?" -> search first.
- **Use /kcheck for modifications**: Mandatory for HIGH-risk canary files, recommended for any
  constrained system (detection, export, labeling, classification).
- **Skip for**: pure code mechanics, general knowledge, test files, documentation-only changes.
```

And deleting the existing "Mid-Session Knowledge Checks" section since it would be fully superseded.

Plus a one-line cross-reference in the Session Rhythm "Work" phase:
```markdown
- **Work**: Do the task. **Search vault before domain reasoning (see Core Rules: Knowledge Activation).** Surface connections.
```

## What We Need From You

1. **Review the overall CLAUDE.md structure** (attached) — is 439 lines too bloated? Are there sections that could be consolidated or moved to reference docs?

2. **Evaluate our proposed placement** — is the Behavioral Contract the right location? Should it be even higher (in the "STOP - READ BEFORE DOING ANYTHING" header)?

3. **Evaluate the wording** — is the rule clear enough? Too verbose? Missing edge cases?

4. **Consider whether CLAUDE.md needs a broader refactor** — the user suspects it's getting too big. If so, what should stay in CLAUDE.md (high-priority behavioral rules) vs. what could move to reference docs that get loaded on-demand?

Give us back a concrete recommendation: either the exact text to add/remove/change, or a broader restructuring plan if you think CLAUDE.md needs more than a surgical edit.
