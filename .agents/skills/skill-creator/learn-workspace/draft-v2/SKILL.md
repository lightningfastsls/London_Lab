---
name: learn
description: Research a topic and grow your knowledge graph. Investigates topics using web search, checks existing vault knowledge to avoid redundancy, captures findings as structured inbox source files with full provenance, and chains to the processing pipeline. Use this skill whenever the user wants to research something, investigate a topic, look something up, or add external knowledge to the vault. Triggers on "/learn", "/learn [topic]", "research this", "find out about", "look up", "what do we know about".
version: "2.0"
user-invocable: true
allowed-tools: Read, Write, Edit, Grep, Glob, Bash, WebSearch, WebFetch, mcp__qmd__search, mcp__qmd__vector_search, mcp__qmd__deep_search, mcp__qmd__get
context: fork
argument-hint: "[topic] -- what to research (e.g., 'VAE clustering for USV classification')"
---

## EXECUTE NOW

**Target: $ARGUMENTS**

Parse immediately:
- If target is a topic string: research that topic
- If target is a URL: fetch and capture that specific source
- If target is empty: ask the user what they want to learn about
- If target contains `--quick` or `--deep`: override auto-detected depth

**Execute these phases in order. The research phase (Phase 2) is the core of this skill -- spend most of your effort there.**

1. Quick setup and vault check (Phase 1)
2. **Research thoroughly** (Phase 2) -- this is the main event
3. Format and capture (Phase 3)
4. Report and chain (Phase 4)

**START NOW.**

---

# Learn

Research a topic and bring structured knowledge into the vault. This is the only skill that creates new knowledge from external sources -- everything else in the pipeline operates on material already inside the vault.

## Philosophy

**Thoroughness first, formatting second.** The quality of your research -- breadth of sources, depth of analysis, verification of claims -- is what matters most. The pipeline formatting is important but secondary: a brilliantly structured inbox file with shallow research is worth less than a comprehensive investigation with good provenance.

Do your research thoroughly, then format the results. Not the other way around.

## Phase 1: Setup and Vault Check

This phase should be fast. Don't let setup overhead eat into research time.

### 1a. Assess Research Depth

Decide how deep to go based on the topic:

| Signal | Depth | Example |
|--------|-------|---------|
| Specific factual question, single concept | **quick** | "what FFT window does DeepSqueak use" |
| Moderate scope, comparing options, how-to | **moderate** | "Python libraries for USV detection" |
| Broad survey, theoretical, multi-faceted | **deep** | "state of the art in bioacoustic deep learning 2025" |

If `--quick` or `--deep` is in the arguments, use that override. Tell the user which depth you chose briefly.

### 1b. Check Existing Knowledge (parallel with depth assessment)

Check what the vault already knows so you can focus on gaps:
- **Quick depth**: Use `mcp__qmd__search(query="<topic keywords>")` -- fast keyword search is enough
- **Moderate/deep**: Use `mcp__qmd__deep_search(query="<topic>")` and skim relevant topic maps in `notes/index.md`

Report briefly: "Found 3 existing notes on X. I'll focus on Y." Then move on immediately.

### 1c. Read Config (only what you need)

Glance at `ops/derivation-manifest.md` for the vocabulary mapping (inbox folder name, reduce command name). Don't read the full config chain -- you can check `ops/config.yaml` for chaining mode at the end when you need it.

## Phase 2: Research

**This is the core of the skill. Spend most of your effort here.**

The goal is to produce research that is comprehensive, well-sourced, and cross-verified. A future /reduce pass will mine this for atomic claims, so every substantive finding you include becomes a potential knowledge graph node.

### Research Depth Targets

These are **minimum floors**, not ceilings. If a topic demands more sources or deeper investigation, go further.

| Depth | Min. search queries | Min. sources to fetch | Min. output scope | What "done" looks like |
|-------|--------------------|-----------------------|-------------------|----------------------|
| **quick** | 2 | 2-3 | Answer the specific question | Definitive answer verified from multiple independent sources (paper + code + docs). Include parameter tables, version numbers, configuration details. |
| **moderate** | 3-5 | 5-8 | Comprehensive comparison | All relevant options covered with pros/cons. Benchmark data where available. Actionable recommendations. Install commands, repo URLs, key publications for every tool. |
| **deep** | 5-8 | 8-15 | Full survey | All major sub-topics covered. Foundation models, methods, tools, benchmarks, open challenges. Structured with sections. No significant area left unaddressed. |

### Research Strategy

**Cross-verify claims.** Don't trust a single source. For factual claims (parameter values, performance numbers), verify from at least two independent sources -- e.g., the published paper AND the source code, or two independent benchmarks. The baseline without-skill naturally does this; you should too.

**Chase primary sources.** When a blog post or secondary source mentions a paper, tool, or benchmark, find the original. Primary sources have parameter values, code snippets, and methodology details that secondary sources summarize away.

**Dig into source code when relevant.** For tools and libraries, check the actual GitHub repository. Source code provides configuration defaults, variable names, and implementation details that documentation often omits. This is especially valuable for quick lookups where a single parameter value is the answer.

**Cover the landscape.** For moderate and deep topics, map the full space of options before narrowing. If comparing tools, find ALL the relevant ones first, then analyze each. Don't stop at the first 3-4 results.

**Search from multiple angles.** Use different query phrasings to reach different parts of the literature:
- Technical terms vs. common names
- Tool names vs. method names
- Recent year-specific queries vs. general queries
- "comparison" / "benchmark" / "review" / "alternative to" queries

### Source Quality Hierarchy

Prefer sources in this order:
1. **Primary research** -- peer-reviewed papers, preprints (arXiv, bioRxiv), source code
2. **Official documentation** -- library docs, tool READMEs, API references
3. **Expert writing** -- technical blog posts from researchers, conference talks
4. **Community knowledge** -- Stack Overflow, GitHub issues, forum discussions

### Provenance Tracking

As you research, keep a running log of every source consulted:
```
Source Log:
1. [URL] -- fetched, high relevance: brief note on key finding
2. [URL] -- fetched, medium: what was found
3. [URL] -- search result, not fetched: why skipped
4. [URL] -- blocked/failed: note the issue
```

This log becomes part of the final capture. Full traceability: every claim should be traceable to its source.

## Phase 3: Format and Capture

**Only start this phase after your research is complete.** Don't interleave formatting with research -- it causes you to stop researching too early.

Write a single inbox source file at `inbox/{topic-slug}-research-{YYYY-MM-DD}.md` (or use the vocabulary.inbox folder name from config).

### Source Capture Structure

```markdown
---
description: "{one-line summary of what was researched and found}"
source_type: article
url: "{primary source URL, or 'multiple -- see source log'}"
author: "{primary author, or 'multiple sources'}"
date_accessed: "{YYYY-MM-DD}"
status: unprocessed
research_tool: "web-search"
research_query: "{primary search query used}"
research_depth: "{quick|moderate|deep}"
---

# {Topic Title}

{Opening paragraph: the core finding, dense with insight.}

---

## {Section per major finding or sub-topic}

{Content synthesized from sources. Restate in your own framing.
Include specific numbers, tool names, parameter values.
Cite sources inline: "According to [Author, Year]..."}

---

## Source Log

| # | URL | Status | Relevance | Key Finding |
|---|-----|--------|-----------|-------------|
| 1 | {url} | fetched | high | {what was found} |

## Research Context

- **Query**: {original user request}
- **Depth**: {quick|moderate|deep} (auto-detected | overridden)
- **Existing vault knowledge**: {summary of Phase 1b findings}
- **Knowledge gap addressed**: {what new ground this covers}
```

### Quality Checks

Before saving, verify:
- Description is under 200 characters
- At least one concrete finding (not just "there are many approaches")
- Source log has entries for ALL sources consulted (fetched, skipped, and failed)
- No verbatim text copied from sources
- Research depth matches the targets in Phase 2 (did you actually hit the minimums?)

## Phase 4: Report and Chain

Tell the user:
1. What was researched and at what depth
2. How many sources consulted
3. Key findings (2-3 bullets)
4. Where the file was saved
5. What vault knowledge was relevant

Then check `ops/config.yaml` for `processing.chaining`:
- **manual**: Mention `/reduce {file}` or `/seed {file}` as next steps
- **suggested**: Recommend /reduce actively with an estimate of extractable notes
- **automatic**: Invoke /seed immediately

## Edge Cases

**URL as input:** Fetch just that URL, skip vault check, capture as source.

**Topic already well-covered:** Present what's known, ask if the user wants updates or a specific gap explored.

**Research yields nothing:** Report honestly. Don't create an empty inbox file.

**Multiple sub-topics:** Handle in a single capture with clear section breaks.
