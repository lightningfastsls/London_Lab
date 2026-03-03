---
name: learn
description: Research a topic and grow your knowledge graph. Investigates topics using web search, checks existing vault knowledge to avoid redundancy, captures findings as structured inbox source files with full provenance, and chains to the processing pipeline. Use this skill whenever the user wants to research something, investigate a topic, look something up, or add external knowledge to the vault. Triggers on "/learn", "/learn [topic]", "research this", "find out about", "look up", "what do we know about".
version: "1.0"
user-invocable: true
allowed-tools: Read, Write, Edit, Grep, Glob, Bash, WebSearch, WebFetch, mcp__qmd__search, mcp__qmd__vector_search, mcp__qmd__deep_search, mcp__qmd__get
context: fork
argument-hint: "[topic] -- what to research (e.g., 'VAE clustering for USV classification')"
---

## Runtime Configuration (Step 0 -- before any processing)

Read these files to configure domain-specific behavior:

1. **`ops/derivation-manifest.md`** -- vocabulary mapping, platform hints
   - Use `vocabulary.notes` for the notes folder name
   - Use `vocabulary.inbox` for the inbox folder name
   - Use `vocabulary.cmd_reduce` for the next-phase command name
   - Use `vocabulary.topic_map` / `vocabulary.topic_maps` for MOC references

2. **`ops/config.yaml`** -- processing depth, pipeline chaining, research config
   - `processing.chaining`: manual | suggested | automatic
   - `research.primary`: preferred research tool
   - `research.default_depth`: fallback depth if auto-detection is ambiguous

3. **`templates/source-capture.md`** -- inbox file template (schema + structure)

If these files don't exist, use universal defaults.

## EXECUTE NOW

**Target: $ARGUMENTS**

Parse immediately:
- If target is a topic string: research that topic
- If target is a URL: fetch and capture that specific source
- If target is empty: ask the user what they want to learn about
- If target contains `--quick` or `--deep`: override auto-detected depth

**Execute these steps in order:**

1. Assess research depth (Phase 1)
2. Check existing vault knowledge (Phase 2)
3. Research the topic (Phase 3)
4. Synthesize and capture (Phase 4)
5. Report and chain (Phase 5)

**START NOW.** Reference below explains each phase.

---

# Learn

Research a topic from the outside world and bring structured knowledge into the vault. This is the only skill that creates new knowledge from external sources -- everything else in the pipeline (/reduce, /reflect, /reweave) operates on material already inside the vault.

## Philosophy

**Research is not collecting links.** The goal is to produce a single, comprehensive source capture that a future /reduce pass can mine for atomic claims. The quality of the inbox file directly determines the quality of the notes that come out of it.

This means: read sources carefully, synthesize across them, capture in your own framing (not verbatim), and preserve the provenance chain so every claim stays traceable back to its origin.

**Depth adapts to the topic.** A specific factual question ("what sample rate does MUPET use?") needs a quick lookup, not a deep literature review. A broad exploration ("how do researchers classify rodent USVs in 2025?") deserves multiple sources and synthesis. The skill auto-detects this from the topic structure.

## Phase 1: Assess Research Depth

Decide how deep to go based on the topic itself. The right depth prevents both shallow answers to complex questions and wasteful over-research of simple lookups.

**Auto-detection heuristics:**

| Signal | Depth | Example |
|--------|-------|---------|
| Specific factual question, single concept | **quick** | "what FFT window does DeepSqueak use" |
| Moderate scope, comparing options, how-to | **moderate** | "Python libraries for USV detection" |
| Broad survey, theoretical, multi-faceted | **deep** | "state of the art in bioacoustic deep learning 2025" |

| Depth | Search queries | Sources to fetch | Synthesis effort |
|-------|---------------|-----------------|-----------------|
| quick | 1-2 targeted queries | 1-2 pages | Brief, factual |
| moderate | 2-4 queries from different angles | 3-5 pages | Compare, contrast, recommend |
| deep | 4-6 queries covering sub-topics | 5-8 pages | Comprehensive survey with structure |

If `--quick` or `--deep` is in the arguments, use that override. Otherwise auto-detect. If ambiguous, fall back to `ops/config.yaml` `research.default_depth`.

Tell the user which depth you chose and why, briefly: "This looks like a moderate-depth topic -- I'll search from a few angles and synthesize 3-5 sources."

## Phase 2: Check Existing Knowledge

Before going to the web, check what the vault already knows. This prevents redundant research and helps focus the search on genuine gaps.

1. **Semantic search** the vault for the topic:
   ```
   mcp__qmd__deep_search(query="<topic>", collection="<notes_collection>")
   ```

2. **Scan topic maps** for relevant areas:
   - Read `{vocabulary.notes}/index.md` to find relevant topic maps
   - Skim any topic maps that might cover this area

3. **Assess the gap:**
   - If the vault has strong coverage: tell the user what's already known and ask if they want to research further anyway
   - If partial coverage: note what's known and focus the research on gaps
   - If no coverage: proceed to full research

Report findings briefly: "Found 3 existing notes on USV clustering, but nothing on VAE-based approaches. I'll focus the research there."

## Phase 3: Research

Execute the research using web search and web fetch. The specific approach depends on the depth level from Phase 1.

### Search Strategy

Build search queries that are specific enough to return useful results. Consider:
- The user's domain context (this is a USV/bioacoustics research project)
- What the vault already knows (from Phase 2)
- Different angles on the topic (theoretical, practical, comparative)

**For each depth level:**

**Quick:** Run 1-2 WebSearch queries. Fetch the most relevant result with WebFetch. Extract the answer.

**Moderate:** Run 2-4 WebSearch queries from different angles. Fetch 3-5 of the most promising results with WebFetch. Cross-reference findings.

**Deep:** Run 4-6 WebSearch queries covering sub-topics. Fetch 5-8 sources. Look for primary sources (papers, official docs) over secondary ones (blog posts, tutorials). Build a structured understanding.

### Source Quality Hierarchy

Prefer sources in this order:
1. **Primary research** -- peer-reviewed papers, preprints (arXiv, bioRxiv)
2. **Official documentation** -- library docs, tool READMEs, API references
3. **Expert writing** -- technical blog posts from researchers, conference talks
4. **Community knowledge** -- Stack Overflow, GitHub issues, forum discussions

### Provenance Tracking

As you research, maintain a running list of every source consulted:

```
Source Log:
1. [URL] -- fetched, relevant: brief note on what was found
2. [URL] -- fetched, not relevant: why
3. [URL] -- search result, not fetched: why skipped
```

This log becomes part of the source capture metadata. The goal is full traceability: anyone reading the inbox file should be able to follow the research trail back to the original sources.

## Phase 4: Synthesize and Capture

Write a single comprehensive inbox source file. The file goes in `{vocabulary.inbox}/` and follows the `source-capture.md` template.

### Filename Convention

Use a descriptive, kebab-case filename:
```
{vocabulary.inbox}/{topic-slug}-research-{YYYY-MM-DD}.md
```

Example: `inbox/vae-clustering-usv-classification-research-2026-02-27.md`

### Source Capture Format

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

# {Topic}: {synthesis title}

{Opening paragraph: the single most important takeaway, written as a clear claim.
This paragraph should be dense with insight -- a reader who stops here should still
get the core message.}

---

## {Section per major finding or sub-topic}

{Content synthesized from sources. Restate in your own framing -- no verbatim copying.
Include specific numbers, tool names, parameter values where relevant.
Cite sources inline: "According to [Author, Year] / [Tool Documentation]..."}

---

## Source Log

| # | URL | Status | Relevance | Key Finding |
|---|-----|--------|-----------|-------------|
| 1 | {url} | fetched | high | {what was found} |
| 2 | {url} | fetched | medium | {what was found} |
| 3 | {url} | skipped | low | {why skipped} |

## Research Context

- **Query**: {original user request}
- **Depth**: {quick|moderate|deep} (auto-detected | overridden)
- **Existing vault knowledge**: {summary of Phase 2 findings}
- **Knowledge gap addressed**: {what new ground this covers}
```

### Quality Checks Before Saving

Before writing the file, verify:
- [ ] Description is under 200 characters and adds context beyond the title
- [ ] At least one concrete finding (not just "there are many approaches")
- [ ] Source log has entries for all sources consulted
- [ ] No verbatim text copied from sources (paraphrase everything)
- [ ] Frontmatter matches the source-capture template schema

## Phase 5: Report and Chain

### Summary Report

Tell the user:
1. What was researched (topic + depth used)
2. How many sources were consulted
3. Key findings (2-3 bullet points)
4. Where the file was saved
5. What existing vault knowledge was relevant (if any)

### Pipeline Chaining

Based on `ops/config.yaml` `processing.chaining`:

- **manual**: "Next step: `{vocabulary.cmd_reduce} {inbox_file_path}` to extract atomic notes from this research. Or run `/seed {inbox_file_path}` to queue it for batch processing."
- **suggested**: Same output as manual, but recommend it more actively: "I'd suggest running `{vocabulary.cmd_reduce}` next to extract the insights while the context is fresh."
- **automatic**: Invoke /seed on the inbox file immediately, then report that it's been queued.

## Edge Cases

### URL as Input

If the user provides a URL instead of a topic:
1. Skip Phase 1 depth assessment (treat as quick/moderate based on page length)
2. Skip Phase 2 vault check (the user has a specific source in mind)
3. In Phase 3, fetch just that URL with WebFetch
4. Proceed normally through Phase 4 and 5

### Topic Already Well-Covered

If Phase 2 reveals the vault already has strong coverage:
1. Present what's known (list relevant notes with brief descriptions)
2. Ask: "The vault already covers this well. Want me to research anyway for updates, or focus on a specific gap?"
3. If the user says proceed, focus on recent developments or specific angles not yet covered

### Research Yields Nothing Useful

If web searches return nothing relevant:
1. Report honestly: "I couldn't find substantive information on [topic]."
2. Suggest alternative angles or related topics that might work better
3. Do not create an inbox file with no real content -- an empty source capture just creates pipeline busywork

### Multiple Distinct Sub-Topics

If the topic naturally splits into clearly separate domains (e.g., "learn about both VAE clustering AND Raven Pro export format"), handle them in a single source capture but use clear section breaks. The /reduce pass will extract separate atomic notes from each section.
