# Handoff: The Vault Retrieval Quality Problem

> **For:** Web Claude research session
> **From:** Investigation 4.3 (qmd retrieval precision benchmark)
> **Goal:** Find practical solutions for making a 526-note knowledge vault useful to AI agents at session start, not just searchable

---

## The Problem In One Paragraph

We have a knowledge vault (526 atomic markdown notes, each one claim, organized by topic maps and wiki-links) that an AI agent is supposed to use automatically at session start. A hook queries the vault based on the user's active goal threads and surfaces relevant notes. But the search engine (qmd, using BM25 keyword search) either returns too many loosely-related results (wasting agent context window) or nothing at all (missing relevant knowledge). The scores are uniform (~0.93 for everything), so we can't filter by confidence. The result: the agent either gets noise or silence, and the vault's knowledge never reliably activates.

---

## System Architecture

```
User has goals in ops/goals.md:
  - "DeepSqueak Classification Bridge" -- Phase 2 done, Phase 3 in progress
  - "Phase 5.3" -- Validation checkpoint complete

Session starts → Hook extracts goal thread titles
              → Queries qmd with each title (BM25 keyword search)
              → Writes top results to ops/session-relevance.md
              → Agent reads that file → has domain context ready
```

**The vault:**
- 526 atomic notes in markdown, each with YAML frontmatter (`description`, `type`, `confidence`, `topics`)
- ~26 topic maps (Maps of Content) organizing notes hierarchically
- Dense wiki-links between notes (avg ~8.6 links/note)
- Indexed by qmd: BM25 full-text search + vector embeddings (but vec is unreliable, see below)

**The search engine (qmd):**
- Local tool, runs as MCP server or CLI
- 769 total indexed documents across 3 collections
- BM25 (keyword) search works reliably
- Vector (embedding) search crashes frequently, no stable CLI fallback
- Returns results with scores, but scores are meaningless (uniform 0.93)

---

## What We Measured (Benchmark Data)

### BM25 Precision by Query Type

| Query Type | Example | P@1 | Problem |
|------------|---------|-----|---------|
| Domain keywords (3-5 terms) | "USV detection energy threshold" | **90%** | Works well |
| Cross-domain keywords | "LMT behavioral event triggered analysis" | **100%** | Works well |
| Goal-thread full prose (15+ words) | "DeepSqueak Classification Bridge Phase 2 Raven export DONE Phase 3 MATLAB import clustering IN PROGRESS" | **0%** | BM25 dilution — status words drown domain terms |
| Condensed goal keywords | "DeepSqueak Raven classification bridge" | **93% top score, relevant** | Works when noise stripped |
| Operational/meta queries | "arscontexta health maintenance schema compliance" | **0%** | Vault stores domain knowledge, not self-knowledge |

### Self-Retrieval Test (can a note find itself?)

Used each note's `description:` field as a BM25 query. **60% success rate.**

Failures had abstract/conceptual descriptions:
- "LLM agents without tasks show model-specific determinism" → 0 results
- "Spectrograms contain identity information beyond call type" → 0 results
- "Graph traversal theory, emergent inter-note knowledge" → 0 results

Successes had concrete, distinctive vocabulary:
- "Greptile scored 82 percent in own benchmark but 45 percent from Augment Code" → 0.98
- "DSP foundation 300 kHz sample rate" → 0.96
- "512-point FFT at 300 kHz gives 1.7 ms temporal resolution" → 0.95

### Score Distribution

Every successful query returns ~0.93 as the top score regardless of actual relevance. A perfect match scores 0.93. A topically-adjacent-but-wrong result also scores 0.93. Score-based thresholds cannot discriminate.

### Vector Search — Full Problem Breakdown

Vector (embedding) search would theoretically solve BM25's biggest weakness: it matches by meaning, not tokens, so "spectrograms contain identity information" could find notes about "speaker recognition from vocal features" even without shared keywords. This is exactly what the 40% self-retrieval failures need. But vec search has four independent reliability problems:

**Problem V1: MCP server crashes under load ("Object is disposed")**
The qmd MCP server (how Claude Code talks to qmd) disconnects mid-session when vec queries run. This happened twice during our benchmark. The error is "Object is disposed" — likely a resource cleanup bug in the embedding model lifecycle. Once it crashes, all subsequent vec queries silently fail for the rest of the session. There's no auto-reconnect.

**Problem V2: No CLI fallback**
Running `qmd vsearch` from the command line triggers a 1.28GB model download (a quantized LLM for query expansion) and attempts to compile llama.cpp from source using CMake. The build fails because Vulkan SDK components (`glslc`) are missing. Even if the build succeeded, it would run on CPU only (no GPU detected), making it prohibitively slow for a session-start hook. Vec search only works through the MCP server, which has Problem V1.

**Problem V3: Hyphen-as-negation parsing bug**
Hyphens inside domain terms are parsed as the exclusion operator. "VQ-VAE" becomes "VQ minus VAE." This silently breaks queries for many domain terms in our vault: VQ-VAE, self-supervised, cross-domain, pre-trained, semi-supervised, multi-scale. The error message ("Negation (-term) is not supported in vec/hyde queries") is caught by try/catch and swallowed silently.

**Problem V4: Lower precision than BM25 for specific concepts**
Even when vec search works, it returns semantic neighbors rather than exact intent matches. When we queried "CNN model architecture for classifying USVs," vec returned a note about CNN false positive behavior patterns (topically adjacent but wrong), while BM25 returned the actual CNN training note (correct). Vec search is better for discovery ("find things related to X") but worse for retrieval ("find the specific note about X"). For session-start activation, we need retrieval, not discovery.

**Vec search precision data (limited — only 2 usable results before crashes):**

| Query | BM25 Result | Vec Result | Winner |
|-------|------------|------------|--------|
| "USV detection energy threshold" | energy-threshold note (HIT) | energy-threshold note (HIT) | Tie |
| "CNN classification architecture" | CNN training bug note (PARTIAL) | CNN false positive note (MISS) | BM25 |
| "VQ-VAE codebook learning" | Gumbel-softmax collapse note (HIT) | CRASHED | BM25 (by default) |

**The open question:** If vec search were stable, would it actually help? The limited data suggests BM25 beats vec for targeted retrieval. Vec's advantage — bridging vocabulary gaps — would help the 40% of notes that fail BM25 self-retrieval. But we can't test this properly until the stability issues are resolved.

**What fixing vec search would require:**
- V1: qmd upstream fix (MCP server resource management)
- V2: Pre-build llama.cpp for CPU, or get Vulkan SDK installed, or find a way to use the MCP server's already-loaded model from CLI
- V3: qmd upstream fix (hyphen parsing), or strip hyphens from queries before sending (workaround: `$query -replace '-', ' '`)
- V4: Not fixable — this is inherent to embedding-based search. Mitigation: use vec as a complement to BM25, not a replacement

---

## The Core Tension

The hook currently returns the **top 3 keyword results + top 3 vector results per goal thread, capped at 4 per thread.** With 2 active threads, that's up to 8 notes surfaced.

**Problem 1: When it works, results are often noise.** For "DeepSqueak Classification Bridge", BM25 returns notes that contain "DeepSqueak" and "classification" and "bridge" — but these might be about DeepSqueak's MATLAB dependency, not about the current classification bridge workflow. The agent gets 4 notes, reads them, and 1-2 are actually relevant to the current work.

**Problem 2: When the query is too broad or too narrow, you get nothing useful.** Either 0 results (query too diluted) or 15 results all at 0.93 (query too broad, no way to rank).

**Problem 3: The agent's context window is finite.** Every irrelevant note surfaced at session start is ~200-500 tokens wasted. With 4 notes per thread and 2 threads, that's 1600-4000 tokens that could be noise. The agent then has less room for the actual task.

**The question:** How do we get from "here are 4-8 notes that match your keywords" to "here is the specific knowledge you need for what you're about to do"?

---

## Constraints

1. **The hook runs at session start** — before the user's first message, so we don't know the specific task yet, only the goal threads from `goals.md`
2. **BM25 is the only reliable search mode today** — vector search has 4 independent problems (V1-V4 above), but some are fixable. Solutions should work with BM25-only but could leverage vec if it becomes stable.
3. **Scores are meaningless** — can't threshold on confidence (uniform 0.93 for everything)
4. **The hook is PowerShell on Windows** — calling `node qmd.js` for search
5. **Latency matters** — the hook runs synchronously before the session starts, so slow queries delay the user
6. **The vault is well-structured** — topic maps, wiki-links, YAML frontmatter with `type`, `confidence`, `topics` fields. This structure is available for smarter retrieval.
7. **40% of notes are invisible to BM25** — notes with abstract/conceptual descriptions can't be found by keyword search. Vec search would theoretically bridge this gap but is currently broken.

---

## What We Haven't Tried (Solution Space)

### Direction 1: Smarter query construction

The current approach: take goal thread title, strip non-word chars, search. Could we do better?

- Extract only nouns/domain terms from the goal description (not just first sentence)
- Use the topic map names as query context (if the goal mentions "classification", search within the classification topic map)
- Maintain a mapping of goal thread → relevant topic maps (manually curated)

### Direction 2: Post-retrieval filtering/re-ranking

Accept that BM25 returns broad results, then filter:

- Check if the result's `type:` field matches the goal context (e.g., for an implementation goal, prefer `type: decision` or `type: finding` over `type: open-question`)
- Check term overlap between the result title and the query (lightweight re-ranking)
- Use the wiki-link graph: if result A links to result B, and both match the query, prefer A (it's more connected to the topic)

### Direction 3: Topic-map-based activation instead of search

Skip search entirely. Instead:
- Map each goal thread to its most relevant topic map (manually or by keyword matching against topic map names)
- Read the topic map file directly — it's a curated, human-organized index of the best notes for that domain
- Surface the top N notes from the topic map, perhaps weighted by `confidence` or incoming link count

This would be a fundamentally different approach: using the vault's own organizational structure instead of text search.

### Direction 4: Two-phase activation

Phase 1 (at session start, before user message): Surface 2-3 topic-map-level summaries, not individual notes. This gives the agent domain orientation without note-level noise.

Phase 2 (after user's first message): Run a targeted search using the actual task description. This is when you know what the user wants and can construct a precise query.

### Direction 5: Cached relevance profiles

Pre-compute a relevance profile for each goal thread:
- When a goal thread is created/updated in `goals.md`, run a batch search and store the top 10 results
- At session start, just read the cached profile — no live search needed
- Refresh the cache when the vault changes (new notes added) or the goal thread changes

### Direction 6: LLM-based re-ranking

After BM25 returns 10-15 candidates, pass them to a lightweight LLM (or even a simple heuristic model) to re-rank by relevance to the goal thread. This is the standard RAG pattern but adds latency and API cost.

### Direction 7: Fix or replace vector search

The 40% BM25 self-retrieval failure rate is fundamentally a vocabulary mismatch problem that keyword search cannot solve. Options:

- **Fix qmd vec search:** Report V1 (MCP crashes) and V3 (hyphen parsing) upstream. Pre-build the llama.cpp binary for CPU. This restores the intended dual-mode search. Risk: upstream fixes may take time; MCP stability is not in our control.
- **Replace qmd vec with a different embedding engine:** Use a Python-based embedding (e.g., sentence-transformers) that runs reliably. Pre-compute embeddings for all 526 notes, store in a simple FAISS/numpy index. Query at session start. This gives us full control but adds a Python dependency to a PowerShell hook.
- **Replace qmd entirely:** Use a search engine that handles both keyword and semantic search reliably in a single tool. Candidates: LanceDB, ChromaDB, or even SQLite FTS5 with a separate embedding step.
- **Augment BM25 with keyword expansion:** Instead of fixing vec, make BM25 work for abstract descriptions by adding keyword aliases. For each note, generate 3-5 alternative keyword phrases and store them in a searchable field. This is manual/semi-automated but doesn't require any new infrastructure.

### Direction 8: Hybrid structural + search approach

Combine directions 3 and 1:
1. Identify which topic maps are relevant to each goal thread (by keyword overlap with topic map names/descriptions)
2. Within those topic maps, BM25 search for the most relevant individual notes
3. This scopes the search to the right domain area, then uses BM25 for precision within that area

---

## What Would "Solved" Look Like?

A solved system would:
1. Surface 2-4 notes per goal thread that are **actually useful** for the session's likely work
2. **Not surface noise** — every surfaced note should be worth the context tokens it costs
3. **Work without vector search** — BM25 + vault structure only
4. **Add < 3 seconds latency** to session start
5. **Degrade gracefully** — if unsure, surface nothing rather than noise (the agent can always search manually)

---

## Existing Vault Structure Available for Solutions

Each note has YAML frontmatter:
```yaml
description: "one-line summary adding info beyond the title"
type: finding | decision | method | open-question | tool | bridge-note
confidence: proven | likely | speculative
topics:
  - "[[classification]]"
  - "[[signal-processing]]"
```

Topic maps are markdown files listing notes with context phrases:
```markdown
## Core Ideas
- [[512-point FFT at 300 kHz gives 1.7 ms temporal resolution]] -- core STFT parameter choice
- [[300 kHz sample rate provides comfortable Nyquist headroom]] -- canonical sample rate
```

Wiki-links create a dense graph (~8.6 links/note average, 0 orphans).

The `goals.md` format:
```markdown
## Active Threads
- **DeepSqueak Classification Bridge** -- Phase 2 (Raven export) DONE, Phase 3 (MATLAB import+clustering) IN PROGRESS
- Phase 5.3 -- Validation checkpoint COMPLETE (2026-03-21). Scored 19/25.
```

---

## Questions to Answer

1. Which direction (or combination) gives the best relevance-per-token ratio?
2. Is topic-map-based activation (Direction 3) sufficient on its own, or does it need search as a complement?
3. How do we handle goal threads that span multiple topic maps (e.g., "DeepSqueak Classification Bridge" touches classification-tools, training-methodology, and signal-processing)?
4. Should we pre-compute relevance or compute live? What are the latency/freshness tradeoffs?
5. Is there a standard technique from RAG/information retrieval literature for this exact problem (known-corpus, known-queries, no reliable scoring)?
6. Is fixing vec search worth the effort, or should we bypass it entirely with a different approach (keyword expansion, structural activation, or a replacement embedding engine)?
7. For the 40% of notes invisible to BM25 — is the right fix to improve their descriptions (make them BM25-friendly), add a semantic search layer, or accept that those notes are only findable through wiki-link traversal?
