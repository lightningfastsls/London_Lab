# Context: How We Got Here — The arscontexta Retrieval Investigation

> **Purpose:** Background context for Web Claude. This explains what arscontexta is, how the investigation unfolded, what we found, and what the current live state looks like. Read this BEFORE the problem handoff.

---

## What Is arscontexta?

arscontexta is a knowledge management system built for an AI agent (Claude Code) working on a USV (ultrasonic vocalization) research project. The core idea: build a knowledge vault of atomic notes that the agent can search and use automatically, so it doesn't start every session cold.

**The vault today:**
- 526 atomic markdown notes, each containing one claim (e.g., "512-point FFT at 300 kHz gives 1.7 ms temporal resolution with 586 Hz frequency bins")
- 26 topic maps (Maps of Content) — curated index files that organize notes by domain area
- Dense wiki-links between notes (average 8.6 links per note, 0 orphan notes)
- YAML frontmatter on every note: `description`, `type` (finding/decision/method/open-question/tool), `confidence` (proven/likely/speculative), `topics` (which topic maps it belongs to)
- Indexed by qmd, a local search engine offering BM25 keyword search and vector embedding search

**The domain:** Mouse ultrasonic vocalizations recorded at 300 kHz. The vault covers signal processing (STFT parameters, detection algorithms), machine learning (CNN classification, VQ-VAE representation learning), tool interoperability (DeepSqueak, Raven, MATLAB bridges), and methodology (knowledge graph operations, retrieval patterns).

---

## What Is the Activation Loop?

The vault is supposed to work as a feedback loop:

```
Session starts
  → Hook reads user's active goal threads from ops/goals.md
  → Hook queries qmd (search engine) with each goal thread
  → Top results written to ops/session-relevance.md
  → Agent reads that file → starts session with domain context
  → Agent works, discovers new knowledge
  → Agent writes discoveries back to vault
Session ends
```

**The problem we investigated:** The "query qmd → surface results" step was broken. `ops/session-relevance.md` always showed "No strong matches above relevance threshold" for every goal thread.

---

## How the Investigation Unfolded

### Phase 1: We assumed it was a search quality problem

The initial hypothesis (from a prior health audit) was that BM25 search suffered from "query term dilution" — the goal thread descriptions are long prose with status words, and BM25 scoring gets diluted when queries have too many low-signal terms.

We designed a benchmark: 10 queries across domain, operational, cross-domain, and goal-thread categories, tested in both BM25 (keyword) and vec (embedding) modes.

### Phase 2: We discovered it was an execution failure

While running the benchmark, we discovered the real primary cause: **the qmd command literally could not execute from the hook.**

The hook is a PowerShell script (`.ps1`) running on Windows. It calls `& qmd search ...`. The `qmd` command resolves to an npm-generated shim that tries to invoke `/bin/sh.exe` — a Unix path that doesn't exist on Windows. Every qmd call failed with `CommandNotFoundException`. The `catch {}` blocks silently swallowed the error.

**Knowledge activation had never worked.** Not once, across ~347 sessions. The "No strong matches" output looked like the system ran and found nothing, but it was a total execution failure disguised as an empty result.

### Phase 3: We fixed the execution failure and benchmarked the actual search

Fix: call `node qmd.js` directly instead of the broken npm shim. This was implemented and deployed.

Then we ran the full benchmark against qmd to understand the actual search quality. Results:

**BM25 (keyword search) — what we found:**

| Query Type | Precision@1 | Key Finding |
|------------|-------------|-------------|
| Domain keywords (3-5 terms) | 90% | Works well when queries are tight |
| Cross-domain keywords | 100% | Specific domain terms work great |
| Goal-thread full prose | 0% | Status words ("DONE", "IN PROGRESS") kill BM25 |
| Goal-thread condensed (4 keywords) | Hit | Stripping noise transforms 0→5 results |
| Self-retrieval (description as query) | 60% | 40% of notes invisible to keyword search |
| Operational/meta queries | 33% | Vault doesn't store self-knowledge |

**Vector (embedding) search — what we found:**

| Problem | Detail |
|---------|--------|
| MCP server crashes | "Object is disposed" under load, disconnected twice during benchmark |
| No CLI fallback | Requires 1.28GB model download + llama.cpp build from source |
| Hyphen-as-negation | "VQ-VAE" parsed as "VQ minus VAE", silently fails |
| Lower precision than BM25 | Returns semantic neighbors, not exact matches |
| Scores also uniform | ~0.93 for everything, just like BM25 |

**Score uniformity:** Every query — BM25 or vec — returns ~0.93 as the top score. A perfect match scores 0.93. A completely wrong result also scores 0.93. Threshold-based filtering is useless.

### Phase 4: We fixed the query construction

Secondary fixes applied to session-orient.ps1:
- Strip status words (DONE, IN PROGRESS, Phase N) from queries
- Preserve periods in version numbers ("Phase 5.3" no longer becomes "Phase 53")
- Handle stderr pollution in JSON parsing

### Phase 5: The fix works — and reveals the deeper problem

After all fixes, session-orient now actually executes and returns results. Here is the LIVE output from `ops/session-relevance.md` as of today (2026-03-22):

```markdown
# Session Relevance Brief
<!-- Generated: 2026-03-22 12:17 -->
<!-- Threads: 2 active from goals.md -->

## DeepSqueak Classification Bridge
- **classification** (score: 0.69, vector)
- **DeepSqueak uses monolithic Faster R-CNN detection whereas our two-stage pipeline allows independent tuning of recall and precision** (score: 0.69, vector)
- **VQ-VAE investigation of language-like sequential structure in USVs is a separate deeper question from courtship degradation** (score: 0.67, vector)
- **VocalMat represents supervised classification with predefined categories achieving 86 percent accuracy on 11 USV types** (score: 0.67, vector)

## Phase 5.3
- **Pasteur USV cloud platform enables online testing of detection methods without local infrastructure** (score: 0.6, vector)
- **experimental-methods** (score: 0.59, vector)
- **normalization statistics must be computed on training set only to prevent data leakage** (score: 0.59, vector)
- **representation-learning** (score: 0.59, vector)
```

**This is the problem in action.** 8 notes surfaced. Let's evaluate each:

For "DeepSqueak Classification Bridge" (goal: import Raven detection tables into DeepSqueak for syllable classification):
1. "classification" topic map — too vague, it's the whole domain not the specific workflow
2. "DeepSqueak uses monolithic Faster R-CNN..." — about DeepSqueak's detection architecture, not the classification bridge import workflow
3. "VQ-VAE investigation of language-like sequential structure..." — **completely unrelated** (VQ-VAE research, not DeepSqueak bridging)
4. "VocalMat represents supervised classification..." — about a different tool entirely

For "Phase 5.3" (goal: arscontexta health validation checkpoint — this is about the vault itself, not the domain):
1. "Pasteur USV cloud platform..." — **completely unrelated** (a cloud platform for USV analysis)
2. "experimental-methods" topic map — wrong domain entirely
3. "normalization statistics must be computed on training set..." — **completely unrelated** (ML training technique)
4. "representation-learning" topic map — wrong domain entirely

**Relevance score: ~1 out of 8 is even loosely on-topic.** The rest is noise that wastes ~1500-3000 tokens of the agent's context window.

Notes that SHOULD have been surfaced for "DeepSqueak Classification Bridge":
- "timestamp proximity matching with configurable tolerance bridges detection systems..." — the actual bridge mechanism
- "Raven selection table format is the standard interchange format between bioacoustic analysis tools" — the format being imported
- "DeepSqueak built-in classification enables pre-VQ-VAE repertoire comparison between wild and lab populations" — the classification workflow
- "Reading DeepSqueak mat outputs in Python uses scipy loadmat..." — the technical bridge step

These notes exist in the vault and ARE findable by BM25 with the right keywords (we proved this in the benchmark). But the hook's vec search returns semantic neighbors instead.

---

## The Goal Thread Format (What Queries Are Built From)

The hook parses `ops/goals.md` which looks like this:

```markdown
## Active Threads
- **DeepSqueak Classification Bridge** -- Phase 2 (Raven export) DONE, Phase 3 (MATLAB import+clustering) IN PROGRESS. Resume: open MATLAB -> DeepSqueak -> Import from Raven (5 files in raven_tables/). See PROJECTS.md Section 6 for full steps.
- Phase 5.3 -- Validation checkpoint COMPLETE (2026-03-21). Scored 19/25 (up from 18/25). Maintenance overhead improved (score 3→4). /rethink threshold lowered to 7.
```

The hook extracts:
- **Title:** "DeepSqueak Classification Bridge"
- **Description:** "Phase 2 (Raven export) DONE, Phase 3 (MATLAB import+clustering) IN PROGRESS..."

It uses the title for BM25 keyword search and title+description for vector search.

---

## What a Vault Note Looks Like (Concrete Examples)

### Example 1: Atomic note (a specific claim)

```markdown
---
description: "Energy detector uses -60 dB threshold to catch even faint USVs, accepting high false positive rate for downstream CNN filtering"
type: decision
confidence: proven
topics:
  - "[[detection]]"
---

# Energy threshold at negative 60 dB is deliberately low to maximize recall in the first stage

The energy detector threshold is set to -60.0 dB, deliberately low to maximize recall in the first detection stage. This permissive threshold means many non-USV candidates will be generated, but since [[two-stage detection uses permissive energy detector followed by CNN precision filter]], the CNN classifier will reject most false positives.

---

Source:
- DECISIONS.md (ADR-003)

Relevant Notes:
- [[two-stage detection uses permissive energy detector followed by CNN precision filter]]
- [[peak energy mode detects narrow-band USVs better than mean energy across the frequency band]]
```

**Key characteristics:** Title IS the claim. Description adds info beyond the title. Wiki-links connect to related notes. Topics field points to the topic map it belongs to. ~200-400 words.

### Example 2: Topic map (curated index of notes)

```markdown
---
description: DeepSqueak interop, Python USV classification tools landscape, and Raven interchange for bridging detection pipelines
type: moc
parent_map: classification
topics: "[[classification]]"
---

# classification-tools

Tools and interoperability for USV classification. DeepSqueak remains the dominant MATLAB tool despite its GUI-only design.

## DeepSqueak
- [[DeepSqueak requires MATLAB 2020a plus seven toolboxes and has no Python port]] -- MATLAB lock-in
- [[DeepSqueak v3.1 added VAE-based contour-invariant clustering as upgrade over k-means]] -- VAE clustering
- [[DeepSqueak Excel export provides 16 per-call metrics including principal frequency bandwidth slope and tonality]] -- richest output
- [[Reading DeepSqueak mat outputs in Python uses scipy loadmat for v5 format or h5py for v7.3 HDF5 format]] -- Python interop

## Raven Interchange
- [[Raven selection table format is the standard interchange format between bioacoustic analysis tools]] -- standard format
- [[timestamp proximity matching with configurable tolerance bridges detection systems]] -- bridge mechanism
```

**Key characteristics:** Curated by a human/agent, with context phrases after each link. Organized by sub-topic. This is the existing structure that could be used for retrieval instead of (or alongside) text search.

### Example 3: Self-retrieval failure note

A note whose description CANNOT find itself via BM25:

```yaml
description: "different model architectures exhibit distinct unconstrained behavioral patterns suggesting architecture-specific biases in code generation"
type: finding
```

The description uses abstract language ("unconstrained behavioral patterns", "architecture-specific biases") that doesn't share BM25 tokens with the note body, which likely discusses specific models by name. A vec search would understand the semantic overlap, but BM25 can't bridge the vocabulary gap.

---

## The Topic Map Hierarchy

```
index.md (root)
├── signal-processing (DSP, STFT, 300 kHz, detection parameters)
├── detection (energy detector, two-stage pipeline, thresholds)
├── detection-landscape (competing tools: VocalMat, MUPET, A-MUD, USVSEG)
├── classification (CNN pipeline, labeling, training, performance)
│   ├── classification-tools (DeepSqueak, Python tools, Raven)
│   └── classification-methodology (clustering, repertoire comparison)
├── representation-learning (VQ-VAE, transformer, codebook, embedding)
├── model-adaptation (LoRA, PEFT, hypernetworks)
├── training-methodology (data prep, evaluation, augmentation)
├── experimental-methods (dataset prep, splits, metrics)
├── agent-memory (knowledge activation, retrieval, RAG patterns)
├── graph-structure (wiki-links, traversal, emergent knowledge)
├── methodology (how the system processes knowledge)
└── ... (26 maps total)
```

Each topic map contains 10-50 notes with contextual descriptions. This hierarchy IS the domain model — it encodes how knowledge areas relate to each other.

---

## Summary of What's Known

| Layer | Status | Problem |
|-------|--------|---------|
| Vault content | Healthy | 526 notes, well-linked, schema-compliant |
| Vault structure | Healthy | 26 topic maps, 0 orphans, curated hierarchy |
| Search execution | Fixed | Was totally broken (npm shim), now calls node directly |
| Query construction | Partially fixed | Status words stripped, periods preserved, but still crude |
| BM25 retrieval | Works for keywords | 90% P@1 when queries are 3-5 domain terms |
| BM25 self-retrieval | 60% | 40% of notes invisible due to vocabulary mismatch |
| Vec retrieval | Unreliable + imprecise | 4 independent bugs, returns semantic neighbors not targets |
| Score calibration | Broken | Uniform 0.93/0.69, thresholds useless |
| End-to-end result quality | **Poor** | 1/8 notes relevant in live test. Noise dominates. |

**The investigation proved the plumbing works. The problem is now: how do you select the RIGHT 2-4 notes from a 526-note vault given only a goal thread title and description?**
