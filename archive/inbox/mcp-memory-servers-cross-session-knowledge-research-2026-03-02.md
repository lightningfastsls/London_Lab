---
description: "Survey of MCP memory servers, auto-memory patterns, knowledge graph vs KV approaches, forgetting strategies, and evaluation criteria for cross-session agent knowledge persistence"
source_type: article
url: "multiple -- see source log"
author: "multiple sources"
date_accessed: "2026-03-02"
status: unprocessed
research_tool: "web-search"
research_query: "MCP memory servers for AI coding agents: persistent cross-session knowledge patterns"
research_depth: "deep"
---

# MCP Memory Servers for AI Coding Agents: Cross-Session Knowledge Persistence

The explosion of MCP (Model Context Protocol) memory servers in 2025-2026 reflects a fundamental challenge: AI coding agents start every session with amnesia. At least 96 knowledge-and-memory MCP servers are listed on Glama alone, ranging from Anthropic's reference implementation (9 tools, JSONL storage) to production systems like Mem0's OpenMemory (Postgres + Qdrant, launched May 2025) handling agent memory across eight IDE platforms. The landscape has fragmented into three architectural camps -- unstructured markdown (Claude Code auto-memory), structured knowledge graphs (Zep/Graphiti, Memento, mcp-neuralmemory), and hybrid vector+graph systems (Memorix, mcp-memory-service) -- each with distinct trade-offs around what to remember, what to forget, and how to scope knowledge across projects and sessions.

---

## 1. Claude Code Auto-Memory: The Baseline

Claude Code ships two complementary memory systems. CLAUDE.md files are human-written instructions loaded into every session. Auto-memory (MEMORY.md) is Claude's own notebook -- it decides what to save based on whether information would be useful in future conversations.

Auto-memory stores to `~/.claude/projects/<project>/memory/MEMORY.md` with optional topic files. Only the first 200 lines of MEMORY.md load at session start (hard-coded: `var U_ = 'MEMORY.md', pZ = 200`). Topic files like `debugging.md` or `patterns.md` load on-demand when Claude reads them. The system is machine-local and scoped per git repository (all worktrees share one memory directory).

What auto-memory captures: build commands, debugging insights, architecture notes, code style preferences, workflow habits. What it misses: the diagnostic path -- which approaches were tried, why some failed, specific conditions that made one work. That detail disappears when the session ends. As Peterson (2026) argues, auto-memory is configuration, not learning -- it trains Claude to operate in a specific workspace but does not build understanding.

After 10 sessions, MEMORY.md typically contains 30% redundant entries. There is no automatic consolidation, decay, or pruning. The user must manually edit or Claude must be told to reorganize. Subagents can maintain their own auto-memory since early 2026.

Key limitation: auto-memory lacks team sharing. Custom agents offer memory scoping (project, local, user) but the main Claude Code session remains isolated.

---

## 2. Anthropic's Official Memory MCP Server (Reference Implementation)

Anthropic's `@modelcontextprotocol/server-memory` provides the canonical MCP memory pattern. It stores a knowledge graph as JSONL (default `memory.jsonl`) with three primitives:

- **Entities**: nodes with name, type, and observations (atomic facts as strings)
- **Relations**: directed edges in active voice between entities
- **Observations**: discrete facts attached to entities, independently addable/removable

Nine tools: create_entities, create_relations, add_observations, delete_entities, delete_observations, delete_relations, read_graph, search_nodes, open_nodes.

No built-in decay, expiration, or automatic consolidation. Memory persists until explicitly deleted. No scoping mechanism beyond separate JSONL files. This simplicity makes it a reference point but not a production solution -- the AIMultiple benchmark found MCP systems "couldn't separate the contexts of two projects" when using single-file storage, causing cross-project contamination.

---

## 3. Major MCP Memory Server Implementations (Feb 2026)

### 3.1 mcp-memory-service (doobidoo) -- 1.4K stars, v10.20.2

The most feature-rich open-source option. SQLite-vec locally with optional Cloudflare Workers + Vectorize for cloud sync. Knowledge graph with typed edges (causes, fixes, contradicts). Hybrid BM25 + vector search. Local ONNX embeddings (MiniLM-L6-v2) with adaptive batching (4-16x GPU speedup). 5ms retrieval latency, graph traversal in 5-25ms (30x faster than sequential lookup). Autonomous consolidation compresses old memories via decay-based identification of stale information. Dream-inspired consolidation with daily/weekly/monthly scheduling. REST API with 15 endpoints supporting LangGraph, CrewAI, AutoGen.

### 3.2 Memorix (AVIDS2) -- 146 stars, v0.9.29

Cross-agent memory bridge supporting 8 IDE agents (Cursor, Claude Code, Windsurf, Codex, Copilot, Kiro, OpenCode, Gemini CLI). Orama search engine (BM25 + optional vector). Progressive 3-tier disclosure (search, timeline, detail) saving approximately 10x tokens. 25 MCP tools across 5 categories. 9 observation types (gotchas, decisions, discoveries). Auto-scoped by git remote per project. All agents read/write `~/.memorix/data/`. TypeScript (87%), Apache 2.0.

### 3.3 Agent-Recall (mnardit) -- 6 stars, 25 commits

Battle-tested in production with 30+ concurrent AI agents at a digital agency. SQLite at `~/.agent-recall/frames.db`. Bitemporal knowledge graph with entities, slots (scoped key-value pairs), observations, and relations. Namespace-based scope hierarchy for data isolation. LLM-generated briefings at session start summarize hundreds of facts into structured sections (Key People, Current Tasks, Recent Decisions). Adaptive cache with stale-detection triggering automatic regeneration. SessionStart hook integration for Claude Code. Minimal dependencies (pyyaml, click only).

### 3.4 OpenMemory / Mem0 -- launched May 2025, 200+ upvotes on Product Hunt in 48 hours

Local-first MCP server by Mem0. Dockerized stack: FastAPI + Postgres + Qdrant. Memory types: user_preferences, implementation, troubleshooting, component_context, project_overview, incident_rca. Access control table with allow/deny rules between apps and memories. Per-project memory scoping. APIs: add_memories, search_memory, list_memories, delete_all_memories. Supports Cursor, VS Code, JetBrains, Claude, OpenAI, LangGraph, LlamaIndex.

### 3.5 Memento MCP (gannonh) -- 404 stars

Neo4j 5.13+ backend combining graph and vector storage in one database. Entities with vector embeddings and complete version history. Relations with strength indicators (0.0-1.0), confidence levels, temporal awareness, and configurable confidence decay (default 30-day half-life). Point-in-time graph retrieval tracks how knowledge evolved. Hybrid semantic + keyword search.

### 3.6 mcp-neuralmemory (Hexecu) -- 14 stars, v0.1.10

Neo4j graph database with Gemini API for LLM integration. Tracks goals/status, constraints/rules, strategies/outcomes (marked success/failure), user preferences, code-to-goal relationships. K-hop graph traversal for context retrieval. Both implicit pattern inference and explicit outcome tracking.

### 3.7 Memory Journal MCP -- 11 stars

SQLite with FTS5 and vector embeddings via @xenova/transformers. Triple search: full-text (FTS5), semantic/vector, and date-range. Deep GitHub integration: 15 tools for Issues, PRs, Kanban, Milestones, Actions, Insights. 39 total MCP tools, 15 workflow prompts, 21 resources. Links specs to implementations to tests to PRs in a knowledge graph.

### 3.8 Ember MCP -- 26 stars, v3.0.0

Most sophisticated forgetting mechanism. HESTIA scoring: `score = similarity * shadow_penalty * (0.6 + 0.3 * importance + 0.1 * recency)`. Shadow-Decay: when memories are contradicted, shadow_load increases and penalty applies `(1.0 - shadow_load)^2.0` -- quadratic fading without deletion. Voronoi-based drift detection flags when topic clusters shift (e.g., Redux to Zustand migration). Drift detection flags only; no auto-shadow. SQLite + all-MiniLM-L6-v2 (384 dims).

### 3.9 claude-mem (thedotmack) -- 32.1K stars (part of Claude Code ecosystem)

Plugin architecture with 5 lifecycle hooks (SessionStart, UserPromptSubmit, PostToolUse, Stop, SessionEnd). SQLite + Chroma for hybrid search. Smart compression via Claude agent-sdk generating semantic summaries. Progressive disclosure: compact indices (~50-100 tokens) before full details (~500-1000 tokens/result), approximately 10x token savings. Worker service on port 37777 with web UI.

### 3.10 Hierarchical Memory MCP (Anthropic variant)

Three-tier temporal hierarchy: working memory (30 min TTL), short-term memory (7 day TTL), long-term memory (1 year TTL). Forgetting curve with importance scoring. Automatic consolidation promotes important short-term memories to long-term.

---

## 4. Architectural Paradigms: Knowledge Graph vs Key-Value vs Vector

### 4.1 Knowledge Graphs

Entities + relations + observations. Best for multi-hop reasoning, relationship tracking, temporal coherence. Zep/Graphiti (20K GitHub stars, MCP Server 1.0) is the leading implementation -- a temporally-aware knowledge graph engine with formal graph G = (N, E, phi) supporting bi-temporal modeling. Graphiti improved efficiency through entropy-gated fuzzy matching for entity deduplication, using deterministic IR front-ends before falling back to LLMs. On DMR benchmark: 94.8% accuracy (vs MemGPT 93.4%). On LongMemEval: 18.5% accuracy improvement, 90% latency reduction.

Graph-based approaches outperform traditional methods in multi-hop reasoning (traversing explicit relational links), temporal coherence (preventing logical hallucinations), personalization (capturing interaction patterns), and hallucination reduction (grounding in structured, verifiable content). However, graph traversal scales poorly with large memories, extraction quality depends on NER accuracy, and real-time graph updates are resource-intensive.

### 4.2 Key-Value / Flat Storage

JSONL, markdown files, SQLite rows. Best for fast deterministic retrieval of user preferences, session state, configuration. Auto-memory falls here. Simple, human-readable, low overhead. Cannot reason across relationships. No inherent temporal or causal structure.

### 4.3 Vector Stores

Embedding-based semantic search. Best for finding conceptually similar memories when exact keywords are unknown. Qdrant, Chroma, LanceDB common backends. Excellent for unstructured text but cannot perform logical joins or structured queries. Most production systems combine vector search with at least one other paradigm.

### 4.4 Hybrid Approaches

The convergence point. mcp-memory-service combines BM25 + vector + typed graph edges. SimpleMem uses LanceDB with multi-view indexing (dense embeddings, BM25 sparse, SQL metadata). Memorix combines Orama BM25 with optional vectors and a knowledge graph layer. The research consensus from the graph-based memory taxonomy survey (Feb 2026) is that no single paradigm suits all scenarios -- success depends on aligning graph structure with specific memory types and application requirements.

---

## 5. What to Remember vs What to Forget

### 5.1 The Forgetting Imperative

Research shows agents using "add-all" memory strategies exhibit sustained performance decline after initial phases. The problem is not insufficient memory but unmanaged memory. Forgetting is essential: it prevents error propagation through decay, removes unsuccessful patterns through relevance-based pruning, and maintains retrieval precision.

Three forgetting strategies dominate:
- **Temporal decay**: fixed windows, LRU eviction, or exponential decay (mcp-ai-memory uses active/dormant/archived/expired states). Predictable but may discard still-useful old information.
- **Importance-based pruning**: score memories by relevance/success and prune low-scoring ones. Ember MCP's HESTIA scoring is the most sophisticated example.
- **Contradiction-based shadowing**: when new facts contradict old ones, increase shadow load on old facts rather than deleting them (Ember's quadratic shadow penalty). Preserves audit trail.

### 5.2 Scoping Strategies

Memory should be scoped by project (git remote, directory), by agent identity (different agents may need different knowledge), and by task (user preference vs troubleshooting vs architecture). OpenMemory implements per-project scoping with access control tables. Agent-Recall uses namespace-based scope hierarchy. Memorix auto-scopes by git remote. The AIMultiple benchmark found cross-project contamination is a real problem in single-scope systems.

### 5.3 Consolidation Patterns

- **Dream-inspired**: mcp-memory-service runs daily/weekly/monthly consolidation cycles
- **Threshold-based**: memory-mcp consolidates when memories exceed 80 items or after every 10 extractions
- **Hierarchical promotion**: Anthropic's hierarchical variant promotes important short-term memories to long-term based on importance scoring
- **Semantic compression**: SimpleMem's three-stage pipeline achieves 30x token reduction through structured compression, online synthesis, and intent-aware retrieval

---

## 6. Auto-Memory vs MCP-Based Approaches

| Dimension | Auto-Memory (CLAUDE.md) | MCP Memory Servers |
|-----------|------------------------|-------------------|
| Who writes | Claude (auto) or user (CLAUDE.md) | Agent via MCP tools |
| Storage | Markdown files, 200-line limit | SQLite, Neo4j, Qdrant, JSONL |
| Search | File reading only | Semantic, BM25, graph traversal |
| Forgetting | Manual only | Decay, shadow, consolidation |
| Scoping | Per-repo, machine-local | Per-project, per-agent, cross-tool |
| Team sharing | Via CLAUDE.md in version control | Via shared database |
| Setup cost | Zero | Docker, databases, API keys |
| Token cost | ~200 lines loaded always | Progressive disclosure (10-20x savings) |
| Cross-IDE | No (Claude Code only) | Yes (Memorix supports 8 agents) |
| Diagnostic path | Lost at session end | Preserved if hooks capture it |

The complementary pattern emerging in practice: CLAUDE.md for stable rules and project architecture, auto-memory for learned patterns, MCP memory for structured cross-session knowledge, diagnostic paths, and cross-tool sharing.

---

## 7. Evaluation Criteria and Benchmarks

### 7.1 MemBench (ACL Findings 2025)

Evaluates memory from multiple aspects: effectiveness (accuracy), efficiency (processing times), capacity (scaling). Covers factual memory and reflective memory at different levels. Four metrics: accuracy, recall, capacity, temporal efficiency. Tests both participation and observation scenarios.

### 7.2 MemoryAgentBench (ICLR 2026)

Four core competencies: Accurate Retrieval (needle-in-haystack extraction), Test-Time Learning (in-context adaptation), Long-Range Understanding (global summarization), Conflict Resolution (updating prior facts with new evidence). Uses incremental chunking and multi-competency sampling to simulate realistic multi-turn, multi-session interactions.

### 7.3 LOCOMO Benchmark

Used by Mem0 for evaluation. Mem0 achieves 26% relative improvement in LLM-as-a-Judge metric over OpenAI memory, 91% lower p95 latency, 90% token cost savings. Four question categories: single-hop, temporal, multi-hop, open-domain. Graph-enhanced Mem0 variant scores approximately 2% higher than base configuration.

### 7.4 DMR and LongMemEval

Used by Zep/Graphiti. DMR: 94.8% accuracy (vs MemGPT 93.4%). LongMemEval: 18.5% accuracy improvement with 90% latency reduction. Better reflects enterprise temporal reasoning requirements.

### 7.5 AIMultiple MCP Memory Benchmark

Tested 4 MCP servers with LangChain ReAct agent + GPT-4. Measured operation accuracy (percentage of turns with correct memory operations). Tested read-on-resume, read-before-write behaviors. Critical finding: cross-project context separation failed. Single-project implementations performed adequately.

### 7.6 Practical Evaluation Criteria

From the research and practical implementations, these criteria emerge for evaluating cross-session memory tools:
1. **Retrieval precision**: does it return what's relevant, not everything?
2. **Token efficiency**: progressive disclosure vs loading all memories
3. **Scope isolation**: project separation, agent separation
4. **Forgetting quality**: does it manage stale/contradicted knowledge?
5. **Setup friction**: zero-config vs Docker + databases
6. **Cross-tool portability**: does memory survive IDE switches?
7. **Auditability**: can humans inspect and edit what's stored?
8. **Consolidation strategy**: how does memory scale over months?

---

## 8. Research Frontiers in Agent Memory (2025-2026)

### 8.1 Memory as Operating System

MemOS (Shanghai Jiao Tong / Zhejiang, May 2025) treats memory as a managed system resource with three-layer architecture (API, scheduling/management, storage/infrastructure). MemCube units encapsulate content + metadata (provenance, versioning) that can be composed, migrated, and fused. On LOCOMO: 159% improvement in temporal reasoning over OpenAI memory, 38.97% accuracy gain, 60.95% token reduction. v2.0 (Dec 2025) added multi-modal memory, tool memory for planning, cross-project KB sharing.

### 8.2 Unified Agentic Memory (AgeMem, Jan 2026)

Exposes memory operations (store, retrieve, update, summarize, discard) as tool-based actions the LLM agent autonomously decides when to use. Three-stage progressive reinforcement learning with step-wise GRPO for sparse rewards. On Qwen2.5-7B: 41.96 average score vs Mem0's 37.14 (13% improvement). Key insight: treating LTM and STM as separate components with heuristic controllers limits adaptability.

### 8.3 Efficient Lifelong Memory (SimpleMem, Jan 2026)

Three-stage pipeline: semantic structured compression, online semantic synthesis (instant in-session deduplication), intent-aware retrieval planning. 26.4% average F1 improvement, 30x token reduction. 64% performance boost over claude-mem on LOCOMO. Multi-view LanceDB indexing with text-embedding-3-small (1536 dims) + BM25 + SQL metadata.

### 8.4 Graph-Based Memory Taxonomy (Feb 2026)

Four-dimensional classification: temporal scope (short/long), functional role (knowledge/experience), structure (non-structural/structural), cognitive type (semantic/procedural/associative/working/episodic/sentiment). Five graph structures compared: knowledge graphs (best for factual relations, struggles with n-ary), hierarchical graphs (compressed experiences), temporal graphs (dynamic events), hypergraphs (complex n-ary interactions), hybrid architectures. Key finding: graph-based approaches reduce hallucination by grounding outputs in structured, verifiable memory content.

### 8.5 Memory Lifecycle Model

The "Memory in the Age of AI Agents" survey (Dec 2025) establishes agent memory as distinct from LLM memory, RAG, and context engineering. Three-dimensional taxonomy: forms (token-level, parametric, latent), functions (factual, experiential, working), dynamics (formation, evolution, retrieval). Five research frontiers: memory automation, RL integration, multimodal memory, multi-agent memory coordination, trustworthiness.

---

## Source Log

| # | URL | Status | Relevance | Key Finding |
|---|-----|--------|-----------|-------------|
| 1 | https://github.com/doobidoo/mcp-memory-service | fetched | high | Most feature-rich MCP memory, hybrid search, autonomous consolidation |
| 2 | https://github.com/moonx010/hive-memory | fetched | medium | Cross-project memory but only 4 stars, 1 commit |
| 3 | https://github.com/mnardit/agent-recall | fetched | high | Production-tested (30+ agents), bitemporal KG, session briefings |
| 4 | https://github.com/AVIDS2/memorix | fetched | high | 8-IDE cross-agent memory bridge, progressive disclosure |
| 5 | https://github.com/Hexecu/mcp-neuralmemory | fetched | medium | Neo4j + Gemini, k-hop traversal, outcome tracking |
| 6 | https://github.com/gannonh/memento-mcp | fetched | high | Neo4j with confidence decay (30-day half-life), version history |
| 7 | https://github.com/neverinfamous/memory-journal-mcp | fetched | medium | Triple search + deep GitHub integration, 39 tools |
| 8 | https://github.com/modelcontextprotocol/servers/tree/main/src/memory | fetched | high | Official reference: 9 tools, JSONL, entities/relations/observations |
| 9 | https://code.claude.com/docs/en/memory | fetched | high | Full auto-memory docs: 200-line limit, scoping, topic files |
| 10 | https://github.com/thedotmack/claude-mem | fetched | high | 5 lifecycle hooks, SQLite + Chroma, progressive disclosure |
| 11 | https://mem0.ai/openmemory | fetched (CSS-heavy) | high | Postgres + Qdrant, per-project scoping, access control |
| 12 | https://github.com/Arkya-AI/ember-mcp | fetched | high | HESTIA scoring, shadow-decay, Voronoi drift detection |
| 13 | https://arxiv.org/abs/2504.19413 | fetched | high | Mem0: 26% over OpenAI, 91% lower latency, 90% token savings |
| 14 | https://arxiv.org/html/2602.05665 | fetched | high | Graph memory taxonomy: 4 dimensions, 5 structures compared |
| 15 | https://arxiv.org/abs/2501.13956 | fetched | high | Zep/Graphiti: temporal KG, DMR 94.8%, LongMemEval +18.5% |
| 16 | https://arxiv.org/abs/2512.13564 | fetched | high | Agent memory survey: 3D taxonomy, 5 frontiers |
| 17 | https://dev.to/anajuliabit/mem0-vs-zep-vs-langmem-vs-memoclaw-ai-agent-memory-comparison-2026-1l1k | fetched | high | Comparison table of 4 major memory solutions |
| 18 | https://aimultiple.com/memory-mcp | fetched | high | MCP benchmark: cross-project separation failed |
| 19 | https://glama.ai/mcp/servers/categories/knowledge-and-memory | fetched | medium | 96 servers listed, landscape overview |
| 20 | https://giuseppegurgone.com/claude-memory | fetched | medium | Auto-memory internals: hard-coded 200-line limit |
| 21 | https://arxiv.org/abs/2601.02553 | search | high | SimpleMem: 64% over claude-mem, 30x token reduction |
| 22 | https://arxiv.org/abs/2601.01885 | search | high | AgeMem: unified LTM/STM, 13% over Mem0 |
| 23 | https://github.com/BAI-LAB/MemoryOS | search | medium | MemoryOS: EMNLP 2025 Oral, STM/MTM/LPM hierarchy |
| 24 | https://github.com/MemTensor/MemOS | search | high | MemOS: 159% temporal reasoning improvement |
| 25 | https://arxiv.org/abs/2506.21605 | search | medium | MemBench: ACL 2025, 4 evaluation metrics |
| 26 | https://arxiv.org/abs/2507.05257 | search | medium | MemoryAgentBench: ICLR 2026, 4 competencies |
| 27 | https://www.emergentmind.com/topics/memory-mechanisms-in-llm-based-agents | search | medium | Overview of memory mechanism research |
| 28 | https://medium.com/@joe.njenga/anthropic-just-added-auto-memory-to-claude-code-memory-md-i-tested-it-0ab8422754d2 | search | medium | Auto-memory testing and observations |
| 29 | https://medium.com/@brentwpeterson/automatic-memory-is-not-learning-4191f548df4c | search (403) | medium | Auto-memory is configuration, not learning |
| 30 | https://github.com/scanadi/mcp-ai-memory | search | medium | Exponential decay with active/dormant/archived/expired states |

## Research Context

- **Query**: MCP memory servers for AI coding agents: persistent cross-session knowledge patterns, auto-memory vs explicit memory, structured knowledge graphs vs unstructured key-value stores, evaluation criteria for cross-session tools
- **Depth**: deep (auto-detected)
- **Existing vault knowledge**: Strong coverage of session handoff patterns and agent memory architecture from arscontexta methodology (9 relevant notes on session continuity, cognitive offloading, operational vs knowledge memory separation). No coverage of specific MCP server implementations, auto-memory mechanics, or evaluation benchmarks.
- **Knowledge gap addressed**: Complete landscape of MCP memory server implementations (10+ servers analyzed), comparison framework for auto-memory vs MCP approaches, evaluation criteria from 5 benchmarks, forgetting/scoping strategies, research frontiers in agent memory (2025-2026)
