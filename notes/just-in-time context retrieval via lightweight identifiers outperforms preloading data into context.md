---
description: "Anthropic calls it progressive disclosure through iterative discovery — maintaining file paths and queries rather than ingesting full content avoids attention dilution"
type: pattern
confidence: likely
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-cognition]]"
  - "[[context-management]]"
---

# Just-in-time context retrieval via lightweight identifiers outperforms preloading data into context

Rather than loading all potentially relevant information into context upfront, maintaining lightweight identifiers (file paths, URLs, database queries) and retrieving content only when needed produces better results. Anthropic describes this as "progressive disclosure through iterative discovery." Claude Code exemplifies the approach — using Bash primitives for codebase navigation rather than full codebase ingestion.

The mechanism is straightforward: since [[structured coherent text creates more context interference than shuffled unstructured text across all tested models]], preloading related but not-yet-needed content creates exactly the kind of semantically relevant distraction that degrades attention. Every file loaded "just in case" competes with the information actually needed right now. Since [[LLMs exhibit a U-shaped attention curve where information in the middle of context degrades by more than 30 percent]], this preloaded content occupies middle positions where it is least accessible anyway.

JIT retrieval also synergizes with the [[practical guidance converges on 60-70 percent of maximum context window as effective usable capacity]] rule. By keeping context lean between retrieval operations, the effective capacity stays within the reliable range. The model uses context for reasoning about what it has retrieved rather than storing everything it might need.

CodeRabbit's production experience validates this principle specifically for code review: their key learning is that "more context isn't always better" — excessive or noisy inputs overwhelm models, creating false positives. Rather than loading entire files, they use AST-based symbol lookup and GraphRAG to curate only the relevant context, demonstrating that aggressive curation outperforms comprehensive loading even when the full context is available.

The trade-off is latency: JIT retrieval requires tool calls for each retrieval, adding round-trip time. But since [[context quality degradation follows a different curve than latency degradation with quality declining gradually while latency increases exponentially]], the quality-over-speed trade-off typically favors JIT: slightly more wall-clock time for substantially better reasoning quality.

---

Source: context-window-degradation-llms-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[structured coherent text creates more context interference than shuffled unstructured text across all tested models]] -- the mechanism that makes preloading harmful
- [[subagent architecture isolates context degradation by giving each task a focused window and compressing summaries upward]] -- complementary pattern
- [[practical guidance converges on 60-70 percent of maximum context window as effective usable capacity]] -- the constraint JIT helps maintain
- [[progressive disclosure in memory retrieval saves 10-20x tokens by loading compact indices before full content]] -- the same JIT principle applied cross-session: load entity names first, full content on demand
- [[MCP Tool Search reduces context pollution by 85-95 percent through lazy loading of tool descriptions via lightweight search index]] -- JIT applied to tool definitions: lightweight index replaces full descriptions (67K to 8.7K tokens)

Topics:
- [[agent-cognition]]
