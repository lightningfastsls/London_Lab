---
description: "Git diff pre-bundling plus Difftastic syntax-aware diffing eliminates per-file API calls and formatting noise — a common anti-pattern is repeated individual file reads"
type: method
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-governance]]"
  - "[[context-management]]"
---

# Pre-bundling diffs into single context reduces review tool calls from 100-plus to a few

A simple but high-impact optimization for AI code review is pre-bundling context via git diff instead of having the review agent make individual tool calls to read each changed file. This reduces tool calls from 100+ to a handful, with proportional savings in latency, token cost, and context window consumption.

The optimization has two layers. First, git diff provides all changed content in a single operation rather than requiring file-by-file reads. Second, syntax-aware diffing with Difftastic ignores formatting and whitespace changes, reducing the token footprint to only semantically meaningful changes. Intelligent chunking then splits large PRs by module boundaries so each review chunk is coherent.

The common anti-pattern — making repeated API calls for each diff instead of bundling diffs together — is widespread because it maps to the natural tool-use pattern of "read file, analyze, read next file." But this sequential approach wastes context on tool call overhead and creates unnecessary round-trips. Pre-bundling avoids this by preparing the review context before the review begins, following the principle that since [[just-in-time context retrieval via lightweight identifiers outperforms preloading data into context]], preparation and retrieval should be deliberate rather than reactive.

An additional finding from deployment experience: 1000-line diffs overwhelm context regardless of model capability, while small, focused diffs produce consistently useful feedback. This suggests that the optimization should work in both directions — not just pre-bundling efficiently, but also splitting oversized PRs into reviewable chunks. The bundling and splitting are complementary: bundle related changes together, but split when the total exceeds what a model can effectively process.

---

Source: ai-code-review-optimization-multi-agent-architectures-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[just-in-time context retrieval via lightweight identifiers outperforms preloading data into context]] -- deliberate retrieval over reactive tool calls
- [[model cascading routes 70-90 percent of review to cheap models achieving 60-87 percent cost reduction]] -- complementary cost optimization
- [[practical guidance converges on 60-70 percent of maximum context window as effective usable capacity]] -- why oversized diffs fail

Topics:
- [[agent-governance]]
- [[context-management]]
