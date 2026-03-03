---
description: "MEMORY.md loads only first 200 lines at session start with no built-in deduplication or decay — after 10 sessions approximately 30 percent of entries are redundant requiring manual editing"
type: finding
confidence: proven
created: 2026-03-02
meta_state: current
topics: "[[agent-memory]]"
---

# auto-memory 200-line hard-coded limit and lack of automatic consolidation creates growing redundancy without manual intervention

Claude Code's auto-memory stores to `~/.claude/projects/<project>/memory/MEMORY.md` with optional topic files (e.g., `debugging.md`, `patterns.md`). Only the first 200 lines of MEMORY.md load at session start — this is hard-coded (`var U_ = 'MEMORY.md', pZ = 200`). Topic files load on-demand when Claude reads them.

After approximately 10 sessions, MEMORY.md typically contains ~30% redundant entries. There is no automatic consolidation, decay, or pruning. The user must manually edit or explicitly instruct Claude to reorganize. This contrasts sharply with MCP memory servers that implement autonomous consolidation — since [[dream-inspired consolidation cycles compress old memories on daily weekly monthly schedules to manage long-term growth]], the problem of growing redundancy has known solutions.

The 200-line limit creates an implicit token budget (~2000-4000 tokens depending on line density). This is both a feature and a constraint: it prevents runaway context consumption but forces information loss when memory exceeds the budget. Subagents can maintain their own auto-memory since early 2026, and memory is scoped per git repository (all worktrees share one memory directory), but the system remains machine-local with no team sharing capability.

---

Source: mcp-memory-servers-cross-session-knowledge-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[Claude Code auto-memory captures configuration not learning because it preserves workspace patterns but loses diagnostic reasoning paths]] -- the qualitative limitation
- [[context compaction quality degrades cumulatively with multiple compressions regardless of implementation]] -- the 200-line limit forces implicit compaction

Topics:
- [[agent-memory]]
