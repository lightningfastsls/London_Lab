---
description: "Minimal friction design — directories for skills, agents, commands, hooks, MCP config, and LSP servers with no compilation needed — drove rapid ecosystem growth plus official marketplace with 55-plus Anthropic Verified plugins"
type: finding
confidence: proven
created: 2026-03-02
meta_state: current
topics: "[[agent-memory]]"
---

# Claude Code plugin architecture grew from zero to 9000 plugins in under a year through directory-based design with no build step or registry approval

Claude Code's plugin system uses a directory-based architecture: `.claude-plugin/` (manifest), `skills/`, `agents/`, `commands/`, `hooks/`, `.mcp.json`, `.lsp.json`, `settings.json`. No build step, no compilation, no registry approval required for distribution. This minimal friction drove growth from zero to 9,000+ plugins indexed by the community in under a year, plus 6,500+ Claude Code plugin repositories.

The official marketplace (`anthropics/claude-plugins-official`) hosts 55+ plugins with an "Anthropic Verified" badge for quality/safety-reviewed submissions. Enterprise self-hosted marketplace also available. Submission via `claude.ai/settings/plugins/submit` or `platform.claude.com/plugins/submit`.

Skills are model-invoked (Claude uses them based on context). Hooks are event-driven enforcement. LSP server support adds language intelligence. Settings can set a default agent for the plugin.

The directory-based approach trades curation quality for ecosystem velocity — any directory with the right structure is a valid plugin. This creates a discovery problem (9,000+ plugins with varying quality) that MCP Tool Search partially addresses since [[MCP Tool Search reduces context pollution by 85-95 percent through lazy loading of tool descriptions via lightweight search index]] — at least the token cost of discovery is managed.

This vault's own arscontexta plugin demonstrates the pattern: 16 skills, custom hooks, MCP config, all in a directory structure with no build process.

---

Source: claude-code-mcp-ecosystem-march-2026-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[MCP Tool Search reduces context pollution by 85-95 percent through lazy loading of tool descriptions via lightweight search index]] -- manages the scaling cost of many plugins
- [[8600 MCP servers and 6500 Claude Code plugin repositories exist as of March 2026 reflecting rapid open ecosystem growth]] -- the landscape context

Topics:
- [[agent-memory]]
