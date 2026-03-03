---
description: "anthropics/claude-code-action v1.0 simplified configuration with automatic mode detection — supports Anthropic API, Amazon Bedrock, Google Vertex AI, and Microsoft Foundry authentication"
type: finding
confidence: proven
created: 2026-03-02
meta_state: current
---

# Claude Code GitHub Actions provides official automated PR review with automatic mode detection responding to mentions assignments and triggers

Anthropic's `anthropics/claude-code-action` is the official GitHub Action for automated PR review and issue handling. v1.0 introduced breaking changes simplifying configuration. The action supports automatic mode detection — it responds to @claude mentions, issue assignments, and configurable automation triggers without manual mode specification.

Authentication supports four providers: Anthropic API (direct), Amazon Bedrock, Google Vertex AI, and Microsoft Foundry — enabling enterprise deployments across cloud environments. A companion action `anthropics/claude-code-security-review` provides security-focused vulnerability scanning.

Some developers have migrated from Cursor's BugBot ($40/user) to Claude Code GitHub Actions, citing better bug detection and lower cost. Claude Code uses 5.5x fewer tokens than Cursor for identical review tasks — a significant efficiency advantage.

Since [[automated code review increases PR closure time by 42 percent despite 74 percent comment acceptance rate]], the mere availability of automated review does not guarantee improved outcomes. The token efficiency advantage matters because since [[context quality over quantity is the defining trend with tools optimizing for less context that matters more not more context that dilutes]], fewer tokens for the same review quality means more context budget available for issue resolution.

---

Source: claude-code-mcp-ecosystem-march-2026-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[automated code review increases PR closure time by 42 percent despite 74 percent comment acceptance rate]] -- the review overhead caveat
- [[context quality over quantity is the defining trend with tools optimizing for less context that matters more not more context that dilutes]] -- why token efficiency matters

Topics:
- [[agent-governance]]
