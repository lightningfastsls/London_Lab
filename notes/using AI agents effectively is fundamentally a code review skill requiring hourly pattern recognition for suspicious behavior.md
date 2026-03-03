---
description: "Goedecke (GitHub) compares AI agents to 3-year engineers — good at producing code, lacking design judgment — about once per hour catching a suspicious action saves hours of wasted effort"
type: pattern
confidence: likely
created: 2026-03-01
meta_state: current
topics:
  - "[[code-review-governance]]"
  - "[[agent-cognition]]"
---

# Using AI agents effectively is fundamentally a code review skill requiring hourly pattern recognition for suspicious behavior

Sean Goedecke (GitHub) makes a provocative reframing: using AI agents correctly is fundamentally a code review process. He compares current AI agents to an engineer with three years of experience — good at producing lots of code but lacking the depth of judgment needed for design decisions and architectural choices.

The key practice insight: about once per hour, he notices an agent doing something suspicious that, when investigated, saves hours of wasted effort. This is pattern recognition applied to agent behavior rather than code — the same skill that makes a good code reviewer (detecting subtle wrongness) makes an effective AI supervisor.

This has implications for how we think about AI agent workflows. The "vibe coding" approach — prompting an agent and accepting its output without review — optimizes for speed but accumulates the same quality problems that since [[AI code generation caused 4x increase in code cloning and first-ever dominance of copy-paste over moved code]]. Effective agent use requires the human to maintain a code-review-like engagement with the agent's work, not just its final output but its intermediate decisions.

The once-per-hour cadence is interesting — it suggests that effective supervision does not require constant monitoring but does require periodic attention. This maps well to the orientation-then-analytical model of review: the human maintains a background awareness of what the agent is doing (orientation) and intervenes when something triggers analytical scrutiny. The skill is in calibrating what triggers scrutiny — too sensitive and the human is constantly interrupting; too insensitive and issues compound.

For practitioners building AI agent workflows, this finding suggests that workflow design should include natural review checkpoints — moments where the human examines agent work before the agent continues. This vault's approval-gate pattern (ANALYSIS -> APPROVAL_PENDING -> EXECUTION) is exactly this: a structural checkpoint that forces human review at the most consequential transition point. Since [[externalized reasoning at approval gates forces agents to improve their plans before executing them]], the checkpoint serves double duty: it enables human oversight AND improves the agent's own planning quality.

---

Source: ai-code-review-optimization-multi-agent-architectures-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[AI code generation caused 4x increase in code cloning and first-ever dominance of copy-paste over moved code]] -- what happens without review engagement
- [[externalized reasoning at approval gates forces agents to improve their plans before executing them]] -- the structural checkpoint pattern
- [[code review follows orientation then analytical phases where skipping orientation degrades analytical quality]] -- the cognitive model that explains the hourly pattern recognition

Topics:
- [[agent-governance]]
- [[agent-cognition]]
