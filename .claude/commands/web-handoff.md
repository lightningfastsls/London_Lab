# Web Claude Handoff

Generate a context summary for continuing this conversation in web Claude (claude.ai).

The user wants to switch to web Claude (often for Opus access) to have deeper conceptual discussions about topics from this session. Generate a summary that:

1. Provides essential project context
2. Summarizes what was accomplished in this session
3. Highlights the topic/question the user wants to explore deeper
4. Is formatted for easy copy-paste into claude.ai

## What to Include

- **Project context**: Brief description of USV Detection Pipeline, key files if relevant
- **Session summary**: What was discussed/implemented (focus on concepts, not line-by-line changes)
- **Current state**: Where things stand now
- **Topic for discussion**: What the user wants to explore (from $ARGUMENTS or infer from recent context)

## What NOT to Include

- Exact code implementations (unless conceptually important)
- File paths that won't make sense outside this repo
- Internal workflow details (CLAUDE.md rules, agents, etc.)
- Token optimization concerns

## Output Format

```
## Context for Web Claude

**Project**: [1-2 sentence description]

**What we worked on**:
[3-5 bullet points summarizing the session]

**Current state**:
[Brief description of where things stand]

---

**Topic I want to explore**:
[The conceptual topic/question for deeper discussion]

**Relevant background**:
[Any context that helps web Claude understand the topic]
```

If $ARGUMENTS is provided, use that as the topic to explore. Otherwise, infer from recent conversation what conceptual topic would benefit from deeper discussion.

Topic to explore: $ARGUMENTS
