---
description: Deep guide to /arscontexta:ask, /arscontexta:architect, /rethink, and /remember
type: manual
generated_from: "arscontexta-0.8.0"
---
# Meta-Skills

Meta-skills help your system evolve and learn from its own operation.

## /arscontexta:ask -- Query the Research Graph

Ask questions about knowledge management methodology:
- "Why does my system use atomic notes?"
- "How should I handle contradicting findings?"
- "What are the risks of my current configuration?"

Routes through a 3-tier knowledge base: WHY (research claims), HOW (guidance docs), WHAT IT LOOKS LIKE (domain examples).

## /arscontexta:architect -- Evolution Advice

Get research-backed recommendations for system changes:
- Analyzes health reports and friction patterns
- Proposes specific changes with justification
- Never auto-implements -- proposals require your approval

## /rethink -- Review Observations and Tensions

When observations (ops/observations/) or tensions (ops/tensions/) accumulate:
- Triages each item: PROMOTE (to notes/), IMPLEMENT (update CLAUDE.md), ARCHIVE, or KEEP PENDING
- Detects methodology drift
- Resolves or dissolves contradictions

## /remember -- Capture Friction

When something goes wrong or feels off during work:
- Captures the observation in ops/observations/
- Categorizes: friction | surprise | process-gap | methodology
- Automatic detection from session transcripts also available

## Rule Zero: Methodology as Spec

/remember treats CLAUDE.md as the executable specification. When you capture a friction signal, the system checks whether it represents a deviation from spec (fix behavior) or a spec gap (update CLAUDE.md).

See [[configuration]] for config changes.
See [[troubleshooting]] for drift-related issues.
