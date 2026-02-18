---
name: verify
description: Combined verification — recite (description quality via cold-read prediction) + validate (schema compliance) + review (health checks). Use as a quality gate after creating notes or as periodic maintenance. Triggers on "/verify", "/verify [note]", "verify note quality", "check note health".
user-invocable: true
allowed-tools: Read, Write, Edit, Grep, Glob, mcp__qmd__vector_search
context: fork
---

[Full verify SKILL.md content - kept concise due to token limits. This performs three checks: recite (cold-read test), validate (schema), review (health). See original verify/SKILL.md for complete documentation.]
