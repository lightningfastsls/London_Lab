---
name: refactor
description: Plan vault restructuring from config changes. Compares config.yaml against derivation.md, identifies dimension shifts, shows restructuring plan, executes on approval. Triggers on "/refactor", "restructure vault".
version: "1.0"
generated_from: "arscontexta-v1.6"
user-invocable: true
context: fork
allowed-tools: Read, Write, Edit, Grep, Glob, Bash
argument-hint: "[dimension|--dry-run] — focus on specific dimension or preview without approval prompt"
---

[Full refactor SKILL.md content - Configuration-driven restructuring. See original refactor/SKILL.md for complete documentation.]
