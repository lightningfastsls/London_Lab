---
name: ralph
description: Queue processing with fresh context per phase. Processes N tasks from the queue, spawning isolated subagents to prevent context contamination. Supports serial, parallel, batch filter, and dry run modes. Triggers on "/ralph", "/ralph N", "process queue", "run pipeline tasks".
version: "1.0"
generated_from: "arscontexta-v1.6"
user-invocable: true
context: fork
allowed-tools: Read, Write, Edit, Grep, Glob, Bash, Task
argument-hint: "N [--parallel] [--batch id] [--type extract] [--dry-run] — N = number of tasks to process"
---

[Full ralph SKILL.md content - Queue orchestrator with subagent spawning. See original ralph/SKILL.md for complete documentation.]
