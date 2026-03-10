---
name: seed
description: Add a source file to the processing queue. Checks for duplicates, creates archive folder, moves source from inbox, creates extract task, and updates queue. Triggers on "/seed", "/seed [file]", "queue this for processing".
version: "1.0"
generated_from: "arscontexta-v1.6"
user-invocable: true
context: fork
model: sonnet
allowed-tools: Read, Write, Edit, Grep, Glob, Bash
argument-hint: "[file] — path to source file to seed for processing"
---

[Full seed SKILL.md content - Entry point for processing pipeline. See original seed/SKILL.md for complete documentation.]
