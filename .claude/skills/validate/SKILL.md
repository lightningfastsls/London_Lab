---
name: validate
description: Schema validation for notes. Checks against domain-specific templates. Validates required fields, enum values, description quality, and link health. Non-blocking — warns but doesn't prevent capture. Triggers on "/validate", "/validate [note]", "check schema", "validate note", "validate all".
user-invocable: true
allowed-tools: Read, Grep, Glob
context: fork
model: sonnet
---

[Full validate SKILL.md content - Schema validation tool. See original validate/SKILL.md for complete documentation.]
