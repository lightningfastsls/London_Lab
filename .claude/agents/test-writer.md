---
name: test-writer
description: "Deprecated — use test-architect (before implementation) or test-hardener (after implementation)"
model: sonnet
tools:
  - Read
  - Grep
  - Glob
  - Edit
  - Write
  - Bash
---

# Test Writer (Deprecated)

**This agent has been superseded by two specialized agents:**

- **`test-architect`** — Writes tests BEFORE implementation from ROADMAP specs. Use when a
  module is about to be implemented. The test-architect produces failing tests that define
  the executable specification.
- **`test-hardener`** — Writes adversarial tests AFTER implementation to find coverage gaps.
  Use after a module passes its initial tests and review.

If you need tests for existing code, use `test-hardener`.
If you need tests before building a new module, use `test-architect`.
