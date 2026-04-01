---
paths:
  - "**/*.py"
---
## Rename/Signature Change Protocol
When renaming any function, type, class, or export, search separately for:
1. Direct calls and references
2. Type-level references (interfaces, generics, type annotations)
3. String literals containing the name (routes, logs, test descriptions)
4. Dynamic imports / require() calls
5. Re-exports and barrel files (__init__.py)
6. Test files: mocks, fixtures, monkeypatches
7. Config files (pyproject.toml, setup.cfg)
If results from any search seem suspiciously few, assume truncation and re-run with narrower scope.

## Search Truncation
If any search returns suspiciously few results (<5 when you'd expect >10), assume truncation.
Re-run directory-by-directory. State when you suspect truncation occurred.
