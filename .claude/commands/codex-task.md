# Codex Task Handoff

Generate a task description for OpenAI Codex.

The user wants to hand off a task to Codex. Based on the current context and the task described in $ARGUMENTS, generate:

1. A clear, self-contained prompt for Codex
2. Any file paths Codex will need to reference
3. Expected output or acceptance criteria

Format the output so the user can copy-paste it directly to Codex.

## Output Format

```
CODEX TASK: [task title]

CONTEXT:
- Project: USV Detection Pipeline (Python)
- Working directory: [path]

FILES TO REFERENCE:
- [list relevant file paths]

TASK:
[Clear, detailed instructions]

EXPECTED OUTPUT:
- [List what Codex should produce]

ACCEPTANCE CRITERIA:
- [ ] [Checklist items]
```

Task to hand off: $ARGUMENTS
