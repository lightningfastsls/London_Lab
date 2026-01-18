# Plan Mode

Enter planning mode for a new task.

Before implementing anything, create a plan that addresses:

1. **Task Understanding**: What exactly needs to be done?
2. **Reference Documents**: Which docs in this project are relevant?
   - `usv_signal_processing_reference.md` - for signal processing decisions
   - `USV_DETECTION_IMPLEMENTATION_PLAN.md` - for pipeline stages
   - `IMPLEMENTATION_PROGRESS.md` - for current state
3. **Signal Processing Considerations**: Any USV-specific nuances?
4. **Existing Code**: What modules/functions already exist that relate to this?
5. **Implementation Steps**: Ordered list of what to do
6. **Verification**: How will we know it works?
7. **Codex Candidates**: Are there subtasks that could be deferred to Codex?

## Output Format

```markdown
# Plan: [Task Title]

## Understanding
[What needs to be done]

## Relevant References
- [ ] Read: [doc name] - [why needed]

## Signal Processing Notes
[Any relevant constraints or decisions]

## Related Code
- `path/to/file.py` - [what it does]

## Implementation Steps
1. [ ] Step 1
2. [ ] Step 2
...

## Verification
- [ ] How to test this works

## Codex Candidates
- [Tasks that could be handed to Codex]
```

Task: $ARGUMENTS
