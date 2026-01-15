# New Task

Create a new task folder and draft the task brief.

## Steps
1. Ask for the task title if not provided
2. Run `python tools/new_task.py "<Task Title>"` to create the folder
3. Fill out `tasks/<date>_<slug>/00_task_brief.md` with:
   - **Goal**: What we want to achieve (1-2 sentences)
   - **Scope**: What's included / what's NOT included
   - **Constraints**: Technical or process limitations
   - **Acceptance Criteria**: How we know it's done (checkboxes)
   - **Files to Touch**: List of files that will be modified
   - **Staged Plan**: Break into Stage 1, Stage 2, etc.

## Template
```markdown
# Task: <Title>

## Goal
<What we want to achieve>

## Scope
**In scope:**
- ...

**Out of scope:**
- ...

## Constraints
- ...

## Acceptance Criteria
- [ ] ...
- [ ] ...

## Files to Touch
- `path/to/file.py`

## Staged Plan
### Stage 1: <name>
- ...

### Stage 2: <name>
- ...
```
