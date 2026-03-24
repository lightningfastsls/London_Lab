# Claude Code Setup Plan — For Your Project

This is a step-by-step implementation plan for setting up Claude Code in a new project. **Don't just copy everything from the USV repo** — each project needs different things. Work through this plan with Claude Code itself, and it will help you figure out what you actually need.

---

## Before You Start

Open Claude Code in your project directory and paste this entire file as your first message. Claude will walk you through each phase, asking questions to understand your project before building anything.

---

## Phase 1: Discovery Interview

**Claude: Before building anything, ask the user these questions to understand their needs.**

### Project Understanding
1. What is this project? (language, framework, what it does)
2. How do you run it? (build command, dev server, test command)
3. What's the directory structure? (where's the source code, tests, configs)
4. Are there any environment variables or secrets needed?
5. Is this a solo project or a team project?

### Workflow Understanding
6. What's your biggest friction point right now? (bugs, boilerplate, testing, deployment)
7. Do you want Claude to just write code when asked, or do you want it to explain and get approval first?
8. How careful should Claude be? (move fast vs. safety-first)
9. Do you use git? What's your branching strategy?
10. Do you have CI/CD? What does it run?

### Scope Understanding
11. Do you want just coding help, or also project management / knowledge tracking?
12. Are there domain-specific rules Claude should always follow? (e.g., "always use TypeScript strict mode", "never use ORM raw queries", "all API responses must be typed")
13. Are there common mistakes in your stack that Claude should watch for?
14. Do you have existing documentation, ADRs, or style guides Claude should know about?

**Based on the answers, Claude should recommend which phases below to implement.** Not every project needs every phase.

---

## Phase 2: CLAUDE.md (Everyone Needs This)

Create `CLAUDE.md` in the project root. Structure it based on the discovery answers:

```markdown
# CLAUDE.md

## Project Overview
[What it is, one paragraph]

## Environment Setup
[Exact commands to install, run, test, build]

## Project Structure
[Key directories and what lives where]

## Coding Conventions
[Language-specific rules, naming, patterns]

## Common Mistakes to Avoid
[Stack-specific pitfalls — from question 13]

## Domain Rules
[From question 12 — things Claude must always/never do]
```

### Quality Checklist
- [ ] Can Claude run the project from these instructions alone?
- [ ] Are the test commands exact (copy-paste-able)?
- [ ] Are domain rules specific, not vague? ("use `zod` for validation" not "validate inputs")
- [ ] Is it under 200 lines? (Longer = Claude skims. Be concise.)

---

## Phase 3: Permissions (Everyone Needs This)

Create `.claude/settings.local.json`:

```json
{
  "permissions": {
    "allow": [
      "Read(*)",
      "Edit",
      "Write"
    ],
    "deny": [
      "Read(.env)",
      "Read(.env.*)",
      "Read(**/secrets/**)",
      "Write(.env)",
      "Write(.env.*)"
    ]
  }
}
```

**Claude: Ask the user which tools they use regularly, then add those to the allow list.** Common additions:

| If they use... | Add to allow |
|----------------|-------------|
| git | `"Bash(git *)"` |
| npm/yarn/pnpm | `"Bash(npm *)"`, `"Bash(npx *)"` |
| python/pip | `"Bash(python*)"`, `"Bash(pip *)"`, `"Bash(pytest*)"` |
| GitHub CLI | `"Bash(gh *)"` |
| Docker | `"Bash(docker *)"` |
| Make | `"Bash(make *)"` |
| cargo (Rust) | `"Bash(cargo *)"` |
| go | `"Bash(go *)"` |

---

## Phase 4: Slash Commands (If They Repeat Workflows)

**Claude: Ask the user:**
- What do you do over and over? (commit, deploy, run specific tests, generate boilerplate)
- Are there multi-step workflows you wish were one command?

Create `.claude/commands/<name>.md` for each repeated workflow.

### Starter Commands (Almost Everyone Wants These)

**`/commit`** — Smart commit with message generation:
```markdown
# Commit Changes

1. Run `git status` and `git diff` to understand what changed
2. Draft a concise commit message (what and why, not how)
3. Stage relevant files (NEVER use `git add -A` without reviewing)
4. Create the commit
5. Show the result
```

**`/test`** — Run tests with context:
```markdown
# Run Tests

1. Run the full test suite: [INSERT EXACT COMMAND]
2. If failures: read the failing test file and the source it tests
3. Explain what failed and why
4. Suggest a fix
```

**`/deploy`** (if applicable):
```markdown
# Deploy

1. Run tests first — abort if any fail
2. [INSERT DEPLOY STEPS]
3. Verify deployment succeeded
```

---

## Phase 5: Custom Agents (If They Want Code Review)

**Claude: Ask the user:**
- Do you want an independent reviewer to check your code before committing?
- Are there specialized domains in the project? (e.g., security, performance, accessibility, database queries)
- What are the most common bugs or issues in code review?

Create `.claude/agents/<name>.md` for each specialist.

### Starter Agent: Code Reviewer

```markdown
---
name: reviewer
description: Reviews code changes for bugs, security issues, and style violations
tools: Read, Grep, Glob
---

# Code Reviewer

You are reviewing code in [PROJECT NAME]. You have fresh context — you haven't
seen the implementation happen.

## Check for:
1. **Bugs** — logic errors, off-by-one, null handling, race conditions
2. **Security** — injection, auth bypass, exposed secrets, OWASP top 10
3. **[DOMAIN-SPECIFIC]** — [from discovery question 12]
4. **Tests** — are edge cases covered? Are tests actually asserting the right things?

## Report format:
- **BLOCKER** — must fix before merging
- **WARNING** — should fix soon
- **SUGGESTION** — nice to have

For each finding: what, where (file:line), why it matters, suggested fix.
```

**Claude: Adapt the checklist based on the user's stack and domain rules.**

---

## Phase 6: Hooks (If They Want Guardrails)

**Claude: Ask the user:**
- Should Claude be allowed to write code without your approval? (If no → add a PreToolUse hook)
- Do you want session context when you start? (If yes → add a SessionStart hook)
- Are there files/directories that need special treatment when modified? (If yes → add PostToolUse hooks)

### Hook Ideas by Need

**"I want approval before code changes"** — PreToolUse hook on Edit/Write that checks for approval.

**"I want to see project status when I start"** — SessionStart hook that runs `git status`, shows recent commits, checks for failing tests, or reads a status file.

**"I want auto-formatting after edits"** — PostToolUse hook on Edit/Write that runs the project's formatter (prettier, black, rustfmt, etc.).

**"I want auto-testing after edits"** — PostToolUse hook on Edit/Write that runs tests (use `async: true` so it doesn't block).

### Example: Auto-Format Hook (settings.local.json)

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [{
          "type": "command",
          "command": "npx prettier --write $CLAUDE_FILE_PATH || true"
        }]
      }
    ]
  }
}
```

### Platform Note

- **Mac/Linux**: Hook commands run in bash directly
- **Windows**: You may need `.cmd` wrappers that call PowerShell (see the USV repo's `.claude/hooks/` for examples)

---

## Phase 7: AGENTS.md (If They Use Subagents)

**Only needed if Phase 5 created agents.** Copy the essential parts of CLAUDE.md into AGENTS.md so subagents also know the project rules. Focus on:

- Project overview
- Environment setup
- Coding conventions
- Domain rules

You do NOT need to copy workflow rules (approval state machine, etc.) — those are for the main session.

---

## Phase 8: Output Style (Personal Preference)

**Claude: Ask the user:**
- Do you want explanations with your code, or just the code?
- Are you learning this stack, or are you an expert who just wants speed?

Add to `settings.local.json`:

| Style | Best for |
|-------|----------|
| `"Explanatory"` | Learning, unfamiliar codebases, educational projects |
| `"Concise"` | Experienced devs who want speed |
| (default) | Balanced — no setting needed |

```json
{
  "outputStyle": "Explanatory"
}
```

---

## Phase 9: Knowledge Management (Only If They Need It)

**This is advanced. Most projects don't need this.** It's useful for:
- Long-running research projects
- Projects with many architectural decisions to track
- Solo devs who work across many sessions and lose context

**Claude: Ask the user:**
- Do you lose context between sessions? (If yes, start with just auto-memory)
- Do you need to track research, decisions, or domain knowledge? (If yes, consider arscontexta)
- How long will this project last? (If < 1 month, probably just use auto-memory)

### Level 1: Auto-Memory Only

Just use Claude Code's built-in memory. Tell Claude "remember that we always use X" and it saves to `~/.claude/projects/<hash>/memory/MEMORY.md`. No setup needed.

### Level 2: Session Continuity

Add a SessionStart hook that reads a status file, and a Stop hook that writes one. This gives Claude "where we left off" context.

### Level 3: Full Knowledge Graph (arscontexta)

Run `/arscontexta:setup` to scaffold a complete knowledge system. This is the full Zettelkasten setup with topic maps, processing pipelines, and 22 skills. Only do this if you have genuine knowledge management needs.

---

## Summary: What to Build Based on Project Type

| Project Type | Recommended Phases |
|-------------|-------------------|
| Quick script / small tool | 2 (CLAUDE.md) + 3 (Permissions) |
| Web app (solo) | 2 + 3 + 4 (Commands) + 5 (Reviewer agent) |
| Web app (team) | 2 + 3 + 4 + 5 + 7 (AGENTS.md) |
| Learning project | 2 + 3 + 8 (Explanatory style) |
| Long-running research | 2 + 3 + 4 + 5 + 6 (Hooks) + 9 (Knowledge mgmt) |
| Safety-critical / careful work | 2 + 3 + 5 + 6 (approval hooks) + 7 |

---

## How to Use This Plan

1. Open Claude Code in your project
2. Paste this file as your first message
3. Claude will start with the Discovery Interview (Phase 1)
4. Based on your answers, Claude will recommend which phases to implement
5. Work through each recommended phase together
6. Skip what you don't need — you can always add more later
