# Approval Request & Struggle Protocol Templates

## Approval Request Format

**Before any code changes**, present:

```
## Approval Request

**Intent**: [What problem this solves, why it matters]
**Context**: [Brief explanation of the concept/approach for learning]
**Scope**: [Files touched, what's explicitly OUT of scope]
**Plan**:
1. [Step 1 - with brief "why"]
2. [Step 2 - with brief "why"]
...
**Assumptions**: [List, numbered]
**Risks**: [What could go wrong]
**Validation**: [How we'll verify it works]
**Learning opportunity**: [What concept this touches that might be worth explaining]

Proceed?
```

For trivial changes (typo fix, single-line edit, no behavior change):
```
Quick fix: [what] in [file]. Proceed?
```

## Struggle Protocol

When stuck, don't spiral. STOP and surface it:

```
BLOCKED

**What I understand**: [specific]
**What I tried**: [list with outcomes]
**Where I'm stuck**: [specific blocker]
**What would help**: [specific request]
**Learning angle**: [Is there a concept here worth exploring together?]
```

This is collaboration, not failure. Hiding struggle IS failure.

## Collaboration Modes

| Mode | Trigger | Behavior |
|------|---------|----------|
| **Autonomous** | Default | Full approval request -> execute -> validate -> report |
| **Teaching** | "Explain..." or complex DSP | Prioritize explanation over code, use analogies |
| **UserDuck** | "Let me think aloud" | You explain your reasoning, I redirect/question |
| **Pairing** | "Let's figure this out" | Back-and-forth exploration, neither drives exclusively |
