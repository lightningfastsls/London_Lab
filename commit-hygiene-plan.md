# Commit Hygiene Plan

## Context

Solo developer on MHS platform (Next.js monorepo — agent portal, client portal, management dashboard, Cloudy Claude AI layer). Current workflow: work on main, dispatch `/implement` to Claude Code sessions for modules, run review agents before considering work done. Moving toward more granular commits with better messages as the primary improvement — worth more than any tooling addition.

## Goal

Make `git log --oneline` a useful debugging and archaeology tool six months from now. Make every commit a clean rollback target.

---

## Phase 1: Commit Message Convention

Discover the current commit history and establish a convention that fits the project.

### Discovery

- Run `git log --oneline -40` to see current message patterns
- Run `git shortlog -s --all` to see commit volume
- Check if there's an existing convention or if messages are ad-hoc
- Look at the project structure (`ls src/` or equivalent) to understand the module boundaries that should appear in commit scopes

### Convention to implement

Use Conventional Commits with project-specific scopes:

```
<type>(<scope>): <description>

[optional body — the "why", not the "what"]
```

**Types** (keep it to these — more is noise):
- `feat` — new capability visible to users or agents
- `fix` — bug fix
- `refactor` — restructure without behavior change
- `security` — auth, permissions, RLS, input validation
- `data` — Madaf pipeline, ETL, schema migrations
- `config` — env, deployment, CI, dependencies
- `docs` — documentation, comments, READMEs

**Scopes** (discover actual directory/module names from the repo, but expect something like):
- `agent-portal`, `client-portal`, `manager-dashboard` — the three pillars
- `cloudy` — Cloudy Claude AI layer
- `auth` — Supabase Auth, WhatsApp OTP, role boundaries
- `pricing` — pricing engine, agorot calculations, tier logic
- `madaf` — Madaf data pipeline, ETL, sync
- `tecdoc` — TecDoc catalog integration
- `db` — Prisma schema, migrations, RLS policies
- `api` — tRPC routers, shared API logic

**Examples of good messages:**
```
feat(agent-portal): add customer search with phone number lookup
fix(pricing): round agorot before tier comparison, not after
security(auth): restrict agent API routes to own customer list only
data(madaf): handle Windows-1255 encoding in product descriptions
refactor(api): extract shared pagination logic from item routers
config: upgrade Next.js to 15.x, update middleware matcher
```

**The body is for "why":**
```
fix(pricing): use integer division for mid-margin tier calculation

Floating point multiplication was producing 1-agora rounding errors
on orders with 7+ line items. Switched to integer math throughout
the pricing pipeline. See issue with order #4821.
```

### Implementation

- Create or update a `CONTRIBUTING.md` (or a section in the repo's `CLAUDE.md`) with this convention
- The convention should be written so that Claude Code sessions producing commits also follow it — add to CLAUDE.md instructions:
  ```
  ## Commit Messages
  Use Conventional Commits: <type>(<scope>): <description>
  Types: feat, fix, refactor, security, data, config, docs
  Scopes: [list discovered from repo]
  Body (optional): explain WHY, not what. The diff shows what.
  Keep subject line under 72 characters.
  ```

---

## Phase 2: Granularity Guidelines

Add to CLAUDE.md so both you and Claude Code sessions follow the same rules.

### The rule of thumb

**One commit = one reviewable decision.** If you'd explain two different things when describing the commit, it's two commits.

### Specific patterns

**Split these into separate commits:**
- A schema migration and the code that uses it
- A bug fix and a refactor you noticed while fixing the bug
- A new feature and the tests for that feature (debatable — but for this project, tests as a separate commit makes the feature commit's diff cleaner)
- Dependency updates and code changes that use the new dependency

**Keep these as one commit:**
- A component and its direct styles (same logical unit)
- A tRPC router and its Zod input schema (one can't exist without the other)
- An RLS policy and the migration that adds it (they're deployed together)

**Size heuristic:** if a commit touches more than 5-6 files, ask whether it's actually one logical change. Sometimes it is (a rename ripples everywhere). Often it's two changes that happened to be made at the same time.

### For Claude Code sessions specifically

Add to CLAUDE.md:
```
## Commit Granularity
Commit after each logical unit of work, not at the end of the task.
If implementing a feature that touches auth + UI + API, commit each
layer separately. Prefer more smaller commits over fewer large ones.
Never bundle a refactor with a feature in the same commit.
```

---

## Phase 3: Lightweight Automated Enforcement

A git hook that validates commit messages and warns about suspiciously large commits. NOT a blocking gate — just a nudge.

### Discovery

- Check if `.git/hooks/` has any existing hooks
- Check if the project uses husky, lint-staged, or similar tools
- Check the package.json for any existing git hook configuration
- Determine whether husky is already a dependency or if a bare shell hook is simpler

### Implementation: commit-msg hook

Create a `commit-msg` hook that validates the convention. This runs after you write the message but before the commit is finalized.

**What it checks:**
1. Message matches `<type>(<scope>): <description>` or `<type>: <description>` pattern
2. Type is one of the allowed types
3. Subject line is under 72 characters
4. First letter of description is lowercase (convention consistency)

**What it does on failure:**
- Prints a clear warning showing what's wrong and what format is expected
- **Does NOT block the commit** — use a soft warning, not `exit 1`. The hook should print "⚠️ Commit message doesn't match convention" but allow it through. Hard blocks get bypassed with `--no-verify` and then you lose the nudge entirely.

**Exception:** If using commitlint + husky and you prefer hard enforcement, that's fine — but start soft for two weeks to build the habit without friction, then optionally switch to hard mode.

### Implementation: pre-push hook (optional — evaluate after Phase 1-2 are habitual)

A pre-push hook that runs a quick sanity check before code leaves your machine.

**What it checks:**
1. Count of commits being pushed — if more than 10 unpushed commits, print a warning ("You're pushing 14 commits — consider whether these should be reviewed first")
2. Total diff size — if the diff is over 500 lines added, print a summary of files changed as a reminder to eyeball it
3. **Does NOT run the full review agent pipeline** — that's too slow for a push hook and you'll bypass it immediately

**What it does NOT do:**
- Run tests (that's CI's job)
- Run linting (that should be in pre-commit or editor-level)
- Block the push (soft warnings only)

### File placement

- If using husky: hooks go in `.husky/` and are version-controlled (good — Claude Code sessions get them too)
- If bare hooks: they go in `.git/hooks/` and are NOT version-controlled (bad — each clone needs setup)
- **Recommendation:** use husky if the project already has it or is Node-based (which it is). This way the hooks are in the repo and Claude Code sessions running in the repo will also have them active.

---

## Phase 4: First-Week Calibration

After implementing phases 1-3, do a one-week check:

- Run `git log --oneline -20` and evaluate: could you understand what happened this week from the log alone?
- Check if any commits should have been split
- Check if any commit messages are vague ("fix stuff", "updates")
- Adjust scope list if modules have been added or the initial list was wrong
- If the soft warnings from the hook are working, optionally switch to hard enforcement

---

## Execution Order

1. **Phase 1 first** — discover repo structure, write the convention, add to CLAUDE.md
2. **Phase 2 immediately after** — add granularity guidelines to CLAUDE.md
3. **Phase 3 after you've manually followed the convention for a few days** — automate what you've already internalized
4. **Phase 4 one week after Phase 3** — calibrate

Phase 1 + 2 can be done in one session. Phase 3 is a separate session. Phase 4 is a manual check.

---

## What This Does NOT Cover

- PR workflow — intentionally excluded. Evaluate separately after the commit hygiene is habitual. Good commits make PRs more useful if you add them later; bad commits make PRs pointless regardless.
- Full automated review — your existing agent pipeline handles this. The hooks here are just format/size nudges, not code review.
- Branch strategy — you're on main for now. If you move to PRs later, the commit convention and granularity carry over unchanged.
